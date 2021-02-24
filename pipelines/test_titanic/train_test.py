from typing import NamedTuple

from kfp.components import InputPath, OutputPath, InputBinaryFile, OutputBinaryFile


def download_data(data_url, data_path: OutputPath()):
    import pandas as pd
    df = pd.read_csv(data_url)
    df.to_csv(data_path, index=False)


def preprocess_data(
    input_data_path: InputPath(str),
    labels_file: OutputBinaryFile(bytes),
    features_file: OutputBinaryFile(bytes),
):
    import pandas as pd
    df = pd.read_csv(input_data_path)

    import numpy as np
    labels = df['survived'].values
    np.save(labels_file, labels)

    features = df[['age', 'n_siblings_spouses', 'parch', 'fare']].values
    np.save(features_file, features)


def training(
    train_labels_file: InputBinaryFile(bytes),
    train_features_file: InputBinaryFile(bytes),
    trained_model_file: OutputBinaryFile(bytes),
):
    import numpy as np
    labels = np.load(train_labels_file)
    features = np.load(train_features_file)

    from sklearn import svm
    model = svm.SVC()
    model.fit(features, labels)

    import pickle
    pickle.dump(model, trained_model_file)


def testing(
    trained_model_file: InputBinaryFile(bytes),
    features_file: InputBinaryFile(bytes),
    predicted_file: OutputBinaryFile(bytes),
):
    import pickle
    model = pickle.load(trained_model_file)

    import numpy as np
    features = np.load(features_file)

    predicted = model.predict(features)
    np.save(predicted_file, predicted)


def evaluate_prediction(
    predicted_file: InputBinaryFile(bytes),
    labels_file: InputBinaryFile(bytes),
) -> NamedTuple(
    'EvaluateResultOutput',
    [
        ('mlpipeline_ui_metadata', 'UI_metadata'),
        ('mlpipeline_metrics', 'Metrics'),
    ],
):
    '''Evaluate test results.
    Reference: https://github.com/kubeflow/pipelines/issues/2395#issuecomment-543984161
    '''
    import numpy as np
    predicted = np.load(predicted_file)
    true_labels = np.load(labels_file)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(predicted, true_labels)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    cm = confusion_matrix(true_labels, predicted)
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((target_index, predicted_index, count))
    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    vocab = list(df_cm['target'].unique())
    cm_string = df_cm.to_csv(columns=['target', 'predicted', 'count'], header=False, index=False)

    import json
    metrics = {
        'metrics': [{
            'name': 'accuracy-score',
            'numberValue':  accuracy,
            'format': 'PERCENTAGE',
        }]
    }

    metadata = {
        'outputs' : [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'storage': 'inline',
            'source': cm_string,
            'labels': list(map(str, vocab)),
        }]
    }

    from collections import namedtuple
    output = namedtuple(
        'EvaluateResultOutput',
        [
            'mlpipeline_ui_metadata',
            'mlpipeline_metrics',
        ],
    )
    return output(json.dumps(metadata), json.dumps(metrics))
