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
    '''
    Reference: https://towardsdatascience.com/predicting-the-survival-of-titanic-passengers-30870ccc7e8
    '''
    import pandas as pd
    df = pd.read_csv(input_data_path)

    import numpy as np
    labels = df['survived'].values
    np.save(labels_file, labels)

    df_features = df[[
        'sex',
        'age',
        'n_siblings_spouses',
        'parch',
        'fare',
        'class',
        'deck',
        'embark_town',
        'alone',
    ]].copy()

    # Preprocess sex column
    genders = {'male': 0, 'female': 1}
    df_features['sex'] = df_features['sex'].map(genders)

    # Preprocess age column
    df_features['age'] = df_features['age'].astype(int)
    df_features.loc[ df_features['age'] <= 11, 'age'] = 0
    df_features.loc[(df_features['age'] > 11) & (df_features['age'] <= 18), 'age'] = 1
    df_features.loc[(df_features['age'] > 18) & (df_features['age'] <= 22), 'age'] = 2
    df_features.loc[(df_features['age'] > 22) & (df_features['age'] <= 27), 'age'] = 3
    df_features.loc[(df_features['age'] > 27) & (df_features['age'] <= 33), 'age'] = 4
    df_features.loc[(df_features['age'] > 33) & (df_features['age'] <= 40), 'age'] = 5
    df_features.loc[(df_features['age'] > 40) & (df_features['age'] <= 66), 'age'] = 6
    df_features.loc[ df_features['age'] > 66, 'age'] = 6

    # Preprocess class column
    classes = {'First': 1, 'Second': 2, 'Third': 3}
    df_features['class'] = df_features['class'].map(classes)

    # Preprocess class deck
    deck = {'unknown': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'U': 8}
    df_features['deck'] = df_features['deck'].map(deck)
    df_features['deck'] = df_features['deck'].fillna(0)

    # Preprocess class embark_town
    embark_town = {'Southampton': 1, 'Queenstown': 2, 'Cherbourg': 3}
    df_features['embark_town'] = df_features['embark_town'].map(embark_town)
    df_features['embark_town'] = df_features['embark_town'].fillna(0)

    # Preprocess class alone
    alone = {'y': 1, 'n': 0}
    df_features['alone'] = df_features['alone'].map(alone)

    print(df_features)

    np.save(features_file, df_features.values)


def training(
    train_labels_file: InputBinaryFile(bytes),
    train_features_file: InputBinaryFile(bytes),
    trained_model_file: OutputBinaryFile(bytes),
):
    import numpy as np
    labels = np.load(train_labels_file)
    features = np.load(train_features_file)

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
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
