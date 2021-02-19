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


def calculate_accuracy(
    predicted_file: InputBinaryFile(bytes),
    labels_file: InputBinaryFile(bytes),
) -> float:
    import numpy as np
    predicted = np.load(predicted_file)
    true_labels = np.load(labels_file)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(predicted, true_labels)
    return accuracy
