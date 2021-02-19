import os

from kfp import dsl
from kfp.components import func_to_container_op
from dotenv import load_dotenv

from kubeflow_study.utils.client import get_kfp_client
from pipelines.test_titanic.train_test import download_data
from pipelines.test_titanic.train_test import preprocess_data
from pipelines.test_titanic.train_test import training
from pipelines.test_titanic.train_test import testing


# Load .env
load_dotenv()

download_data_op = func_to_container_op(
    download_data,
    packages_to_install=['pandas'],
)

preprocess_data_op = func_to_container_op(
    preprocess_data,
    packages_to_install=['pandas'],
)

training_op = func_to_container_op(
    training,
    packages_to_install=['scikit-learn', 'numpy'],
)

testing_op = func_to_container_op(
    testing,
    packages_to_install=['scikit-learn', 'numpy'],
)


@dsl.pipeline(name='Titanic Pipeline')
def train_and_test_pipeline():
    TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
    download_train_data_task = download_data_op(
        TRAIN_DATA_URL,
    )
    preprocess_train_data_task = preprocess_data_op(
        input_data=download_train_data_task.output,
    ).after(download_train_data_task)
    training_task = training_op(
        train_labels=preprocess_train_data_task.outputs['labels'],
        train_features=preprocess_train_data_task.outputs['features'],
    ).after(preprocess_train_data_task)

    TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
    download_test_data_task = download_data_op(
        TEST_DATA_URL,
    )
    preprocess_test_data_task = preprocess_data_op(
        input_data=download_test_data_task.output,
    ).after(download_test_data_task)
    testing_task = testing_op(
        trained_model=training_task.outputs['trained_model'],
        features=preprocess_test_data_task.outputs['features']
    ).after(training_task, preprocess_test_data_task)


client = get_kfp_client(
    host=os.getenv('KUBEFLOW_HOST'),
    username=os.getenv('KUBEFLOW_USERNAME'),
    namespace=os.getenv('KUBEFLOW_NAMESPACE'),
)
client.create_run_from_pipeline_func(
    train_and_test_pipeline,
    arguments={},
    experiment_name=os.getenv('KUBEFLOW_EXPERIMENT_NAME'),
)
