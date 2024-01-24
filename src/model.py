import os
import sys
import mlflow
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv, find_dotenv
from category_encoders.ordinal import OrdinalEncoder
from sklearn.model_selection import train_test_split
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
load_dotenv(find_dotenv())

sys.path.append(os.getcwd())
from src.utils import read_config

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', None)


random_seed = 123
np.random.seed(random_seed)


def split_data_into_train_test(data, target_col):
    train_data, test_data = train_test_split(data, random_state=random_seed, test_size=0.3, stratify=data[target_col])

    return train_data, test_data


def train_model(experiment, train_data, test_data, target_col, numerical_columns, categorical_columns, features_perc):
    encoding_method = OrdinalEncoder(handle_missing="return_nan", handle_unknown="value")
    experiment.setup(train_data,
                     target=target_col,
                     session_id=random_seed,
                     encoding_method=encoding_method,
                     feature_selection=True,
                     fix_imbalance=True,
                     n_features_to_select=features_perc,
                     numeric_features=numerical_columns,
                     categorical_features=categorical_columns,
                     data_split_stratify=[target_col],
                     fold=5,
                     verbose=False)
    boost_model = experiment.create_model('catboost')
    # tuned_model = experiment.tune_model(boost_model, search_library='optuna', search_algorithm='tpe')
    tuned_model = experiment.tune_model(boost_model)
    return tuned_model


def evaluate_metrics(data, target_col, output_path, name=''):
    """
    Calculate metrics
    :param data:
    :param target_col:
    :param output_path:
    :param name:
    :return:
    """
    score = accuracy_score(y_true=data[target_col], y_pred=data['prediction_label'])
    print(f'Accuracy score for {name}:\n', score)
    report = classification_report(y_true=data[target_col], y_pred=data['prediction_label'])
    print(f'Classification report for {name} results:\n', report)
    cm = confusion_matrix(y_true=data[target_col], y_pred=data['prediction_label'])
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp_cm.plot()
    plt.savefig(os.path.join(output_path, f'{name}_results.png'))
    return None


def main():
    config = read_config(['model'])
    input_file = os.path.join(config['input']['path'], config['input']['file_name'])
    output_path = config['output']['path']
    data = pd.read_parquet(input_file)
    feats_perc = config['feats_perc']

    # set mlflow tracking uri
    # mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    target_col = config['target_column']
    float_columns = data.select_dtypes(include=['float64']).columns
    int_columns = data.select_dtypes(include=['int']).columns
    cat_columns = data.select_dtypes(include=['category']).columns
    numerical_columns = list(float_columns[float_columns != target_col])
    categorical_columns = list(cat_columns[cat_columns != target_col]) + list(int_columns[int_columns != target_col])

    train_data, test_data = split_data_into_train_test(data, target_col)
    # experiment = RegressionExperiment()
    experiment = ClassificationExperiment()
    tuned_model = train_model(experiment, train_data, test_data, target_col, numerical_columns, categorical_columns,
                              feats_perc)
    experiment.save_model(tuned_model, f'tuned_model')
    predicted_target_train = experiment.predict_model(tuned_model, data=train_data)
    predicted_target_test = experiment.predict_model(tuned_model, data=test_data)

    os.makedirs(output_path, exist_ok=True)
    evaluate_metrics(predicted_target_train, target_col, output_path, name='train')
    evaluate_metrics(predicted_target_test, target_col, output_path, name='test')

    experiment.interpret_model(tuned_model)
    experiment.plot_model(tuned_model, plot='feature')


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
