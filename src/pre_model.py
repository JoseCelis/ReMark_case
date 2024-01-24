import os
import sys
import mlflow
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg', force=True)
from dotenv import load_dotenv, find_dotenv
from pycaret.classification import ClassificationExperiment
# from pycaret.regression import RegressionExperiment
load_dotenv(find_dotenv())

from category_encoders.ordinal import OrdinalEncoder


sys.path.append(os.getcwd())
from src.utils import read_config

random_seed = 123
np.random.seed(random_seed)

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', None)


def look_for_best_model(data, target_col, numerical_columns, categorical_columns, experiment, features_perc,
                        analysis_report):
    encoding_method = OrdinalEncoder(handle_missing="return_nan", handle_unknown="value")
    experiment.setup(data,
                     target=target_col,
                     session_id=random_seed,
                     encoding_method=encoding_method,
                     feature_selection=True,
                     n_features_to_select=features_perc,
                     verbose=False,
                     categorical_features=categorical_columns,
                     numeric_features=numerical_columns,
                     fold=5,
                     # experiment_name='batch1', log_experiment=True,
                     )
    best_model = experiment.compare_models()
    text = f'\n\nfeatures_perc {features_perc}\n'
    text += str(experiment.pull())
    analysis_report.write(text)
    return best_model


def run_pipeline(data, target_col, numerical_columns, categorical_columns, analysis_report):
    for features_perc in [x / 10.0 for x in range(1, 2)]:
        # experiment = RegressionExperiment()
        experiment = ClassificationExperiment()
        best_model = look_for_best_model(data, target_col, numerical_columns, categorical_columns, experiment,
                                         features_perc, analysis_report)
    # os.makedirs('models', exist_ok=True)
    # experiment.save_model(best_model, 'looking_for_best_model')


def main():
    config = read_config(['model'])
    input_file = os.path.join(config['input']['path'], config['input']['file_name'])

    data = pd.read_parquet(input_file)
    # set mlflow tracking uri
    # mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    # mlflow.set_tracking_uri("http://localhost:5000")
    target_col = config['target_column']
    float_columns = data.select_dtypes(include=['float64']).columns
    int_columns = data.select_dtypes(include=['int']).columns
    cat_columns = data.select_dtypes(include=['category']).columns
    numerical_columns = list(float_columns[float_columns != target_col])
    categorical_columns = list(cat_columns[cat_columns != target_col]) + list(int_columns)

    os.makedirs(config["output"]["path"], exist_ok=True)
    report_file = os.path.join(config["output"]["path"], config["output"]["report"])
    analysis_report = open(report_file, 'w')
    run_pipeline(data, target_col, numerical_columns, categorical_columns, analysis_report)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
