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
from pycaret.classification import *
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
load_dotenv(find_dotenv())

sys.path.append(os.getcwd())
from src.utils import read_config


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

    experiment = ClassificationExperiment()
    experiment.setup(data=data, target=target_col)
    tuned_model = experiment.load_model('tuned_model')
    #experiment.interpret_model(tuned_model)
    experiment.plot_model(tuned_model, plot='feature')


if __name__ == '__main__':
    logging.basicConfig()

    logging.getLogger().setLevel(logging.INFO)
    main()
