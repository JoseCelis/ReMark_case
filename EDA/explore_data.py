import os
import sys
import logging
import pandas as pd

sys.path.append(os.getcwd())
from src.utils import read_config

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', None)


def check_variables(data, report):
    """
    Check the dtypes of the variables, and describe them
    :param data:
    :param report:
    :return:
    """
    text = 'Checking the variables types in raw_data\n'
    text += str(data.dtypes)
    text += '\n\nChecking some statistics of the variables\n'
    text += str(data.describe(include='all'))
    text += '\n\ncounting the number of null values per column'
    text += str(data.isna().sum().to_frame().rename(columns={0: 'N null values'}))
    text += '\n\n'
    report.write(text)
    return None


def check_duplicates(data, report):
    """
    Check the duplicates iin the table
    :param data:
    :param report:
    :return:
    """
    text = 'Checking for dupliicated rows in raw_data\n'
    text += str(data.duplicated().sum())
    text += '\n\nData length with and without duplicates\n'
    text += str(f'{len(data)}, {len(data.drop_duplicates())}')
    text += '\n\ncounting the number of null values per column'
    text += str(data.isna().sum().to_frame().rename(columns={0: 'N null values'}))
    text += '\n\n'
    report.write(text)
    return None


def check_possible_options(data, report, target_col):
    """
    Compares the number of rows before and after removing null values
    :param data:
    :param analysis_report:
    :return:
    """
    text = 'Deciding what to do with the empty cells:\n'
    text += f'initial length of data file {len(data)}\n'
    text += f'length of data file after dropping rows with null values {len(data.dropna(axis=0))}\n'
    text += f'initial distributon of the target variable \n{data[target_col].value_counts()}\n'
    text += (f'distributon of the target variable after dropping rows wth null values \n'
             f'{data.dropna(axis=0)[target_col].value_counts()}\n')
    text += str(data.dropna(axis=0).describe(include='all'))
    report.write(text)
    return None


def main():
    config = read_config(['eda'])
    input_file = os.path.join(config['input']['path'], config['input']['file_name'])
    target_col = config['target_col']

    data = pd.read_csv(input_file, encoding='utf8')
    os.makedirs(config['output']['path'], exist_ok=True)
    report_file = os.path.join(config["output"]["path"], config["output"]["report"])

    analysis_report = open(report_file, 'w')
    check_variables(data, analysis_report)
    check_duplicates(data, analysis_report)
    check_possible_options(data, analysis_report, target_col)
    analysis_report.close()
    logging.info(f'detailed explanation of the analysis can be found in file {report_file}')


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    main()
