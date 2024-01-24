import os
import re
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
sys.path.append(os.getcwd())
from src.utils import read_config

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', None)


def drop_duplicated_rows(data):
    """
    drop duplicated rows in raw data
    :param data:
    :return:
    """
    data = data.drop_duplicates()
    return data


def detect_and_fill_empty_cells(data):
    """
    check for null values in object columns and fill the
    :param data:
    :return:
    """
    object_columns = data.select_dtypes('object').columns
    data[object_columns] = data[object_columns].replace(' ', np.nan).replace('', np.nan).replace('|', np.nan)
    if data.isna().any().any():
        logging.info(f'{data.isna().sum()} null values detected.')
        # iin the EDA I noticed the distributions do not change after dropping nulls
        data = data.dropna(axis=0)
    else:
        logging.info(f'No null values detected.')
    return data


def modify_type_of_columns(data, to_cat_cols=None, to_float_cols=None, to_int_cols=None, to_str_cols=None,
                           to_date_time_cols=None):
    """
    convert type of columns on request
    :return:
    """
    logging.info(f'some of the variables type will be modified on request')
    formated_data = data.copy()
    if to_cat_cols:
        formated_data.loc[:, to_cat_cols] = data[to_cat_cols].astype('category')
    if to_float_cols:
        for col in to_float_cols:
            formated_data[col] = data[col].str.replace(',', '').str.extract(r'(\d+[.\d]*)').astype(float)
    if to_int_cols:
        for col in to_int_cols:
            formated_data[col] = data[col].str.replace(',', '').str.extract(r'(\d+[.\d]*)').astype(int)
    if to_str_cols:
        formated_data.loc[:, to_str_cols] = data[to_str_cols].astype(str)
    if to_date_time_cols:
        for col in to_date_time_cols:
            formated_data.loc[:, col] = pd.to_datetime(data[col])
    return formated_data


def process_target_column(data, target_column, target_map):
    """
    converts the target column to cat
    :param data:
    :param target_column:
    :param target_map:
    :return:
    """
    logging.info(f'The target column {target_column} will be converted to categorical')
    data[target_column] = data[target_column].astype(float)
    data[target_column] = np.where(data[target_column] < 3.0, 3.0, data[target_column])
    data[target_column] = data[target_column].round(decimals=0).astype(int).astype('category')
    return data


def process_text_column(data, text_column):
    """
    Vectorize the text column
    :param data:
    :return:
    """
    count_vect = CountVectorizer(max_df=0.90, min_df=0.02, stop_words='english')
    category_counts = count_vect.fit_transform(data[text_column])
    vect_text = pd.DataFrame(category_counts.toarray(), columns=count_vect.get_feature_names_out(), index=data.index)
    vect_text = (vect_text > 0).astype(int)
    data.drop(columns=[text_column], inplace=True)
    data = pd.merge(vect_text, data, left_index=True, right_index=True)
    return data


def process_category_column(data, column):
    cat_df = data[column].str.split('|', expand=True).copy()
    cat_df.columns = 'catlevel' + cat_df.columns.astype(str)
    cat_df = cat_df.replace('&|,', '', regex=True)
    cat_df = cat_df.astype('category')
    data.drop(columns=column, inplace=True)
    data = pd.merge(cat_df, data, left_index=True, right_index=True)
    return data


def process_raw_data(data, config):
    target_col = config['target_col']
    drop_columns = config['drop_columns']
    target_map = config['target_map']

    data.drop(columns=drop_columns, inplace=True)
    data = drop_duplicated_rows(data)
    data = detect_and_fill_empty_cells(data)
    data = process_target_column(data, target_col, target_map)
    data = modify_type_of_columns(data, to_float_cols=['discounted_price', 'actual_price', 'discount_percentage',
                                                       'rating_count'])
    data = process_text_column(data, text_column='product_name')
    data = process_category_column(data, 'category')
    data = data.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    return data


def main():
    config = read_config(['preprocess'])
    input_file = os.path.join(config['input']['path'], config['input']['file_name'])
    output_file = os.path.join(config['output']['path'], config['output']['file_name'])

    data = pd.read_csv(input_file)
    data = process_raw_data(data, config)
    # parquet format preserves the dtypes unlike csv format
    os.makedirs(config['output']['path'], exist_ok=True)
    data.to_parquet(output_file)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
