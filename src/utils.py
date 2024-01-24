import yaml
import operator
from functools import reduce


def read_config(key_list=None):
    """
    read the configs/settings.yaml file located in
    :param key_list:
    :return:
    """
    with open('configs/settings.yaml', 'r') as yaml_file:
        yaml_dict = yaml.load(yaml_file, Loader=yaml.FullLoader)
    if key_list:
        return reduce(operator.getitem, key_list, yaml_dict)
    else:
        return yaml_dict
