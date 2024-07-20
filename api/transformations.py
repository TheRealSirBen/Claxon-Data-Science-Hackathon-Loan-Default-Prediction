import sys
from os import environ

import dill
from numpy import concatenate
from numpy import inf as np_inf
from numpy import linspace
from numpy import log
from numpy import nan as nan_value
from pandas import DataFrame
from pandas import concat
from pandas import cut
from pandas import to_datetime
from pandas import to_numeric
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from _init_ import start_app

start_app()


def get_steps() -> list[dict]:
    # Load the pickle file
    pickle_file_name = '{}/{}'.format(environ.get('TRANSFORMATIONS_DIR'), environ.get('TRANSFORMATIONS_MODULE'))
    with open(pickle_file_name, 'rb') as f:
        transformation_module = dill.load(f)
    f.close()

    transformations = transformation_module.transformations
    transformations_steps: list[dict] = list()
    for func, args, kwargs in transformations:
        transformations_steps.append({'name': func.__name__, 'kwargs': kwargs})

    return transformations_steps


def transform_data(df: DataFrame) -> DataFrame:
    transformations_steps = get_steps()

    transformation_module = TransformationModule()
    for transformation in transformations_steps:
        func_name = transformation.get('name')
        kwargs = transformation.get('kwargs', {})

        # Find the function in the global namespace
        transformation_func = globals().get(func_name)

        if transformation_func is None:
            # If not found in globals, check in the current module
            transformation_func = getattr(sys.modules[__name__], func_name, None)

        if transformation_func is None:
            raise ValueError(f"Transformation function '{func_name}' not found")

        transformation_module.add_transformation(transformation_func, **kwargs)

    df = transformation_module.apply_transformations(df)
    return df


class TransformationModule:
    def __init__(self):
        self.transformations = []

    def add_transformation(self, func, *args, **kwargs):
        """
        Add a transformation function to the list of transformations.
        """
        self.transformations.append((func, args, kwargs))

    def apply_transformations(self, _df: DataFrame):
        """
        Apply all stored transformations to the DataFrame in order.
        """
        for func, args, kwargs in self.transformations:
            print('\nApplying transformation: {}'.format(func.__name__))
            print('Before transformation: {}'.format(_df.shape))
            _df = func(_df, *args, **kwargs)
            print('After transformation: {}'.format(_df.shape))
        return _df

    def __getstate__(self):
        """
        Called when pickling the object. Ensures all attributes are picklable.
        """
        return self.__dict__

    def __setstate__(self, state):
        """
        Called when unpickling the object. Restores the object's state.
        """
        self.__dict__.update(state)


def find_duplicate_columns_list(_df: DataFrame, identity_perc_threshold: float) -> list[tuple]:
    number_of_rows = _df.shape[0]
    _columns_list = _df.columns.to_list()

    duplicate_set_list: list = list()

    # Iterate through all columns
    for _anchor_column in _columns_list:

        # Create a check column list which excludes the anchor column
        check_columns = [column for column in _columns_list if column != _anchor_column]

        # Iterate through all check columns
        for check_column in check_columns:

            # Test whether anchor and check column are identical
            df_check = _df[_anchor_column] == _df[check_column]
            identical_rows = df_check.sum()

            # When the proportion of number of identical rows is at least the threshold
            if (identical_rows / number_of_rows) * 100 >= identity_perc_threshold:
                duplicate_set_list.append((_anchor_column, check_column))

    return duplicate_set_list


def drop_dataframe_columns(_df: DataFrame, drop_column_list: list) -> DataFrame:
    _df = _df.drop(columns=drop_column_list)
    return _df


def convert_from_text_to_date(_df: DataFrame, _columns_list: list) -> DataFrame:
    for _column in _columns_list:
        _df[_column] = to_datetime(_df[_column])

    return _df


def convert_from_text_to_numeric(_df: DataFrame, _columns_list: list) -> DataFrame:
    for _column in _columns_list:
        _df[_column] = to_numeric(_df[_column], errors='coerce')

    return _df


def convert_to_text(_df: DataFrame, _columns_list: list) -> DataFrame:
    for _column in _columns_list:
        _df[_column] = _df[_column].astype(str)

    return _df


def treat_missing_by_mode_imputation(_df: DataFrame, _columns_list: list):
    for column in _columns_list:
        replace_value = _df[column].mode()
        _df[column] = _df[column].fillna(replace_value[0])

    return _df


def treat_missing_by_mean_imputation(_df: DataFrame, _columns_list: list):
    for column in _columns_list:
        replace_value = _df[column].mean()
        _df[column] = _df[column].fillna(replace_value)

    return _df


def clean_text_columns(_df: DataFrame, column_names: list) -> DataFrame:
    for col in column_names:
        _df[col] = _df[col].str.strip().str.lower()
    return _df


def make_value_corrections(_df: DataFrame, instructions: list[dict]) -> DataFrame:
    for instruction in instructions:
        column = instruction.get('column')
        value = instruction.get('value')
        to_correct = instruction.get('to correct')

        _df[column] = _df[column].replace(to_correct, value)

    return _df


def replace_empty_or_blank_values(_df: DataFrame):
    """
    Replaces empty spaces with NaN values in the DataFrame
    """
    _columns_list = _df.columns.to_list()
    for _column in _columns_list:
        _df[_column] = _df[_column].replace(r'^\s*$', nan_value, regex=True)

    return _df


def log_transform_columns(_df: DataFrame, _columns_list_original: list, _columns_list_transformed: list) -> DataFrame:
    _df[_columns_list_transformed] = _df[_columns_list_original].applymap(lambda x: log(x + 1))
    return _df


def winsorize_columns(_df: DataFrame, _columns_list: list, outlier_threshold: tuple) -> DataFrame:
    for _column in _columns_list:
        _df[_column] = winsorize(_df[_column], outlier_threshold)

    return _df


def normalize_by_standard_scaling(_df: DataFrame, _columns_list: list) -> DataFrame:
    scaler = StandardScaler()
    _df[_columns_list] = scaler.fit_transform(_df[_columns_list])

    return _df


def bin_data_with_outliers(_df: DataFrame, column_name: str, _bin_details: dict) -> DataFrame:
    # Unpack variables
    min_category_cap = _bin_details.get('min_category_cap')
    max_category_cap = _bin_details.get('max_category_cap')
    num_bins = _bin_details.get('num_bins')

    # Compute the bin edges
    bins = linspace(min_category_cap, max_category_cap, num_bins - 1)

    # Ensure max_category_cap is not duplicated in the bins
    if max_category_cap not in bins:
        bins = concatenate((bins, [max_category_cap]))
    else:
        bins = bins.tolist()
        bins[-1] = max_category_cap

    edges = [-np_inf] + list(bins) + [np_inf]

    # Generate bin labels
    bin_labels = []
    for i in range(len(edges) - 1):
        if i == 0:
            bin_label = f"below {edges[i + 1]}"
        elif i == len(edges) - 2:
            bin_label = f"above {edges[i]}"
        else:
            bin_label = f"{edges[i]} to {edges[i + 1]}"
        bin_labels.append(bin_label)

    # Bin the data, including outliers in the first and last bins
    binned_data = cut(_df[column_name], bins=edges, labels=bin_labels, include_lowest=True)

    # Add the binned data as a new column
    binned_column_name = "binned_{}".format(column_name)
    _df[binned_column_name] = binned_data

    return _df


def one_hot_encode_columns(_df: DataFrame, _columns_list: list) -> DataFrame:
    print('Original dimensions are {}'.format(_df.shape))

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(handle_unknown='ignore')

    # Fit and transform the categorical columns
    encoded_columns = encoder.fit_transform(_df[_columns_list])

    # Create a DataFrame from the one-hot encoded columns
    column_names = encoder.get_feature_names_out(_columns_list)

    # Convert to dense array if it's sparse
    if hasattr(encoded_columns, 'toarray'):
        encoded_columns = encoded_columns.toarray()

    encoded_df = DataFrame(encoded_columns, columns=column_names, index=_df.index)
    print('Encoded columns have {} dimensions'.format(encoded_df.shape))

    # Concatenate the one-hot encoded columns with the original DataFrame
    print('Removing {} existing columns'.format(len(_columns_list)))

    _df = concat([_df.drop(columns=_columns_list), encoded_df], axis=1)
    print('Merged columns have {} dimensions'.format(_df.shape))

    return _df
