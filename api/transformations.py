from os import environ

import dill
from pandas import DataFrame

from _init_ import start_app

start_app()


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


if __name__ == '__main__':
    get_steps()
