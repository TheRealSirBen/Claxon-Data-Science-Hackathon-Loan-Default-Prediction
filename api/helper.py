from os import environ

import joblib
from pandas import DataFrame
from pandas import to_datetime

from _init_ import datadrift_logger


def check_date_columns(df: DataFrame, columns: list) -> list:
    """
    Check if the specified columns in the DataFrame can be converted to datetime.
    """
    not_convertible_columns: list = list()
    for column in columns:
        if column in df.columns:
            try:
                to_datetime(df[column])
            except (ValueError, TypeError):
                not_convertible_columns.append(column)
        else:
            not_convertible_columns.append(column)
    return not_convertible_columns


def compare_features(model, input_df: DataFrame) -> tuple[list, list]:
    """
    Compares features in the model with features in the transformed data.
    """
    # For models that use feature_names_in_
    model_features: list = list()
    if hasattr(model, 'feature_names_in_'):
        model_features = model.feature_names_in_.tolist()

    # For models that use feature_name_
    if hasattr(model, 'feature_name_'):
        # Retrieve feature names
        model_features = model.feature_name_

    transformed_data_features = input_df.columns.tolist()

    model_features_set = set(model_features)
    transformed_features_set = set(transformed_data_features)

    new_features = list(transformed_features_set - model_features_set)
    missing_model_features = list(model_features_set - transformed_features_set)

    return new_features, missing_model_features


def predict_loan(input_data: DataFrame):
    # Read model
    model_pickle_file = '{}/{}'.format(environ.get('MODEL_DEPLOYMENT_DIR'), environ.get('ML_MODEL_FILE'))
    with open(model_pickle_file, 'rb') as f:
        model = joblib.load(f)
    f.close()

    # Get feature landscape
    new_features, missing_model_features = compare_features(model, input_data)

    # Treat model missing features
    for feature in missing_model_features:
        input_data[feature] = 0

    # Treat new generated features
    for feature in new_features:
        input_data.drop(feature, axis=1, inplace=True)
        datadrift_logger.info('new - {}'.format(feature))

    # Run prediction for positive outcome
    positive_probabilities = model.predict_proba(input_data)[:, 1]

    # Create a DataFrame with the name 'probability'
    result_df = DataFrame(positive_probabilities, columns=['probability'])

    # Display the result
    return result_df
