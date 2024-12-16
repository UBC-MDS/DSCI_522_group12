from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
from deepchecks.tabular import Dataset
import pandera as pa
import pandas as pd

def check_duplicates(df):
    """
    Checks for duplicates in the dataframe after dropping the 'id' column.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe to check for duplicates.

    Returns
    -------
    bool
        Returns True if there are no duplicates (i.e., no duplicate rows in the dataset),
        otherwise False.

    Examples
    --------
    >>> check_duplicates(df)
    True

    Notes
    -----
    - The 'id' column is excluded from the check because it is a unique identifier.
    - The function uses `duplicated()` to detect duplicate rows and sums them to check for any duplicates.

    """
    if 'id' in df.columns:
        return not bool(df.drop('id', axis=1).duplicated().sum())
    else:
        return not bool(df.duplicated().sum())
    

def validate_data(df, missing_data_threshold=0.2):
    """
    Validates the input dataframe against a predefined schema for data quality checks.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe to validate.
    missing_data_threshold : float
        The maximum acceptable proportion of missing values in any column.

    Returns
    -------
    None
        Prints out validation results and failure cases if any.

    Raises
    ------
    pa.errors.SchemaErrors
        If the dataframe fails any of the validation checks.

    Examples
    --------
    >>> validate_data(df, missing_data_threshold=0.2)
    Congratulations! Data validation passed!

    Notes
    -----
    - The function checks for duplicates, missing values, and column-level constraints.
    - If any validation checks fail, an exception is raised with details about the failed checks.

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Variable 'df' must be a pandas DataFrame")    
    if df.empty:
        raise ValueError("Dataframe must contain observations.")
    if not isinstance(missing_data_threshold, float):
        raise TypeError("missing_data_threshold should be a float data type")
    if (missing_data_threshold) < 0 or (missing_data_threshold) > 1:
        raise ValueError("missing_data_threshold should be a value between 0 and 1")

    # Define the schema
    schema = pa.DataFrameSchema(
        {
            "gender": pa.Column(str, pa.Check.isin(["Male", "Female"]), nullable=False),
            "customer_type": pa.Column(str, pa.Check.isin(["Loyal Customer", "Disloyal Customer"]), nullable=False),
            "age": pa.Column(int, pa.Check.between(0, 100), nullable=False),
            "type_of_travel": pa.Column(str, pa.Check.isin(["Business travel", "Personal Travel"]), nullable=False),
            "class": pa.Column(str, pa.Check.isin(["Eco", "Eco Plus", "Business"]), nullable=False),
            "flight_distance": pa.Column(int, pa.Check.greater_than(0), nullable=False),
            "inflight_wifi_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "time_convenient": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "ease_of_online_booking": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "gate_location": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "food_and_drink": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "online_boarding": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "seat_comfort": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "inflight_entertainment": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "on_board_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "leg_room_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "baggage_handling": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "checkin_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "inflight_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "cleanliness": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "departure_delay_in_minutes": pa.Column(int, pa.Check.greater_than_or_equal_to(0), nullable=True),
            "arrival_delay_in_minutes": pa.Column(float, pa.Check.greater_than_or_equal_to(0), nullable=True),
            "satisfaction": pa.Column(str, pa.Check.isin(["neutral or dissatisfied", "satisfied"]), nullable=False),
        },
        checks=[
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found!"),
            pa.Check(lambda df: (df.isna().sum() / len(df) < missing_data_threshold).all(), error=f"Some columns have more than {missing_data_threshold*100}% missing values."),
            pa.Check(check_duplicates, error = "There are duplicates observations in the dataset!")
        ])

    # Check the data with the above defined schema
    schema.validate(df, lazy=True)
    print("Congratulations! Data validation passed!\n")



def validate_for_correlations(train_data, feature_target_threshold=0.92, feature_feature_threshold=0.9):
    """
    Validates the feature-target and feature-feature correlations in the training data.

    Parameters
    ----------
    train_data : pd.DataFrame
        The input training dataframe.
    feature_target_threshold : float, optional
        The threshold for the maximum correlation between features and the target variable. Default is 0.92.
    feature_feature_threshold : float, optional
        The threshold for the maximum correlation between features. Default is 0.9.

    Returns
    -------
    None
        Prints out validation results and raises exceptions if any checks fail.

    Raises
    ------
    ValueError
        If there are features violating the specified correlation thresholds.

    Examples
    --------
    >>> validate_for_correlations(train_data, feature_target_threshold=0.95, feature_feature_threshold=0.85)
    Congratulations! Feature-Target Correlations Passed!
    Congratulations! Feature-Feature Correlations Passed!
    """
    # Make the Dataset class to check
    dataset_to_check = Dataset(train_data, label="satisfaction", cat_features=[])

    # Check for the feature-target correlations
    check_feat_target_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(feature_target_threshold)
    check_feat_target_corr_result = check_feat_target_corr.run(dataset=dataset_to_check)

    # Check for the feature-feature correlations
    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(threshold = feature_feature_threshold, n_pairs = 0)
    check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=dataset_to_check)

    # If the condition didn't pass, raise a Value Error
    if not check_feat_target_corr_result.passed_conditions():
        raise ValueError(f"There is at least one feature having a correlation higher or equal to {feature_target_threshold} with the target variable!")

    # If the condition didn't pass, raise a Value Error
    if not check_feat_feat_corr_result.passed_conditions():
        raise ValueError("There are at least two features having a correlation higher or equal to {feature_feature_threshold}!")
    
    # Print about successful validation pass
    print("Congratulations! Feature-Target Correlations Passed!\n")
    print("Congratulations! Feature-Feature Correlations Passed!\n")