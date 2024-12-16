import pandas as pd
import numpy as np

def clean_column_names(df):
    """
    Clean column names in a pandas DataFrame.

    Converts column names to lowercase, replaces spaces/dashes with underscores,
    and renames specific problematic columns for clarity.

    Parameters:
    -----------
    df : pd.DataFrame
        The input pandas DataFrame.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with cleaned column names.

    Raises:
    -------
    TypeError
        If the input is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(r'\s+', '_', regex=True)
        .str.replace(r'-', '_', regex=True)
    )
    df = df.rename({"departure/arrival_time_convenient": "time_convenient"}, axis=1)
    return df


def correct_precision_after_scaling(df, ordinal_features):
    """
    Correct floating-point precision errors in scaled ordinal features.

    Converts the specified ordinal features to float32 to reduce precision errors.

    Parameters:
    -----------
    df : pd.DataFrame
        The input pandas DataFrame.
    ordinal_features : list
        List of ordinal feature column names.

    Returns:
    --------
    None
        The function modifies the input DataFrame in place.

    Raises:
    -------
    TypeError
        If the input df is not a pandas DataFrame or if ordinal_features is not a list.
    KeyError
        If any feature in ordinal_features is not in the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input df must be a pandas DataFrame")
    if not isinstance(ordinal_features, list):
        raise TypeError("ordinal_features must be a list")
    if not all(feature in df.columns for feature in ordinal_features):
        raise KeyError("Some features in ordinal_features are not present in the DataFrame")
    
    df[ordinal_features] = df[ordinal_features].astype("float32")
