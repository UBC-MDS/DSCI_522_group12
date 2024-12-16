from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

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
    """
    df.columns = (
        df.columns
        .str.lower()
        .str.replace(r'\s+', '_', regex=True)
        .str.replace(r'-', '_', regex=True)
    )
    df = df.rename({"departure/arrival_time_convenient": "time_convenient"}, axis=1)
    return df

def validate_data(df, missing_data_threshold=0.05):
    """
    Validate a pandas DataFrame for missing values.

    Checks if any column in the DataFrame has missing values exceeding the specified threshold.

    Parameters:
    -----------
    df : pd.DataFrame
        The input pandas DataFrame.
    missing_data_threshold : float, optional
        The maximum allowed proportion of missing values in any column (default is 0.05).

    Raises:
    -------
    ValueError
        If any column exceeds the missing data threshold.
    """
    missing_data_ratio = df.isnull().mean()
    if missing_data_ratio.max() > missing_data_threshold:
        raise ValueError(f"Data has missing values exceeding {missing_data_threshold * 100}%.")

def create_column_transformer(categorical_cols, ordinal_cols, numerical_cols, drop_cols):
    """
    Create a column transformer for preprocessing data.

    The transformer applies one-hot encoding, scaling, and dropping columns as specified.

    Parameters:
    -----------
    categorical_cols : list
        List of categorical column names to apply one-hot encoding.
    ordinal_cols : list
        List of ordinal column names to apply min-max scaling.
    numerical_cols : list
        List of numerical column names to apply standard scaling.
    drop_cols : list
        List of column names to drop.

    Returns:
    --------
    ColumnTransformer
        A scikit-learn ColumnTransformer for preprocessing data.
    """
    return make_column_transformer(
        (OneHotEncoder(drop='first', handle_unknown='ignore', dtype=np.int32), categorical_cols),
        (MinMaxScaler(), ordinal_cols),
        (StandardScaler(), numerical_cols),
        ('drop', drop_cols),
        remainder='passthrough'
    )

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
    """
    df[ordinal_features] = df[ordinal_features].astype("float32")
