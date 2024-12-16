import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_preprocessing import clean_column_names, correct_precision_after_scaling

@pytest.fixture
def valid_sample_data():
    """Fixture for valid sample data."""
    data = {
        "Gender": ["Male", "Female"],
        "Customer Type": ["Loyal", "Disloyal"],
        "Age": [25, 32],
        "Flight Distance": [500, 700],
        "Inflight Wifi Service": [3, 2],
        "Departure/Arrival Time Convenient": [2, 4],
        "Satisfaction": ["Satisfied", "Neutral"]
    }
    return pd.DataFrame(data)


# Tests for clean_column_names
def test_clean_column_names_valid(valid_sample_data):
    """Test clean_column_names with valid input."""
    cleaned_df = clean_column_names(valid_sample_data)
    expected_columns = [
        "gender", "customer_type", "age", "flight_distance", 
        "inflight_wifi_service", "time_convenient", "satisfaction"
    ]
    assert list(cleaned_df.columns) == expected_columns


def test_clean_column_names_invalid_string():
    """Test clean_column_names with string input."""
    with pytest.raises(TypeError):
        clean_column_names("This is a string")


def test_clean_column_names_invalid_list():
    """Test clean_column_names with list input."""
    with pytest.raises(TypeError):
        clean_column_names(["Column1", "Column2"])


def test_clean_column_names_invalid_none():
    """Test clean_column_names with None as input."""
    with pytest.raises(TypeError):
        clean_column_names(None)


# Tests for correct_precision_after_scaling
def test_correct_precision_after_scaling_valid(valid_sample_data):
    """Test correct_precision_after_scaling with valid input."""
    df = valid_sample_data[["Inflight Wifi Service", "Departure/Arrival Time Convenient"]].copy()
    correct_precision_after_scaling(df, ["Inflight Wifi Service", "Departure/Arrival Time Convenient"])
    assert df["Inflight Wifi Service"].dtype == np.float32
    assert df["Departure/Arrival Time Convenient"].dtype == np.float32


def test_correct_precision_after_scaling_invalid_dataframe():
    """Test correct_precision_after_scaling with string as DataFrame input."""
    with pytest.raises(TypeError):
        correct_precision_after_scaling("This is not a DataFrame", ["Column1"])


def test_correct_precision_after_scaling_invalid_ordinal_features():
    """Test correct_precision_after_scaling with non-list ordinal features."""
    df = pd.DataFrame({"Column1": [1, 2, 3]})
    with pytest.raises(TypeError):
        correct_precision_after_scaling(df, "Not a list")


def test_correct_precision_after_scaling_missing_column(valid_sample_data):
    """Test correct_precision_after_scaling with missing column."""
    df = valid_sample_data.copy()
    with pytest.raises(KeyError):
        correct_precision_after_scaling(df, ["Non_Existing_Column"])


def test_correct_precision_after_scaling_invalid_none():
    """Test correct_precision_after_scaling with None as input."""
    with pytest.raises(TypeError):
        correct_precision_after_scaling(None, ["Column1"])
