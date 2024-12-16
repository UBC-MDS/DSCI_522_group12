import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from src.clean_column_names import clean_column_names
from src.clean_column_names import validate_data
from src.clean_column_names import create_column_transformer
from src.clean_column_names import correct_precision_after_scaling
from sample_data import valid_sample_data, invalid_sample_data


def test_clean_column_names_valid_data():
    """Test cleaning column names using valid sample data."""
    cleaned = clean_column_names(valid_sample_data)
    expected_columns = [
        "gender", "customer_type", "age", "type_of_travel", "class", "flight_distance",
        "inflight_wifi_service", "time_convenient", "ease_of_online_booking", "gate_location",
        "food_and_drink", "online_boarding", "seat_comfort", "inflight_entertainment",
        "on_board_service", "leg_room_service", "baggage_handling", "checkin_service",
        "inflight_service", "cleanliness", "departure_delay_in_minutes", "arrival_delay_in_minutes",
        "satisfaction"
    ]
    assert list(cleaned.columns) == expected_columns


def test_clean_column_names_edge_cases():
    """Test cleaning column names with special characters."""
    df = pd.DataFrame(columns=["C@lumn$Name!", "N#ame-With*Spaces"])
    cleaned = clean_column_names(df)
    assert list(cleaned.columns) == ["c_lumn_name_", "n_ame_with_spaces"]


def test_validate_data_valid():
    """Test validate_data on valid data without errors."""
    validate_data(valid_sample_data, missing_data_threshold=0.1)  


def test_validate_data_missing_threshold():
    """Test validate_data raises error when missing data exceeds the threshold."""
    incomplete_data = valid_sample_data.copy()
    incomplete_data.loc[0, "age"] = None 
    with pytest.raises(ValueError, match="Data has missing values exceeding"):
        validate_data(incomplete_data, missing_data_threshold=0.05)


def test_create_column_transformer():
    """Test creating a column transformer with sample column lists."""
    categorical_cols = ["gender", "customer_type"]
    ordinal_cols = ["inflight_wifi_service", "time_convenient"]
    numerical_cols = ["age", "flight_distance"]
    drop_cols = ["arrival_delay_in_minutes"]

    transformer = create_column_transformer(categorical_cols, ordinal_cols, numerical_cols, drop_cols)
    assert isinstance(transformer, ColumnTransformer)
    assert len(transformer.transformers) == 4


def test_correct_precision_after_scaling():
    """Test precision correction on ordinal features."""
    df = valid_sample_data[["inflight_wifi_service", "time_convenient"]].copy()
    correct_precision_after_scaling(df, ["inflight_wifi_service", "time_convenient"])
    assert df["inflight_wifi_service"].dtype == "float32"
    assert df["time_convenient"].dtype == "float32"

