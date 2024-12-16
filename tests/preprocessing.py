import pytest
import pandas as pd
import numpy as np
from sample_data import valid_sample_data
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from src.data_preprocessing import clean_column_names, correct_precision_after_scaling

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


def test_correct_precision_after_scaling():
    """Test precision correction on ordinal features."""
    df = valid_sample_data[["inflight_wifi_service", "time_convenient"]].copy()
    correct_precision_after_scaling(df, ["inflight_wifi_service", "time_convenient"])
    assert df["inflight_wifi_service"].dtype == "float32"
    assert df["time_convenient"].dtype == "float32"

