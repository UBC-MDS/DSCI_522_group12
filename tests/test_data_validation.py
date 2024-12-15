from sample_data import valid_sample_data, \
                        invalid_sample_data
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.data_validation_utils import validate_data
import pytest
import pandera as pa
import pandas as pd
import numpy as np

default_missing_value_perc = 0.2

# Check if correctly validates a dataset
def test_validate_data_success():
    sample_data = valid_sample_data.copy()
    try:
        validate_data(sample_data, missing_data_threshold=0.2)
    except Exception as e:
        pytest.fail(f"Validation failed unexpectedly: {e}")

# Check if correctly identifies missing values
def test_validate_data_failure_missing():
    sample_data = valid_sample_data.copy()
    sample_data.loc[:, "arrival_delay_in_minutes"] = np.NaN
    with pytest.raises(pa.errors.SchemaErrors, match=f"Some columns have more than {default_missing_value_perc*100}% missing values."):
        validate_data(sample_data, missing_data_threshold=default_missing_value_perc)

# Check if correctly identifies duplicates    
def test_validate_data_failure_duplicates():
    sample_data = valid_sample_data.copy()
    sample_data.loc[3] = sample_data.loc[0]  
    with pytest.raises(pa.errors.SchemaErrors, match="There are duplicates observations in the dataset!"):
        validate_data(sample_data, missing_data_threshold=default_missing_value_perc)

# Check if correctly identifies invalid categorical values
def test_validate_data_failure_invalid_categorical():
    sample_data = valid_sample_data.copy()
    sample_data.loc[:, "type_of_travel"] = "Leisure"
    sample_data.loc[:, "class"] = "Luxury"
    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(sample_data, missing_data_threshold=default_missing_value_perc)

# Check if correctly identifies out of range numeric values
def test_validate_data_failure_out_of_range_numeric():
    sample_data = invalid_sample_data.copy()
    sample_data.loc[:, "age"] = -5  
    sample_data.loc[:, "flight_distance"] = -100  
    with pytest.raises(pa.errors.SchemaErrors):
        validate_data(sample_data, missing_data_threshold=default_missing_value_perc)

# Check for invalid data types passed as the dataframe
def test_valid_data_type():
    data_as_np = valid_sample_data.to_numpy()
    with pytest.raises(TypeError):
        validate_data(data_as_np)

    with pytest.raises(TypeError):
        validate_data([1, 2, 3])

    with pytest.raises(TypeError):
        validate_data("DataFrame")

# Check for empty data frames
def test_valid_data_empty_data_frame():
    empty_data_frame = valid_sample_data.copy().iloc[0:0]
    with pytest.raises(ValueError):
        validate_data(empty_data_frame)

# Check for valid missing value threshold type
def test_valid_missing_threshold_type():
    with pytest.raises(TypeError):
        validate_data(valid_sample_data, missing_data_threshold="hi")

    with pytest.raises(TypeError):
        validate_data(valid_sample_data, missing_data_threshold=[1, 2])

# Check for valid missing value threshold values
def test_valid_missing_threshold_value():
    with pytest.raises(ValueError):
        validate_data(valid_sample_data, missing_data_threshold=1.5)

    with pytest.raises(ValueError):
        validate_data(valid_sample_data, missing_data_threshold=-0.6)


