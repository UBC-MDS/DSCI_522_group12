import pytest
from unittest.mock import patch
from pathlib import Path
import shutil
import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.download_read_combine_data import download_read_combine_data
from sample_data import sample_train_data, sample_test_data



# Define the test directory
test_dir = Path('tests/test_dir')
train_csv = test_dir / 'train.csv'
test_csv = test_dir / 'test.csv'

valid_url = "teejmahal20/airline-passenger-satisfaction"
invalid_url = "invalid/url"

@pytest.fixture(scope="module")
def setup_data():
    # Ensure the test directory exists
    os.makedirs(test_dir, exist_ok=True)

    # Generate sample train and test data if they don't exist
    if not train_csv.exists() or not test_csv.exists():
        # Sample train data 
        train_data = sample_train_data.copy() 
        # Sample test data 
        test_data = sample_test_data.copy()
        
        # Write to CSV files
        train_data.to_csv(train_csv)
        test_data.to_csv(test_csv)
    
    # Yield to allow the test to run
    yield
    
    # Clean up by removing the test directory after the test is done
    shutil.rmtree(test_dir)

# Mock test case
@patch('kagglehub.dataset_download')
def test_download_and_combine(mock_download, setup_data):
    # Mock the return value of the dataset download function
    mock_download.return_value = str(test_dir)  # Mock the download path
    
    # Call the function to download, read, and combine data
    download_read_combine_data(
        valid_url,  
        save_to=str(test_dir),
        file_to="combined.csv",
        force_save=True
    )

    # Verify that the combined file was created
    combined_file = test_dir / 'combined.csv'
    assert combined_file.exists(), "The combined dataset file was not created"

    # Verify that raw files were created
    raw_train_file = test_dir / 'raw/satisfaction_train.csv'
    raw_test_file = test_dir / 'raw/satisfaction_test.csv'
    
    assert raw_train_file.exists(), "The raw train dataset was not created"
    assert raw_test_file.exists(), "The raw test dataset was not created"

# Mock test case for invalid URL
@patch('kagglehub.dataset_download')
def test_download_and_combine_invalid_url(mock_download, setup_data):
    # Mock the return value of the dataset download function
    mock_download.return_value = str(test_dir)  # Mock the download path
    
    # Call the function to download, read, and combine data with an invalid URL
    with pytest.raises(ValueError, match="Invalid URL or dataset not found"):
        download_read_combine_data(
            invalid_url,  
            save_to=str(test_dir),
            file_to="combined.csv",
            force_save=True
        )