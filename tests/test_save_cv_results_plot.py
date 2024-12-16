import pytest
from unittest.mock import patch
from pathlib import Path
import shutil
import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_cv_results_plot   import save_cv_results_plot  
from sample_data import sample_train_data, sample_test_data


# Test Fixture for valid cv_results
@pytest.fixture
def valid_cv_results():
    return {
        "param_decisiontreeclassifier__max_depth": [1, 2, 3, 4],
        "mean_val_score": [0.7, 0.8, 0.85, 0.9],
        "mean_train_score": [0.8, 0.85, 0.9, 0.95],
        "se_val_score": [0.01, 0.02, 0.015, 0.01],
        "se_train_score": [0.02, 0.015, 0.01, 0.005]
    }

# Test Case 1: Check if the function saves a plot
def test_save_cv_results_plot(valid_cv_results):
    plot_save_path = './test_plots'
    save_cv_results_plot(valid_cv_results, eval_metric='f1', plot_save_path=plot_save_path)
    file_to_check = Path(plot_save_path) / 'cv_results_plot.png'
    assert file_to_check.exists(), f"Plot was not saved at {file_to_check}"
    os.remove(file_to_check)
    os.rmdir(plot_save_path)

# Test Case 2: Test invalid cv_results input (missing keys)
def test_invalid_cv_results():
    
    invalid_cv_results = {
        "param_decisiontreeclassifier__max_depth": [1, 2, 3, 4],
        "mean_val_score": [0.7, 0.8, 0.85, 0.9],
        
    }

    with pytest.raises(KeyError):
        save_cv_results_plot(invalid_cv_results, eval_metric='f1', plot_save_path='./test_plots')

# Test Case 3: Check if directory is created
def test_directory_creation(valid_cv_results):
    plot_save_path = './test_plots/subdir'
    
    
    if Path(plot_save_path).exists():
        for file in Path(plot_save_path).iterdir():
            file.unlink()
        os.rmdir(plot_save_path)

    save_cv_results_plot(valid_cv_results, eval_metric='precision', plot_save_path=plot_save_path)
    assert Path(plot_save_path).exists(), f"Directory {plot_save_path} was not created."
    file_to_check = Path(plot_save_path) / 'cv_results_plot.png'
    assert file_to_check.exists(), f"Plot was not saved at {file_to_check}"
    file_to_check.unlink()
    os.rmdir(plot_save_path)


# Test Case 4: Test empty cv_results dictionary
def test_empty_cv_results():
    empty_cv_results = {}
    with pytest.raises(KeyError):
        save_cv_results_plot(empty_cv_results, eval_metric='f1', plot_save_path='./test_plots')

# Test Case 5: Check if plot_save_path is a Path object
def test_path_object_input(valid_cv_results):
    plot_save_path = Path('./test_plots')
    save_cv_results_plot(valid_cv_results, eval_metric='recall', plot_save_path=plot_save_path)
    file_to_check = plot_save_path / 'cv_results_plot.png'
    assert file_to_check.exists(), f"Plot was not saved at {file_to_check}"
    file_to_check.unlink()
    plot_save_path.rmdir()

# Test Case 6: Test eval_metric capitalization
def test_eval_metric_case(valid_cv_results):
    
    save_cv_results_plot(valid_cv_results, eval_metric='F1', plot_save_path='./test_plots')
    file_to_check = Path('./test_plots') / 'cv_results_plot.png'
    assert file_to_check.exists(), f"Plot was not saved at {file_to_check}"
    file_to_check.unlink()
    os.rmdir('./test_plots')