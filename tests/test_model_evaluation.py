import os
import pytest
import pandas as pd
from unittest.mock import MagicMock
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_evaluation import check_directory_exists, plot_save_confusion_matrix, evaluate_model
from sample_data import valid_sample_data


# === Fixtures === #
@pytest.fixture
def tmp_dir(tmp_path):
    """Fixture for creating a temporary directory."""
    return tmp_path

@pytest.fixture
def dummy_model():
    """Fixture for creating a mock model with `classes_` attribute."""
    model = MagicMock()
    model.classes_ = ['satisfied', 'neutral or dissatisfied']
    return model

@pytest.fixture
def dummy_data():
    """Fixture for creating dummy observed and predicted data."""
    y_obs = pd.Series(['satisfied', 'neutral or dissatisfied', 'satisfied', 'satisfied'])
    y_pred = pd.Series(['satisfied', 'satisfied', 'satisfied', 'neutral or dissatisfied'])
    return y_obs, y_pred


# === Tests for `check_directory_exists` === #
def test_check_directory_exists_existing(tmp_dir):
    """Test when the directory already exists."""
    path = tmp_dir / "existing_dir"
    path.mkdir()
    assert check_directory_exists(path) == path

def test_check_directory_exists_non_existing(tmp_dir):
    """Test when the directory does not exist."""
    path = tmp_dir / "new_dir"
    result = check_directory_exists(path)
    assert result.exists()
    assert result == path

def test_check_directory_exists_nested(tmp_dir):
    """Test creating a nested directory."""
    nested_path = tmp_dir / "level1" / "level2" / "level3"
    result = check_directory_exists(nested_path)
    assert result.exists()
    assert result == nested_path

def test_check_directory_exists_invalid_path(tmp_path):
    """Test for invalid path."""
    invalid_path = tmp_path / "invalid:/"
    with pytest.raises(OSError, match="The filename, directory name, or volume label syntax is incorrect:"):
        check_directory_exists(invalid_path)


# === Tests for `plot_save_confusion_matrix` === #
def test_plot_save_confusion_matrix(dummy_model, dummy_data, tmp_dir):
    """Test creating and saving a confusion matrix plot."""
    y_obs, y_pred = dummy_data
    plot_save_confusion_matrix(y_obs, y_pred, dummy_model, tmp_dir)
    assert(tmp_dir/"confusion_matrix.png").exists()

def test_plot_save_confusion_matrix_invalid_path(dummy_model, dummy_data, tmp_path):
    y_obs, y_pred = dummy_data
    invalid_path = tmp_path / "invalid:/"
    with pytest.raises(OSError):
        plot_save_confusion_matrix(y_obs, y_pred, dummy_model, invalid_path)

def test_plot_save_confusion_matrix_empty_data(dummy_model, tmp_dir):
    """Test confusion matrix with empty data."""
    y_obs, y_pred = pd.Series(dtype="object"), pd.Series(dtype="object")
    with pytest.raises(ValueError, match="zero-size array to reduction operation maximum which has no identity"):
        plot_save_confusion_matrix(y_obs, y_pred, dummy_model, tmp_dir)

def test_plot_save_confusion_matrix_incorrect_datatype(dummy_model, tmp_dir):
    """Test confusion matrix with incorrect dataype."""
    y_obs, y_pred = "object 1", "object 2"
    with pytest.raises(ValueError):
        plot_save_confusion_matrix(y_obs, y_pred, dummy_model, tmp_dir)

def test_plot_save_confusion_matrix_missing_classes(dummy_data, tmp_dir):
    """Test when the model has target and `classes_` attribute mismatch"""
    y_obs, y_pred = dummy_data
    incomplete_model = MagicMock()
    incomplete_model.classes_ = ['satisfied']
    with pytest.raises(ValueError):
        plot_save_confusion_matrix(y_obs, y_pred, incomplete_model, tmp_dir)


# === Tests for `evaluate_model` === #
def test_evaluate_model(tmp_dir):
    """Test evaluating model and saving metrics."""
    y_obs, y_pred = valid_sample_data['satisfaction'], valid_sample_data['satisfaction']

    evaluate_model(y_obs, y_pred, tmp_dir)

    # Ensure results are saved
    assert (tmp_dir / "test_scores.csv").exists()
    assert (tmp_dir / "classification_report.csv").exists()

    # Check the metrics in test_scores.csv
    test_scores = pd.read_csv(tmp_dir / "test_scores.csv")
    assert "Accuracy" in test_scores.columns
    assert "Recall" in test_scores.columns
    assert "Precision" in test_scores.columns
    assert "F1-Score" in test_scores.columns

def test_evaluate_model_incorrect_datatype(tmp_dir):
    """Test confusion matrix with incorrect dataype."""
    y_obs, y_pred = "object 1", "object 2"
    with pytest.raises(ValueError):
        evaluate_model(y_obs, y_pred, tmp_dir)

def test_evaluate_model_mismatched_length(dummy_data, tmp_dir):
    """Test evaluating model with mismatched input lengths."""
    y_obs, y_pred = dummy_data
    y_obs = y_obs[:-1]
    with pytest.raises(ValueError):
        evaluate_model(y_obs, y_pred, tmp_dir)
