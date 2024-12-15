import pytest
import sys
import os
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eda_plots import save_target_distribution, \
                          save_correlation_matrix, \
                          save_continuous_feat_target_plots, \
                          save_cat_feat_target_plots
from pathlib import Path
from sample_data import valid_sample_data
import shutil

tmp_path = Path("./tmp_path/temp_figures/")
if not tmp_path.exists():
    tmp_path.mkdir(parents=True, exist_ok=True)

def test_save_target_distribution_train_data_type():
    with pytest.raises(AssertionError, match="The variable 'train_data' should be a pandas dataframe. You have"):
        save_target_distribution(train_data="not_a_dataframe", save_path=tmp_path, target_column="satisfaction")

def test_save_target_distribution_save_path_type():
    with pytest.raises(AssertionError, match="The variable 'save_path' should be a string or Path class. You have"):
        save_target_distribution(train_data=pd.DataFrame(), save_path=123, target_column="satisfaction")

def test_save_target_distribution_target_column_type():
    with pytest.raises(AssertionError, match="The variable 'target_column' should be a string. You have"):
        save_target_distribution(train_data=pd.DataFrame(), save_path=tmp_path, target_column=123)

def test_save_target_distribution_missing_target_column():
    df = pd.DataFrame({'other_column': [1, 2, 3]})
    with pytest.raises(AssertionError, match="The satisfaction column should be in the training data. It is missing..."):
        save_target_distribution(train_data=df, save_path=tmp_path, target_column="satisfaction")

def test_save_target_distribution_created_and_saved_correctly():
    sample_data = valid_sample_data.copy()
    save_target_distribution(sample_data, tmp_path, "satisfaction")
    saved_file = tmp_path / "target_variable_distribution.png"
    assert saved_file.exists(), "The plot file was not created."

def test_save_correlation_matrix_train_data_type():
    with pytest.raises(AssertionError, match="The variable 'train_data' should be a pandas dataframe. You have"):
        save_correlation_matrix(train_data="not_a_dataframe", save_path=tmp_path)

def test_save_correlation_matrix_save_path_type():
    with pytest.raises(AssertionError, match="The variable 'save_path' should be a string or Path class. You have"):
        save_correlation_matrix(train_data=pd.DataFrame(), save_path=123)

def test_save_correlation_matrix_directory_exists():
    with pytest.raises(AssertionError, match="The path .* doesn't exist or is not a directory."):
        save_correlation_matrix(train_data=valid_sample_data, save_path="non_existent_directory")

def test_save_correlation_matrix_created_and_saved_correctly():
    sample_data = valid_sample_data.copy()
    save_correlation_matrix(sample_data, tmp_path)
    saved_file = tmp_path / "correlation_matrix.png"
    assert saved_file.exists(), "The correlation matrix plot file was not created."

def test_save_continuous_feat_target_plots_train_data_type():
    with pytest.raises(AssertionError, match="The variable 'train_data' should be a pandas dataframe. You have"):
        save_continuous_feat_target_plots(train_data="not_a_dataframe", save_path=tmp_path)

def test_save_continuous_feat_target_plots_save_path_type():
    with pytest.raises(AssertionError, match="The variable 'save_path' should be a string or Path class. You have"):
        save_continuous_feat_target_plots(train_data=pd.DataFrame(), save_path=123)

def test_save_continuous_feat_target_plots_target_column_type():
    with pytest.raises(AssertionError, match="The variable 'target_column' should be a string. You have"):
        save_continuous_feat_target_plots(train_data=pd.DataFrame(), save_path=tmp_path, target_column=123)

def test_save_continuous_feat_target_plots_missing_target_column():
    df = pd.DataFrame({'other_column': [1, 2, 3]})
    with pytest.raises(AssertionError, match="The satisfaction column should be in the training data. It is missing..."):
        save_continuous_feat_target_plots(train_data=df, save_path=tmp_path, target_column="satisfaction")

def test_save_continuous_feat_target_plots_created_and_saved_correctly():
    sample_data = valid_sample_data.copy()
    save_continuous_feat_target_plots(sample_data, tmp_path)
    saved_file = tmp_path / "numeric_feat_target_plots.png"
    assert saved_file.exists(), "The numeric feature vs target plots file was not created."

def test_save_cat_feat_target_plots_train_data_type():
    with pytest.raises(AssertionError, match="The variable 'train_data' should be a pandas dataframe. You have"):
        save_cat_feat_target_plots(train_data="not_a_dataframe", save_path=tmp_path)

def test_save_cat_feat_target_plots_save_path_type():
    with pytest.raises(AssertionError, match="The variable 'save_path' should be a string or Path class. You have"):
        save_cat_feat_target_plots(train_data=pd.DataFrame(), save_path=123)

def test_save_cat_feat_target_plots_target_column_type():
    with pytest.raises(AssertionError, match="The variable 'target_column' should be a string. You have"):
        save_cat_feat_target_plots(train_data=pd.DataFrame(), save_path=tmp_path, target_column=123)

def test_save_cat_feat_target_plots_missing_target_column():
    df = pd.DataFrame({'other_column': [1, 2, 3]})
    with pytest.raises(AssertionError, match="The satisfaction column should be in the training data. It is missing..."):
        save_cat_feat_target_plots(train_data=df, save_path=tmp_path, target_column="satisfaction")

def test_save_cat_feat_target_plots_created_and_saved_correctly():
    sample_data = valid_sample_data.copy()
    save_cat_feat_target_plots(sample_data, tmp_path)
    saved_file = tmp_path / "cat_feat_target_plots.png"
    assert saved_file.exists(), "The categorical feature vs target plots file was not created."

def test_empty_the_tmp_path():
    # Clean up the temporary directory
    parent_path = tmp_path.parent
    shutil.rmtree(parent_path)

    assert not parent_path.exists(), "tmp_path was not removed."