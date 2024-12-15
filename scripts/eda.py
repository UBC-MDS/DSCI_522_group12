import click
import pandas as pd
from pathlib import Path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.eda_plots import save_target_distribution, \
                          save_correlation_matrix, \
                          save_continuous_feat_target_plots, \
                          save_cat_feat_target_plots

from src.data_validation_utils import validate_for_correlations

@click.command()
@click.option('--train-data-path',
              type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
              help="Path to the training data set.")
@click.option('--plot-to',
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
              help="Path to directory where the plots from the eda will be saved to.")
def main(train_data_path, plot_to):
    # Convert the path to Path class
    train_data_path = Path(train_data_path)

    # Check if the train data path exists and is a file
    assert (train_data_path.is_file() and \
            train_data_path.exists() and \
            train_data_path.suffix == '.csv'), \
    "The argument '--train-data-path' should point to the train data. Valid train data is a .csv file."

    # Read the training data
    train_data = pd.read_csv(train_data_path)

    # Define the path where the plot should be saved
    plot_to_path = Path(plot_to)

    # If the path doesn't exist, create it
    if not plot_to_path.exists():
        plot_to_path.mkdir(parents=True, exist_ok=True)

    # Check if the plot_to_path is a directory
    assert plot_to_path.is_dir(), "The argument '--plot-to' should be a directory."

    # Create and save the target distribution plot
    save_target_distribution(train_data=train_data, save_path=plot_to_path, target_column="satisfaction")

    # Validate the train data not to have anomalous correlations
    validate_for_correlations(train_data, feature_target_threshold=0.92, feature_feature_threshold=0.9)

    # Create and save the correlation matrix plot
    save_correlation_matrix(train_data=train_data, save_path=plot_to_path)

    # Create and save the continuous features vs. target variable plot
    save_continuous_feat_target_plots(train_data=train_data, save_path=plot_to_path, target_column="satisfaction")
    
    # Create and save the categorical features vs. target variable plot
    save_cat_feat_target_plots(train_data=train_data, save_path=plot_to_path, target_column="satisfaction")

if __name__ == '__main__':
    main()