import click
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
from deepchecks.tabular import Dataset


def save_target_distribution(train_data, save_path, target_column="satisfaction"):
    # Define the figure size
    plt.figure(figsize=(5, 5))

    # Create the countplot
    ax = sns.countplot(data=train_data, x = target_column, hue = target_column, palette=["lightcoral", "lightgreen"], legend=False)
    
    # Create the title
    plt.title("Target Variable Distribution")

    # Change the xlabel
    plt.xlabel(target_column.title())
    
    # Change the ylabel
    plt.ylabel("Count")

    # Rotate the xticks
    plt.xticks(rotation=20)

    # Add the text above the countplot
    for p in ax.patches:
        height = p.get_height()  
        ax.text(p.get_x() + p.get_width() / 2, height + 0.1,  
                f'{int(height)}', 
                ha='center', va='bottom', fontsize=10)
    
    # Have a tight layout
    plt.tight_layout()

    # If the path is not a Path class, make it
    if not isinstance(save_path, Path):
        save_path = Path(save_path)
    
    # Define the file where the plot will be saved to
    file_to_save = save_path / 'target_variable_distribution.png'

    # Save the figure
    plt.savefig(file_to_save)

    # Print about successful save
    print(f"Target variable distribution plot saved in: \033[1m{file_to_save}\033[0m\n")


def save_correlation_matrix(train_data, save_path):
    # Take only the columns having a float data type
    numeric_data = train_data.select_dtypes(include=['float'])
    
    # Calculate the correlation across the variables
    correlation_matrix = numeric_data.corr()

    # Define the plot and its size
    plt.figure(figsize=(12, 8))
    
    # Create a heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    # Add title
    plt.title('Correlation Heatmap')

    # Have a tight layout
    plt.tight_layout()

    # If the path is not a Path class, make it
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    # Define the file where the plot will be saved to
    file_to_save = save_path / 'correlation_matrix.png'

    # Save the figure
    plt.savefig(file_to_save)

    # Print about successful save
    print(f"Correlation matrix saved in: \033[1m{file_to_save}\033[0m\n")


def validate_for_correlations(train_data, feature_target_threshold=0.92, feature_feature_threshold=0.9):
    # Make the Dataset class to check
    dataset_to_check = Dataset(train_data, label="satisfaction", cat_features=[])

    # Check for the feature-target correlations
    check_feat_target_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(feature_target_threshold)
    check_feat_target_corr_result = check_feat_target_corr.run(dataset=dataset_to_check)

    # Check for the feature-feature correlations
    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(threshold = feature_feature_threshold, n_pairs = 0)
    check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=dataset_to_check)

    # If the condition didn't pass, raise a Value Error
    if not check_feat_target_corr_result.passed_conditions():
        raise ValueError(f"There is at least one feature having a correlation higher or equal to {feature_target_threshold} with the target variable!")

    # If the condition didn't pass, raise a Value Error
    if not check_feat_feat_corr_result.passed_conditions():
        raise ValueError("There are at least two features having a correlation higher or equal to {feature_feature_threshold}!")
    
    # Print about successful validation pass
    print("Congratulations! Feature-Target Correlations Passed!\n")
    print("Congratulations! Feature-Feature Correlations Passed!\n")


def save_continuous_feat_target_plots(train_data, save_path, target_column="satisfaction"):
    # Take the continuous variables
    continuous_vars = ['age', 'flight_distance', 'departure_delay_in_minutes']

    # Define the number of rows and columns
    n_rows = 2
    n_cols = 2
    
    # Create the plot and define the size
    fig = plt.figure(figsize=(15, 8))

    # Create the gridspec with rows and columns and height ratios
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[1, 1])

    # For continuous variable except the last one
    for i, column in enumerate(continuous_vars[:-1]):

        # Add a subplot to the gridspec in the first row
        ax = fig.add_subplot(gs[0, i])

        # Create a density plot with target variable providing the color
        sns.kdeplot(data=train_data, x=column, hue=target_column, fill=True, ax=ax, common_norm=False)
        
        # Add the title
        ax.set_title(f'Density Plot of {column}')

        # Change the xlabel
        ax.set_xlabel(column)

        # Change the ylabel
        ax.set_ylabel('Density')

    # For the last variable add a subplot in the second row
    ax = fig.add_subplot(gs[1, :])

    # Create a density plot with target variable providing the color 
    sns.kdeplot(data=train_data, x=continuous_vars[2], hue=target_column, fill=True, ax=ax, common_norm=False)
    
    # Add the title
    ax.set_title(f'Density Plot of {continuous_vars[-1]}')
    
    # Change the xlabel
    ax.set_xlabel(continuous_vars[-1])

    # Change the ylabel
    ax.set_ylabel('Density')

    # Have a tight layout
    plt.tight_layout()

    # If the path is not a Path class, make it
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    # Define the file where the plot will be saved to
    file_to_save = save_path / 'numeric_feat_target_plots.png'

    # Save the plot
    plt.savefig(file_to_save)

    # Print about the successful save
    print(f"Numeric features vs. Target variable plots saved in: \033[1m{file_to_save}\033[0m\n")


def save_cat_feat_target_plots(train_data, save_path, target_column="satisfaction"):
    # Take the continuous variables
    continuous_vars = ['age', 'flight_distance', 'departure_delay_in_minutes']

    # Get the categorical columns by taking the set difference of all numeric columns and the continuous ones
    categorical_cols = list(set(train_data.select_dtypes(include=['number']).columns) - set(continuous_vars))
    
    # Define the number of columns and rows
    n_cols = 3
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols 

    # Define the subplots with the number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    # Converts the 2D grid to 1D
    axes = axes.flatten()

    # For categorical column
    for i, column in enumerate(categorical_cols):

        # Create a count plot with target column as the color
        sns.countplot(data=train_data, x=column, hue=target_column, ax=axes[i])

        # Add title
        axes[i].set_title(f'Count Plot of {column}')

        # Change the xlabel
        axes[i].set_xlabel(column)

        # Change the ylabel
        axes[i].set_ylabel('Count')  

    # For each subplot, turn off the axis lines, ticks, and labels
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Have a tight layout
    plt.tight_layout()

    # If the path is not a Path class, make it
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    # Define the file where the plot will be saved to
    file_to_save = save_path / 'cat_feat_target_plots.png'

    # Save the file
    plt.savefig(file_to_save)

    # Print about the successful save
    print(f"Categorical features vs. Target variable plots saved in: \033[1m{file_to_save}\033[0m\n")


@click.command()
@click.option('--train-data-path',
              type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
              help="Path to the training data set.")
@click.option('--plot-to',
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
              help="Path to directory where the plots from the eda will be saved to.")
def main(train_data_path, plot_to):
    # Read the training data
    train_data = pd.read_csv(train_data_path)

    # Define the path where the plot should be saved
    plot_to_path = Path(plot_to)

    # If the path doesn't exist, create it
    if not plot_to_path.exists():
        plot_to_path.mkdir(parents=True, exist_ok=True)

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