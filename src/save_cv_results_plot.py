import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from pathlib import Path


def save_cv_results_plot(cv_results, eval_metric, plot_save_path):
    """
    Creates and saves a plot visualizing the mean validation and training scores along with error bounds for different hyperparameter values from cross-validation results.

    Parameters:
    - cv_results (dict): A dictionary containing the cross-validation results with the following keys:
      - "param_decisiontreeclassifier__max_depth": Hyperparameter values for max depth.
      - "mean_val_score": Mean validation scores across cross-validation folds.
      - "mean_train_score": Mean training scores across cross-validation folds.
      - "se_val_score": Standard error of validation scores.
      - "se_train_score": Standard error of training scores.
    - eval_metric (str): The evaluation metric used for scoring (e.g., "precision", "recall", "f1").
    - plot_save_path (str or Path): The directory where the plot should be saved.

    Returns:
    - None: The plot is saved to the specified path as a PNG file.

    Side Effects:
    - Saves a plot showing the mean validation and training scores with error bars for different values of the hyperparameter "max_depth".
    - If the directory does not exist, it is created.
    """
    # Get the parameters and their respective scores and standard deviations for both train and validation sets
    parameters = cv_results["param_decisiontreeclassifier__max_depth"]
    mean_validation_scores = cv_results["mean_val_score"]
    mean_train_scores = cv_results["mean_train_score"]
    validation_error = cv_results["se_val_score"]
    train_error = cv_results["se_train_score"]

    # Define the plot and its size
    plt.figure(figsize=(8, 6))

    # Plot the line plot of the mean scores for both sets
    plt.plot(parameters, mean_validation_scores, color="black")
    plt.plot(parameters, mean_train_scores, color="black")

    # Plot the error bar for the validation scores
    plt.errorbar(
        parameters, mean_validation_scores, 
        yerr=validation_error, 
        fmt='o',  
        ecolor='gray',  
        elinewidth=1.5, 
        capsize=4,
        label=f"Mean Validation Score ({eval_metric.title()})"
    )

    plt.errorbar(
        parameters, mean_train_scores, 
        yerr=train_error, 
        fmt='o',  
        ecolor='gray',  
        elinewidth=1.5, 
        capsize=4,
        label=f"Mean Validation Score ({eval_metric.title()})"
    )

    # Change the labels, add grid and change the xticks to match the hyperparameter values
    plt.xlabel("Parameter Max Depth", fontsize=12)
    plt.ylabel(eval_metric.title(), fontsize=12)
    plt.title("Mean Validation Score with Error Bounds", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)  
    plt.xticks(list(parameters))

    # Add a legend and tight layout
    plt.legend()
    plt.tight_layout()

    # If the plot save path is not a Path class, make it
    if not isinstance(plot_save_path, Path):
        plot_save_path = Path(plot_save_path)

    # If the path doesn't exist, create it
    if not plot_save_path.exists():
        plot_save_path.mkdir(parents=True, exist_ok=True)

    # Define the file where the plot will be saved to
    file_to_save = plot_save_path / 'cv_results_plot.png'

    # Save the figure, close it
    plt.savefig(file_to_save)
    plt.close()

    # Print about successful save
    print(f"CV results plot saved in: \033[1m{file_to_save}\033[0m\n")
