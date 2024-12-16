import click
import os
import pandas as pd
import pickle
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_evaluation import check_directory_exists, plot_save_confusion_matrix,\
    evaluate_model


@click.command()
@click.option('--test-path', 
                type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
                help='File path to the testing data')
@click.option('--pipeline', 
                type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
                help='File path to the fit best model pipeline')
@click.option('--results-to',
                type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True), 
                help="Directory path to save results to")
@click.option('--plots-to',
                type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
                help="Directory path to save the plots to")
def main(pipeline, test_path, results_to, plots_to):
    """
    Main function to evaluate a trained model on test data, save evaluation metrics,
    and generate plots.

    This function reads the test dataset, loads a pre-trained model pipeline, evaluates 
    the model's performance on the test dataset, and saves the results (including metrics 
    and plots) to the specified directories.
    
    Parameters
    ----------
    pipeline : str
        File path to the pickled model pipeline to be evaluated.
    test_path : str
        File path to the testing dataset in CSV format.
    results_to : str
        Directory path where evaluation metrics and classification reports will be saved as CSV files.
    plots_to : str
        Directory path where evaluation plots will be saved.

    Returns
    -------
    None
        This function saves the plot to the directory without returning any value.
    """

    results_to = check_directory_exists(results_to)
    plots_to = check_directory_exists(plots_to)

    # Prepare the test set
    test_data = pd.read_csv(test_path)
    X_test = test_data.drop(columns=['satisfaction'])
    y_test = test_data['satisfaction'].values.ravel()

    # Predict and evaluate on the test set
    final_model = pickle.load(open(pipeline, "rb"))
    y_test_pred = final_model.predict(X_test)
    evaluate_model(y_test, y_test_pred, results_to)
    plot_save_confusion_matrix(y_test, y_test_pred, final_model, plots_to)

if __name__ == "__main__":
    try:
        main(standalone_mode=False)  # Prevents sys.exit()
        print("Congratulations! Model Evaluation Done!")
    except Exception as e:
        print(f"The following error occurred: {e}")
        sys.exit(1)