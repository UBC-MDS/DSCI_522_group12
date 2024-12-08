import click
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,\
    classification_report, confusion_matrix,\
    ConfusionMatrixDisplay
import sys
from pathlib import Path

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
    # Read the test data
    test_data = pd.read_csv(test_path)

    # Convert the results path and plots path to Path class
    results_to = Path(results_to)
    plots_to = Path(plots_to)

    # If the paths do not exist, create them
    if not results_to.exists():
        results_to.mkdir(parents=True, exist_ok=True)
    
    if not plots_to.exists():
        plots_to.mkdir(parents=True, exist_ok=True)

    # Prepare the test set
    X_test = test_data.drop(columns=['satisfaction'])
    y_test = test_data['satisfaction']
    y_test = y_test.values.ravel()

    # Predict on the test set
    final_model = pickle.load(open(pipeline, "rb"))
    y_test_pred = final_model.predict(X_test)
    
    scoring_metrics = pd.DataFrame({
        "Accuracy": [accuracy_score(y_test, y_test_pred)],
        "Recall": [recall_score(y_test, y_test_pred, pos_label = 'satisfied')],
        "Precision": [precision_score(y_test, y_test_pred, pos_label = 'satisfied')],
        "F1-Score": [f1_score(y_test, y_test_pred, pos_label = 'satisfied')]
    })

    # Save the test scores
    test_scores_save_path = results_to / "test_scores.csv"
    scoring_metrics.to_csv(test_scores_save_path, index=False)
    print(f"Test scores saved in the directory: \033[1m{test_scores_save_path}\033[0m\n")

    # Create a classification report and save it
    class_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    class_report_save_path = results_to / "classification_report.csv"
    class_report.to_csv(class_report_save_path, index=False)
    print(f"Classification report saved in the directory: \033[1m{class_report_save_path}\033[0m\n")
    
    # Create and save the confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_model.classes_)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    conf_matrix_save_path = plots_to / "confusion_matrix.png"
    disp.figure_.savefig(conf_matrix_save_path)
    print(f"Confusion matrix saved in the directory: \033[1m{conf_matrix_save_path}\033[0m\n")

if __name__ == "__main__":
    try:
        main(standalone_mode=False)  # Prevents sys.exit()
        print("Congratulations! Model Evaluation Done!")
    except Exception as e:
        print(f"The following error occurred: {e}")
        sys.exit(1)