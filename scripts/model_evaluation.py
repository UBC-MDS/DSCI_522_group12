import click
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,\
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
import sys

@click.command()
@click.option('--test_path', 
                type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
                help='File path to the testing data')
@click.option('--pipeline', 
                type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
                help='File path to the fit best model pipeline')
@click.option('--results_to',
              type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True), 
              help="Directory pat to save results to")
def main(pipeline, test_path, results_to):

    test_data = pd.read_csv(test_path)
    
    # Prepare the test set
    X_test = test_data.drop(columns=['satisfaction'])
    y_test = test_data['satisfaction']
    y_test = y_test.values.ravel()

    # Predict on the test set
    final_model = pickle.load(open(pipeline, "rb"))
    y_test_pred = final_model.predict(X_test)
    
    acc_score = pd.DataFrame({
        "Accuracy": [accuracy_score(y_test, y_test_pred)]
    })
    acc_score.to_csv(os.path.join(results_to, "test_scores.csv"), index=False)
    class_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    class_report.to_csv(os.path.join(results_to, "classification_report.csv"), index=False)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_model.classes_)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    disp.figure_.savefig(os.path.join(results_to, "confusion_matrix.png"))

    # Precision-Recall Curve
    pr_curve = PrecisionRecallDisplay.from_estimator(
        estimator=final_model,  
        X=X_test,               
        y=y_test,              
        pos_label='satisfied', 
    )
    plt.title("Precision-Recall Curve")
    pr_curve.figure_.savefig(fname=os.path.join(results_to,'pr_curve.png'))
    
    # ROC Curve
    roc_curve = RocCurveDisplay.from_estimator(
        estimator=final_model,  
        X=X_test,              
        y=y_test,               
        name="Decision Tree",  
        pos_label="satisfied"       
    )
    plt.title("ROC Curve")
    roc_curve.figure_.savefig(os.path.join(results_to, "roc_curve.png"))

    print(f"Model evaluation results saved in the directory: {results_to}")

if __name__ == "__main__":
    try:
        main(standalone_mode=False)  # Prevents sys.exit()
        print("Congratulations! Model Evaluation passed!")
    except Exception as e:
        print(f"The following error occurred: {e}")
        sys.exit(1)