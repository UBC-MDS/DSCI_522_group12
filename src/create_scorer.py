import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score

def create_scorer(eval_metric, pos_label='satisfied'):
    """
    Creates a custom scoring function for model evaluation based on the specified evaluation metric.

    Parameters:
    - eval_metric (str): The evaluation metric to use. It should be one of ['precision', 'recall', 'f1', 'accuracy'].
    - pos_label (str, optional): The positive class label to consider for metrics like precision, recall, and F1 score. Defaults to 'satisfied'.

    Returns:
    - sklearn.metrics._scorer.make_scorer: A scorer function that can be used in model evaluation (e.g., cross-validation, grid search).

    Raises:
    - ValueError: If the provided eval_metric is not valid.
    """
    metrics = {
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score,
        'accuracy': accuracy_score
    }
    
    # If eval metric is not in metrics, raise a value error
    if eval_metric not in metrics:
        raise ValueError(f"Invalid metric name. Available metrics are {list(metrics.keys())}.")
    
    # Return the scorer
    return make_scorer(metrics[eval_metric], pos_label=pos_label)
