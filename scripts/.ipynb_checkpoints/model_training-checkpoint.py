import click
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path
import sys
import os
import requests
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_cv_results_plot import save_cv_results_plot
from src.create_scorer import create_scorer




@click.command()
@click.option('--preprocessor-path', 
                type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
                help='File path to the preprocess object')
@click.option('--pipeline-to', 
                type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True),
                help='Directory path to save the model pipeline to')
@click.option('--train-path', 
                type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
                help='File path to the training data')
@click.option('--eval-metric',
                type=click.Choice(['accuracy', 'precision', 'recall', 'f1'], case_sensitive=False),
                help='Evaluation metric to use for cross-validation'   
              )
@click.option('--plot-save-path',
              type=str,
              help='Path to save the cross-validation results plot')
@click.option('--cv-results-save-path',
              type=str,
              help='Path to save the cv results dataframe')
@click.option('--seed', type=int, help="Random seed", default=123)
def main(preprocessor_path, pipeline_to, train_path, eval_metric, plot_save_path, cv_results_save_path, seed):
    '''
    Fits the Decision Tree Clasifier model, performs hyper-paramter tuning
    and saves the pipeline
    '''
    # Define a random seed
    np.random.seed(seed)

    # Read the train data
    train_data = pd.read_csv(train_path)

    # Read the preprocessor
    preprocessor = pickle.load(open(preprocessor_path, "rb"))

    # Define and create the eval metric function
    eval_metric_scorer = create_scorer(eval_metric)

    # Define the maximum depth parameter range to tune
    max_depth_params = list(range(6, 27, 3))

    # Define the number of cross-validation iterations to do
    cv = 30 

    # Create the param grid dictionary
    param_grid = {
        'decisiontreeclassifier__max_depth': max_depth_params,  
    }

    # Make the pipeline using the preprocessor and DecisionTreeClassifier
    dt_pipe = make_pipeline(preprocessor, DecisionTreeClassifier(random_state=123))

    # Instantiate the GridSearchCV class and add the attributes
    grid_search = GridSearchCV(
        estimator=dt_pipe,
        param_grid=param_grid,
        scoring=eval_metric_scorer,  
        cv=cv,  
        n_jobs=-1,  
        return_train_score=True 
    )

    # Prepare the features and the target variable 
    X_train = train_data.drop(columns=['satisfaction'])
    y_train = train_data['satisfaction']

    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Take the best performing model
    final_model = grid_search.best_estimator_

    # Convert cv results to a dataframe
    cv_results = pd.DataFrame(grid_search.cv_results_)

    # Take only the mean scores and std for both validation and train sets
    # Calculate the standard error of the mean score across the folds
    cv_results = cv_results[[
            "param_decisiontreeclassifier__max_depth",
            "mean_test_score",
            "std_test_score",
            "mean_train_score",
            "std_train_score"
        ]].assign(
        se_val_score=cv_results.std_test_score / cv**0.5,
        se_train_score=cv_results.std_train_score / cv**0.5
        )
    
    # Rename the 'test' to 'validation'
    cv_results = cv_results.rename({"mean_test_score":"mean_val_score", 
                                    "std_test_score":"std_val_score"}, 
                                    axis=1)

    # Produce and save the cv results plot
    save_cv_results_plot(cv_results=cv_results, eval_metric=eval_metric, plot_save_path=plot_save_path)

    # If the cv results save path is not a Path class, make it
    if not isinstance(cv_results_save_path, Path):
        cv_results_save_path = Path(cv_results_save_path)
    
    # If the path doesn't exist, create it
    if not cv_results_save_path.exists():
        cv_results_save_path.mkdir(parents=True, exist_ok=True)

    # Save cv results table
    cv_results.to_csv(cv_results_save_path / "cv_results.csv", index=False)

    # If the path is not a Path class, make it
    if not isinstance(pipeline_to, Path):
        pipeline_to = Path(pipeline_to)

    if not pipeline_to.exists():
        pipeline_to.mkdir(parents=True, exist_ok=True)

    # Try to save the model
    try:
        file_name = "model_pipeline.pickle"
        pipeline_save_path = pipeline_to / file_name
        with open(pipeline_save_path, 'wb') as f:
            pickle.dump(final_model, f)
        print(f"Model pipeline saved in the directory: \033[1m{pipeline_save_path}\033[0m\n")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

if __name__ == "__main__":
    try:
        main(standalone_mode=False)  # Prevents sys.exit()
        print("Congratulations! Model Training Done!\n")
    except Exception as e:
        print(f"The following error occurred: {e}")
        sys.exit(1)