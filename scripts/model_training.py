import click
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import sys

@click.command()
@click.option('--preprocessor_path', 
                type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
                help='File path to the preprocess object')
@click.option('--pipeline_to', 
                type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True),
                help='Directory path to save the model pipeline to')
@click.option('--train_path', 
                type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
                help='File path to the training data')
@click.option('--seed', type=int, help="Random seed", default=123)
def main(preprocessor_path, pipeline_to, train_path, seed):
    '''
    Fits the Decision Tree Clasifier model, performs hyper-paramter tuning
    and saves the pipeline
    '''

    np.random.seed(seed)
    train_data = pd.read_csv(train_path)
    preprocessor = pickle.load(open(preprocessor_path, "rb"))
    param_grid = {
        'decisiontreeclassifier__max_depth': [10, 12, 15, 18],  
    }

    dt_pipe = make_pipeline(preprocessor, DecisionTreeClassifier(random_state=123))
    grid_search = GridSearchCV(
        estimator=dt_pipe,
        param_grid=param_grid,
        scoring='accuracy',  
        cv=5,  
        n_jobs=-1,  
        return_train_score=True 
    )

    X_train = train_data.drop(columns=['satisfaction'])
    y_train = train_data['satisfaction']
    grid_search.fit(X_train, y_train)
    final_model = grid_search.best_estimator_

    try:
        file_name = "model_pipeline.pickle"
        with open(os.path.join(pipeline_to, file_name), 'wb') as f:
            pickle.dump(final_model, f)
        print(f"Model '{file_name}' saved in the directory: {pipeline_to}")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

if __name__ == "__main__":
    try:
        main(standalone_mode=False)  # Prevents sys.exit()
        print("Congratulations! Model Training passed!")
    except Exception as e:
        print(f"The following error occurred: {e}")
        sys.exit(1)