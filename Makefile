# all: 

# python scripts/data_download.py \
#     --url="teejmahal20/airline-passenger-satisfaction" \
#     --save-to="./data/" \
#     --file-to="combined_dataset.csv"

# python scripts/data_preparation.py \
#     --raw-data="./data/combined_dataset.csv" \
#     --test-size=0.2 \
#     --data-to="./data/" \
#     --preprocessor-to="./results/models/"

# python scripts/eda.py \
#     --train-data-path="./data/processed/scaled_satisfaction_train.csv" \
#     --plot-to="./results/figures/"

# python scripts/model_training.py \
#     --preprocessor-path="./results/models/preprocessor.pickle" \
#     --pipeline-to="./results/models/" \
#     --train-path="./data/raw/satisfaction_train.csv" \
#     --eval-metric="f1" \
#     --plot-save-path="./results/figures/" \
#     --cv-results-save-path="./results/tables/"

results/figures/tables/test_scores.csv results/tables/classification_report.csv results/figures/confusion_matrix.png: scripts/model_evaluation.py results/models/model_pipeline.pickle
	python scripts/model_evaluation.py \
        --pipeline="results/models/model_pipeline.pickle" \
        --test-path="data/raw/satisfaction_test.csv" \
        --results-to="results/tables/" \
        --plots-to="results/figures/"

clean:
	rm -rf results/figures/ \
        results/tables/