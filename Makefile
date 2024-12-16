.PHONY: clean

all: report/airline-customer-satisfaction-predictor.html airline-customer-satisfaction-predictor.pdf report/airline-customer-satisfaction-predictor_files

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



# Data Download Step
data/raw/combined_dataset.csv: scripts/data_download.py
	python scripts/data_download.py \
    	--url="teejmahal20/airline-passenger-satisfaction" \
    	--save-to="./data/" \
    	--file-to="combined_dataset.csv"

# Data Preparation Step
data/processed/scaled_satisfaction_train.csv results/models/preprocessor.pickle: data/raw/combined_dataset.csv scripts/data_preparation.py
	python scripts/data_preparation.py \
    	--raw-data="./data/combined_dataset.csv" \
    	--test-size=0.2 \
    	--data-to="./data/" \
    	--preprocessor-to="./results/models/"

# Exploratory Data Analysis (EDA) Step
results/figures/target_variable_distribution.png results/figures/numeric_feat_target_plots.png results/figures/cat_feat_target_plots.png results/figures/correlation_matrix.png: data/processed/scaled_satisfaction_train.csv scripts/eda.py
	python scripts/eda.py \
    	--train-data-path="./data/processed/scaled_satisfaction_train.csv" \
    	--plot-to="./results/figures/"



# Target to train a model and save the pipeline
results/models/model_pipeline.pickle: results/models/preprocessor.pickle scripts/model_training.py data/raw/satisfaction_train.csv
	python scripts/model_training.py \
    	--preprocessor-path="./results/models/preprocessor.pickle" \
    	--pipeline-to="./results/models/" \
    	--train-path="./data/raw/satisfaction_train.csv" \
    	--eval-metric="f1" \
    	--plot-save-path="./results/figures/" \
    	--cv-results-save-path="./results/tables/"


# Model evaluation target
results/tables/test_scores.csv results/tables/classification_report.csv results/figures/confusion_matrix.png: scripts/model_evaluation.py results/models/model_pipeline.pickle
	# Run the model evaluation script
	python scripts/model_evaluation.py \
        --pipeline="results/models/model_pipeline.pickle" \
        --test-path="data/raw/satisfaction_test.csv" \
        --results-to="results/tables/" \
        --plots-to="results/figures/"

# report generation(html and pdf) and copy html to docs folder
report/airline-customer-satisfaction-predictor.html airline-customer-satisfaction-predictor.pdf report/airline-customer-satisfaction-predictor_files: results/tables/test_scores.csv\
data/combined_dataset.csv\
results/figures/target_variable_distribution.png\
results/figures/numeric_feat_target_plots.png\
results/figures/cat_feat_target_plots.png\
results/figures/correlation_matrix.png\
results/tables/cv_results.csv\
results/figures/cv_results_plot.png\
results/tables/classification_report.csv\
results/figures/confusion_matrix.png
	quarto render report/airline-customer-satisfaction-predictor.qmd --to html
	quarto render report/airline-customer-satisfaction-predictor.qmd --to pdf
	cp report/airline-customer-satisfaction-predictor.html docs/airline_passenger_satisfaction_predictor.html

clean:
	rm -rf results/figures/ \
        results/tables/ \
        data/raw/combined_dataset.csv \
        data/processed/scaled_satisfaction_train.csv \
        results/models/model_pipeline.pickle \
        results/models/preprocessor.pickle