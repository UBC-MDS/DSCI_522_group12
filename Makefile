.PHONY: clean

all: report/airline-customer-satisfaction-predictor.html report/airline-customer-satisfaction-predictor.pdf report/airline-customer-satisfaction-predictor_files

# Data Download Step
data/combined_dataset.csv: scripts/data_download.py
	python scripts/data_download.py \
    	--url="teejmahal20/airline-passenger-satisfaction" \
    	--save-to="./data/" \
    	--file-to="combined_dataset.csv"

# Data Preparation Step
data/processed/scaled_satisfaction_train.csv data/processed/scaled_satisfaction_test.csv results/models/preprocessor.pickle: data/combined_dataset.csv scripts/data_preparation.py
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
report/airline-customer-satisfaction-predictor.html report/airline-customer-satisfaction-predictor.pdf report/airline-customer-satisfaction-predictor_files: data/combined_dataset.csv\
results/figures/target_variable_distribution.png\
results/figures/numeric_feat_target_plots.png\
results/figures/cat_feat_target_plots.png\
results/tables/test_scores.csv\
results/figures/correlation_matrix.png\
results/tables/cv_results.csv\
results/figures/cv_results_plot.png\
results/tables/classification_report.csv\
results/figures/confusion_matrix.png
	quarto render report/airline-customer-satisfaction-predictor.qmd --to html
	quarto render report/airline-customer-satisfaction-predictor.qmd --to pdf
	mkdir -p docs
	cp report/airline-customer-satisfaction-predictor.html docs/airline_passenger_satisfaction_predictor.html
	cp -r report/airline-customer-satisfaction-predictor_files docs/airline-customer-satisfaction-predictor_files

clean:
	rm  data/combined_dataset.csv
	rm -rf data/raw \
		data/processed \
		results/models/preprocessor.pickle \
		results/models/model_pipeline.pickle \
		results/models/
	rm -rf results/figures/ \
        results/tables/
	rm -rf report/airline-customer-satisfaction-predictor.html \
        report/airline-customer-satisfaction-predictor.pdf \
        report/airline-customer-satisfaction-predictor_files
