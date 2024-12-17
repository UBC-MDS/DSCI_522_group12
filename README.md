# US Airline Customer Satisfaction Predictor
Authors: Hrayr Muradyan, Azin Piran, Sopuruchi Chisom, Shengjia Yu.

# About

This project aims to predict US airline customer satisfaction based on several factors.

**Customer Satisfaction** in this context refers to a passenger's level of contentment with their airline experience, encompassing factors such as in-flight amenities, operational performance, and overall flight experience. Customer satisfaction levels were compiled from passengers' survey responses and represented as a categorical variable with two distinct categories: neutral/dissatisfied, and satisfied.

Understanding customer satisfaction is critical for airlines as it provides directions to improve the service and equipment. Additionally, it fosters building customer loyalty; this is beneficial as loyal customers often promote the airline through word-of-mouth or positive reviews, reducing the cost of acquiring new customers (Sadegh Eshaghi, 2024).  

We developed a classification model using a decision tree algorithm to predict airline customer satisfaction using features like in-flight service quality, seat comfort, and demographic information. Customers were categorized as either satisfied (positive rating) or neutral/dissatisfied (negative rating). The decision tree model performed well on an unseen test dataset, demonstrating strong overall F1-score of 0.94.

These results indicate that the decision tree model effectively detects patterns in customer satisfaction, making it a valuable tool for analyzing key factors that influence satisfaction. While the results are very promising, there are several limitations of the project that should be addressed. 

Firstly, the dataset contains only US airline observations which limits its usage to only US-based airline scenarios. Secondly, the dataset is relatively old, being about 5 years old. The airline industry might have faced significant changes. Thirdly, collecting detailed information about customer experiences, such as seat comfort ratings, can be challenging. And lastly, the exact method customer satisfaction was measured is unknown.

Thus, for further research, a collection of a new data set involving international airlines and an exploration of alternative ways
for quantifying subjective binary customer satisfaction is recommended.

The dataset used to answer this question was sourced in Kaggle, posted by [@teejmahal20](https://www.kaggle.com/teejmahal20) (TJ Klein). 
It is important to note that the dataset was originally posted by [@johndddddd](https://www.kaggle.com/datasets/johndddddd/customer-satisfaction), 
which is then modified and cleaned by [@teejmahal20](https://www.kaggle.com/teejmahal20). 
The full dataset can be found [here](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction). 
Additionally, the dataset contains **only** US airline data, as mentioned in the original source.

# Report

The final report can be found: [here](https://ubc-mds.github.io/airline-customer-satisfaction-predictor/docs/airline_passenger_satisfaction_predictor.html)


# Dependencies
This project requires the following Python packages and versions:

- [Docker](https://www.docker.com/): Consistent, reproducible containers.
- [VS Code](https://code.visualstudio.com/download): Lightweight, versatile code editor.
- [Jupyter Extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter): Interactive coding.

For the recent versions of the dependencies, view the [environment file](https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/blob/main/dsci522_environment.yml).

# Usage

To run the analysis in a dedicated computational environment set up using Docker, please follow these steps:

### Step 1: Clone the repository
Outlined are 2 options for cloning the repository- through https or ssh.

> **Note:** The instructions contained in this section assume the commands are executed in a unix-based shell.

Using Https:
```bash
git clone https://github.com/UBC-MDS/airline-customer-satisfaction-predictor.git
```

Using SSH:
```bash
git clone git@github.com:UBC-MDS/airline-customer-satisfaction-predictor.git
```

### Step 2: Setup Docker Computational Environment

1. **Navigate to the root directory of the project**: 
    <br/>In the terminal/command line navigate to the root directory of your local copy of this project.
    ```bash
    cd <repo_directory>
    ```

2. **Launch the docker container image for the computational environment**:

    ```bash
    docker-compose up
    ```
    - The terminal logs should display an output similar to: **Jupyter Server 2.14.2 is running at:**
    - Locate the URL starting with `http://127.0.0.1:8888/lab?token=` and click (or copy and paste in the browser) on the http address in the logs to access the Jupyter application from your web browser.
    <br/>Example link: `http://127.0.0.1:8888/lab?token=9f22c04a7fe732fdb2d2d98f1c2c0b74a89a5a6a1d60b45b`
    

### Step 3: Run the Analysis

#### The first method (**Recommended**):

In the root directory of the project run the following:

```bash
make all
```

The Makefile will run all the necessary files to generate the results and the report.
This is the recommended option because it checks if all the dependencies have generated for each consecutive step.

Additionally, if you want to erase everything generated, you can run the following:
```bash
make clean
```

#### The second method

Run the following commands in the root directoy of the project in the order provided:

```bash
python scripts/data_download.py \
    --url="teejmahal20/airline-passenger-satisfaction" \
    --save-to="./data/" \
    --file-to="combined_dataset.csv"

python scripts/data_preparation.py \
    --raw-data="./data/combined_dataset.csv" \
    --test-size=0.2 \
    --data-to="./data/" \
    --preprocessor-to="./results/models/"

python scripts/eda.py \
    --train-data-path="./data/processed/scaled_satisfaction_train.csv" \
    --plot-to="./results/figures/"

python scripts/model_training.py \
    --preprocessor-path="./results/models/preprocessor.pickle" \
    --pipeline-to="./results/models/" \
    --train-path="./data/raw/satisfaction_train.csv" \
    --eval-metric="f1" \
    --plot-save-path="./results/figures/" \
    --cv-results-save-path="./results/tables/"

python scripts/model_evaluation.py \
    --pipeline="./results/models/model_pipeline.pickle" \
    --test-path="./data/raw/satisfaction_test.csv" \
    --results-to="./results/tables/" \
    --plots-to="./results/figures/"

quarto render report/airline-customer-satisfaction-predictor.qmd --to html
quarto render report/airline-customer-satisfaction-predictor.qmd --to pdf
mkdir -p docs && cp report/airline-customer-satisfaction-predictor.html docs/airline_passenger_satisfaction_predictor.html
```

#### The third option 

The last option is to open the notebook `notebooks/terminal_commands_notebook.ipynb` and run all the cells in order which will execute all the commands mentioned above.

### Clean up
To shut down the docker container and clean up the resources, interrupt the terminal by **Ctrl + C**. Then type:
```bash
docker compose rm
```

# LICENSE

The code in this repository is licensed under the MIT license. Refer to the [LICENSE](LICENSE) file for more details.

# References

TJ, Klein. (2020, February). "Airline Passenger Satisfaction". Retrieved November 20, 2024 from [Kaggle Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data).

M. Sadegh Eshaghi, Mona Afshardoost, Gui Lohmann, Brent D Moyle, Drivers and outcomes of airline passenger satisfaction: A Meta-analysis, Journal of the Air Transport Research Society, Volume 3, 2024, 100034, ISSN 2941-198X, https://doi.org/10.1016/j.jatrs.2024.100034.