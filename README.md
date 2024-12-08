# US Airline Customer Satisfaction Predictor
Authors: Hrayr Muradyan, Azin Piran, Sopuruchi Chisom, Shengjia Yu.

# About

In this project we try to predict US airline customer satisfaction based on several factors like: gender, age, travel class, etc.
Understanding customer satisfaction is very important for airlines as it provides directions to improve the service and equipment. 
The right improvement, subsequently, will translate to an increase in revenue. 

Additionally, it will be easier to build customer loyalty.
This is essential as loyal customers often promote the airline through word-of-mouth or positive reviews, reducing the cost of acquiring new customers (Sadegh Eshaghi, 2024). 

Thus, the reasons to conduct the this analysis are abundant. 
<br/>The main question is: <b>can we accurately predict the customer satisfaction from the information we have? </b>

The dataset we use to answer this question was sourced in Kaggle, posted by [@teejmahal20](https://www.kaggle.com/teejmahal20) (TJ Klein). It is important to note that the dataset was originally posted by [@johndddddd](https://www.kaggle.com/datasets/johndddddd/customer-satisfaction), which is then modified and cleaned by [@teejmahal20](https://www.kaggle.com/teejmahal20). The full dataset can be found [here](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction). Additionally, the dataset contains **only** US airline data, as mentioned in the original source.

# Report

The final report can be found: [here](https://ubc-mds.github.io/airline-customer-satisfaction-predictor/docs/airline_passenger_satisfaction_predictor.html)

# Dependencies
This project requires the following Python packages and versions:

- **ipykernel**: Used for interactive computing in Jupyter notebooks.
- **matplotlib**: A library for creating static, animated, and interactive visualizations in Python.
- **numpy**: A package for numerical computing and handling arrays.
- **pandas**: A powerful data manipulation and analysis library.
- **python**: The programming language required to run the project.
- **scikit-learn**: A library for machine learning algorithms and data mining.
- **seaborn**: A Python visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics.
- **conda-lock**: A tool for generating deterministic, reproducible conda environments.
- **jupyterlab**: An Interactive Development Environment to write, debug, and test code.


For the recent versions of the dependencies, view the [environment file](https://github.com/UBC-MDS/airline-customer-satisfaction-predictor/blob/main/dsci522_environment.yml).

### Installation using Conda Lock

To ensure a reproducible environment with exact dependency versions, you can use the `conda-lock` file. Follow these steps to set up the environment using the lock file:

#### Step 1: Install `conda-lock`

First, make sure that `conda-lock` is installed on your system. If you don't have it installed, you can install it via Conda:

```bash
conda install conda-lock
```
#### Step 2: Install dependencies using the `conda-lock` file

After `conda-lock` is installed, use the `conda-lock` file to install the environment. Run the following command in the directory containing the 'conda-lock.yml' file:

```bash
conda-lock install
```
#### Step 3: Create and activate the environment
Once the dependencies are installed, create the environment using:

```bash
conda env create --file conda-lock.yml
```
Then, activate the environment:

```bash
conda activate <your-environment-name>
```

# Usage
The steps below outline how to set up and run the analysis. 

### Step 1: Clone the repository

Using Https:
```bash
git clone https://github.com/UBC-MDS/airline-customer-satisfaction-predictor.git
```

Using SSH
```bash
git clone git@github.com:UBC-MDS/airline-customer-satisfaction-predictor.git
```

### Step 2: Setup Docker Computational Environment

> **Note:** The instructions contained in this section assume the commands are executed in a unix-based shell.
    
1. **Navigate to the root directory of the project**: 
    - In the terminal/command line navigate to the root directory of your local copy of this project.
    ```bash
    cd <repo_directory>
    ```
2. **Launch the docker container image for the computational environment**:

    ```bash
    docker-compose up
    ```
    - The terminal logs should display an output similar to: **Jupyter Server 2.14.2 is running at:**
    - Locate the URL starting with `http://127.0.0.1:8888/lab?token=` and click (or copy and paste in the browser) on the http address in the logs to access the Jupyter application from your web browser.

    Example link: `http://127.0.0.1:8888/lab?token=9f22c04a7fe732fdb2d2d98f1c2c0b74a89a5a6a1d60b45b`
    


### Step 3: Run the Analysis

Open a terminal and run the following commands in the order provided:

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
```

Alternatively you can open the notebook `notebooks/terminal_commands_notebook.ipynb` and run all the cells in order which will execute all the commands mentioned above.

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