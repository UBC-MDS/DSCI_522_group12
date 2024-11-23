# Airline Customer Satisfaction Predictor
Authors: Hrayr Muradyan, Azin Piran, Sopuruchi Chisom, Shengjia Yu.

# About

In this project we try to predict the airline customer satisfaction based on several factors like: gender, age, travel class, etc.
Understanding customer satisfaction is very important for airlines as it provides directions to improve the service and equipment. 
The right improvement, subsequently, will increase the revenue. Additionally, it will be easier to build customer loyalty.
Loyal customers often promote the airline through word-of-mouth or positive reviews, reducing the cost of acquiring new customers.
Thus, the reasons to conduct the project are many. The main question is: <b>can we accurately predict the customer satisfaction from the information we have? </b>

The dataset we use to answer this question was sourced in Kaggle, posted by [@teejmahal20](https://www.kaggle.com/teejmahal20) (TJ Klein). It is important to note that the dataset was originally posted by [@johndddddd](https://www.kaggle.com/datasets/johndddddd/customer-satisfaction), which is then modified and cleaned by [@teejmahal20](https://www.kaggle.com/teejmahal20). The full dataset can be found [here](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction).
# Report

The final report can be found: [here](https://ubc-mds.github.io/airline-customer-satisfaction-predictor/airline_passenger_satisfaction_predictor.html)

# Dependencies
This project requires the following Python packages and versions:

- **ipykernel**: Used for interactive computing in Jupyter notebooks.
- **matplotlib**: A library for creating static, animated, and interactive visualizations in Python.
- **numpy** (version 1.22): A package for numerical computing and handling arrays.
- **pandas** (version 1.3): A powerful data manipulation and analysis library.
- **python** (version 3.9): The programming language required to run the project.
- **category_encoders**: A package that provides various encoding techniques for categorical data.
- **scikit-learn**: A library for machine learning algorithms and data mining.
- **seaborn**: A Python visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics.
- **conda-lock**: A tool for generating deterministic, reproducible conda environments.

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

# LICENSE

The code in this repository is licensed under the MIT license. Refer to the [LICENSE](LICENSE) file for more details.

# References

1. **Mean and Standard Deviation of Cross-Validation Scores**  
   DSCI 571 Lab 4. (2024, October). "Model Evaluation and Cross-Validation". Retrieved November 20, 2024 from [DSCI 571 Lab 4](https://github.ubc.ca/mds-2024-25/DSCI_571_sup-learn-1_students/tree/master/labs/lab4/solutions).

2. **Confusion Matrix and ROC Curve**  
   DSCI 573 Lecture 1. (2024, September). "Introduction to Model Performance Metrics". Retrieved November 20, 2024 from [DSCI 573 Lecture 1](https://pages.github.ubc.ca/mds-2024-25/DSCI_573_feat-model-select_students/lectures/01_classification-metrics.html).

3. **Dataset**  
   TJ, Klein. (2020, February). "Airline Passenger Satisfaction". Retrieved November 20, 2024 from [Kaggle Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data).


