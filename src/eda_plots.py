import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

def save_target_distribution(train_data, save_path, target_column="satisfaction"):
    """
    Saves a count plot of the target variable distribution to the specified path.

    Parameters
    ----------
    train_data : pd.DataFrame
        The training dataset containing the target column.
    save_path : str or pathlib.Path
        The directory where the plot should be saved. If a string is provided, it 
        will be converted to a Path object.
    target_column : str, optional
        The name of the target column to visualize, by default "satisfaction".

    Returns
    -------
    None
        This function does not return anything. It saves the plot as a PNG file in the 
        specified directory and prints the file's location.

    Notes
    -----
    - The function generates a count plot for the distribution of the target column.
    - Counts are displayed as text above each bar in the plot.
    - The saved plot is named `target_variable_distribution.png`.

    Examples
    --------
    >>> save_target_distribution(train_data, "plots", target_column="satisfaction")
    Target variable distribution plot saved in: plots/target_variable_distribution.png
    """

    assert isinstance(train_data, pd.DataFrame), f"The variable 'train_data' should be a pandas dataframe. You have {type(train_data)}."
    assert (isinstance(save_path, str) or isinstance(save_path, Path)), f"The variable 'save_path' should be a string or Path class. You have {type(save_path)}."
    assert isinstance(target_column, str), f"The variable 'target_column' should be a string. You have {type(target_column)}."
    assert (target_column in train_data.columns), f"The {target_column} column should be in the training data. It is missing..."

    # Define the figure size
    plt.figure(figsize=(5, 5))

    # Create the countplot
    ax = sns.countplot(data=train_data, x = target_column, hue = target_column, palette=["lightcoral", "lightgreen"], legend=False)
    
    # Create the title
    plt.title("Target Variable Distribution")

    # Change the xlabel
    plt.xlabel(target_column.title())
    
    # Change the ylabel
    plt.ylabel("Count")

    # Rotate the xticks
    plt.xticks(rotation=20)

    # Add the text above the countplot
    for p in ax.patches:
        height = p.get_height()  
        ax.text(p.get_x() + p.get_width() / 2, height + 0.1,  
                f'{int(height)}', 
                ha='center', va='bottom', fontsize=10)
    
    # Have a tight layout
    plt.tight_layout()

    # If the path is not a Path class, make it
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    assert (save_path.exists() and save_path.is_dir()), f"The path {save_path} doesn't exist or is not a directory."
    
    # Define the file where the plot will be saved to
    file_to_save = save_path / 'target_variable_distribution.png'

    # Save the figure
    plt.savefig(file_to_save)

    # Print about successful save
    print(f"Target variable distribution plot saved in: \033[1m{file_to_save}\033[0m\n")


def save_correlation_matrix(train_data, save_path):
    """
    Saves a heatmap of the correlation matrix for numeric columns in the dataset to the specified path.

    Parameters
    ----------
    train_data : pd.DataFrame
        The training dataset containing numeric columns for which the correlation matrix is computed.
    save_path : str or pathlib.Path
        The directory where the correlation heatmap should be saved. If a string is provided, 
        it will be converted to a Path object.

    Returns
    -------
    None
        This function does not return anything. It saves the heatmap plot as a PNG file in the 
        specified directory and prints the file's location.

    Notes
    -----
    - Only columns with float data types are considered for the correlation matrix.
    - The heatmap includes annotations with correlation values, uses the `coolwarm` colormap, and 
      has a format of two decimal places for the annotations.
    - The saved heatmap is named `correlation_matrix.png`.

    Examples
    --------
    >>> save_correlation_matrix(train_data, "plots")
    Correlation matrix saved in: plots/correlation_matrix.png
    """

    assert isinstance(train_data, pd.DataFrame), f"The variable 'train_data' should be a pandas dataframe. You have {type(train_data)}."
    assert (isinstance(save_path, str) or isinstance(save_path, Path)), f"The variable 'save_path' should be a string or Path class. You have {type(save_path)}."

    # Take only the columns having a float data type
    numeric_data = train_data.select_dtypes(include=['float'])
    
    # Calculate the correlation across the variables
    correlation_matrix = numeric_data.corr()

    # Define the plot and its size
    plt.figure(figsize=(12, 8))
    
    # Create a heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    # Add title
    plt.title('Correlation Heatmap')

    # Have a tight layout
    plt.tight_layout()

    # If the path is not a Path class, make it
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    assert (save_path.exists() and save_path.is_dir()), f"The path {save_path} doesn't exist or is not a directory."

    # Define the file where the plot will be saved to
    file_to_save = save_path / 'correlation_matrix.png'

    # Save the figure
    plt.savefig(file_to_save)

    # Print about successful save
    print(f"Correlation matrix saved in: \033[1m{file_to_save}\033[0m\n")



def save_continuous_feat_target_plots(train_data, save_path, target_column="satisfaction"):
    """
    Saves density plots of continuous variables against a target variable to the specified path.

    Parameters
    ----------
    train_data : pd.DataFrame
        The training dataset containing continuous features and the target column.
    save_path : str or pathlib.Path
        The directory where the plots should be saved. If a string is provided, 
        it will be converted to a Path object.
    target_column : str, optional
        The name of the target column used to differentiate densities by hue, 
        by default "satisfaction".

    Returns
    -------
    None
        This function does not return anything. It saves the plots as a PNG file in the 
        specified directory and prints the file's location.

    Notes
    -----
    - This function generates density plots for three continuous variables:
      `age`, `flight_distance`, and `departure_delay_in_minutes`.
    - Two plots are created:
      - One row with density plots for `age` and `flight_distance`.
      - Another row with a density plot for `departure_delay_in_minutes`.
    - The saved plot is named `numeric_feat_target_plots.png`.

    Examples
    --------
    >>> save_continuous_feat_target_plots(train_data, "plots", target_column="satisfaction")
    Numeric features vs. Target variable plots saved in: plots/numeric_feat_target_plots.png
    """

    assert isinstance(train_data, pd.DataFrame), f"The variable 'train_data' should be a pandas dataframe. You have {type(train_data)}."
    assert (isinstance(save_path, str) or isinstance(save_path, Path)), f"The variable 'save_path' should be a string or Path class. You have {type(save_path)}."
    assert isinstance(target_column, str), f"The variable 'target_column' should be a string. You have {type(target_column)}."
    assert (target_column in train_data.columns), f"The {target_column} column should be in the training data. It is missing..."

    # Take the continuous variables
    continuous_vars = ['age', 'flight_distance', 'departure_delay_in_minutes']

    # Define the number of rows and columns
    n_rows = 2
    n_cols = 2
    
    # Create the plot and define the size
    fig = plt.figure(figsize=(15, 8))

    # Create the gridspec with rows and columns and height ratios
    gs = fig.add_gridspec(n_rows, n_cols, height_ratios=[1, 1])

    # For continuous variable except the last one
    for i, column in enumerate(continuous_vars[:-1]):

        # Add a subplot to the gridspec in the first row
        ax = fig.add_subplot(gs[0, i])

        # Create a density plot with target variable providing the color
        sns.kdeplot(data=train_data, x=column, hue=target_column, fill=True, ax=ax, common_norm=False)
        
        # Add the title
        ax.set_title(f'Density Plot of {column}')

        # Change the xlabel
        ax.set_xlabel(column)

        # Change the ylabel
        ax.set_ylabel('Density')

    # For the last variable add a subplot in the second row
    ax = fig.add_subplot(gs[1, :])

    # Create a density plot with target variable providing the color 
    sns.kdeplot(data=train_data, x=continuous_vars[2], hue=target_column, fill=True, ax=ax, common_norm=False)
    
    # Add the title
    ax.set_title(f'Density Plot of {continuous_vars[-1]}')
    
    # Change the xlabel
    ax.set_xlabel(continuous_vars[-1])

    # Change the ylabel
    ax.set_ylabel('Density')

    # Have a tight layout
    plt.tight_layout()

    # If the path is not a Path class, make it
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    assert (save_path.exists() and save_path.is_dir()), f"The path {save_path} doesn't exist or is not a directory."

    # Define the file where the plot will be saved to
    file_to_save = save_path / 'numeric_feat_target_plots.png'

    # Save the plot
    plt.savefig(file_to_save)

    # Print about the successful save
    print(f"Numeric features vs. Target variable plots saved in: \033[1m{file_to_save}\033[0m\n")


def save_cat_feat_target_plots(train_data, save_path, target_column="satisfaction"):
    """
    Saves count plots of categorical features against a target variable to the specified path.

    Parameters
    ----------
    train_data : pd.DataFrame
        The training dataset containing categorical features and the target column.
    save_path : str or pathlib.Path
        The directory where the plots should be saved. If a string is provided,
        it will be converted to a Path object.
    target_column : str, optional
        The name of the target column used to differentiate counts by hue,
        by default "satisfaction".

    Returns
    -------
    None
        This function does not return anything. It saves the plots as a PNG file in the 
        specified directory and prints the file's location.

    Notes
    -----
    - Categorical features are identified as numeric columns not listed among continuous variables.
    - Count plots are generated for all categorical features, colored by the target variable.
    - The saved plot is named `cat_feat_target_plots.png`.
    - Subplots are organized dynamically based on the number of categorical features:
        - Each row contains up to 3 plots.
        - Empty subplots in the grid are turned off.

    Examples
    --------
    >>> save_cat_feat_target_plots(train_data, "plots", target_column="satisfaction")
    Categorical features vs. Target variable plots saved in: plots/cat_feat_target_plots.png
    """

    assert isinstance(train_data, pd.DataFrame), f"The variable 'train_data' should be a pandas dataframe. You have {type(train_data)}."
    assert (isinstance(save_path, str) or isinstance(save_path, Path)), f"The variable 'save_path' should be a string or Path class. You have {type(save_path)}."
    assert isinstance(target_column, str), f"The variable 'target_column' should be a string. You have {type(target_column)}."
    assert (target_column in train_data.columns), f"The {target_column} column should be in the training data. It is missing..."

    # Take the continuous variables
    continuous_vars = ['age', 'flight_distance', 'departure_delay_in_minutes']

    # Get the categorical columns by taking the set difference of all numeric columns and the continuous ones
    categorical_cols = list(set(train_data.select_dtypes(include=['number']).columns) - set(continuous_vars))
    
    # Define the number of columns and rows
    n_cols = 3
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols 

    # Define the subplots with the number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    # Converts the 2D grid to 1D
    axes = axes.flatten()

    # For categorical column
    for i, column in enumerate(categorical_cols):

        # Create a count plot with target column as the color
        sns.countplot(data=train_data, x=column, hue=target_column, ax=axes[i])

        # Add title
        axes[i].set_title(f'Count Plot of {column}')

        # Change the xlabel
        axes[i].set_xlabel(column)

        # Change the ylabel
        axes[i].set_ylabel('Count')  

    # For each subplot, turn off the axis lines, ticks, and labels
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Have a tight layout
    plt.tight_layout()

    # If the path is not a Path class, make it
    if not isinstance(save_path, Path):
        save_path = Path(save_path)

    assert (save_path.exists() and save_path.is_dir()), f"The path {save_path} doesn't exist or is not a directory."

    # Define the file where the plot will be saved to
    file_to_save = save_path / 'cat_feat_target_plots.png'

    # Save the file
    plt.savefig(file_to_save)

    # Print about the successful save
    print(f"Categorical features vs. Target variable plots saved in: \033[1m{file_to_save}\033[0m\n")


def test_save_cat_feat_target_plots_train_data_type():
    with pytest.raises(AssertionError, match="The variable 'train_data' should be a pandas dataframe. You have"):
        save_cat_feat_target_plots(train_data="not_a_dataframe", save_path="plots")

def test_save_cat_feat_target_plots_save_path_type():
    with pytest.raises(AssertionError, match="The variable 'save_path' should be a string or Path class. You have"):
        save_cat_feat_target_plots(train_data=pd.DataFrame(), save_path=123)

def test_save_cat_feat_target_plots_target_column_type():
    with pytest.raises(AssertionError, match="The variable 'target_column' should be a string. You have"):
        save_cat_feat_target_plots(train_data=pd.DataFrame(), save_path="plots", target_column=123)

def test_save_cat_feat_target_plots_missing_target_column():
    df = pd.DataFrame({'other_column': [1, 2, 3]})
    with pytest.raises(AssertionError, match="The satisfaction column should be in the training data. It is missing..."):
        save_cat_feat_target_plots(train_data=df, save_path="plots", target_column="satisfaction")

def test_save_cat_feat_target_plots_created_and_saved_correctly():
    sample_data = valid_sample_data.copy()
    save_cat_feat_target_plots(sample_data, tmp_path)
    saved_file = tmp_path / "cat_feat_target_plots.png"
    assert saved_file.exists(), "The categorical feature vs target plots file was not created."