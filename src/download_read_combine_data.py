import pandas as pd
import kagglehub
from pathlib import Path

def download_read_combine_data(url, save_to, file_to, force_save=False):
    """
    Downloads a dataset from Kaggle, reads the train and test CSV files, 
    combines them into a single dataset, and saves it to a specified directory.
    Additionally, saves the train and test datasets separately in a 'raw' folder 
    within the given directory.

    Parameters:
    - url (str): The URL or dataset identifier from Kaggle.
    - save_to (str): The directory where the combined dataset will be saved.
    - file_to (str): The name of the file to save the combined dataset.
    - force_save (bool, optional): If True, will overwrite existing files. Defaults to False.
    """
    # Try to download the dataset
    try:
        data_saved_path = Path(kagglehub.dataset_download(url))
    except ValueError:
        print("The dataset link is invalid, please use kaggle's link 'teejmahal20/airline-passenger-satisfaction'")
    except:
        print("An unknown error occurred. Please try to provide 'teejmahal20/airline-passenger-satisfaction' as the link.")
    
    # Get the paths of the initial train and test sets
    train_data_path = data_saved_path / "train.csv"
    test_data_path = data_saved_path / "test.csv" 

    # Read the datasets
    train_data = pd.read_csv(train_data_path, index_col="Unnamed: 0")
    test_data = pd.read_csv(test_data_path, index_col="Unnamed: 0")

    # Combine the dataset
    dataset = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    
    # Convert the string to Path
    save_to = Path(save_to)
    
    # If the path doesn't exist, create it
    if not save_to.exists():
        save_to.mkdir(parents=True, exist_ok=True)

    # Save the combined dataset
    combined_file_path = save_to / file_to
    if combined_file_path.exists() and not force_save:
        print(f"""
                The file "{combined_file_path}" already exists... 
                The script will not overwrite the file. 
                If you want to force save the file, specify argument --force_save=True
                Terminating the script...
        """)
        return

    if save_to.is_dir():
        dataset.to_csv(combined_file_path, index=False)
    else:
        raise ValueError("The argument save_to is not a directory!")

   
    # Create a 'raw' directory inside the 'save_to' folder if it doesn't exist
    raw_folder = save_to / "raw"
    if not raw_folder.exists():
        raw_folder.mkdir(parents=True, exist_ok=True)

    # Save the train and test datasets separately in the 'raw' folder
    raw_train_file_path = raw_folder / "satisfaction_train.csv"
    raw_test_file_path = raw_folder / "satisfaction_test.csv"

    if not raw_train_file_path.exists() or force_save:
        train_data.to_csv(raw_train_file_path, index=False)
        print(f"Raw train dataset saved at: {raw_train_file_path}")

    if not raw_test_file_path.exists() or force_save:
        test_data.to_csv(raw_test_file_path, index=False)
        print(f"Raw test dataset saved at: {raw_test_file_path}")

    print(f"The combined dataset was successfully saved in the directory: \033[1m{save_to}\033[0m\n")
