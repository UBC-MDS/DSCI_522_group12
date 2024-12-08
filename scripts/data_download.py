import pandas as pd
import click
import kagglehub
from pathlib import Path

def download_read_combine_data(kaggle_url):
    # Try to download the dataset
    try:
        data_saved_path = Path(kagglehub.dataset_download(kaggle_url))
    except ValueError:
        print("The dataset link is invalid, please use kaggle's link 'teejmahal20/airline-passenger-satisfaction'")
    except:
        print("An unknown error occurred. Please try to provide 'teejmahal20/airline-passenger-satisfaction' as the link.")
    
    # Get the paths of the initial train and test sets
    train_data_path = data_saved_path / "train.csv"
    test_data_path = data_saved_path / "test.csv" 

    # Read the datasets
    train_data = pd.read_csv(train_data_path, index_col = "Unnamed: 0")
    test_data = pd.read_csv(test_data_path, index_col = "Unnamed: 0")

    # Combine the dataset and return it

    return pd.concat([train_data, test_data], axis=0).reset_index(drop=True)


@click.command()
@click.option('--url', 
              type=str, 
              help="URL of the dataset to download")
@click.option('--save-to', 
              type=click.Path(exists=False, dir_okay=True, file_okay=False, writable=True), 
              help="The path to save the downloaded dataset")
@click.option('--file-to', 
              type=str, 
              help="The file name to save the dataset into")
@click.option('--force-save', 
              type=bool, 
              help="Do you want to overwrite the file if it exists?", 
              default=False)
def main(url, save_to, file_to, force_save):
    # Convert the string to Path
    save_to = Path(save_to)

    # Download, read, combine and get the data
    dataset = download_read_combine_data(url)

    # If the path doesn't exist, create it
    if not save_to.exists():
        save_to.mkdir(parents=True, exist_ok=True)

    # The path to save the dataset
    file_to_save = save_to / file_to

    # If the file exists and force save is False do nothing
    if file_to_save.exists() and not force_save:
        print(f"""
                The file "{file_to_save}" already exists... 
                The script will not overwrite the file. 
                If you want to force save the file, specify argument --force_save=True
                Terminating the script...
        """)
        return

    # If save_to is a directory, not a file save the dataset, if not, raise an error
    if save_to.is_dir():
        dataset.to_csv(file_to_save, index=False)
    else:
        raise ValueError("The argument save_to is not a directory!")
    
    print(f"The dataset was successfully downloaded and saved in the directory: \033[1m{save_to}\033[0m\n")
    
if __name__ == '__main__':
    main()

