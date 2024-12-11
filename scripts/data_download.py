import click
import os
import zipfile
import requests
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.download_read_combine_data import download_read_combine_data


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
    
    """Downloads the data from the web to a local filepath and combine it."""
    download_read_combine_data(url, save_to, file_to, force_save)


if __name__ == '__main__':
    main()

