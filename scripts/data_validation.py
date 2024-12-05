import click
import os
import pandas as pd
import pandera as pa
from sklearn.model_selection import train_test_split


def clean_column_names(df):
    """
    Clean column names by converting them to lowercase, replacing spaces and dashes with underscores. 
    
    Parameters:
    df (pd.DataFrame): The input pandas DataFrame whose column names need to be cleaned.

    Returns:
    pd.DataFrame: A DataFrame with cleaned column names.
    """
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(r'\s+', '_', regex=True)
    df.columns = df.columns.str.replace(r'-', '_', regex=True)
    df = df.rename({"departure/arrival_time_convenient":"time_convenient"}, axis=1)
    
    return df

# running it 
"""
# cd into the root of this directory
# download into relative path from the root and create directory if it doesn't exist
curl -L --create-dirs -o ./data/raw/airline-passenger-satisfaction.zip\
    https://www.kaggle.com/api/v1/datasets/download/teejmahal20/airline-passenger-satisfaction

# unzip the folders
unzip data/raw/airline-passenger-satisfaction.zip -d data/raw/
rm data/raw/airline-passenger-satisfaction.zip
mkdir './data/val/'

# run the scipt
python scripts/data_validation.py './data/raw/train.csv' './data/raw/test.csv' './data/val/' './data/val/'
"""


@click.command()
@click.argument('raw_train_path', type=click.Path(exists=True, dir_okay=True, readable=True))
@click.argument('raw_test_path', type=click.Path(exists=True, dir_okay=True, readable=True))
@click.argument('val_train_path', type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True))
@click.argument('val_test_path', type=click.Path(exists=True, dir_okay=True, file_okay=False, writable=True))
def main(raw_train_path, raw_test_path, val_train_path, val_test_path):
    train_data = pd.read_csv(raw_train_path, index_col='Unnamed: 0')
    test_data = pd.read_csv(raw_test_path, index_col='Unnamed: 0')

    combined_dataset = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
    assert len(combined_dataset) == (len(train_data) + len(test_data)), \
        "Combined dataset length should equal the length of the train and test datasets"
    combined_dataset = clean_column_names(combined_dataset)

    # Data Validation
    print(">>"*30)
    missing_data_threshold = 0.05
    schema = pa.DataFrameSchema(
        {
            "gender": pa.Column(str, pa.Check.isin(["Male", "Female"]), nullable=False),
            "customer_type": pa.Column(str, pa.Check.isin(["Loyal Customer", "Disloyal Customer"]), nullable=False),
            "age": pa.Column(int, pa.Check.between(0, 100), nullable=False),
            "type_of_travel": pa.Column(str, pa.Check.isin(["Business travel", "Personal Travel"]), nullable=False),
            "class": pa.Column(str, pa.Check.isin(["Eco", "Eco Plus", "Business"]), nullable=False),
            "flight_distance": pa.Column(int, pa.Check.greater_than(0), nullable=False),
            "inflight_wifi_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "time_convenient": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "ease_of_online_booking": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "gate_location": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "food_and_drink": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "online_boarding": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "seat_comfort": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "inflight_entertainment": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "on_board_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "leg_room_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "baggage_handling": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "checkin_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "inflight_service": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "cleanliness": pa.Column(int, pa.Check.between(0, 5), nullable=True),
            "departure_delay_in_minutes": pa.Column(int, pa.Check.greater_than_or_equal_to(0), nullable=True),
            "arrival_delay_in_minutes": pa.Column(float, pa.Check.greater_than_or_equal_to(0), nullable=True),
            "satisfaction": pa.Column(str, pa.Check.isin(["neutral or dissatisfied", "satisfied"]), nullable=False),
        },
        checks=[
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."),
            pa.Check(lambda df: (df.isna().sum() / len(df) < missing_data_threshold).all(), error=f"Some columns have more than {missing_data_threshold*100}% missing values.")
        ])

    combined_dataset["customer_type"] = combined_dataset["customer_type"].str.title()
    try:
        schema.validate(combined_dataset, lazy=True)
        print("Data validation passed.")
    except pa.errors.SchemaErrors as e:
        print(e.failure_cases)

    # check for duplicates
    all_ids = combined_dataset.id
    if (len(set(all_ids)) == len(all_ids)):
        print("No duplicates found")
    else:
        print("There are duplicate observations")
    print(f"Number of duplicated observations = {combined_dataset.drop('id', axis=1).duplicated().sum()}")
    print("<<"*30)

    # splitting
    test_size = 0.2
    random_state = 123
    train_data, test_data = train_test_split(combined_dataset, test_size = test_size, random_state = random_state)
    train_data.to_csv(os.path.join(val_train_path, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(val_test_path, 'test.csv'), index=False)
    print(">>"*30)
    print("Validated and re-split data written out")
    print("Train path: ", val_train_path)
    print("Test path: ", val_test_path)
    print("<<"*30)

if __name__ == "__main__":
    main()