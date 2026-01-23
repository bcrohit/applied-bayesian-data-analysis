import pandas as pd
from pathlib import Path

# 


REQ_COLUMNS = [
    # 'eva',
    'station_name',
    'train_name',
    'delay_in_min',
    'time',
    'is_canceled'
]


def load_month(path):
    """Data is huge, ~15M records per month. Use only necessary columns."""
    return pd.read_parquet(path, columns=REQ_COLUMNS)

def load_data_as_df(data_dir='data'):
    """Load last three month data and make ready a pandas dataframe."""
    data_dir = Path(data_dir)

    files = [
        data_dir / "data-2025-10.parquet",
        data_dir / "data-2025-11.parquet",
        data_dir / "data-2025-12.parquet",
    ]

    df = pd.concat([load_month(f) for f in files], ignore_index=True)

    return df

def basic_clean(df): # !NOTE: Should we reatin cancelled ones?
    """Remove cancelled trains, to focus delay magnitude""" 
    

    df = df[
        (df["is_canceled"] == False) &
        (df["delay_in_min"].notna())
    ]

    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date

    return df

def add_route_identifier(df):
    """Create a new column to uniquely identify the route"""

    df["route_id"] = (
        df["train_name"].astype(str) +  '→'  + df["station_name"].astype(str)
    )

    return df