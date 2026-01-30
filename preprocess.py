import pandas as pd
from pathlib import Path

REQ_COLUMNS = [
    'station_name',
    'xml_station_name',
    'final_destination_station',
    'train_name',
    'delay_in_min',
    'time',
    'is_canceled',
    'train_type'
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
    """Basic Cleaning"""

    # If station name is null, replace it with xml_station_name
    df["station_name"] = df["station_name"].fillna(df["xml_station_name"])

        # Remove cancelled trains, to focus delay magnitude
    df = df[
        (df["is_canceled"] == False) &
        (df["delay_in_min"].notna())
    ]

    # Have proper date formatting
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date

    # These columns are no longer required
    df.drop(['is_canceled', 'xml_station_name'], axis=1, inplace=True)

    return df

def filter_data(df, train_type_threshold=1000):

    # Only focus on RE and ICE
    df = df[~df['train_type'].isin(['S', 'Bus'])]
    df = df[~df['train_name'].str.contains(r'\bS[0-9]+\b', regex=True)]

    # Remove less frequent train types
    frequent_types = df['train_type'].value_counts()
    valid_types = frequent_types[frequent_types >= train_type_threshold].index
    df = df[df['train_type'].isin(valid_types)]
    
    return df

def add_route_identifier(df, col_unique='route_id'):
    """Create a new column to uniquely identify the route"""

    df[col_unique] = (
        df["train_type"].astype(str)
        + "_" + df["train_name"].astype(str)
        + "→" + df["station_name"].astype(str)
    )

    return df


def sample_data(df, top_n_routes=150, col_unique='route_id'):
    """Sample data: only include top and busiest routes"""

    route_counts = (
        df.groupby(col_unique)
        .size()
        .sort_values(ascending=False)
    )

    major_routes = route_counts.head(top_n_routes).index

    df = df[df[col_unique].isin(major_routes)]

    return df


def compute_daily_route_delays(df):
    """
    Aggregate a per-train dataframe to daily route-level delay statistics.
    Returns: pandas.DataFrame
    """

    daily_route_delays = (
        df.groupby(['route_id', 'date'])
        .agg(
            mean_delay=('delay_in_min', "mean"),
            sd_delay=('delay_in_min', "std"),
            n_trains=('delay_in_min', "size"),
        )
        .reset_index()
    )

    daily_route_delays['date'] = pd.to_datetime(daily_route_delays['date'])
    daily_route_delays["weekday"] = daily_route_delays['date'].dt.weekday
    daily_route_delays["is_weekend"] = daily_route_delays["weekday"] >= 5

    return daily_route_delays