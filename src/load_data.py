import pandas as pd

from config import (
    MASTER_DATA_FILE,
    MASTER_DICTIONARY_FILE,
    PROCESSED_DIR,
    PREPROCESSING_DIR,
    MARKET_DIR,
    WEATHER_DIR,
    GENERATION_DIR,
    WEATHER_LOCATIONS,
    TIMESTAMP_COL,
)

# canonical processed loaders

def load_master_data(set_index = False):
    '''
    loads the canonical alberta energy master dataset
    (created June 5th, hourly, utc, thoroughly inspected)
    
    file: data/processed/alberta_energy_master_2020_2025_hourly_utc.csv
    
    coverage: 2020-01-01 07:00 UTC to 2025-07-31 23:00 UTC
    
    contains:
        - timestamp
        - AECO gas prices
        - pool price and market variables
        - generation capacity / availability / generation by fuel type
        - intertie capability / offers
        - weather aggregates
        - outage MW by fuel type
    '''
    df = pd.read_csv(MASTER_DATA_FILE, parse_dates=[TIMESTAMP_COL])
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc = True)

    if set_index:
        df = df.set_index(TIMESTAMP_COL).sort_index()

    return df

def load_data_dictionary():
    '''
    loads the data dictionary for the canonical master datasheet
    '''
    return pd.read_csv(MASTER_DICTIONARY_FILE)

def load_source_columns(source):
    '''
    loads only columns from one soruce group in the master dataset
    
    examples: price, gen, intertie, outages, aeco
    '''
    df = load_master_data(set_index=False)

    prefix = f'{source}__'
    cols = [TIMESTAMP_COL] + [c for c in df.columns if c.startswith(prefix)]

    if len(cols) == 1:
        raise ValueError(f'No columns found for source prefix: {prefix}')
    
    return df[cols]

# backward-compatible aliases

def load_model_data(set_index = False):
    '''
    backward-compatible alias
    
    older notebooks used load_model_data()
    new notebooks should prefer load_master_data()
    '''
    return load_master_data(set_index = set_index)

def load_complete_data(set_index=False):
    '''
    backward-compatible alias
    
    older notebooks used load_model_data()
    new notebooks should prefer load_master_data()
    '''

    return load_master_data(set_index=set_index)

# preprocessed subsystem loaders

def load_preprocessed_price(set_index=False):
    df = pd.read_csv(
    PREPROCESSING_DIR / "pool_price_2020_2025_hourly_utc.csv",
    parse_dates=[TIMESTAMP_COL],
    )
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    return df.set_index(TIMESTAMP_COL).sort_index() if set_index else df

def load_preprocessed_aeco(set_index=False):
    df = pd.read_csv(
        PREPROCESSING_DIR / "aeco_hourly_2020_2025.csv",
        parse_dates=[TIMESTAMP_COL],
    )
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    return df.set_index(TIMESTAMP_COL).sort_index() if set_index else df

def load_preprocessed_generation(set_index=False):
    df = pd.read_csv(
        PREPROCESSING_DIR / "generation_aligned_2020_2025.csv",
        parse_dates=[TIMESTAMP_COL],
    )
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    return df.set_index(TIMESTAMP_COL).sort_index() if set_index else df

def load_preprocessed_intertie_weather(set_index=False):
    df = pd.read_csv(
        PREPROCESSING_DIR / "intertie_weather_2020_2025_hourly_utc_clean.csv",
        parse_dates=[TIMESTAMP_COL],
    )
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    return df.set_index(TIMESTAMP_COL).sort_index() if set_index else df

def load_preprocessed_outages(set_index=False):
    df = pd.read_csv(
        PREPROCESSING_DIR / "outages_2020_2025_preprocessed.csv",
        parse_dates=[TIMESTAMP_COL],
    )
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
    return df.set_index(TIMESTAMP_COL).sort_index() if set_index else df

# raw loaders retained for rebuilding / auditing

def load_raw_pool_price_data():
    """
    loads raw pool price data
    """
    df = pd.read_csv(MARKET_DIR / "pool_price_2020_2025.csv")
    for col in ["Date_Begin_GMT", "Date_Begin_Local"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    return df

def load_raw_aeco_data():
    """
    loads raw AECO monthly gas price data
    """
    df = pd.read_csv(
        MARKET_DIR / "aeco_monthly_2020_2025.csv",
        parse_dates=["date"],
    )
    return df[["date", "aeco_price_cad_gj"]]

def load_raw_generation_full():
    """
    loads raw generation full-data export
    """
    return pd.read_csv(
        GENERATION_DIR / "Gen Chart_Full Data_data.csv",
        encoding="utf-16",
        sep="\t",
    )

def load_raw_outage_data():
    """
    loads raw outage table
    """
    return pd.read_csv(
        MARKET_DIR / "Outage Table.csv",
        encoding="utf-16",
        sep="\t",
    )

def load_raw_weather_data():
    """
    loads raw weather files into a dictionary of DataFrames
    """
    weather = {}

    for location in WEATHER_LOCATIONS:
        path = WEATHER_DIR / f"weather_{location}_2020_2025.csv"
        weather[location] = pd.read_csv(path, parse_dates=["time"])
    return weather