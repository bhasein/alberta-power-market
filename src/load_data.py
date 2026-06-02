import pandas as pd

from config import (
    MARKET_DIR,
    WEATHER_DIR,
    PROCESSED_DIR,
    WEATHER_LOCATIONS
)


# ---------------------------------------------------
# Raw Pool Price Data
# ---------------------------------------------------

def load_pool_price_data():
    """
    Load raw AESO pool price file.
    Returns full dataframe with all 236 columns.
    """

    df = pd.read_csv(
        MARKET_DIR / "pool_price_2020_2025.csv"
    )

    df["Date_Begin_Local"] = pd.to_datetime(
        df["Date_Begin_Local"]
    )

    return df


# ---------------------------------------------------
# AECO Monthly Gas Prices
# ---------------------------------------------------

def load_aeco_data():
    """
    Load AECO monthly natural gas reference prices.
    Source: Alberta government
    Units: CAD/GJ
    """

    df = pd.read_csv(
        MARKET_DIR / "aeco_monthly_2020_2025.csv",
        parse_dates=["date"]
    )

    return df


# ---------------------------------------------------
# Clean Market Dataset
# ---------------------------------------------------

def load_market_data():
    """
    Load and clean AESO market data.
    Extracts: timestamp, pool_price, demand, import/export flows.
    Import/export columns are needed for net_imports feature engineering.
    Drops nulls.
    """

    raw_df = load_pool_price_data()

    market_df = raw_df[[
        "Date_Begin_Local",
        "ACTUAL_POOL_PRICE",
        "ACTUAL_AIL",
        "IMPORT_BC",
        "IMPORT_MT", 
        "IMPORT_SK",
        "EXPORT_BC",
        "EXPORT_MT",
        "EXPORT_SK"
    ]].rename(columns={
        "Date_Begin_Local": "timestamp",
        "ACTUAL_POOL_PRICE": "pool_price",
        "ACTUAL_AIL": "demand"
    })

    return market_df.dropna()


# ---------------------------------------------------
# Weather Data
# ---------------------------------------------------

def load_weather_data():
    """
    Load and merge hourly weather data for all
    Alberta locations defined in config.
    Applies 6-hour timezone correction.
    Prefixes columns by location name.
    """

    weather_dfs = []

    for location in WEATHER_LOCATIONS:

        path = WEATHER_DIR / f"weather_{location}_2020_2025.csv"

        w = pd.read_csv(
            path,
            parse_dates=["time"]
        )

        w = w.rename(columns={"time": "timestamp"})

        # Open-Meteo returned UTC-6 instead of UTC-7
        # subtract 6 hours to align with Alberta local time
        w["timestamp"] = w["timestamp"] - pd.Timedelta(hours=6)

        # prefix all columns with location name
        # so calgary_temperature_2m, edmonton_wind_speed_100m etc
        w.columns = [
            f"{location}_{col}"
            if col != "timestamp"
            else col
            for col in w.columns
        ]

        weather_dfs.append(w)

    # merge all locations on timestamp
    weather = weather_dfs[0]

    for w in weather_dfs[1:]:
        weather = pd.merge(
            weather,
            w,
            on="timestamp",
            how="inner"
        )

    return weather


# ---------------------------------------------------
# Merged Market + Weather Dataset
# ---------------------------------------------------

def load_merged_data():
    """
    Merge market and weather data on timestamp.
    Returns fully joined hourly dataset.
    """

    market_df = load_market_data()
    weather_df = load_weather_data()

    merged_df = pd.merge(
        market_df,
        weather_df,
        on="timestamp",
        how="inner"
    )

    return (
        merged_df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ---------------------------------------------------
# AECO Merge Helper
# ---------------------------------------------------

def merge_aeco(df):
    """
    Merge monthly AECO gas prices into hourly dataframe.
    Matches on year-month period.
    Each hour in a given month gets the same gas price.
    """

    aeco = load_aeco_data()

    # create year-month period for matching
    df["period"] = df["timestamp"].dt.to_period("M")
    aeco["period"] = aeco["date"].dt.to_period("M")

    df = df.merge(
        aeco[["period", "aeco_price_cad_gj"]],
        on="period",
        how="left"
    )

    df = df.drop(columns=["period"])

    return df

# ---------------------------------------------------
# Generation Mix Data
# ---------------------------------------------------

def load_generation_mix():
    """
    Load hourly generation by fuel type from AESO market statistics.
    Source: AESO Annual Market Statistics — Generation tab
    Coverage: 2020-01-01 to 2025-07-31
    Fuel types: Coal, Cogeneration, Combined Cycle, Dual Fuel,
                Gas Fired Steam, Hydro, Other, Simple Cycle,
                Solar, Storage, Wind
    """

    df = pd.read_csv(
        MARKET_DIR / "Gen Chart_data.csv",
        encoding="utf-16",
        sep="\t"
    )

    # filter to system generation only
    df = df[df["Measure Names"] == "System Generation"].copy()

    # parse timestamp
    df["timestamp"] = pd.to_datetime(df["Date - MST"], format='mixed')

    # keep only what we need
    df = df[["timestamp", "Fuel Type", "Measure Values"]]

    # pivot — each fuel type becomes its own column
    df = df.pivot_table(
        index="timestamp",
        columns="Fuel Type",
        values="Measure Values",
        aggfunc="mean"
    ).reset_index()

    # clean column names
    df.columns.name = None
    df.columns = [
        "timestamp" if c == "timestamp"
        else f"gen_{c.lower().replace(' ', '_')}"
        for c in df.columns
    ]

    # fill NaN with 0 — fuel types not yet online
    df = df.fillna(0)

    # trim to analysis window
    df = df[
        (df["timestamp"] >= "2020-01-01") &
        (df["timestamp"] <= "2025-07-31")
    ].reset_index(drop=True)

    return df

# ---------------------------------------------------
# Generation Mix Data
# ---------------------------------------------------

def merge_generation_mix(df):
    """
    Merge hourly generation mix into main dataframe.
    Joins on timestamp using left merge to preserve
    all rows in the main dataset.
    Forward fills the 23 known missing timestamps.
    """

    gen = load_generation_mix()

    df = pd.merge(
        df,
        gen,
        on="timestamp",
        how="left"
    )

    # forward fill small gaps
    gen_cols = [c for c in df.columns if c.startswith("gen_")]
    df[gen_cols] = df[gen_cols].ffill()

    return df


# ---------------------------------------------------
# Saved Processed Datasets
# ---------------------------------------------------

def load_processed_data():
    """
    Load saved merged dataset from processed folder.
    Use this instead of rebuilding from raw each session.
    """

    return pd.read_csv(
        PROCESSED_DIR / "alberta_merged_2020_2025.csv",
        parse_dates=["timestamp"]
    )

def load_model_data():
    """
    load fully featured model-ready dataset.
    includes: pool_price, demand, weather, HDD, CDD,
    wind_index, net_imports, AECO gas prices,
    generation mix by fuel type (coal, wind, solar, gas etc),
    renewable_share, thermal_share, time features.
    coverage: 2020-01-01 to 2025-07-31
    """

    return pd.read_csv(
        PROCESSED_DIR / "alberta_model_ready_2020_2025.csv",
        parse_dates=["timestamp"]
    )
