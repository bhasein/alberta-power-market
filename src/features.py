import numpy as np
import pandas as pd

# helpers

def _require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing: 
        raise KeyError(f'missing required columns: {missing}')
    
def _safe_divide(num, den):
    return np.where(den == 0, np.nan, num / den)

# time / calendar features

def add_time_features(df):
    df = df.copy()

    _require_columns(df, ['timestamp'])

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc = True)

    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year

    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # cyclic encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df

# weather features

def add_weather_features(df, base_temp=18):
    '''
    uses master dataset weather aggregates
    
    required: 
        - intertie__temp_mean
        - intertie__wind_speed_mean
        
    creates: 
        - HDD, CDD, wind_index
    '''
    df = df.copy()

    _require_columns(df, [
        'intertie__temp_mean',
        'intertie__wind_speed_mean',
    ])

    df['HDD'] = np.maximum(0, base_temp - df['intertie__temp_mean'])
    df['CDD'] = np.maximum(0, df['intertie__temp_mean'] - base_temp)

    df['wind_index'] = df['intertie__wind_speed_mean']

    return df

# intertie / import features

def add_intertie_features(df):
    df = df.copy()

    import_cols = [
        "price__IMPORT_BC",
        "price__IMPORT_MT",
        "price__IMPORT_SK",
    ]

    export_cols = [
        "price__EXPORT_BC",
        "price__EXPORT_MT",
        "price__EXPORT_SK",
    ]

    _require_columns(df, import_cols + export_cols + ["price__ACTUAL_AIL"])

    df["total_imports"] = df[import_cols].sum(axis=1)
    df["total_exports"] = df[export_cols].sum(axis=1)

    df["net_imports"] = df["total_imports"] - df["total_exports"]

    df["import_dependence"] = _safe_divide(
        df["total_imports"],
        df["price__ACTUAL_AIL"]
    )

    return df

# generation / system-state features

def add_generation_features(df):
    '''
    build genreation mix features from master dataset fuel-type columns
    uses actual system generation where available
    '''
    df = df.copy()

    gen_cols = [
        c for c in df.columns
        if c.startswith('gen__') and c.endswith('_system_generation')
    ]

    if not gen_cols:
        raise KeyError("No generation columns ending in '_system_generation' found")

    df['gen_total'] = df[gen_cols].sum(axis=1)

    def col(name):
        return f'gen__gen_{name}_system_generation'

    wind = col('wind')
    solar = col('solar')
    hydro = col('hydro')
    coal = col('coal')
    cogeneration = col('cogeneration')
    combined_cycle = col("combined_cycle")
    simple_cycle = col("simple_cycle")
    gas_fired_steam = col("gas_fired_steam")
    dual_fuel = col("dual_fuel")

    for c in [wind, solar, hydro, coal, cogeneration, combined_cycle,
            simple_cycle, gas_fired_steam, dual_fuel]:
        if c not in df.columns:
            df[c] = 0.0

    df["gen_wind"] = df[wind]
    df["gen_solar"] = df[solar]
    df["gen_hydro"] = df[hydro]
    df["gen_coal"] = df[coal]
    df["gen_renewable"] = (
        df["gen_wind"]
        + df["gen_solar"]
        + df["gen_hydro"]
    )

    df["gen_thermal"] = (
        df[coal]
        + df[cogeneration]
        + df[combined_cycle]
        + df[simple_cycle]
        + df[gas_fired_steam]
        + df[dual_fuel]
    )

    df["renewable_share"] = _safe_divide(df["gen_renewable"], df["gen_total"])
    df["variable_renewable_share"] = _safe_divide(
        df["gen_wind"] + df["gen_solar"],
        df["gen_total"]
    )

    df["thermal_share"] = _safe_divide(df["gen_thermal"], df["gen_total"])
    df["coal_share"] = _safe_divide(df["gen_coal"], df["gen_total"])

    return df

def add_net_load_features(df):
    '''
    net load = ail minus variable renewable generation
    '''
    df = df.copy()

    _require_columns(df, [
        'price__ACTUAL_AIL',
        'gen_wind',
        'gen_solar',
    ])

    df['demand'] = df['price__ACTUAL_AIL']

    df['net_load'] = (
        df['demand']
        - df['gen_wind']
        - df['gen_solar']
    )

    df['residual_load'] = (
        df['demand']
        - df['gen_renewable']
    )

    return df

# outage / reserve proxy feautures

def add_outage_features(df):
    df = df.copy()

    if 'outages__outage_total' not in df.columns:
        raise KeyError('missing outages__outage_total')
    
    df['outage_total'] = df['outages__outage_total']

    return df

def add_reserve_proxy_features(df):
    '''
    approximate system cushion
    proxy - not a true AESO reserve margin
    uses: 
    - available generation + net imports - demand

    does not apply 2600 MW behind-the-fence offset by default
    offset should be tested separately
    '''
    df = df.copy()

    avail_cols = [
        c for c in df.columns
        if c.startswith("gen__") and c.endswith("_system_available")
    ]

    if not avail_cols:
        raise KeyError("No generation availability columns found.")

    _require_columns(df, [
        "price__ACTUAL_AIL",
        "net_imports",
    ])

    df["available_generation_total"] = df[avail_cols].sum(axis=1)

    df["reserve_proxy"] = (
        df["available_generation_total"]
        + df["net_imports"]
        - df["price__ACTUAL_AIL"]
    )

    df["reserve_proxy_pct"] = _safe_divide(
        df["reserve_proxy"],
        df["price__ACTUAL_AIL"]
    )

    return df

# ramp / volatility features

def add_ramp_features(df):
    df = df.copy()

    _require_columns(df, [
        "demand",
        "net_load",
        "gen_wind",
        "gen_solar",
    ])

    df["demand_ramp"] = df["demand"].diff()
    df["abs_demand_ramp"] = df["demand_ramp"].abs()

    df["net_load_ramp"] = df["net_load"].diff()
    df["abs_net_load_ramp"] = df["net_load_ramp"].abs()

    df["wind_ramp"] = df["gen_wind"].diff()
    df["solar_ramp"] = df["gen_solar"].diff()

    df["wind_instability_24h"] = df["gen_wind"].rolling(24).std()
    df["solar_instability_24h"] = df["gen_solar"].rolling(24).std()

    df["residual_load_vol_24h"] = df["residual_load"].rolling(24).std()
    df["residual_load_vol_7d"] = df["residual_load"].rolling(24 * 7).std()
   
    return df

# stress score / interaction features

def add_stress_features(df):
    '''
    continuous physical stress indicators

    stress_score_norm:
        - high net load
        - low renewable share
        - low wind gen
    reproduces the original notebook 04 stress score and is intended as the primary interpretable market stress metric

    
    stress_score_extended:
        - higher net load
        - lower renewable share
        - lower wind generation
        - lower reserve proxy
    '''
    df = df.copy()

    _require_columns(df, [
        "net_load",
        "renewable_share",
        "gen_wind",
        "reserve_proxy_pct",
    ])

    df["net_load_pct"] = df["net_load"].rank(pct=True)
    df["low_renewable_pct"] = 1 - df["renewable_share"].rank(pct=True)
    df["low_wind_pct"] = 1 - df["gen_wind"].rank(pct=True)
    df["low_reserve_pct"] = 1 - df["reserve_proxy_pct"].rank(pct=True)

    # validated core score from Notebook 04
    df["stress_score"] = (
        df["net_load_pct"]
        + df["low_renewable_pct"]
        + df["low_wind_pct"]
    )

    df["stress_score_norm"] = df["stress_score"] / 3

    # optional extended score for experiments
    df["stress_score_extended"] = (
        df["net_load_pct"]
        + df["low_renewable_pct"]
        + df["low_wind_pct"]
        + df["low_reserve_pct"]
    ) / 4

    return df

def add_interaction_features(df):
    df = df.copy()

    _require_columns(df, [
        "abs_net_load_ramp",
        "net_load",
        "reserve_proxy_pct",
        "gen_wind",
        "renewable_share",
        "total_imports",
    ])

    df["ramp_x_netload"] = df["abs_net_load_ramp"] * df["net_load"]

    # Higher when ramp is large AND reserve cushion is low
    df["ramp_x_low_reserve"] = (
        df["abs_net_load_ramp"] * (1 - df["reserve_proxy_pct"].rank(pct=True))
    )

    df["wind_x_netload"] = df["gen_wind"] * df["net_load"]
    df["renewable_x_netload"] = df["renewable_share"] * df["net_load"]
    df["imports_x_netload"] = df["total_imports"] * df["net_load"]

    return df

# targets and lags

def add_scarcity_target(df, price_col='price__ACTUAL_POOL_PRICE', threshold=300):
    df = df.copy()

    _require_columns(df, [price_col])

    df['scarcity_event'] = (df[price_col] > threshold).astype(int)

    return df

def add_forecast_targets(df, horizons=(1, 3, 6, 12, 24)):
    df = df.copy()

    _require_columns(df, ['scarcity_event'])

    for h in horizons:
        df[f'scarcity_t_plus_{h}'] = df['scarcity_event'].shift(-h)
    
    return df

def add_lag_features(df, features, lags=(1, 3, 6, 12, 24)):
    """
    add explicit historical lag features
    example:
        net_load_lag1 = net_load observed 1 hour ago
    """
    df = df.copy()

    _require_columns(df, features)

    lagged = {}

    for feature in features:
        for lag in lags:
            lagged[f"{feature}_lag{lag}"] = df[feature].shift(lag)

    lagged_df = pd.DataFrame(lagged, index=df.index)

    return pd.concat([df, lagged_df], axis=1)

# main builder

def build_features(
    df, 
    add_targets = True,
    add_lags = False,
    lag_base_features = None,
    lags = (1, 3, 6, 12, 24),
):
    '''
    build model features from the canonical master dataset
    this replaces the older proxy-heavy feature pipeline
    '''
    df = df.copy()

    df = add_time_features(df)
    df = add_weather_features(df)
    df = add_intertie_features(df)
    df = add_generation_features(df)
    df = add_net_load_features(df)
    df = add_outage_features(df)
    df = add_reserve_proxy_features(df)
    df = add_ramp_features(df)
    df = add_stress_features(df)
    df = add_interaction_features(df)

    if add_targets:
        df = add_scarcity_target(df)
        df = add_forecast_targets(df)

    if add_lags:
        if lag_base_features is None:
            lag_base_features = [
                "net_load",
                "gen_wind",
                "gen_solar",
                "renewable_share",
                "reserve_proxy_pct",
                "total_imports",
                "stress_score_norm",
                "wind_instability_24h",
                "abs_net_load_ramp",
            ]
        df = add_lag_features(df, lag_base_features, lags=lags)

    return df
