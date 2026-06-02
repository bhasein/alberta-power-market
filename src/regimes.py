import numpy as np
import pandas as pd


def add_reserve_regime(df):

    # relaxed thresholds — catches moderate stress not just extremes
    df["regime_high_net_load"] = (
        df["net_load"] > df["net_load"].quantile(0.75)  # was 0.85
    ).astype(int)

    df["regime_low_renewable"] = (
        df["renewable_share"] < df["renewable_share"].quantile(0.25)  # was 0.15
    ).astype(int)

    # high wind instability — volatile wind precedes many scarcity events
    df["regime_high_wind_instability"] = (
        df["wind_instability_24h"] > df["wind_instability_24h"].quantile(0.75)
    ).astype(int)

    # high import dependence — already leaning on neighbors
    df["regime_high_imports"] = (
        df["total_imports"] > df["total_imports"].quantile(0.75)
    ).astype(int)

    # original combined tightness
    df["regime_system_tight"] = (
        (df["regime_high_net_load"] == 1) &
        (df["regime_low_renewable"] == 1)
    ).astype(int)

    # extended stress regime — catches moderate conditions
    df["regime_extended_stress"] = (
        (df["regime_high_net_load"] == 1) &
        (df["regime_low_renewable"] == 1) |
        (df["regime_high_wind_instability"] == 1) &
        (df["regime_high_imports"] == 1)
    ).astype(int)

    return df


def add_renewable_regime(df):
    """
    captures renewable-driven system fragility
    """

    df['low_wind'] = (
        df['gen_wind'] < df['gen_wind'].quantile(0.2)
    ).astype(int)

    df['high_renewable_share'] = (
        df['renewable_share'] > df['renewable_share'].quantile(0.8)
    ).astype(int)

    df['renewable_stress_regime'] = (
        (df['low_wind'] == 1) & 
        (df['reserve_proxy'] < df['reserve_proxy'].quantile(0.2))
    ).astype(int)

    return df


def add_ramp_regime(df):
    ramp_threshold = df["abs_net_load_ramp"].quantile(0.9)

    df["high_ramp_event"] = (
        df["abs_net_load_ramp"] > ramp_threshold
    ).astype(int)

    # ramp stress only matters when system is already tight
    df["ramp_stress_regime"] = (
        (df["high_ramp_event"] == 1) &
        (df["regime_system_tight"] == 1)
    ).astype(int)

    return df


def add_scarcity_label(df, price_threshold=300):
    """
    This is NOT a feature.
    This is the outcome variable for evaluation.
    """

    df["scarcity_event"] = (df["pool_price"] > price_threshold).astype(int)

    return df


def build_regimes(df):
    df = add_reserve_regime(df)
    df = add_renewable_regime(df)
    df = add_ramp_regime(df)
    df = add_scarcity_label(df)

    df["system_state"] = np.select(
        [
            df["regime_system_tight"] == 1,
            df["renewable_stress_regime"] == 1,
            df["ramp_stress_regime"] == 1,
        ],
        [
            "system_tight",
            "renewable_stress",
            "ramp_stress",
        ],
        default="normal"
    )

    return df