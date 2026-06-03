import numpy as np
import pandas as pd


def add_hdd_cdd(df):
    df["HDD"] = np.maximum(0, 18 - df["calgary_temperature_2m"])
    df["CDD"] = np.maximum(0, df["calgary_temperature_2m"] - 18)
    return df


def add_wind_index(df):
    df["wind_index"] = (
        df["lethbridge_wind_speed_100m"] +
        df["medicine_hat_wind_speed_100m"]
    ) / 2
    return df


def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    return df


def add_net_imports(df):
    """
    Net interchange with adjacent provinces/states.
    Positive = Alberta net importer (drawing from BC/SK/MT).
    Negative = Alberta net exporter (sending surplus out).
    Range observed 2020-2025: -1082 to +1118 MW.
    """
    df["net_imports"] = (
        df["IMPORT_BC"] + df["IMPORT_MT"] + df["IMPORT_SK"]
    ) - (
        df["EXPORT_BC"] + df["EXPORT_MT"] + df["EXPORT_SK"]
    )
    return df


def add_generation_features(df):
    """
    Generation mix features from hourly fuel type data.

    renewable_share  — wind + solar fraction of total gen.
                       High = price suppression risk.
                       Low  = scarcity risk.
    thermal_share    — dispatchable gas fraction; proxy for
                       marginal cost of generation.
    coal_share       — tracks retirement trajectory 2020-2025.
    """
    gen_cols = [
        "gen_coal", "gen_cogeneration", "gen_combined_cycle",
        "gen_dual_fuel", "gen_gas_fired_steam", "gen_hydro",
        "gen_other", "gen_simple_cycle", "gen_solar",
        "gen_storage", "gen_wind",
    ]
    df["gen_total"] = df[gen_cols].sum(axis=1)

    df["renewable_share"] = (
        df["gen_wind"] + df["gen_solar"]
    ) / df["gen_total"]

    df["thermal_share"] = (
        df["gen_combined_cycle"] +
        df["gen_gas_fired_steam"] +
        df["gen_simple_cycle"] +
        df["gen_cogeneration"]
    ) / df["gen_total"]

    df["coal_share"] = df["gen_coal"] / df["gen_total"]
    return df


def add_ramp_features(df):
    df = df.sort_values("timestamp")

    # --- net load (must be saved as column before diffing) ---
    # demand minus variable renewables
    # represents dispatchable generation requirement
    df["net_load"] = df["demand"] - df["gen_wind"] - df["gen_solar"]

    # --- base ramps ---
    df["demand_ramp"] = df["demand"].diff()
    df["net_load_ramp"] = df["net_load"].diff()
    df["gen_ramp"] = df["gen_total"].diff()

    # --- absolute ramps ---
    df["abs_demand_ramp"] = df["demand_ramp"].abs()
    df["abs_net_load_ramp"] = df["net_load_ramp"].abs()
    df["abs_gen_ramp"] = df["gen_ramp"].abs()

    # --- normalized ramps ---
    df["demand_ramp_pct"] = df["demand_ramp"] / df["demand"]
    df["net_load_ramp_pct"] = df["net_load_ramp"] / df["demand"]

    # --- renewable instability ---
    df["wind_ramp"] = df["gen_wind"].diff()
    df["solar_ramp"] = df["gen_solar"].diff()

    df["wind_instability_24h"] = df["wind_ramp"].rolling(24).std()
    df["solar_instability_24h"] = df["solar_ramp"].rolling(24).std()

    # --- ramp stress conditional on reserve tightness ---
    # reserve_proxy must exist before this runs
    # (add_reserve_features must run before add_ramp_features)
    df["ramp_stress"] = (
        df["abs_net_load_ramp"] /
        (df["reserve_proxy"].abs() + 1e-6)
    )

    return df


def add_residual_load_features(df):
    """
    Residual load and rolling volatility — key regime signals.

    residual_load     — ACTUAL_AIL minus ALL tracked generation.
                        What the system still needs to balance
                        (storage dispatch, imports, untracked gen).
                        Distinct from net_load which only strips renewables.

    vol_24h           — 24-period rolling std dev of residual load.
                        Intraday stress signal.
    vol_7d            — 168-period rolling std dev.
                        Weekly regime signal.
    regime            — Binary label split at vol_24h median.
                        'stressed' | 'stable'. Useful for stratified
                        backtesting and model diagnostics.

    gen_total must be computed before calling this (add_generation_features first).
    """
    df["residual_load"] = df["demand"] - df["gen_total"]

    df["residual_load_vol_24h"] = (
        df["residual_load"]
        .rolling(24, min_periods=12)
        .std()
    )
    df["residual_load_vol_7d"] = (
        df["residual_load"]
        .rolling(24 * 7, min_periods=24)
        .std()
    )

    vol_median = df["residual_load_vol_24h"].median()
    df["regime"] = (
        df["residual_load_vol_24h"]
        .gt(vol_median)
        .map({True: "stressed", False: "stable"})
    )
    return df

def add_import_dependence(df):
    """
    import dependence - fraction of Alberta demand served by imports. 
    
    alberta has limited intertie capacity between BC, SK and MT.
    high import dependence signals the system is leaning heavily on neighbors.
    theres less emergency buffere abailable
    
    during scarcity events, import dependence near maximum intertie 
    capacity means no additional help is avaiable externally. 
    

    total_imports        — raw MW flowing in from all neighbors
    import_dependence    — imports as fraction of total demand
                           0.05 = 5% of demand served by imports
                           typical range: 0 to ~0.15 in normal hours
                           can spike higher during stress events
    """

    df['total_imports'] = (
        df['IMPORT_BC'] + 
        df['IMPORT_MT'] + 
        df['IMPORT_SK']
    )

    df['import_dependence'] = (
        df['total_imports'] / df['demand']
    )

    return df

# 2600 MW systematic gap makes it unreliable as a regime classifier
def add_reserve_features(df):

    COGEN_OFFSET = 2600

    df['reserve_proxy'] = df['gen_total'] + df['total_imports'] + COGEN_OFFSET - df['demand']
    df['reserve_proxy_pct'] = df['reserve_proxy'] / df['demand']
    return df

def add_interaction_features(df):
    # ramp stress conditional on system load
    df["ramp_x_netload"] = (
        df["abs_net_load_ramp"]
        * df["net_load"]
    )

    # ramp stress conditional on reserve margin
    df["ramp_x_reserve"] = (
        df["abs_net_load_ramp"]
        * (-df["reserve_proxy_pct"])
    )

    # wind drop during high net load
    df["wind_x_netload"] = (
        df["gen_wind"]
        * df["net_load"]
    )

    # low renewables during high net load
    df["renewable_x_netload"] = (
        df["renewable_share"]
        * df["net_load"]
    )

    # imports during stress
    df["imports_x_netload"] = (
        df["total_imports"]
        * df["net_load"]
    )

    return df

def build_features(df):
    df = add_hdd_cdd(df)
    df = add_wind_index(df)
    df = add_time_features(df)
    df = add_net_imports(df)
    df = add_generation_features(df)
    df = add_residual_load_features(df)
    df = add_import_dependence(df)       # creates total_imports
    df = add_reserve_features(df)        # creates reserve_proxy
    df = add_ramp_features(df)           # now can use both safely
    df = add_interaction_features(df)
    
    return df