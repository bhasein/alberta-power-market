Alberta Power Market Analysis

A data-driven research project exploring scarcity formation, price dynamics, system tightness, and forecasting opportunities within the Alberta electricity market.

Project Overview

This project investigates the physical and operational drivers of electricity prices in Alberta’s energy-only market. The objective is to move beyond simple price analysis and develop a systematic understanding of how demand, generation availability, outages, renewable production, imports, and system constraints interact to create scarcity conditions.

The project combines electricity market fundamentals with statistical analysis, feature engineering, and predictive modeling to identify the variables most strongly associated with market stress and elevated pool prices.

The long-term goal is to build forecasting frameworks capable of identifying scarcity risk before it occurs.

⸻

Research Questions

The project is organized around several core questions:

* What operational conditions characterize scarcity events?
* Which system variables contain the strongest scarcity information?
* How do scarcity conditions develop through time?
* What role do generation outages, renewable output, and imports play during system stress?
* Can physically meaningful system-state variables outperform traditional market indicators?
* How far in advance can scarcity events be forecasted?

⸻

Dataset

The analysis is built from an integrated hourly master dataset covering January 2020 through July 2025.

The dataset contains approximately 49,000 hourly observations and includes:

Market Variables

* Alberta Pool Price
* Hour-Ahead Pool Price Forecast
* Alberta Internal Load (AIL)
* AECO Natural Gas Prices

Generation Availability

* Fuel-specific available capacity
* System-wide available generation
* Thermal availability metrics

Outages

* Total outages
* Combined-cycle outages
* Simple-cycle outages
* Cogeneration outages
* Fuel-specific outage categories

Interties

* Imports and exports
* Net imports
* Historical intertie capability data

Renewable Generation

* Wind generation
* Solar generation
* Renewable share of supply

Derived Features

* Net load
* Thermal capacity margin
* Thermal availability percentage
* Scarcity event labels
* Stress score metrics
* Interaction features

⸻

Notebook Structure

Phase 1 — Market Exploration

Notebook 01

Initial market familiarization and exploratory analysis.

Notebook 02

Weather, load, and renewable generation analysis.

Notebook 03

Price drivers and descriptive market relationships.

Notebook 04

Early scarcity modeling and feature evaluation.

Notebook 05

Interaction effects, stress indicators, and system-state analysis.

Notebook 06

Initial scarcity forecasting experiments.

⸻

Phase 2 — Scarcity and System Tightness

Notebook 07 — Supply Constraints and Scarcity

Major findings:

* Thermal capacity margin emerged as the strongest scarcity variable tested (AUC = 0.813).
* Net load remains the dominant demand-side scarcity driver.
* Renewable withdrawal materially increases scarcity risk.
* Simple-cycle outages contain more scarcity information than aggregate outage measures.
* Imports behave primarily as indicators of system stress.
* Scarcity is best understood as a tightening supply-demand balance rather than a single-variable phenomenon.

Notebook 08 — Scarcity Formation and Temporal Dynamics (in progress)

Focus areas:

* Scarcity event development through time
* Lead-lag analysis of key operational variables
* Regime stability testing across years
* Import indicator versus predictor analysis
* Forecast-error feature construction
* Missed-event analysis

Notebook 09 — Scarcity Forecasting (planned)

Planned topics:

* Walk-forward validation
* Logistic regression baseline
* Multi-horizon forecasting
* Probability calibration
* Forecast performance across market regimes

⸻

Key Findings To Date

Thermal Capacity Margin

Thermal capacity margin is currently the strongest scarcity indicator identified in the project.

AUC: 0.813

The results suggest scarcity is fundamentally driven by the relationship between available dispatchable generation and system demand rather than either variable independently.

Net Load

Net load remains the strongest individual demand-side feature.

AUC: 0.780

Scarcity frequency increases substantially as net load rises.

Renewable Generation

Lower renewable output is consistently associated with scarcity events.

Renewable share and wind generation remain among the strongest scarcity-related variables examined.

Simple-Cycle Generation

Simple-cycle outages and simple-cycle capacity relative to load repeatedly emerge as meaningful scarcity indicators, particularly during already-stressed operating conditions.

Imports

Imports increase significantly during scarcity events but currently appear more consistent with a system response than a leading indicator.

⸻

Methodology

The analytical framework follows a structured research process:

1. Data characterization
2. Univariate analysis
3. Bivariate analysis
4. Segmented and regime analysis
5. Multivariate analysis
6. Temporal analysis
7. Statistical validation
8. Forecast validation
9. Residual and error analysis

The objective is to build conclusions that remain robust across years, market regimes, and analytical methods.

⸻

Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

⸻

Current Status

The project is transitioning from descriptive scarcity analysis toward forecasting and predictive modeling.

Current work focuses on understanding how scarcity develops before it occurs and identifying which operational signals provide actionable forecasting information.
