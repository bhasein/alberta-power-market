# Alberta Electricity Market Regime & Price Dynamics

A data-driven exploration of Alberta's electricity market, focusing on price formation, weather sensitivity, and regime behavior (normal vs scarcity conditions).

This project builds a progressively more structured model of Alberta pool price dynamics using weather data, demand, and market fundamentals, with the goal of understanding both:
- baseline price behavior under normal system conditions
- extreme price formation during scarcity events

---

## 🔍 Core Research Question

How do weather conditions, demand, and market fundamentals interact to drive electricity prices in Alberta — and why do standard models fail during scarcity events?

---

## ⚙️ Project Structure

The project is organized into sequential notebooks:

### 01 — Market Price Exploration
- Initial analysis of Alberta pool price behavior
- Time series trends, distributions, and seasonality
- Identification of extreme price spikes ($999 events)

### 02 — Weather & Fundamental Relationships
- Engineering weather-driven features:
  - Heating Degree Days (HDD)
  - Cooling Degree Days (CDD)
  - Wind index (southern Alberta corridor)
- Correlation analysis between weather, demand, and price
- First structural insights into price drivers

### 03 — Regime Decomposition & Linear Models
- Baseline linear regression models
- Inclusion of AECO gas prices (fuel benchmark)
- Residual analysis and model failure during scarcity events
- Identification of structural regime differences

### 04 — State-Based Regime Modeling
- Classification of market states (scarcity vs normal)
- Threshold-based regime definitions
- Comparison of model performance across regimes

### 05 — Threshold Regime Systems
- Sensitivity analysis of scarcity thresholds
- Stress testing model assumptions
- Exploration of nonlinear system behavior

---

## 📊 Key Features Engineered

- Demand and net load
- Wind and solar variability
- Heating and cooling degree days
- Ramp rates (system stress indicators)
- Import/export flows
- AECO gas price (fuel benchmark)
- Hourly and seasonal time features

---

## 📈 Key Findings (So Far)

### 1. Alberta is a gas-linked electricity market
AECO gas prices are a dominant structural driver of electricity prices.

### 2. Weather effects are nonlinear and regime-dependent
- Cooling demand (summer) has a stronger price impact than heating demand
- Wind suppresses prices, but mainly during constrained system conditions

### 3. Scarcity events dominate model error
Extreme prices ($500–$999/MWh) are not explained by weather or demand alone.

They are driven by:
- reserve tightness
- outages
- renewable variability
- operational constraints

### 4. Linear models capture baseline behavior only
R² improves significantly when gas prices are included, but:
- ~20–30% of variance explained in simple models
- remaining variance is structural and regime-driven

---

## ⚠️ Key Insight

Alberta electricity prices are best understood as a **two-regime system**:

### 1. Fundamental Regime
- Driven by gas prices, demand, and weather
- Relatively predictable
- Captured reasonably well by linear models

### 2. Scarcity / Stress Regime
- Driven by system constraints and operational stress
- Highly nonlinear and event-driven
- Responsible for extreme price spikes

---

## 🧪 Modeling Approach

- Linear regression baseline models
- Feature engineering for weather and system state
- Residual analysis for model failure detection
- Regime separation via threshold methods

Future work will explore:
- tree-based models (Random Forest, XGBoost)
- regime classification models
- probabilistic forecasting of scarcity events

---

## 🏗️ Data Sources

- Alberta Electric System Operator (AESO) pool price data
- Weather data (Calgary, southern Alberta wind corridor)
- Import/export flows
- AECO natural gas benchmark prices

---

## 🚧 Status

This project is ongoing and evolving from:
> exploratory data analysis → feature engineering → regime modeling → predictive system modeling

---

## 🎯 Long-Term Goal

To build a structured, interpretable model of electricity price formation that can be used for:

- energy market analysis
- infrastructure investment insight
- trading signal research
- scenario and stress testing (AESO-style thinking)

---

## 📌 Author

Brodie Hasein