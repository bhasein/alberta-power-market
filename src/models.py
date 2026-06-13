import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def train_baseline_model(
    df,
    features,
    target="pool_price",
    test_size=0.2,
    random_state=42
):

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    r2 = model.score(X_test, y_test)

    return {
        "model": model,
        "r2": r2,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def prepare_data(df, features, target='scarcity_event'):
    X = df[features].dropna()
    y = df.loc[X.index, target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns


def evaluate_model(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print("\n=== MODEL PERFORMANCE ===")
    print(f"AUC: {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

    print("\nConfusion Matrix:")
    print(cm)

    return auc, precision, recall, cm

def prepare_forecast_data(df, features, target, split_ratio=0.8):
    """
    Time-safe train/test split for forecasting models.

    Uses chronological split instead of random split to prevent
    future information leaking into training data.

    Parameters
    ----------
    df         : DataFrame with lagged features and forward target already built.
    features   : list of lagged feature column names.
    target     : forward-looking target (e.g. scarcity_t_plus_1).
    split_ratio: fraction of data used for training (default 0.8).
    """
    data = pd.concat([df[features], df[target]], axis=1).dropna()
    data = data.sort_index()

    X = data[features]
    y = data[target]

    split_idx = int(len(data) * split_ratio)

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, features



