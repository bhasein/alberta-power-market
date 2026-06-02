from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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