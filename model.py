
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import FEATURE_COLS


def _metrics(y_true, y_pred) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R²": round(r2, 4), "MAPE (%)": round(mape, 2)}


def train_and_predict(
    df_feat: pd.DataFrame,
    model_type: str = "Random Forest",
    forecast_days: int = 30,
    test_size: float = 0.2,
) -> dict:
   
    df = df_feat.copy()
    available = [c for c in FEATURE_COLS if c in df.columns]

    X = df[available].values
    y = df["Close"].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    split = int(len(X_sc) * (1 - test_size))
    X_train, X_test = X_sc[:split], X_sc[split:]
    y_train, y_test = y_sc[:split], y_sc[split:]
    train_dates = df.index[:split]
    test_dates  = df.index[split:]

    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                      random_state=42, n_jobs=-1)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                          learning_rate=0.05, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)

    def inv(arr):
        return scaler_y.inverse_transform(arr.reshape(-1, 1)).ravel()

    train_pred = inv(model.predict(X_train))
    test_pred  = inv(model.predict(X_test))
    y_train_r  = inv(y_train)
    y_test_r   = inv(y_test)

    # ── Walk-forward future forecast ─────────────────────────────────────────
    lag_indices = [available.index(f"Lag_{i}") for i in range(1, 6)
                   if f"Lag_{i}" in available]
    last_row = X_sc[-1].copy()
    future_preds_sc = []

    for _ in range(forecast_days):
        p = model.predict(last_row.reshape(1, -1))[0]
        future_preds_sc.append(p)
        if len(lag_indices) >= 2:
            for j in range(len(lag_indices) - 1, 0, -1):
                last_row[lag_indices[j]] = last_row[lag_indices[j - 1]]
            last_row[lag_indices[0]] = p

    future_pred  = inv(np.array(future_preds_sc))
    future_dates = pd.bdate_range(
        start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days
    )

    fi = None
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_,
                       index=available).sort_values(ascending=False)

    return {
        "model": model,
        "y_train": y_train_r, "y_test": y_test_r,
        "train_pred": train_pred, "test_pred": test_pred,
        "train_dates": train_dates, "test_dates": test_dates,
        "metrics_train": _metrics(y_train_r, train_pred),
        "metrics_test":  _metrics(y_test_r,  test_pred),
        "future_dates": future_dates, "future_pred": future_pred,
        "feature_importances": fi,
        "feature_cols": available,
    }