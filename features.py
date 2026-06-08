
import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    df["MA_10"] = close.rolling(10).mean()
    df["MA_20"] = close.rolling(20).mean()
    df["MA_50"] = close.rolling(50).mean()

    df["EMA_12"]      = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"]      = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI_14"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    bb_mid         = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]

    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low  - prev_close).abs()], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.append(obv[-1] + vol.iloc[i])
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.append(obv[-1] - vol.iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv

    df["Returns"] = close.pct_change()
    for lag in range(1, 6):
        df[f"Lag_{lag}"] = close.shift(lag)

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "MA_10", "MA_20", "MA_50",
    "EMA_12", "EMA_26",
    "MACD", "MACD_Signal",
    "RSI_14",
    "BB_Upper", "BB_Lower", "BB_Width",
    "ATR_14", "OBV", "Returns",
    "Lag_1", "Lag_2", "Lag_3", "Lag_4", "Lag_5",
]