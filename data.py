
import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
   
    df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for '{ticker}'. Check the ticker symbol.")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def get_company_info(ticker: str) -> dict:
    """Return basic company metadata from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "name":     info.get("longName", ticker),
            "sector":   info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "currency": info.get("currency", "USD"),
            "country":  info.get("country", "N/A"),
            "market_cap": info.get("marketCap", None),
            "pe_ratio": info.get("trailingPE", None),
            "52w_high": info.get("fiftyTwoWeekHigh", None),
            "52w_low":  info.get("fiftyTwoWeekLow", None),
        }
    except Exception:
        return {"name": ticker}