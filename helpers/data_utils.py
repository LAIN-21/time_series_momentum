import pandas as pd
import yfinance as yf
from pathlib import Path

def load_yahoo_close(ticker: str, data_dir: str = "data/daily", refresh: bool = False) -> pd.Series:
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    safe_ticker = ticker.replace("^", "").replace("/", "_")
    file_path = data_path / f"{safe_ticker}_adj_close.csv"

    if file_path.exists() and not refresh:
        px = pd.read_csv(file_path, index_col=0, parse_dates=True)["Close"].dropna()
        px.index = pd.to_datetime(px.index).tz_localize(None)
        px = px.sort_index()
        px.name = "Close"
        print(f"Loaded {ticker} from cache: {file_path} | rows={len(px)}")
        return px

    df = yf.download(
        ticker,
        period="max",
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        raise ValueError(f"No data downloaded for ticker '{ticker}'")

    px = df["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]

    px = px.dropna().sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px.name = "Close"

    px.to_frame().to_csv(file_path)
    print(f"Downloaded {ticker} from Yahoo and saved to: {file_path} | rows={len(px)}")

    return px

def daily_to_month_end(px: pd.Series) -> pd.Series:
    px = px.dropna().sort_index()
    px_m = px.resample("ME").last()
    px_m.name = "month_end_price"
    return px_m

def month_end_to_returns(px_m: pd.Series) -> pd.Series:
    r_m = px_m.pct_change()
    r_m.name = "asset_return"
    return r_m