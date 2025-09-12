# app.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

from eth_rebalancer.strategy import run_backtest

# -----------------------------
# Robust price fetchers
# -----------------------------

@st.cache_data(show_spinner=True)
def fetch_yfinance_prices(start_date=None, end_date=None):
    """
    Fetch ETH-USD 1h candles from yfinance over ANY range by chunking.
    Returns DataFrame with ['Date','Close'].
    """
    import yfinance as yf

    if end_date is None: end_date = pd.to_datetime('today')
    if start_date is None: start_date = end_date - pd.Timedelta(days=365*4)

    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # yfinance intraday has server-side limits; chunk to ~59 days per request (safe under 60d windows)
    chunk_days = 59
    frames = []
    cur = start_dt
    while cur <= end_dt:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end_dt) + pd.Timedelta(days=1)  # end is exclusive
        raw = yf.download(
            "ETH-USD",
            start=cur.tz_localize(None),
            end=nxt.tz_localize(None),
            interval="1h",
            auto_adjust=True,
            group_by="column",
            progress=False,
            threads=True,
        )
        if raw is not None and len(raw) > 0:
            frames.append(raw)
        cur = (cur + pd.Timedelta(days=chunk_days + 1)).normalize()

    if not frames:
        raise RuntimeError("yfinance returned no data.")

    raw_all = pd.concat(frames).sort_index()
    raw_all.index = pd.to_datetime(raw_all.index).tz_localize(None)

    # Normalize Close (handle MultiIndex or Adj Close)
    if isinstance(raw_all.columns, pd.MultiIndex):
        try:
            close = raw_all.xs("Close", axis=1, level=-1)
        except Exception:
            try:
                close = raw_all.xs("Close", axis=1, level=0)
            except Exception:
                close = None
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = raw_all["Close"] if "Close" in raw_all.columns else raw_all.get("Adj Close")

    if close is None:
        raise RuntimeError("yfinance data did not contain Close/Adj Close.")

    df = pd.DataFrame({"Date": raw_all.index, "Close": pd.to_numeric(close, errors="coerce")})
    # Strict trim to requested range
    df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59))]
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    # Basic gap handling (optional): forward-fill tiny gaps if any
    df["Close"] = df["Close"].ffill()
    if df["Close"].isna().all():
        raise RuntimeError("yfinance returned only NaNs for Close.")
    return df[["Date", "Close"]]


@st.cache_data(show_spinner=True)
def fetch_binance_prices(start_date=None, end_date=None):
    """
    Fallback: fetch ETHUSDT 1h klines from Binance public API.
    Treats USDT ≈ USD. Returns ['Date','Close'] (UTC).
    """
    if end_date is None: end_date = pd.to_datetime('today')
    if start_date is None: start_date = end_date - pd.Timedelta(days=365*4)

    start_dt = pd.to_datetime(start_date).tz_localize(timezone.utc)
    end_dt = pd.to_datetime(end_date).tz_localize(timezone.utc)

    url = "https://api.binance.com/api/v3/klines"
    interval = "1h"
    limit = 1000  # max per request (~41.6 days of 1h bars)

    frames = []
    cur = int(start_dt.timestamp() * 1000)
    end_ms = int((end_dt + pd.Timedelta(days=1)).timestamp() * 1000)

    while cur < end_ms:
        params = {"symbol": "ETHUSDT", "interval": interval, "startTime": cur, "endTime": end_ms, "limit": limit}
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Binance API error: {r.status_code} {r.text}")

        data = r.json()
        if not data:
            break

        # kline fields: [0] open time ms, [1] open, [2] high, [3] low, [4] close, [5] volume, [6] close time ms, ...
        chunk = pd.DataFrame(data, columns=[
            "open_time","open","high","low","close","volume","close_time","qav","num_trades",
            "taker_base","taker_quote","ignore"
        ])
        chunk["Date"] = pd.to_datetime(chunk["open_time"], unit="ms", utc=True).dt.tz_convert(None)
        chunk["Close"] = pd.to_numeric(chunk["close"], errors="coerce")
        frames.append(chunk[["Date","Close"]])

        # Advance: next start = last close_time + 1 ms
        last_close_time = int(data[-1][6])
        cur = last_close_time + 1
        # Safety to avoid infinite loops
        if len(data) < limit:
            break

    if not frames:
        raise RuntimeError("Binance returned no data.")

    df = pd.concat(frames).drop_duplicates(subset=["Date"]).sort_values("Date")
    # Trim strictly to requested dates (naive)
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date) + pd.Timedelta(hours=23, minutes=59, seconds=59))]
    df = df.reset_index(drop=True)
    if df["Close"].isna().all():
        raise RuntimeError("Binance returned only NaNs for Close.")
    return df[["Date","Close"]]


def fetch_prices(start_date=None, end_date=None):
    """
    Wrapper: try yfinance first (ETH-USD 1h); if it fails, fallback to Binance 1h.
    """
    try:
        return fetch_yfinance_prices(start_date, end_date)
    except Exception as e_yf:
        st.warning(f"yfinance fetch failed, trying Binance fallback… ({e_yf})")
        return fetch_binance_prices(start_date, end_date)


# -----------------------------
# Streamlit UI (unchanged)
# -----------------------------
st.title("ETH Rebalancer Simulator")

# Sidebar inputs (kept as-is)
start_date = st.date_input("Start Date", pd.to_datetime("2021-09-06"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
initial = st.number_input("Initial Capital", 1000, 1000000, 100000)
eth_weight = st.slider("Initial ETH Weight", 0.0, 1.0, 0.5)
sma = st.slider("SMA Window", 50, 400, 200)
alpha = st.slider("Alpha", 0.1, 1.0, 0.5)
threshold = st.slider("Rebalance Threshold", 0.01, 0.1, 0.03)
stable_apy = st.slider("Stablecoin APY", 0.0, 0.2, 0.06)
eth_apy = st.slider("ETH APY", 0.0, 0.2, 0.025)
fee_bps = st.slider("Fee (bps)", 0, 100, 5)
slip_bps = st.slider("Slippage (bps)", 0, 100, 10)
use_bands = st.checkbox("Use ATR Bands")
atr = st.number_input("ATR Window", 5, 50, 14)
atr_k = st.number_input("ATR Multiplier", 0.5, 5.0, 2.0)

# Action
if st.button("Fetch Real ETH Prices & Run Simulation"):
    # Fetch data (same name/shape as before: Date + Close)
    df = fetch_prices(start_date=pd.to_datetime(start_date), end_date=pd.to_datetime(end_date))

    # Build args object (unchanged)
    args = type("Args", (), {
        "initial": initial,
        "eth_weight": eth_weight,
        "sma": sma,
        "alpha": alpha,
        "threshold": threshold,
        "stable_apy": stable_apy,
        "eth_apy": eth_apy,
        "fee_bps": fee_bps,
        "slip_bps": slip_bps,
        "use_bands": use_bands,
        "atr": atr,
        "atr_k": atr_k
    })()

    # Run backtest (your original logic)
    result = run_backtest(df, args)

    # Display results (same components)
    st.subheader("Simulation Summary")
    st.json(result['summary'])

    st.subheader("Portfolio Equity Curve")
    st.line_chart(result['equity_curve'].set_index('Date')['portfolio'])

    st.subheader("All Trades")
    if 'trades' in result:
        st.dataframe(result['trades'])
