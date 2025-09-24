# app.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import pdb

from eth_rebalancer.strategy import run_backtest

# -----------------------------
# Robust price fetchers
# -----------------------------

@st.cache_data(show_spinner=True)
def fetch_yfinance_prices(start_date=None, end_date=None, interval="1h"):
    """
    Fetch ETH-USD candles from yfinance over ANY range by chunking.
    Returns DataFrame with ['Date','Close'].
    
    Args:
        start_date: Start date for data fetching
        end_date: End date for data fetching  
        interval: Yahoo Finance interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
                  Note: For sub-daily intervals, yfinance has ~60 day limits per request
    """
    import yfinance as yf

    if end_date is None: end_date = pd.to_datetime('today')
    if start_date is None: start_date = end_date - pd.Timedelta(days=365*4)

    #pdb.set_trace() 

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
            interval=interval,
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
def fetch_binance_prices(start_date=None, end_date=None, interval="1h"):
    """
    Fallback: fetch ETHUSDT klines from Binance public API with configurable interval.
    Treats USDT ≈ USD. Returns ['Date','Close'] (UTC).
    
    Args:
        start_date: Start date for data fetching
        end_date: End date for data fetching  
        interval: Binance API interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d)
    """
    if end_date is None: end_date = pd.to_datetime('today')
    if start_date is None: start_date = end_date - pd.Timedelta(days=365*4)

    start_dt = pd.to_datetime(start_date).tz_localize(timezone.utc)
    end_dt = pd.to_datetime(end_date).tz_localize(timezone.utc)

    url = "https://api.binance.us/api/v3/klines"
    limit = 1000  # max per request

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


def fetch_prices(start_date=None, end_date=None, interval="1h"):
    """
    Wrapper: try yfinance first (ETH-USD); if it fails, fallback to Binance.
    
    Args:
        start_date: Start date for data fetching
        end_date: End date for data fetching  
        interval: Time interval for candles
    """
    try:
  #     return fetch_yfinance_prices(start_date, end_date, interval)
        return fetch_binance_prices(start_date, end_date, interval)

    except Exception as e_yf:
        st.warning(f"yfinance fetch failed, trying Binance fallback… ({e_yf})")
        return fetch_binance_prices(start_date, end_date, interval)


def calculate_holding_value(df_prices, start_date, end_date, initial_capital):
    """
    Calculate the holding value (final ETH value in dollars) based on start and end date.
    
    Args:
        df_prices: DataFrame with Date and Close columns
        start_date: Selected start date
        end_date: Selected end date
        initial_capital: Initial capital amount
    
    Returns:
        float: Final ETH value in dollars if we held ETH from start to end
    """
    if df_prices.empty:
        return 0.0

    start_date_dt = pd.to_datetime(start_date)
    end_date_dt = pd.to_datetime(end_date)

    # Find the first available price ON or AFTER the start date
    start_filtered = df_prices[df_prices['Date'] >= start_date_dt]
    if start_filtered.empty:
        return 0.0
    initial_eth_price = start_filtered['Close'].iloc[0]

    # Find the last available price ON or BEFORE the end date
    end_filtered = df_prices[df_prices['Date'] <= end_date_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)]
    if end_filtered.empty:
        return 0.0
    final_eth_price = end_filtered['Close'].iloc[-1]

    # Calculate how much ETH you could buy at the start date
    initial_eth_amount = initial_capital / initial_eth_price

    # --- ETH APY compounding for holding ---
    # Try to get eth_apy from Streamlit session state, default to 0.0 if not set
    eth_apy = 0.0
    try:
        import streamlit as st
        eth_apy = st.session_state.get('eth_apy', 0.0)
    except Exception:
        eth_apy = 0.0

    # Calculate the number of days in the holding period
    days_held = (end_date_dt - start_date_dt).days
    if days_held < 0:
        return 0.0
    years_held = days_held / 365.0

    # Compound ETH amount by APY for the holding period
    if eth_apy > 0 and years_held > 0:
        # Assume APY is annualized, compounded once per year
        compounded_eth_amount = initial_eth_amount * ((1 + eth_apy) ** years_held)
    else:
        compounded_eth_amount = initial_eth_amount

    # Calculate final value of that ETH amount at the end date
    final_holding_value = compounded_eth_amount * final_eth_price
    return final_holding_value


# -----------------------------
# Streamlit UI (unchanged)
# -----------------------------
st.title("ETH Rebalancer Simulator")

# Define the absolute allowed date range (past 4 years)
today = datetime.now().date()
allowed_min_date = today - timedelta(days=365*4)  # 4 years ago
allowed_max_date = today

# Interval selection dropdown
interval_options = {
    "1m": "1 minute",
    "3m": "3 minutes", 
    "5m": "5 minutes",
    "15m": "15 minutes",
    "30m": "30 minutes",
    "1h": "1 hour",
    "2h": "2 hours",
    "4h": "4 hours", 
    "6h": "6 hours",
    "8h": "8 hours",
    "12h": "12 hours",
    "1d": "1 day"
}

# Initialize session state for interval
if 'selected_interval' not in st.session_state:
    st.session_state['selected_interval'] = "1h"

selected_interval = st.selectbox(
    "Select Time Interval",
    options=list(interval_options.keys()),
    format_func=lambda x: interval_options[x],
    index=list(interval_options.keys()).index(st.session_state['selected_interval']),
    key='interval_selector'
)

# Update session state if interval changed
if selected_interval != st.session_state['selected_interval']:
    st.session_state['selected_interval'] = selected_interval
    # Clear cache to force data reload with new interval
    st.cache_data.clear()
    st.rerun()

# Fetch full price data for slider range
# Use current start/end dates if they exist in session state, otherwise fetch default range first
if 'start_date' not in st.session_state or 'end_date' not in st.session_state:
    # First time - fetch default range to initialize slider bounds
    temp_df = fetch_prices(interval=selected_interval)
    min_date = temp_df['Date'].min().date()
    max_date = temp_df['Date'].max().date()
    
    # Initialize session state for range and individual dates
    if 'date_range' not in st.session_state:
        st.session_state['date_range'] = (min_date, max_date)
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = min_date
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = max_date
    
    full_df_prices = temp_df
else:
    # Use current start/end dates from session state, but expand range if needed to ensure data availability
    current_start = st.session_state['start_date']
    current_end = st.session_state['end_date']
    
    # Expand the fetch range to ensure we have data for the entire allowed range
    # This prevents issues when users select dates outside the currently loaded range
    fetch_start = min(current_start, allowed_min_date)
    fetch_end = max(current_end, allowed_max_date)
    
    full_df_prices = fetch_prices(start_date=fetch_start, end_date=fetch_end, interval=selected_interval)

# Update min_date and max_date from the fetched data for slider bounds
min_date = full_df_prices['Date'].min().date()
max_date = full_df_prices['Date'].max().date()



# Initial Capital, Start Date, End Date, Holding in one row

# --- UI for Initial Capital, Start Date, End Date (no holding yet) ---
col_init, col_start, col_end, col_holding = st.columns(4)
with col_init:
    initial = st.number_input("Initial Capital", 1000, 1000000, 100000)
with col_start:
    # Clamp default value to allowed range to avoid StreamlitAPIException
    default_start = min(max(st.session_state['start_date'], allowed_min_date), allowed_max_date)
    start_date = st.date_input("Start Date", default_start, min_value=allowed_min_date, max_value=allowed_max_date)
with col_end:
    default_end = min(max(st.session_state['end_date'], allowed_min_date), allowed_max_date)
    end_date = st.date_input("End Date", default_end, min_value=allowed_min_date, max_value=allowed_max_date)


# Two-way sync logic

date_range = (st.session_state['start_date'], st.session_state['end_date'])

# Check for dates outside the allowed 4-year range and show warnings
if start_date < allowed_min_date or start_date > allowed_max_date:
    st.warning(f"Start date is outside the allowed range ({allowed_min_date} to {allowed_max_date}). Please select a date within the past 4 years.")
if end_date < allowed_min_date or end_date > allowed_max_date:
    st.warning(f"End date is outside the allowed range ({allowed_min_date} to {allowed_max_date}). Please select a date within the past 4 years.")


# --- Sync session state for start/end date changes ---
if (start_date != st.session_state['start_date']) or (end_date != st.session_state['end_date']):
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    st.session_state['date_range'] = (start_date, end_date)
    st.rerun()


# --- Now display Holding value using the latest session state ---
# Add eth_apy as a dependency to force Streamlit to rerun this block when eth_apy changes
eth_apy = st.session_state.get('eth_apy', 0.0)
with col_holding:
    # Always use the latest session state for start/end date and eth_apy
    holding_value = calculate_holding_value(full_df_prices, st.session_state['start_date'], st.session_state['end_date'], initial)
    start_date_dt = pd.to_datetime(st.session_state['start_date'])
    end_date_dt = pd.to_datetime(st.session_state['end_date'])
    start_filtered = full_df_prices[full_df_prices['Date'] >= start_date_dt]
    end_filtered = full_df_prices[full_df_prices['Date'] <= end_date_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)]
    start_eth_price = start_filtered['Close'].iloc[0] if not start_filtered.empty else 0.0
    end_eth_price = end_filtered['Close'].iloc[-1] if not end_filtered.empty else 0.0
    st.markdown(
        f"""
        <div style="text-align: left; margin-top: 20px;">
            <p style="font-size: 14px; color: #666; margin-bottom: 2px;">Holding</p>
            <p style="font-size: 16px; font-weight: bold; margin-top: 0px; margin-bottom: 8px;">${holding_value:,.2f}</p>
            <p style="font-size: 11px; color: #888; margin: 0px;">Start: ${start_eth_price:,.2f}</p>
            <p style="font-size: 11px; color: #888; margin: 0px;">End: ${end_eth_price:,.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
        help="Final ETH value in dollars if held from start to end date"
    )


start_date, end_date = st.session_state['start_date'], st.session_state['end_date']
df_prices = full_df_prices[(full_df_prices['Date'] >= pd.to_datetime(start_date)) & (full_df_prices['Date'] <= pd.to_datetime(end_date))]
st.subheader(f"ETH Prices ({start_date} to {end_date})")
st.line_chart(df_prices.set_index('Date')['Close'])

# Date range slider right below the graph
date_range_slider = st.slider(
    "Select Date Range",
    min_value=max(min_date, allowed_min_date),
    max_value=min(max_date, allowed_max_date),
    value=(start_date, end_date),
    format="YYYY-MM-DD"
)


threshold = st.slider("Rebalance Threshold", 0.01, 0.1, 0.05)

# Advanced options expander
with st.expander("Advanced Options"):
    eth_weight = st.slider("Initial ETH Weight", 0.0, 1.0, 0.5)
    sma = st.slider("SMA Window", 50, 400, 200)
    alpha = st.slider("Alpha", 0.1, 1.0, 0.5)
    stable_apy = st.slider("Stablecoin APY", 0.0, 0.2, 0.00)
    eth_apy = st.slider("ETH APY", 0.0, 0.2, 0.00, key='eth_apy_slider')
    fee_bps = st.slider("Fee (bps)", 0, 100, 5)
    slip_bps = st.slider("Slippage (bps)", 0, 100, 10)
    use_bands = st.checkbox("Use ATR Bands")
    atr = st.number_input("ATR Window", 5, 50, 14)
    atr_k = st.number_input("ATR Multiplier", 0.5, 5.0, 2.0)

# --- Sync session state for eth_apy changes ---
if 'eth_apy' not in st.session_state:
    st.session_state['eth_apy'] = eth_apy
elif eth_apy != st.session_state['eth_apy']:
    st.session_state['eth_apy'] = eth_apy
    st.rerun()

# Sync logic for slider
if date_range_slider != (start_date, end_date):
    st.session_state['start_date'], st.session_state['end_date'] = date_range_slider
    st.session_state['date_range'] = date_range_slider
    st.rerun()

if st.button("Run Simulation"):
    # Fetch data (same name/shape as before: Date + Close)
    df = df_prices

    #pdb.set_trace() 

    # Calculate interval-based multipliers
    def calculate_multipliers(interval, stable_apy, eth_apy):
        # Define conversion factors for each specific interval
        interval_to_periods_per_year = {
            "1m": 365 * 24 * 60,      # 525,600 periods per year
            "3m": 365 * 24 * 20,      # 175,200 periods per year (60/3 = 20 periods per hour)
            "5m": 365 * 24 * 12,      # 105,120 periods per year (60/5 = 12 periods per hour)
            "15m": 365 * 24 * 4,      # 35,040 periods per year (60/15 = 4 periods per hour)
            "30m": 365 * 24 * 2,      # 17,520 periods per year (60/30 = 2 periods per hour)
            "1h": 365 * 24,           # 8,760 periods per year
            "2h": 365 * 12,           # 4,380 periods per year (24/2 = 12 periods per day)
            "4h": 365 * 6,            # 2,190 periods per year (24/4 = 6 periods per day)
            "6h": 365 * 4,            # 1,460 periods per year (24/6 = 4 periods per day)
            "8h": 365 * 3,            # 1,095 periods per year (24/8 = 3 periods per day)
            "12h": 365 * 2,           # 730 periods per year (24/12 = 2 periods per day)
            "1d": 365                 # 365 periods per year
        }
        
        # Get periods per year for the specific interval
        periods_per_year = interval_to_periods_per_year.get(interval)
        
        if periods_per_year is None:
            # Fallback for unexpected interval format
            if interval.endswith('m'):  # minutes
                minutes = int(interval[:-1])
                periods_per_year = 365 * 24 * (60 // minutes)
            elif interval.endswith('h'):  # hours
                hours = int(interval[:-1])
                periods_per_year = 365 * (24 // hours)
            elif interval.endswith('d'):  # days
                days = int(interval[:-1])
                periods_per_year = 365 // days
            else:
                # Default to daily if interval format is unexpected
                periods_per_year = 365
        
        stable_mult = 1.0 + stable_apy / periods_per_year
        eth_mult = 1.0 + eth_apy / periods_per_year
        
        return stable_mult, eth_mult
    
    stable_mult, eth_mult = calculate_multipliers(selected_interval, stable_apy, eth_apy)

    # Build args object with interval and multipliers
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
        "atr_k": atr_k,
        "interval": selected_interval,
        "stable_mult": stable_mult,
        "eth_mult": eth_mult
    })()

    # Run backtest (your original logic)
    result = run_backtest(df, args)

    # Display results (same components)
    st.subheader("Simulation Summary")
    st.json(result['summary'])

    st.subheader("Portfolio Equity Curve")
    import matplotlib.pyplot as plt
    eq_curve = result['equity_curve']
    trades = result.get('trades', pd.DataFrame())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq_curve['Date'], eq_curve['portfolio'], label='Portfolio')

    # Add BUY/SELL markers
    if not trades.empty:
        buy_trades = trades[trades['Action'] == 'BUY']
        sell_trades = trades[trades['Action'] == 'SELL']
        ax.scatter(buy_trades['Date'], eq_curve.set_index('Date').loc[buy_trades['Date'], 'portfolio'], color='green', marker='^', s=100, label='BUY')
        ax.scatter(sell_trades['Date'], eq_curve.set_index('Date').loc[sell_trades['Date'], 'portfolio'], color='red', marker='v', s=100, label='SELL')
        # Add text labels
        for _, row in buy_trades.iterrows():
            y = eq_curve.set_index('Date').loc[row['Date'], 'portfolio']
            ax.text(row['Date'], y, 'BUY', color='green', fontsize=8, ha='left', va='bottom', rotation=45)
        for _, row in sell_trades.iterrows():
            y = eq_curve.set_index('Date').loc[row['Date'], 'portfolio']
            ax.text(row['Date'], y, 'SELL', color='red', fontsize=8, ha='right', va='top', rotation=45)
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("All Trades")
    if 'trades' in result:
        st.dataframe(result['trades'])
