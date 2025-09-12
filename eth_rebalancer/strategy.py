import pandas as pd
import numpy as np

def run_backtest(df, args):
    """
    Simple rebalance strategy with SMA bands and optional ATR bands.

    Inputs (from args):
      - initial (float): starting USD capital
      - eth_weight (float): initial fraction in ETH (0..1)
      - sma (int): SMA window (periods)
      - alpha (float): not used in this basic version (kept for compatibility)
      - threshold (float): percent band size for SMA mode (e.g., 0.20 == 20%)
      - stable_apy (float): annual APY on stables (e.g., 0.06)
      - eth_apy (float): annual APY on ETH (e.g., 0.025)
      - fee_bps (int): trade fee in basis points (0..100+)
      - slip_bps (int): slippage in basis points (0..100+)
      - use_bands (bool): if True, use ATR bands; else SMA percent bands
      - atr (int): ATR window (periods) if use_bands=True
      - atr_k (float): ATR multiplier if use_bands=True

    DataFrame requirements:
      - df has columns ['Date','Close'] (Close numeric; Date datetime)

    Strategy (all-in / all-out):
      - If not using ATR bands:
          Upper = SMA * (1 + threshold)
          Lower = SMA * (1 - threshold)
      - If using ATR bands:
          Upper = SMA + atr_k * ATR
          Lower = SMA - atr_k * ATR
        ATR is approximated if High/Low not available:
          ATR ≈ rolling mean of |Close - Close.shift(1)|

      - BUY all (convert all stables to ETH) when Close <= Lower
      - SELL all (convert all ETH to stables) when Close >= Upper
      - Fees/slippage applied on each trade
      - APY compounds per bar (assumes ~daily bars; for sub-daily still compounds per bar)
    """

    # --- Defensive copy and basic hygiene ---
    df = df.copy()
    if 'Date' not in df.columns or 'Close' not in df.columns:
        raise ValueError("Input DataFrame must contain 'Date' and 'Close' columns.")

    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close']).reset_index(drop=True)

    if len(df) == 0:
        return {
            'equity_curve': df.assign(portfolio=np.nan, eth=np.nan, stable=np.nan),
            'trades': pd.DataFrame(columns=['Date','Action','Price','EffPrice','FeePaid_$','ETH_delta','ETH_holdings','Stable_holdings']),
            'summary': {'final_value': 0.0, 'return_pct': 0.0, 'total_trades': 0, 'buys': 0, 'sells': 0}
        }

    # --- Parameters (scalars) ---
    initial_capital = float(getattr(args, 'initial', 100000.0))
    eth_weight      = float(getattr(args, 'eth_weight', 0.5))
    sma_window      = int(getattr(args, 'sma', 200))
    _alpha          = float(getattr(args, 'alpha', 0.5))   # kept for compatibility; not used here
    threshold       = float(getattr(args, 'threshold', 0.20))

    stable_apy      = float(getattr(args, 'stable_apy', 0.06))
    eth_apy         = float(getattr(args, 'eth_apy', 0.025))
    fee_bps         = int(getattr(args, 'fee_bps', 0))
    slip_bps        = int(getattr(args, 'slip_bps', 0))

    use_bands       = bool(getattr(args, 'use_bands', False))
    atr_period      = int(getattr(args, 'atr', 14))
    atr_k           = float(getattr(args, 'atr_k', 2.0))

    # --- Compute SMA ---
    if sma_window > 0:
        df['SMA'] = df['Close'].rolling(window=sma_window, min_periods=1).mean()
    else:
        # If SMA disabled (0), use price itself for bands reference
        df['SMA'] = df['Close']

    # --- Bands: percent or ATR ---
    if use_bands:
        # If High/Low not present, approximate ATR via |diff(Close)| rolling mean
        if 'High' in df.columns and 'Low' in df.columns:
            prev_close = df['Close'].shift(1)
            tr1 = (df['High'] - df['Low']).abs()
            tr2 = (df['High'] - prev_close).abs()
            tr3 = (df['Low']  - prev_close).abs()
            df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['ATR'] = df['TR'].rolling(window=atr_period, min_periods=1).mean()
        else:
            # Fallback ATR proxy: average absolute close-to-close move
            df['ATR'] = df['Close'].diff().abs().rolling(window=atr_period, min_periods=1).mean()

        df['Upper'] = df['SMA'] + atr_k * df['ATR']
        df['Lower'] = df['SMA'] - atr_k * df['ATR']
    else:
        df['Upper'] = df['SMA'] * (1.0 + threshold)
        df['Lower'] = df['SMA'] * (1.0 - threshold)

    # --- Trading costs & compounding ---
    fee_rate  = float(fee_bps) / 10_000.0
    slip_buy  = 1.0 + float(slip_bps) / 10_000.0
    slip_sell = 1.0 - float(slip_bps) / 10_000.0

    # Approx per-bar compounding as "daily"; for sub-daily this compounds each bar.
    stable_mult = 1.0 + stable_apy / 365.0
    eth_mult    = 1.0 + eth_apy    / 365.0

    # --- Initialize portfolio ---
    p0 = float(df['Close'].iloc[0])
    eth_hold    = float((initial_capital * eth_weight) / p0)
    stable_hold = float(initial_capital * (1.0 - eth_weight))

    df['eth'] = 0.0
    df['stable'] = 0.0
    df['portfolio'] = 0.0

    trades = []
    buys = sells = 0

    # --- Main loop ---
    n = len(df)
    for i in range(n):
        price = float(df['Close'].iloc[i])
        date  = df['Date'].iloc[i]
        upper = float(df['Upper'].iloc[i]) if not pd.isna(df['Upper'].iloc[i]) else np.inf
        lower = float(df['Lower'].iloc[i]) if not pd.isna(df['Lower'].iloc[i]) else -np.inf

        # Decision: all-in/all-out around bands
        if price <= lower and stable_hold > 1e-12:
            # BUY all stables -> ETH
            eff_price = price * slip_buy
            # fee paid from notional (stable side)
            spend_after_fee = stable_hold * (1.0 - fee_rate)
            eth_bought = spend_after_fee / eff_price

            eth_hold += eth_bought
            fee_paid = stable_hold - spend_after_fee
            stable_hold = 0.0
            buys += 1

            trades.append({
                'Date': date,
                'Action': 'BUY',
                'Price': price,
                'EffPrice': eff_price,
                'FeePaid_$': fee_paid,
                'ETH_delta': eth_bought,
                'ETH_holdings': eth_hold,
                'Stable_holdings': stable_hold
            })

        elif price >= upper and eth_hold > 1e-12:
            # SELL all ETH -> stables
            eff_price = price * slip_sell
            gross = eth_hold * eff_price
            fee_paid = gross * fee_rate
            proceeds_after_fee = gross - fee_paid

            eth_sold_now = eth_hold
            eth_hold = 0.0
            stable_hold += proceeds_after_fee
            sells += 1

            trades.append({
                'Date': date,
                'Action': 'SELL',
                'Price': price,
                'EffPrice': eff_price,
                'FeePaid_$': fee_paid,
                'ETH_delta': -eth_sold_now,
                'ETH_holdings': eth_hold,
                'Stable_holdings': stable_hold
            })

        # Mark-to-market BEFORE compounding
        df.at[i, 'eth'] = eth_hold
        df.at[i, 'stable'] = stable_hold
        df.at[i, 'portfolio'] = eth_hold * price + stable_hold

        # Apply per-bar APY compounding AFTER snapshot
        eth_hold    *= eth_mult
        stable_hold *= stable_mult

    # --- Results ---
    trades_df = pd.DataFrame(trades) if len(trades) else pd.DataFrame(
        columns=['Date','Action','Price','EffPrice','FeePaid_$','ETH_delta','ETH_holdings','Stable_holdings']
    )

    final_value = float(df['portfolio'].iloc[-1])
    summary = {
        'final_value': final_value,
        'return_pct': (final_value / initial_capital - 1.0) * 100.0,
        'total_trades': len(trades_df),
        'buys': buys,
        'sells': sells
    }

    # Ensure Date dtype is datetime for charting
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])

    return {
        'equity_curve': df,
        'trades': trades_df,
        'summary': summary
    }
