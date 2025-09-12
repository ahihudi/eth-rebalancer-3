# ETH Rebalancer Simulator

This simulator implements a percentage-based ETH rebalancing strategy with stablecoin lending. It allows you to backtest different ETH buy/sell strategies using historical prices and visualize the results interactively.

How it works

Initial allocation:

Part of your capital is in ETH, part in stablecoins.

ETH earns a small lending APY; stablecoins earn a higher APY.

Strategy logic:

Buy ETH when its price drops X% from the last sell.

Sell ETH when its price rises X% from the last buy.

Thresholds can be set between 10–40% to avoid overtrading.

Portfolio update:

Each day, portfolio value = ETH holdings * price + stablecoin holdings.

APYs are compounded daily.

All trades are logged and visualized.

## Parameters Explained

| Parameter                   | Description                                                                        |
| --------------------------- | ---------------------------------------------------------------------------------- |
| **Start Date / End Date**   | The historical range for ETH prices.                                               |
| **Initial Capital**         | Total starting capital in USD.                                                     |
| **Initial ETH Weight**      | Percentage of capital allocated to ETH initially.                                  |
| **SMA Window**              | Number of days for simple moving average (optional).                               |
| **Alpha**                   | Optional parameter for advanced strategies (can be ignored for basic simulation).  |
| **Rebalance Threshold (%)** | Percentage drop/rise to trigger **buy/sell** (10–40%).                             |
| **Stablecoin APY**          | Annual interest rate earned on stablecoins (compounded daily).                     |
| **ETH APY**                 | Annual interest rate earned on ETH holdings (compounded daily).                    |
| **Fee (bps)**               | Trading fees in basis points (0–100). Set 0 for low-fee chains like Base/Arbitrum. |
| **Slippage (bps)**          | Price slippage in basis points (0–100). Usually negligible.                        |
| **Use ATR Bands**           | Optional: apply Average True Range bands for advanced strategy.                    |
| **ATR Window**              | Number of days to calculate ATR.                                                   |
| **ATR Multiplier**          | Multiplier for ATR bands.                                                          |


## Outputs

Simulation Summary:

final_value: total portfolio value at the end.

return_pct: total percentage gain/loss.

total_trades: number of buy/sell actions executed.

Portfolio Equity Curve:

Blue line: portfolio value over time.

Green triangles: BUY trades.

Red triangles: SELL trades.

All Trades Table:

Date, Action (BUY/SELL), ETH bought/sold, remaining stablecoins.

## Notes

The simulator uses Yahoo Finance for historical ETH prices, avoiding API rate limits.

Threshold-based strategy avoids frequent trades and focuses on large price movements.

Fully compatible with low-fee networks like Base and Arbitrum.

You can adjust APYs, thresholds, and SMA to test different strategies.


## Example Use Case

Initial capital: $100,000

ETH weight: 50%

Rebalance threshold: 20%

The app will buy ETH whenever it drops 20% from last sell and sell ETH whenever it rises 20% from last buy. Portfolio and trades are plotted and logged automatically.

## Features
- SMA and ATR band strategies for buy/sell
- Lending APY for stablecoins and ETH
- Slippage and fees included
- CLI simulator
- Streamlit UI for visual exploration

## Usage
Install requirements:
```
pip install -r requirements.txt
```

Run CLI:
```
python eth_sim.py --yfinance --start 2021-09-06 --end 2025-09-06 --initial 100000
```

Run Streamlit app:
```
streamlit run app.py
```
