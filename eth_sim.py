import argparse
from eth_rebalancer.strategy import run_backtest
from eth_rebalancer.data import load_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, default='2021-09-06')
    parser.add_argument('--end', type=str, default='2025-09-06')
    parser.add_argument('--initial', type=float, default=100000)
    parser.add_argument('--eth_weight', type=float, default=0.5)
    parser.add_argument('--sma', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--threshold', type=float, default=0.03)
    parser.add_argument('--stable_apy', type=float, default=0.06)
    parser.add_argument('--eth_apy', type=float, default=0.025)
    parser.add_argument('--fee_bps', type=float, default=5)
    parser.add_argument('--slip_bps', type=float, default=10)
    parser.add_argument('--yfinance', action='store_true')
    parser.add_argument('--use_bands', action='store_true')
    parser.add_argument('--atr', type=int, default=14)
    parser.add_argument('--atr_k', type=float, default=2.0)
    args = parser.parse_args()

    df = load_data(args.start, args.end, args.yfinance)
    result = run_backtest(df, args)
    print(result['summary'])

if __name__ == "__main__":
    main()
