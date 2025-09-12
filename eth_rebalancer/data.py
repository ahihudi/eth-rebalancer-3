import pandas as pd
import yfinance as yf

def load_data(start, end):
    """
    Load ETH price data from Yahoo Finance.
    
    Arguments:
        start (str or datetime): Start date (YYYY-MM-DD or datetime)
        end (str or datetime): End date (YYYY-MM-DD or datetime)
    
    Returns:
        pd.DataFrame: Columns ['Date', 'Close']
    """
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)

    df = yf.download("ETH-USD", start=start_date, end=end_date)
    if df.empty:
        raise ValueError("No data fetched from Yahoo Finance. Check your dates.")

    df = df.reset_index()[['Date', 'Close']]
    return df
