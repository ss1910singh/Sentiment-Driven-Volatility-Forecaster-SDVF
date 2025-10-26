import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta

def get_stock_data(ticker, years=10):
    logging.info(f"Attempting to fetch historical stock data for ticker: {ticker} for up to: {years} years")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(years * 365.25))
    
    try:
        stock_ticker = yf.Ticker(ticker)
        stock_data = stock_ticker.history(start=start_date, end=end_date, interval="1d")
        
        if stock_data.empty:
            logging.warning(f"No data found for the full {years}-year period. Trying 1-year period.")
            start_date_1y = end_date - timedelta(days=365)
            stock_data = stock_ticker.history(start=start_date_1y, end=end_date, interval="1d")

        if stock_data.empty:
            logging.warning(f"No data found for ticker: {ticker}. It might be delisted or invalid.")
            return pd.DataFrame()

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in stock_data.columns:
                stock_data[col] = pd.NA
                
        stock_data = stock_data[required_cols].copy()
        stock_data = stock_data.dropna(subset=['Close'])

        logging.info(f"Successfully fetched {len(stock_data)} data points for {ticker}.")
        return stock_data

    except Exception as e:
        logging.error(f"Failed to fetch data for {ticker}. Reason: {e}", exc_info=True)
        return pd.DataFrame()