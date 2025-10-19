import yfinance as yf
import pandas as pd
import logging

def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    logging.info(f"Fetching historical stock data for ticker: {ticker} for the period: {period}")
    
    try:
        stock_data = yf.download(ticker, period=period, interval="1d", progress=False)
        if stock_data.empty:
            logging.error(f"No data found for ticker '{ticker}'. It may be an invalid symbol.")
            raise ValueError(f"No data retrieved for ticker: {ticker}. Please check the symbol.")

        if stock_data.isnull().values.any():
            logging.warning(f"Missing values detected in the data for {ticker}.")
        logging.info(f"Successfully fetched {len(stock_data)} data points for {ticker}.")
        
        return stock_data

    except Exception as e:
        logging.error(f"An unexpected error occurred while fetching data for {ticker}: {e}")
        raise