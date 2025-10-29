import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(tickers, years=10):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365.25)
    
    print(f"Fetching data for {len(tickers)} companies from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    all_data = []
    
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                print(f"No data found for {ticker}, skipping.")
                continue
            data['Ticker'] = ticker
            all_data.append(data)
            print(f"Successfully fetched data for {ticker}")
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            
    if not all_data:
        print("No data was fetched for any ticker.")
        return None

    combined_data = pd.concat(all_data)
    combined_data.reset_index(inplace=True)
    
    if 'index' in combined_data.columns and 'Date' not in combined_data.columns:
        combined_data.rename(columns={'index': 'Date'}, inplace=True)
        
    print(f"\nTotal rows fetched: {len(combined_data)}")
    return combined_data

if __name__ == "__main__":
    company_tickers = [
        'RELIANCE.NS',   # Reliance Industries
        'TCS.NS',        # Tata Consultancy Services
        'HDFCBANK.NS',   # HDFC Bank
        'INFY.NS',       # Infosys
        'ICICIBANK.NS',  # ICICI Bank
        'HINDUNILVR.NS', # Hindustan Unilever
        'SBIN.NS',       # State Bank of India
        'BHARTIARTL.NS', # Bharti Airtel
        'LT.NS',         # Larsen & Toubro
        'KOTAKBANK.NS'   # Kotak Mahindra Bank
    ]
    
    years_to_fetch = 10
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_to_fetch * 365.25)
    
    print(f"Fetching 10 years of data for {len(company_tickers)} companies...")
    print(f"Data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n")

    files_saved = []

    for ticker in company_tickers:
        try:
            print(f"Fetching data for {ticker}...")
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                print(f"  No data found for {ticker}, skipping.")
                continue
                
            data['Ticker'] = ticker
            data.reset_index(inplace=True)
            if 'index' in data.columns and 'Date' not in data.columns:
                data.rename(columns={'index': 'Date'}, inplace=True)
            
            output_file = f"./data/{ticker}_10_years_data.csv"
            data.to_csv(output_file, index=False)
            print(f"  Successfully saved data to '{output_file}'")
            files_saved.append(output_file)
            
        except Exception as e:
            print(f"  Error fetching data for {ticker}: {e}")

    print(f"\nProcess complete. Saved {len(files_saved)} files:")
    for f in files_saved:
        print(f"- {f}")