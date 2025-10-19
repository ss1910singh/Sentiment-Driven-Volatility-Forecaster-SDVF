import argparse
import logging
from utils.logging_config import setup_logging
from core.data_fetcher import fetch_stock_data
from core.sentiment_analyzer import fetch_recent_news_sentiment, simulate_historical_sentiment
from core.volatility_modeler import fit_garch_model_and_forecast

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Sentiment-Driven Volatility Forecaster (SDVF) - CLI",
        epilog="Example: python main.py AAPL"
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="The stock ticker symbol to analyze (e.g., 'AAPL', 'GOOGL')."
    )
    args = parser.parse_args()
    ticker = args.ticker.upper()
    
    logging.info(f"--- Starting Analysis for {ticker} ---")

    try:
        stock_data = fetch_stock_data(ticker)
        recent_sentiment = fetch_recent_news_sentiment(ticker)
        stock_data_with_sentiment = simulate_historical_sentiment(stock_data)
        forecast_df = fit_garch_model_and_forecast(stock_data_with_sentiment)

        print("\n" + "="*50)
        print(f"Analysis Results for: {ticker}")
        print("="*50)
        print(f"\nRecent News Sentiment Score: {recent_sentiment:.4f}")
        print("(Score ranges from -1: very negative, to +1: very positive)\n")
        
        if not forecast_df.empty:
            print("Volatility Forecast (Next 5 Trading Days):")
            forecast_df['Forecast Date'] = forecast_df['Forecast Date'].dt.strftime('%Y-%m-%d')
            print(forecast_df.to_string(index=False))
            print("\n* Volatility is the annualized standard deviation of stock returns.")
            print("* A higher value indicates a higher risk or price fluctuation.")
            print("\nNote: Forecast data has been saved to 'forecast_output.csv' for Power BI.")
        else:
            print("\nCould not generate a forecast. The model failed to converge or another error occurred.")
            print("Please check the logs for more details.")
            
        print("="*50 + "\n")
        logging.info(f"--- Analysis for {ticker} Complete ---")

    except ValueError as ve:
        logging.error(f"A validation error occurred: {ve}")
        print(f"\nError: {ve}\nPlease ensure you have entered a valid stock ticker.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the analysis: {e}")
        print(f"\nAn unexpected error occurred. Please check logs for details.")

if __name__ == "__main__":
    main()