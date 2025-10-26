import argparse
import logging
import pandas as pd
from core.data_fetcher import get_stock_data
from core.sentiment_analyzer import add_sentiment_to_data
from core.volatility_modeler import fit_ardl_volatility_model
from utils.logging_config import setup_logging
from tabulate import tabulate

def format_change_arrow(change):
    if pd.isna(change):
        return "N/A"
    if change > 0:
        return f"▲ {change:.2f}"
    elif change < 0:
        return f"▼ {change:.2f}"
    else:
        return f"— {change:.2f}"

def main(args):
    setup_logging()
    logging.info(f"--- Starting Analysis for {args.ticker} ---")

    try:
        stock_data = get_stock_data(args.ticker, years=args.years)
        if stock_data.empty:
            logging.error(f"Could not fetch stock data for {args.ticker}. Exiting.")
            return
    except Exception as e:
        logging.error(f"Error in data fetching: {e}", exc_info=True)
        return

    try:
        stock_data_with_sentiment, recent_sentiment, _ = add_sentiment_to_data(stock_data, args.ticker)
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}. Check API key. Exiting.", exc_info=False)
        return

    try:
        forecast_df, fitted_mean_values = fit_ardl_volatility_model(
            stock_data_with_sentiment,
            recent_sentiment,
            ar_lags=args.ar,
            sentiment_lags=args.sentiment,
            forecast_horizon=args.horizon
        )
    except Exception as e:
        logging.error(f"An error occurred during modeling: {e}", exc_info=True)
        forecast_df = pd.DataFrame()
        fitted_mean_values = pd.Series()

    print("\n" + "="*50)
    print(f"Analysis Results for: {args.ticker}")
    print("="*50)

    print(f"\nRecent News Sentiment Score (FinBERT): {recent_sentiment:.4f}")
    print("(Score from -1: very negative, to +1: very positive)")

    if forecast_df.empty:
        print("\nCould not generate a forecast. The model failed to converge or another error occurred.")
        print("Please check the logs for more details.")
    else:
        print(f"\n--- {args.horizon}-Day Advanced Forecast (2-Stage ARDL Model) ---")

        display_df = forecast_df.copy()
        display_df['Price Change'] = display_df['Price Change'].apply(format_change_arrow)
        output_columns = [
            "Date",
            "Predicted Price",
            "Price Change",
            "Lower Band",
            "Upper Band",
            "Predicted Daily Volatility (%)",
            "Predicted Annualized Volatility (%)"
        ]
        display_df = display_df[output_columns]
        print(tabulate(
            display_df,
            headers='keys',
            tablefmt='grid',
            showindex=False,
            floatfmt=(None, ".2f", None, ".2f", ".2f", ".3f", ".3f")
        ))

    print("\n" + "="*50)
    logging.info(f"--- Analysis for {args.ticker} Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a 2-Stage ARDL Volatility Forecast from the command line."
    )
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol (e.g., AAPL, GOOGL)"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=10,
        help="Number of years for historical data."
    )
    parser.add_argument(
        "--ar",
        type=int,
        default=5,
        help="Autoregressive (AR) lags (p)."
    )
    parser.add_argument(
        "--sentiment",
        type=int,
        default=5,
        help="Sentiment (DL) lags (q)."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Forecast horizon in days."
    )

    args = parser.parse_args()
    main(args)