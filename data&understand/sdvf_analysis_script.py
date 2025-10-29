import pandas as pd
import numpy as np
import yfinance as yf
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.stats.mstats import winsorize
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import logging
from tabulate import tabulate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, logging as hf_logging


sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 7) 
plt.rcParams['figure.dpi'] = 100 
hf_logging.set_verbosity_error() 


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='analysis_log.log',
    filemode='w'
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


logging.info("--- Starting Analysis Script ---")

CSV_FILE_PATH = './data/HDFCBANK.NS_10_years_data.csv'
NEWS_API_KEY = "YOUR_NEWS_API_KEY_HERE"
TICKER_SYMBOL = "HDFCBANK.NS"
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
SUMMARIZER_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
MODEL_DIR = "./models"

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except (nltk.downloader.DownloadError, LookupError):
    logging.info("Downloading VADER lexicon...")
    nltk.download('vader_lexicon', quiet=True)
    logging.info("VADER lexicon downloaded.")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("./plots", exist_ok=True)


def load_and_clean_csv(filepath):
    logging.info(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        df.columns = [col.strip().lower() for col in df.columns]
        if 'date' not in df.columns: raise ValueError("CSV must have a 'date' column.")
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as date_err:
            logging.error(f"Error parsing 'date' column: {date_err}. Check format (e.g., YYYY-MM-DD).")
            return None
        df = df.sort_values('date').set_index('date')

        required_cols = ['close', 'high', 'low', 'open', 'volume']
        for col in required_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                 df[col] = np.nan
                 logging.warning(f"Column '{col}' not found.")

        if 'close' in df.columns and df['close'].notna().any():
             df['daily_return'] = df['close'].astype(float).pct_change() * 100
        else:
            df['daily_return'] = np.nan

        df_cleaned = df.dropna(subset=['close'])
        if df_cleaned.empty:
            logging.error(f"No valid 'close' data found in {os.path.basename(filepath)} after cleaning.")
            return None

        logging.info(f"Loaded {len(df_cleaned)} rows. Data range: {df_cleaned.index.min()} to {df_cleaned.index.max()}")
        return df_cleaned
    except FileNotFoundError:
        logging.error(f"Error: CSV file not found at {filepath}"); return None
    except Exception as e:
        logging.error(f"Error loading CSV: {e}"); return None


stock_data_hist = load_and_clean_csv(CSV_FILE_PATH)

if stock_data_hist is not None:
    print("\n--- Historical Data Info ---")
    stock_data_hist.info()
    print("\n--- Historical Data Tail (Last 5 rows) ---")
    print(stock_data_hist.tail())
    print("\n--- Historical Data NaN Counts ---")
    print(stock_data_hist.isnull().sum())
else:
    print("\nFailed to load historical data. Exiting script.")
    exit()

def analyze_sentiment_finbert_notebook(text_list):
    finbert_local_path = os.path.join(MODEL_DIR, FINBERT_MODEL_NAME.replace("/", "_"))
    logging.info(f"Checking for FinBERT model locally at {finbert_local_path}...")
    try:
        if os.path.exists(finbert_local_path):
            logging.info("Loading FinBERT from cache...")
            tokenizer = AutoTokenizer.from_pretrained(finbert_local_path)
            model = AutoModelForSequenceClassification.from_pretrained(finbert_local_path)
        else:
            logging.info("Downloading FinBERT...")
            tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
            logging.info("Saving FinBERT to cache...")
            tokenizer.save_pretrained(finbert_local_path)
            model.save_pretrained(finbert_local_path)

        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        processed_texts = [text[:512] for text in text_list]
        results = nlp(processed_texts)

        sentiment_scores = []
        for result in results:
            label, score = result['label'], result['score']
            if label == 'Positive': sentiment_scores.append(score)
            elif label == 'Negative': sentiment_scores.append(-score)
            else: sentiment_scores.append((score - 0.5) * 0.2)
        final_score = np.mean(sentiment_scores) if sentiment_scores else 0.0
        logging.info(f"FinBERT analysis complete. Average score: {final_score:.4f}")
        return final_score

    except Exception as e:
        logging.error(f"FinBERT analysis failed: {e}. Falling back to VADER.")
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(text)['compound'] for text in text_list]
        return np.mean(scores) if scores else 0.0

def add_sentiment_notebook(df, ticker):
    logging.info(f"Fetching news for {ticker}...")
    recent_sentiment = 0.0
    recent_articles_list = []

    if NEWS_API_KEY == "YOUR_NEWS_API_KEY_HERE" or not NEWS_API_KEY:
        logging.warning("NEWS_API_KEY not set. Skipping live news fetch. Recent sentiment will be 0.")
    else:
        try:
            newsapi = NewsApiClient(api_key=NEWS_API_KEY)
            to_date = datetime.now()
            from_date = to_date - timedelta(days=2)
            all_articles = newsapi.get_everything(q=ticker,
                                                  from_param=from_date.strftime('%Y-%m-%d'),
                                                  to=to_date.strftime('%Y-%m-%d'),
                                                  language='en', sort_by='relevancy', page_size=20)
            articles = all_articles.get('articles', [])
            if articles:
                logging.info(f"Found {len(articles)} articles.")
                texts = [f"{a.get('title','')}. {a.get('description','')}" for a in articles if a.get('title')]
                recent_sentiment = analyze_sentiment_finbert_notebook(texts)
                recent_articles_list = articles
            else:
                logging.warning(f"No recent news articles found for {ticker}.")

        except Exception as e:
            logging.error(f"Failed to fetch/analyze news: {e}. Setting recent sentiment to 0.")

    logging.info("Simulating historical sentiment...")
    n_days = len(df)
    time = np.arange(n_days)
    simulated_trend = recent_sentiment + 0.1 * np.sin(2 * np.pi * time / (252*0.25))
    noise = np.random.normal(0, 0.05, n_days)
    historical_sentiment = np.clip(simulated_trend + noise, -1, 1)
    df['sentiment'] = historical_sentiment
    logging.info("Historical sentiment simulation complete.")
    return df, recent_sentiment, recent_articles_list


stock_data_with_sentiment, recent_sentiment_score, _ = add_sentiment_notebook(stock_data_hist.copy(), TICKER_SYMBOL)

print(f"\nRecent sentiment score (live fetch or default): {recent_sentiment_score:.4f}")
print("--- Data with Simulated Sentiment (Tail) ---")
print(stock_data_with_sentiment[['close', 'sentiment']].tail())

def _preprocess_data_notebook(stock_data, sentiment_lags=5, ar_lags=5):
    logging.info("Starting preprocessing pipeline...")
    ardl_df = pd.DataFrame(index=stock_data.index)
    ardl_df['y'] = stock_data['close'].pct_change() * 100

    logging.info("Calculating and winsorizing returns...")
    returns_not_na = ardl_df['y'].notna()
    if returns_not_na.any():
        original_values = ardl_df.loc[returns_not_na, 'y'].values
        winsorized_values = winsorize(original_values, limits=[0.01, 0.01]).squeeze()
        ardl_df.loc[returns_not_na, 'y'] = winsorized_values
    logging.info("Returns calculated and winsorized.")

    logging.info("Checking stationarity...")
    adf_result = adfuller(ardl_df['y'].dropna())
    logging.info(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
    if adf_result[1] > 0.05: logging.warning("Returns data may not be stationary.")
    else: logging.info("Returns data appears stationary.")

    logging.info("Creating lagged features...")
    for i in range(1, sentiment_lags + 1):
        ardl_df[f'sentiment_lag_{i}'] = stock_data['sentiment'].shift(i)

    ardl_df_cleaned = ardl_df.dropna(subset=['y'])

    final_returns = ardl_df_cleaned['y']
    dl_cols = [col for col in ardl_df_cleaned.columns if 'sentiment_lag' in col]
    final_exog = ardl_df_cleaned[dl_cols]

    combined_data = pd.concat([final_returns, final_exog], axis=1).dropna()
    final_returns = combined_data['y']
    final_exog = combined_data.drop(columns='y')
    if final_exog.empty: final_exog = None

    logging.info(f"Preprocessing complete. Final dataset size: {len(final_returns)} observations.")
    return final_returns, final_exog

AR_LAGS_P = 5
SENTIMENT_LAGS_Q = 5
processed_returns, processed_exog = _preprocess_data_notebook(
    stock_data_with_sentiment,
    sentiment_lags=SENTIMENT_LAGS_Q,
    ar_lags=AR_LAGS_P
)

print("\n--- Processed Data Shapes ---")
print(f"Returns (y): {processed_returns.shape}")
if processed_exog is not None:
    print(f"Exogenous (DL lags): {processed_exog.shape}")
    print("\n--- Processed Exogenous Data (Head) ---")
    print(processed_exog.head())
else:
    print("No exogenous variables (sentiment lags = 0).")


plt.figure()
stock_data_with_sentiment['close'].plot(title=f'{TICKER_SYMBOL} Closing Price Over Time')
plt.ylabel('Price')
plt.savefig("./plots/price_trend.png")
plt.close()
logging.info("Saved price_trend.png")

if 'volume' in stock_data_with_sentiment.columns and stock_data_with_sentiment['volume'].notna().any():
    plt.figure()
    stock_data_with_sentiment['volume'].plot(title=f'{TICKER_SYMBOL} Trading Volume Over Time', kind='bar', alpha=0.5)
    plt.ylabel('Volume')
    ax = plt.gca()
    tick_spacing = len(ax.get_xticks()) // 10
    ax.set_xticks(ax.get_xticks()[::tick_spacing])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("./plots/volume_trend.png")
    plt.close()
    logging.info("Saved volume_trend.png")
else:
    logging.warning("Volume data missing or invalid, skipping volume plot.")


plt.figure()
processed_returns.plot(title=f'{TICKER_SYMBOL} Daily Returns (%)')
plt.ylabel('Return (%)')
plt.savefig("./plots/daily_returns.png")
plt.close()
logging.info("Saved daily_returns.png")

plt.figure()
sns.histplot(processed_returns, kde=True, bins=50)
plt.title(f'{TICKER_SYMBOL} Distribution of Daily Returns')
plt.xlabel('Return (%)')
plt.savefig("./plots/returns_distribution.png")
plt.close()
logging.info("Saved returns_distribution.png")

plt.figure()
stock_data_with_sentiment['sentiment'].plot(title=f'{TICKER_SYMBOL} Simulated Historical Sentiment')
plt.ylabel('Sentiment Score')
plt.ylim(-1.1, 1.1)
plt.savefig("./plots/simulated_sentiment.png")
plt.close()
logging.info("Saved simulated_sentiment.png")


def fit_ardl_volatility_model_notebook(returns, exog_df, stock_data, recent_sentiment, ar_lags=5, sentiment_lags=5, forecast_horizon=10):
    logging.info("Starting new 2-stage ARDL model fitting...")
    try:
        logging.info(f"Fitting ARDL({ar_lags},{sentiment_lags}) model to mean returns...")
        mean_model = ARIMA(returns, order=(ar_lags, 0, 0), exog=exog_df)
        mean_model_fit = mean_model.fit()
        logging.info("Mean ARDL model fitting complete.")
        print("\n--- ARDL Mean Model Results ---")
        print(mean_model_fit.summary())
        residuals = mean_model_fit.resid

        log_sq_residuals = np.log(residuals**2 + 1e-6)
        log_sq_residuals.name = "log_vol"
        vol_data_combined = pd.concat([log_sq_residuals, exog_df], axis=1).dropna()
        final_log_vol = vol_data_combined['log_vol']
        final_vol_exog = vol_data_combined.drop(columns='log_vol')
        if final_vol_exog.empty: final_vol_exog = None

        logging.info(f"Fitting ARDL({ar_lags},{sentiment_lags}) model to log-volatility...")
        vol_model = ARIMA(final_log_vol, order=(ar_lags, 0, 0), exog=final_vol_exog)
        vol_model_fit = vol_model.fit()
        logging.info("Volatility ARDL model fitting complete.")
        print("\n--- ARDL Volatility Model Results ---")
        print(vol_model_fit.summary())

        logging.info(f"Generating volatility forecast for {forecast_horizon} days...")
        future_exog_df = None
        if final_vol_exog is not None:
            future_exog_data = {}
            last_sentiments = stock_data['sentiment'].iloc[-sentiment_lags:].tolist()
            for i in range(forecast_horizon):
                current_sentiments = last_sentiments + [recent_sentiment] * (i + 1)
                for j in range(1, sentiment_lags + 1):
                    col_name = f'sentiment_lag_{j}'
                    if col_name not in future_exog_data: future_exog_data[col_name] = []
                    future_exog_data[col_name].append(current_sentiments[len(last_sentiments) + i - j])
            future_exog_df = pd.DataFrame(future_exog_data, columns=final_vol_exog.columns)

        log_vol_forecast = vol_model_fit.forecast(steps=forecast_horizon, exog=future_exog_df)

        predicted_sq_variance = np.exp(log_vol_forecast)
        predicted_variance = predicted_sq_variance.values
        predicted_volatility = np.sqrt(predicted_variance)
        annualized_volatility = predicted_volatility * np.sqrt(252)

        last_obs_date = final_log_vol.index[-1]
        forecast_dates = pd.to_datetime(last_obs_date) + pd.to_timedelta(np.arange(1, forecast_horizon + 1), 'D')

        forecast_df = pd.DataFrame({
            "Date": forecast_dates.strftime('%Y-%m-%d'),
            "Predicted Daily Volatility (%)": predicted_volatility,
            "Predicted Annualized Volatility (%)": annualized_volatility
        })

        logging.info("Volatility forecast generated.")
        return forecast_df

    except Exception as e:
        logging.error(f"An error occurred during ARDL modeling: {e}", exc_info=True)
        return pd.DataFrame()

FORECAST_HORIZON = 10
final_forecast = fit_ardl_volatility_model_notebook(
    processed_returns,
    processed_exog,
    stock_data_with_sentiment, 
    recent_sentiment_score,
    ar_lags=AR_LAGS_P,
    sentiment_lags=SENTIMENT_LAGS_Q,
    forecast_horizon=FORECAST_HORIZON
)

if not final_forecast.empty:
    print(f"\n--- {FORECAST_HORIZON}-Day Volatility Forecast (2-Stage ARDL) ---")
    print(tabulate(final_forecast, headers='keys', tablefmt='psql', showindex=False, floatfmt=(".0f", ".4f", ".4f")))
    try:
        final_forecast.to_csv("notebook_forecast_output.csv", index=False)
        logging.info("Forecast saved to notebook_forecast_output.csv")
    except Exception as e:
        logging.error(f"Failed to save forecast CSV: {e}")

else:
    print("\n--- Forecast Generation Failed ---")
    print("Check logs for errors during modeling.")