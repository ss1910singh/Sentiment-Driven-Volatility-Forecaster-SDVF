import logging
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import numpy as np
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, logging as hf_logging
hf_logging.set_verbosity_error()

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    logging.info("VADER lexicon not found. Downloading...")
    nltk.download('vader_lexicon')
    logging.info("VADER lexicon downloaded successfully.")
except LookupError:
     logging.info("VADER lexicon lookup error. Attempting download...")
     try:
         nltk.download('vader_lexicon')
         logging.info("VADER lexicon downloaded successfully.")
     except Exception as e:
         logging.error(f"Failed to download VADER lexicon even after lookup error: {e}")
except Exception as e:
    logging.error(f"An unexpected error occurred during NLTK setup: {e}")


from utils.config import NEWS_API_KEY
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
SUMMARIZER_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

def analyze_sentiment_finbert(text_list):
    finbert_local_path = os.path.join(MODEL_DIR, FINBERT_MODEL_NAME.replace("/", "_"))
    logging.info(f"Checking for FinBERT model at: {finbert_local_path}")

    try:
        if os.path.exists(finbert_local_path):
            logging.info("Loading FinBERT model and tokenizer from local cache...")
            tokenizer = AutoTokenizer.from_pretrained(finbert_local_path)
            model = AutoModelForSequenceClassification.from_pretrained(finbert_local_path)
            logging.info("Loaded FinBERT from local cache.")
        else:
            logging.info("Downloading FinBERT model and tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
            logging.info("FinBERT downloaded. Saving to local cache...")
            tokenizer.save_pretrained(finbert_local_path)
            model.save_pretrained(finbert_local_path)
            logging.info(f"FinBERT saved to {finbert_local_path}")

        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        results = nlp(text_list)

        sentiment_scores = []
        for result in results:
            label = result['label']
            score = result['score']
            if label == 'Positive':
                sentiment_scores.append(score)
            elif label == 'Negative':
                sentiment_scores.append(-score)
            else:
                sentiment_scores.append((score - 0.5) * 0.2)
        return np.mean(sentiment_scores) if sentiment_scores else 0.0

    except Exception as e:
        logging.error(f"Error during FinBERT sentiment analysis: {e}", exc_info=True)
        logging.warning("Falling back to VADER sentiment analysis due to FinBERT error.")
        analyzer = SentimentIntensityAnalyzer()
        scores = [analyzer.polarity_scores(text)['compound'] for text in text_list]
        return np.mean(scores) if scores else 0.0

def add_sentiment_to_data(stock_data, ticker):
    logging.info(f"Fetching recent news for sentiment analysis of {ticker}.")

    try:
        to_date = datetime.now()
        from_date = to_date - timedelta(days=5)
        all_articles = newsapi.get_everything(q=ticker,
                                              from_param=from_date.strftime('%Y-%m-%d'),
                                              to=to_date.strftime('%Y-%m-%d'),
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=20)

        articles_list = all_articles['articles']
        if not articles_list:
            logging.warning(f"No recent news articles found for {ticker}.")
            recent_sentiment = 0.0
            recent_articles_for_summary = []
        else:
            texts_to_analyze = [
                f"{article['title']}. {article['description']}"
                for article in articles_list
                if article['title'] and article['description']
            ]
            recent_sentiment = analyze_sentiment_finbert(texts_to_analyze)
            logging.info(f"Found {len(articles_list)} articles for {ticker}. Average FinBERT sentiment: {recent_sentiment:.4f}")
            recent_articles_for_summary = articles_list

    except Exception as e:
        logging.error(f"Failed to fetch or analyze news for {ticker}: {e}. Setting recent sentiment to 0.", exc_info=False)
        recent_sentiment = 0.0
        recent_articles_for_summary = []

    logging.info("Simulating historical sentiment data for modeling.")
    n_days = len(stock_data)
    time = np.arange(n_days)
    simulated_trend = recent_sentiment + 0.1 * np.sin(2 * np.pi * time / (252*0.25))
    noise = np.random.normal(0, 0.05, n_days)
    historical_sentiment = simulated_trend + noise
    historical_sentiment = np.clip(historical_sentiment, -1, 1)
    stock_data['sentiment'] = historical_sentiment
    logging.info("Historical sentiment simulation complete.")
    return stock_data, recent_sentiment, recent_articles_for_summary

def summarize_articles(articles):
    summarizer_local_path = os.path.join(MODEL_DIR, SUMMARIZER_MODEL_NAME.replace("/", "_"))
    logging.info(f"Checking for Summarizer model at: {summarizer_local_path}")

    try:
        if os.path.exists(summarizer_local_path):
            logging.info("Loading Summarizer model and tokenizer from local cache...")
            summarizer = pipeline("summarization", model=summarizer_local_path, tokenizer=summarizer_local_path)
            logging.info("Loaded Summarizer from local cache.")
        else:
            logging.info("Downloading Summarizer model and tokenizer...")
            summarizer = pipeline("summarization", model=SUMMARIZER_MODEL_NAME)
            logging.info("Summarizer downloaded. Saving to local cache...")
            summarizer.tokenizer.save_pretrained(summarizer_local_path)
            summarizer.model.save_pretrained(summarizer_local_path)
            logging.info(f"Summarizer saved to {summarizer_local_path}")

        individual_summaries = []
        texts_for_combined = []
        max_summary_length = 100
        min_summary_length = 20 

        for article in articles:
            content = article.get('content') or article.get('description') or article.get('title')
            if not content or content == '[Removed]': continue
            content = content.split(' [+')[0]
            max_input_length = summarizer.tokenizer.model_max_length
            inputs = summarizer.tokenizer(content, max_length=max_input_length, truncation=True, return_tensors="pt")
            truncated_content = summarizer.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            if len(truncated_content.split()) < min_summary_length * 1.5:
                 continue

            try:
                summary_result = summarizer(truncated_content,
                                            max_length=max_summary_length,
                                            min_length=min_summary_length,
                                            do_sample=False)[0]
                summary_text = summary_result['summary_text']
                individual_summaries.append({
                    'title': article.get('title', 'N/A'),
                    'source': article.get('source', {}).get('name', 'N/A'),
                    'url': article.get('url', '#'),
                    'summary': summary_text
                })
                texts_for_combined.append(summary_text)
            except Exception as e:
                logging.warning(f"Could not summarize article: {article.get('title')}. Error: {e}")
                continue

        combined_summary = "No articles could be summarized."
        if texts_for_combined:
            combined_input = " ".join(texts_for_combined)
            inputs = summarizer.tokenizer(combined_input, max_length=max_input_length, truncation=True, return_tensors="pt")
            truncated_combined_input = summarizer.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            try:
                combined_summary_result = summarizer(truncated_combined_input,
                                                     max_length=150,
                                                     min_length=40,
                                                     do_sample=False)[0]
                combined_summary = combined_summary_result['summary_text']
            except Exception as e:
                 logging.warning(f"Could not create combined summary. Error: {e}")
                 combined_summary = "Could not generate combined summary from individual articles."


        return individual_summaries, combined_summary

    except Exception as e:
        logging.error(f"Error initializing or running summarizer: {e}", exc_info=True)
        return [], "Error generating summaries."

def get_buy_sell_recommendation(historical_sentiment, recent_sentiment):
    if historical_sentiment.empty:
        return "Hold", "Not enough historical sentiment data for a recommendation."

    lookback = min(21, len(historical_sentiment))
    trend_sentiment = historical_sentiment.iloc[-lookback:]
    avg_trend = trend_sentiment.mean()
    
    if recent_sentiment > 0.15 and recent_sentiment > avg_trend * 1.1:
        return "Buy", f"Recent sentiment ({recent_sentiment:.2f}) is positive and shows an improving trend over the last {lookback} days (avg: {avg_trend:.2f})."
    elif recent_sentiment > 0.05:
         return "Hold", f"Recent sentiment ({recent_sentiment:.2f}) is positive but the trend is stable (avg last {lookback} days: {avg_trend:.2f})."
    elif recent_sentiment < -0.15 and recent_sentiment < avg_trend * 0.9:
        return "Sell", f"Recent sentiment ({recent_sentiment:.2f}) is negative and shows a worsening trend over the last {lookback} days (avg: {avg_trend:.2f})."
    elif recent_sentiment < -0.05:
         return "Hold", f"Recent sentiment ({recent_sentiment:.2f}) is negative but the trend is stable (avg last {lookback} days: {avg_trend:.2f})."
    else:
        return "Hold", f"Recent sentiment ({recent_sentiment:.2f}) is neutral (avg last {lookback} days: {avg_trend:.2f})."