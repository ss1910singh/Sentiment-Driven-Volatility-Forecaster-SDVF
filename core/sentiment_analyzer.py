import logging
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
from utils.config import get_api_key

def _initialize_nltk_vader():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        logging.info("NLTK VADER lexicon not found. Downloading...")
        nltk.download('vader_lexicon')
        logging.info("Download complete.")
    
    return SentimentIntensityAnalyzer()

sia = _initialize_nltk_vader()

def analyze_sentiment(text: str) -> float:
    if not text or not isinstance(text, str):
        return 0.0
    return sia.polarity_scores(text)['compound']

def fetch_recent_news_sentiment(ticker: str) -> float:
    logging.info(f"Fetching recent news for sentiment analysis of {ticker}.")
    try:
        api_key = get_api_key()
        newsapi = NewsApiClient(api_key=api_key)
        all_articles = newsapi.get_everything(
            q=ticker,
            language='en',
            sort_by='publishedAt',
            page_size=20
        )
        
        articles = all_articles.get('articles', [])
        
        if not articles:
            logging.warning(f"No recent news articles found for {ticker}.")
            return 0.0

        sentiment_scores = []
        for article in articles:
            title = article.get('title', '')
            description = article.get('description', '')
            combined_text = f"{title}. {description}"
            
            score = analyze_sentiment(combined_text)
            sentiment_scores.append(score)
            
        average_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
        logging.info(f"Found {len(articles)} articles for {ticker}. Average sentiment: {average_sentiment:.4f}")
        return average_sentiment

    except Exception as e:
        logging.error(f"Failed to fetch or analyze news for {ticker}: {e}")
        return 0.0

def simulate_historical_sentiment(stock_data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Simulating historical sentiment data for modeling.")
    sentiment = pd.Series(np.random.randn(len(stock_data)) * 0.1, index=stock_data.index)
    sentiment = sentiment.cumsum()
    sentiment = (sentiment - sentiment.min()) / (sentiment.max() - sentiment.min()) * 2 - 1
    sentiment.iloc[-1] = sentiment.iloc[-2] * 0.5
    stock_data_with_sentiment = stock_data.copy()
    stock_data_with_sentiment['sentiment'] = sentiment
    logging.info("Historical sentiment simulation complete.")
    return stock_data_with_sentiment