import os
from dotenv import load_dotenv
import logging

load_dotenv()

def get_api_key():
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        logging.error("NEWS_API_KEY not found in environment variables.")
    return api_key
