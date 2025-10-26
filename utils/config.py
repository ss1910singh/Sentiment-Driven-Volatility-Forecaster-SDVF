import os
from dotenv import load_dotenv
import logging

def load_api_key():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dotenv_path = os.path.join(project_root, '.env')

    if not os.path.exists(dotenv_path):
        logging.warning(f".env file not found at {dotenv_path}. Make sure it's in the project root.")
        return None
        
    load_dotenv(dotenv_path)
    
    api_key = os.getenv("NEWS_API_KEY")
    
    if not api_key:
        logging.error("NEWS_API_KEY not found in .env file. Please add it.")
        return None
        
    return api_key
NEWS_API_KEY = load_api_key()

