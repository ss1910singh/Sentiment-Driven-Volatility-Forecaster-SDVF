import logging
import sys

def setup_logging():
    log_format = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        stream=sys.stdout,
        force=True
    )
    
    logging.info("Logging configured successfully.")

