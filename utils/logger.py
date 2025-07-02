import logging
import os

def get_logger(name="app"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.FileHandler("app.log", mode="a")
        formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent double logging in Streamlit
    return logger