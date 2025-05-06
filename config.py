import os
import sys
from os import environ as env

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

try:
    POLYGON_API_KEY = env["POLYGON_API_KEY"]
    FINANCIAL_PREP_API_KEY = env["FINANCIAL_PREP_API_KEY"]
    API_KEY = env["API_KEY"]
    API_SECRET = env["API_SECRET"]
    BASE_URL = env["BASE_URL"]
    WANDB_API_KEY = env["WANDB_API_KEY"]
    mongo_url = env["MONGO_URL"]
    # PRICE_DB_PATH = env["PRICE_DB_PATH"]
    PRICE_DB_PATH = os.path.join("PriceData", "price_data.db")
    STRATEGY_DECISIONS_DB_PATH = env["STRATEGY_DECISIONS_DB_PATH"]
    environment = "dev"
except KeyError as e:
    print(f"[error]: {e} required environment variable missing")
    sys.exit(1)
