from pymongo import MongoClient
from config import POLYGON_API_KEY, FINANCIAL_PREP_API_KEY, MONGO_DB_USER, MONGO_DB_PASS, API_KEY, API_SECRET, BASE_URL, mongo_url, environment
from bson.decimal128 import Decimal128
import certifi
ca = certifi.where()

status = "market_status(client)" if environment != "dev" else "open"
print(status)
