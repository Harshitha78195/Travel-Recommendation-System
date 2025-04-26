# database.py
from pymongo import MongoClient
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection handling
_client = None
_db = None

def get_mongo_client():
    """Returns a MongoDB client instance"""
    global _client
    if _client is None:
        try:
            _client = MongoClient(
                Config.MONGO_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            # Test the connection
            _client.server_info()
            logger.info("Successfully connected to MongoDB server")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    return _client

def get_db():
    """Returns your database instance"""
    global _db
    if _db is None:
        client = get_mongo_client()
        _db = client[Config.MONGO_DB_NAME]
        logger.info(f"Using database: {_db.name}")
    return _db

def get_collection(collection_name=None):
    """Returns a collection instance"""
    db = get_db()
    collection = db[collection_name or Config.MONGO_COLLECTION]
    logger.info(f"Using collection: {collection.name}")
    return collection