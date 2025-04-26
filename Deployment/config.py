import os

class Config:
    """Configuration settings for the travel recommender application."""
    
    # Flask app settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-for-development'
    DEBUG = True
    
    # MongoDB settings
    MONGO_URI = os.environ.get('MONGO_URI') or "mongodb://localhost:27017/"
    MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME') or 'Travel_Recommendation'  # Note the correct name
    MONGO_COLLECTION = os.environ.get('MONGO_COLLECTION') or 'africa_data_with_sentiment'