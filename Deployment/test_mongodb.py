# test_data.py
from pymongo import MongoClient
from config import Config

def check_data():
    try:
        client = MongoClient(Config.MONGO_URI)
        db = client[Config.MONGO_DB_NAME]
        collection = db[Config.MONGO_COLLECTION]
        
        print(f"Total documents: {collection.count_documents({})}")
        print("Sample document:")
        print(collection.find_one())
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    check_data()