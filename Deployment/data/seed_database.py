import pandas as pd
import pymongo
from pymongo import MongoClient
import sys
import os
import json
import argparse
from datetime import datetime

# Add the parent directory to the path so we can import from config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def seed_database(csv_file):
    """
    Seed MongoDB with data from a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
    """
    try:
        # Connect to MongoDB
        client = MongoClient(Config.MONGO_URI)
        db = client[Config.MONGO_DB_NAME]
        collection = db[Config.MONGO_COLLECTION]
        
        # Load data from CSV
        print(f"Loading data from {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Clean and prepare data
        print("Preparing data for MongoDB...")
        
        # Convert amenities from string to list if needed
        if 'amenities' in df.columns:
            df['amenities'] = df['amenities'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else
                          ([] if pd.isna(x) else x)
            )
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict(orient='records')
        
        # Clean records for MongoDB
        for record in records:
            # Convert NaN values to None (MongoDB doesn't handle NaN well)
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
        
        # Drop existing collection if it exists
        if Config.MONGO_COLLECTION in db.list_collection_names():
            print(f"Dropping existing collection: {Config.MONGO_COLLECTION}")
            db[Config.MONGO_COLLECTION].drop()
        
        # Insert records
        print(f"Inserting {len(records)} records into MongoDB...")
        result = collection.insert_many(records)
        
        # Create indexes for faster querying
        print("Creating indexes...")
        collection.create_index("type")
        collection.create_index("country")
        collection.create_index("city")
        collection.create_index([("name", pymongo.TEXT), ("description", pymongo.TEXT)])
        
        # Additional indexes for common queries
        collection.create_index("rating")
        collection.create_index("subcategories")
        collection.create_index([("city", pymongo.ASCENDING), ("type", pymongo.ASCENDING)])
        
        print(f"Successfully inserted {len(result.inserted_ids)} records.")
        
        # Create metadata record to track when database was last seeded
        metadata_collection = db["metadata"]
        metadata_collection.update_one(
            {"_id": "db_info"},
            {"$set": {
                "last_seeded": datetime.now(),
                "record_count": len(result.inserted_ids),
                "source_file": os.path.basename(csv_file)
            }},
            upsert=True
        )
        
        print("Database seeding complete!")
        return True
        
    except pymongo.errors.ConnectionFailure as e:
        print(f"Failed to connect to MongoDB: {e}")
        return False
    except pymongo.errors.OperationFailure as e:
        print(f"MongoDB operation failed: {e}")
        return False
    except Exception as e:
        print(f"Error seeding database: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()
            print("MongoDB connection closed.")

def verify_database():
    """
    Verify that the database was seeded correctly by checking record counts.
    """
    try:
        client = MongoClient(Config.MONGO_URI)
        db = client[Config.MONGO_DB_NAME]
        collection = db[Config.MONGO_COLLECTION]
        
        count = collection.count_documents({})
        
        print(f"Database verification:")
        print(f"- Total records: {count}")
        
        # Count by type
        types = collection.distinct("type")
        for type_name in types:
            type_count = collection.count_documents({"type": type_name})
            print(f"- {type_name}: {type_count} records")
        
        # Count by country (top 5)
        countries = collection.distinct("country")
        print(f"- Unique countries: {len(countries)}")
        
        # Count cities
        cities = collection.distinct("city")
        print(f"- Unique cities: {len(cities)}")
        
        return count > 0
    
    except Exception as e:
        print(f"Error verifying database: {e}")
        return False
    finally:
        if 'client' in locals():
            client.close()

def main():
    """
    Main function to parse arguments and run the seeding process.
    """
    parser = argparse.ArgumentParser(description="Seed MongoDB with travel data from CSV.")
    parser.add_argument("csv_file", help="Path to the CSV file containing travel data")
    parser.add_argument("--verify", action="store_true", help="Verify database after seeding")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file not found: {args.csv_file}")
        sys.exit(1)
    
    print(f"Starting database seeding process at {datetime.now()}")
    success = seed_database(args.csv_file)
    
    if success and args.verify:
        print("\nVerifying database contents...")
        verify_database()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()