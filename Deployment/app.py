from flask import Flask, render_template, request, jsonify
from pymongo import MongoClient
from models.recommender import HybridRecommender
import pandas as pd
import numpy as np
from config import Config
import logging
from logging.handlers import RotatingFileHandler
import traceback
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import ObjectId
import atexit
import socket
import sys

app = Flask(__name__)
app.config.from_object(Config)

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# MongoDB Connection Setup
def get_mongo_connection():
    """Establish and return MongoDB connection with error handling"""
    try:
        client = MongoClient(
            app.config['MONGO_URI'],
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=30000
        )
        # Force a connection test
        client.server_info()
        app.logger.info("Successfully connected to MongoDB")
        return client
    except ServerSelectionTimeoutError:
        app.logger.error("MongoDB connection timeout - server not available")
        raise
    except ConnectionFailure as e:
        app.logger.error(f"MongoDB connection failed: {str(e)}")
        raise
    except Exception as e:
        app.logger.error(f"Unexpected MongoDB connection error: {str(e)}")
        raise

# Initialize MongoDB connection at app startup
try:
    mongo_client = get_mongo_connection()
    db = mongo_client[app.config['MONGO_DB_NAME']]
    places_collection = db[app.config['MONGO_COLLECTION']]
    
    # Test collection access
    places_collection.find_one()
    app.logger.info(f"Successfully accessed collection: {places_collection.name}")
    
except Exception as e:
    app.logger.critical(f"Failed to initialize MongoDB connection: {str(e)}")
    raise

def cleanup_resources():
    """Clean up resources before shutdown"""
    app.logger.info("Cleaning up resources...")
    try:
        if 'mongo_client' in globals():
            mongo_client.close()
            app.logger.info("MongoDB connection closed")
    except Exception as e:
        app.logger.error(f"Error during cleanup: {str(e)}")

# Register cleanup function
atexit.register(cleanup_resources)

def convert_for_json(item):
    """
    Convert MongoDB document to JSON-serializable format
    Handles ObjectId, NaN values, numpy types, and other non-serializable types
    """
    if isinstance(item, dict):
        # Handle ObjectId
        if '_id' in item and isinstance(item['_id'], ObjectId):
            item['_id'] = str(item['_id'])
        
        # Handle all values in the item
        for key, value in item.items():
            if isinstance(value, (np.ndarray)):
                # Convert numpy array to list
                item[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                # Recursively process list items
                item[key] = [convert_for_json(v) for v in value]
            elif isinstance(value, dict):
                # Recursively process dict values
                item[key] = convert_for_json(value)
            elif isinstance(value, (np.int64, np.int32)):
                # Convert numpy integers to Python int
                item[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                # Convert numpy floats to Python float
                if np.isinf(value) or np.isnan(value):
                    item[key] = None
                else:
                    item[key] = float(value)
            elif value is None or (isinstance(value, (float, int)) and np.isnan(value)):
                # Handle None and NaN values
                item[key] = None
                
    return item

def init_recommender():
    """Initialize the recommender system with data from MongoDB"""
    try:
        app.logger.info("Initializing recommender system")
        places_cursor = places_collection.find({})
        places_data = list(places_cursor)
        
        if not places_data:
            app.logger.error("No data found in MongoDB collection")
            return None
            
        places_df = pd.DataFrame(places_data)
        
        # Check for required columns
        required_columns = ['name', 'type', 'country', 'city', 'rating']
        for col in required_columns:
            if col not in places_df.columns:
                app.logger.error(f"Missing required column in data: {col}")
                return None
        
        app.logger.info(f"Successfully loaded {len(places_df)} records")
        return HybridRecommender(places_df)
        
    except Exception as e:
        app.logger.error(f"Error initializing recommender: {str(e)}\n{traceback.format_exc()}")
        return None

@app.route('/')
def index():
    """Render the main page with country and city dropdowns"""
    try:
        countries = places_collection.distinct('country')
        cities = places_collection.distinct('city')
        return render_template('index.html', countries=sorted(countries), cities=sorted(cities))
    except Exception as e:
        app.logger.error(f"Error in index route: {str(e)}\n{traceback.format_exc()}")
        return render_template('index.html', countries=[], cities=[], error="Error loading page")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        query = request.form.get('query', '').strip()
        app.logger.info(f"Processing query: '{query}'")
        
        if not query:
            return jsonify({"error": "Please enter a travel request"}), 400
            
        recommender = init_recommender()
        if not recommender:
            return jsonify({"error": "Recommendation service unavailable"}), 500
        
        try:
            recommended_attractions, recommended_hotels = recommender.recommend_hybrid(query, top_n=5)
            
            attractions_list = []
            hotels_list = []
            
            if not recommended_attractions.empty:
                attractions_list = [convert_for_json(a) for a in recommended_attractions.to_dict('records')]
                
            if not recommended_hotels.empty:
                hotels_list = [convert_for_json(h) for h in recommended_hotels.to_dict('records')]
                
            app.logger.info(f"Found {len(attractions_list)} attractions and {len(hotels_list)} hotels")
            
            # Format results
            for attraction in attractions_list:
                attraction['rating_stars'] = '⭐' * int(attraction.get('rating', 0))
                if not attraction.get('description'):
                    attraction['description'] = 'No description available'
                # Format distance if available
                if 'distance' in attraction and attraction['distance'] is not None:
                    attraction['distance'] = float(attraction['distance'])
                # Ensure image URL is properly formatted
                if attraction.get('image') and not attraction['image'].startswith(('http://', 'https://')):
                    attraction['image'] = f"https://{attraction['image']}"
                
            for hotel in hotels_list:
                hotel['rating_stars'] = '⭐' * int(hotel.get('rating', 0))
                if not hotel.get('description'):
                    hotel['description'] = 'No description available'
                # Format amenities
                if isinstance(hotel.get('amenities_cleaned'), list):
                    hotel['amenities_cleaned'] = [a for a in hotel.get('amenities_cleaned', []) if a]
                else:
                    hotel['amenities_cleaned'] = []
                hotel['amenities_formatted'] = ', '.join(hotel.get('amenities_cleaned', [])) or 'No amenities listed'
                # Format distance if available
                if 'distance' in hotel and hotel['distance'] is not None:
                    hotel['distance'] = float(hotel['distance'])
                # Ensure image URL is properly formatted
                if hotel.get('image') and not hotel['image'].startswith(('http://', 'https://')):
                    hotel['image'] = f"https://{hotel['image']}"
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': True,
                    'attractions': attractions_list,
                    'hotels': hotels_list,
                    'query': query,
                    'message': 'No results found' if not attractions_list and not hotels_list else None
                })
            
            return render_template(
                'results.html',
                attractions=attractions_list,
                hotels=hotels_list,
                query=query,
                message='No results found for your query' if not attractions_list and not hotels_list else None
            )
        
        except Exception as e:
            app.logger.error(f"Recommendation error: {str(e)}\n{traceback.format_exc()}")
            error_msg = "An error occurred while processing your request"
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'success': False,
                    'error': error_msg
                }), 500
            return render_template('error.html', message=error_msg), 500
            
    except Exception as e:
        app.logger.error(f"Unexpected error in recommend endpoint: {str(e)}\n{traceback.format_exc()}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'success': False,
                'error': "Internal server error"
            }), 500
        return render_template('error.html', message="An unexpected error occurred"), 500

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """API endpoint for recommendations (accepts JSON)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        query = data.get('query', '').strip()
        app.logger.info(f"API recommendation request for query: '{query}'")
        
        if not query:
            app.logger.warning("Empty API query received")
            return jsonify({"error": "Please provide a query"}), 400
            
        recommender = init_recommender()
        if not recommender:
            app.logger.error("Recommender initialization failed for API request")
            return jsonify({"error": "Service unavailable"}), 500
            
        recommended_attractions, recommended_hotels = recommender.recommend_hybrid(query, top_n=5)
        
        # Initialize empty lists
        attractions_list = []
        hotels_list = []
        
        # Process attractions if not empty
        if not recommended_attractions.empty:
            attractions_list = [convert_for_json(a) for a in recommended_attractions.to_dict('records')]
            
        # Process hotels if not empty
        if not recommended_hotels.empty:
            hotels_list = [convert_for_json(h) for h in recommended_hotels.to_dict('records')]
        
        return jsonify({
            'success': True,
            'attractions': attractions_list,
            'hotels': hotels_list,
            'query': query,
            'message': 'No results found' if not attractions_list and not hotels_list else None
        })
        
    except Exception as e:
        app.logger.error(f"API recommendation error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': "Internal server error"
        }), 500

# Common amenities for the search page
COMMON_AMENITIES = [
    'Free WiFi', 'Pool', 'Restaurant', 'Air Conditioning', 
    'Spa', 'Fitness Center', 'Parking', 'Pet Friendly'
]

def run_server():
    """Run the Flask application with proper socket handling"""
    try:
        app.logger.info("Starting Flask application")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except socket.error as e:
        app.logger.error(f"Socket error: {str(e)}")
    except KeyboardInterrupt:
        app.logger.info("Server shutdown requested")
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
    finally:
        cleanup_resources()
        sys.exit(0)

if __name__ == '__main__':
    run_server()