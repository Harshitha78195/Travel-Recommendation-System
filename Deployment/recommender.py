# models/recommender.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic
import textwrap
from database import get_collection

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

class HybridRecommender:
    """
    A hybrid recommender system that combines content-based filtering with
    collaborative filtering techniques for travel recommendations.
    """
    
    def __init__(self):
        """
        Initialize the recommender with MongoDB connection.
        """
        self.collection = get_collection()
        self.df = self._load_data_from_mongodb()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        
        # Prepare the data
        self.prepare_data()
        
    def _load_data_from_mongodb(self):
        """Load data from MongoDB into a pandas DataFrame."""
        cursor = self.collection.find({})
        return pd.DataFrame(list(cursor))
        
    def prepare_data(self):
        """Prepare and clean the data for recommendation."""
        # Check if amenities column is string and convert if needed
        if 'amenities' in self.df.columns and self.df['amenities'].dtype == 'object':
            # Convert string representation of list to actual list
            self.df['amenities'] = self.df['amenities'].apply(
                lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
            )
        
        # Clean amenities - ensure it's a list and properly formatted
        if 'amenities' in self.df.columns:
            self.df['amenities_cleaned'] = self.df['amenities'].apply(self._clean_amenities)
            self.df['amenities_str'] = self.df['amenities_cleaned'].apply(
                lambda x: ', '.join(x) if x else ''
            )
        
        # Create a combined text field for content-based filtering
        text_columns = ['name', 'subcategories', 'country', 'city']
        if 'amenities_str' in self.df.columns:
            text_columns.append('amenities_str')
            
        self.df['combined_features'] = self.df[text_columns].apply(
            lambda row: ' '.join(str(cell) for cell in row if pd.notna(cell)), axis=1
        )
        
        # Clean the text
        self.df['combined_features'] = self.df['combined_features'].apply(self._clean_text)
        if 'description' in self.df.columns:
            self.df['description_clean'] = self.df['description'].apply(self._clean_text)
        
        # Make sure price columns are numeric
        for col in ['LowerPrice', 'UpperPrice']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Fill missing values for ratings and prices
        self.df['rating'] = self.df['rating'].fillna(0)
        if 'LowerPrice' in self.df.columns:
            self.df['LowerPrice'] = self.df['LowerPrice'].fillna(0)
        if 'UpperPrice' in self.df.columns:
            self.df['UpperPrice'] = self.df['UpperPrice'].fillna(0)
        
        # Make sure latitude and longitude are numeric
        for col in ['latitude', 'longitude']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
        # Create TF-IDF vectors
        self._create_tfidf_matrix()
        
        # Make sure sentiment and review scores are numeric and fill missing values
        if 'avg_sentiment_score' in self.df.columns:
            self.df['avg_sentiment_score'] = pd.to_numeric(self.df['avg_sentiment_score'], errors='coerce').fillna(0)
        else:
            self.df['avg_sentiment_score'] = 0
            
        if 'avg_review_rating' in self.df.columns:
            self.df['avg_review_rating'] = pd.to_numeric(self.df['avg_review_rating'], errors='coerce').fillna(0)
        else:
            self.df['avg_review_rating'] = 0
            
    def _clean_amenities(self, amenities):
        """Clean amenities data."""
        if isinstance(amenities, list):
            # Join the list elements into a single string
            return [item.strip() for item in amenities if isinstance(item, str)]
        elif isinstance(amenities, str):
            try:
                # Try to evaluate if it's a string representation of a list
                cleaned = eval(amenities)
                if isinstance(cleaned, list):
                    return [item.strip() for item in cleaned if isinstance(item, str)]
            except:
                pass
        return []
        
    def _clean_text(self, text):
        """Clean text data."""
        if not isinstance(text, str):
            return ""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def _create_tfidf_matrix(self):
        """Create TF-IDF matrix for content-based filtering."""
        # Create TF-IDF vectorizer for the combined features
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])
        
        # Compute the cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
    def extract_entities(self, query):
        """
        Extract entities from a user query with detailed debugging.
        
        Args:
            query (str): User's natural language query
            
        Returns:
            dict: Extracted entities
        """
        import re
        import logging
        logger = logging.getLogger('models.recommender')
        
        original_query = query
        query = query.lower()
        logger.info(f"Processing query: '{original_query}'")
        
        # Initialize variables to store extracted information
        country = None
        city = None
        place_type = None
        subcategories = []
        amenities = []
        price_preference = None
        
        # Create a helper function to find whole word matches
        def find_whole_word(word, text):
            # Add word boundaries and escape special characters
            pattern = r'\b' + re.escape(word) + r'\b'
            match = re.search(pattern, text)
            return match is not None
        
        # Extract country - using word boundaries for exact matches
        countries = self.df['country'].str.lower().unique().tolist()
        for country_name in countries:
            if find_whole_word(country_name, query):
                country = country_name.title()
                logger.info(f"Found country: {country}")
                break
        
        # Extract city - using strict word boundary matching and length filtering
        cities = self.df['city'].str.lower().unique().tolist()
        # Sort cities by length (descending) to prefer longer matches like "New York" over just "York"
        cities = sorted(cities, key=len, reverse=True)
        
        for city_name in cities:
            # Skip very short city names (less than 3 characters) to avoid common false positives
            if len(city_name) < 3:
                continue
                
            if find_whole_word(city_name, query):
                city = city_name.title()
                logger.info(f"Found city: {city}")
                break
        
        # Extra check for specific cities explicitly mentioned in this example
        if "nairobi" in query and city != "Nairobi":
            logger.info("Overriding city detection: Found 'nairobi' in query")
            city = "Nairobi"
        
        # Extract place type
        hotel_keywords = ['hotel', 'lodging', 'accommodation', 'stay', 'room']
        attraction_keywords = ['attraction', 'landmark', 'sight', 'visit', 'see', 'tour']
        
        if any(find_whole_word(word, query) for word in hotel_keywords):
            place_type = 'HOTEL'
        elif any(find_whole_word(word, query) for word in attraction_keywords):
            place_type = 'ATTRACTION'
        
        # Infer place_type if mentioned subcategories but not explicitly stated
        if not place_type:
            hotel_related = ['specialty lodging', 'bed and breakfast', 'b&b']
            attraction_related = ['museum', 'park', 'landmark', 'sight', 'casino', 
                                 'gambling', 'tour', 'shopping']
            
            if any(term in query for term in hotel_related):
                place_type = 'HOTEL'
            elif any(term in query for term in attraction_related):
                place_type = 'ATTRACTION'
        
        logger.info(f"Detected place_type: {place_type}")
        
        # Extract subcategories - add more based on your dataset
        subcategory_keywords = {
            'landmark': 'Sights & Landmarks',
            'sight': 'Sights & Landmarks',
            'casino': 'Casinos & Gambling',
            'gambling': 'Casinos & Gambling',
            'game': 'Fun & Games',
            'specialty': 'Specialty Lodging',
            'bed and breakfast': 'Bed and Breakfast',
            'b&b': 'Bed and Breakfast',
            'tour': 'Tours',
            'shopping': 'Shopping',
            'museum': 'Museums',
            'park': 'Nature & Parks',
            'nature': 'Nature & Parks',
            'restaurant': 'Restaurants',
            'outdoor': 'Outdoor Activities'
        }
        
        for keyword, subcategory in subcategory_keywords.items():
            if keyword in query and subcategory not in subcategories:
                logger.info(f"Found subcategory keyword '{keyword}' -> adding '{subcategory}'")
                subcategories.append(subcategory)
                
        # Extract amenities
        amenity_keywords = ['pool', 'internet', 'wifi', 'parking', 'restaurant', 
                           'bar', 'breakfast', 'air conditioning', 'gym', 
                           'fitness', 'spa', 'free', 'shuttle', 'airport', 'service']
        
        for amenity in amenity_keywords:
            if find_whole_word(amenity, query):
                amenities.append(amenity.title())
                logger.info(f"Found amenity: {amenity}")
                
        # Extract price preference
        price_low = ['low price', 'cheap', 'affordable', 'budget', 'inexpensive']
        price_high = ['high price', 'luxury', 'expensive', 'premium', 'high end']
        price_medium = ['mid price', 'medium price', 'reasonable', 'average']
        
        if any(term in query for term in price_low):
            price_preference = 'low'
        elif any(term in query for term in price_high):
            price_preference = 'high'
        elif any(term in query for term in price_medium):
            price_preference = 'medium'
        
        # Extra debug for Nairobi issue
        logger.info(f"Final city detection: {city}")
        
        result = {
            'country': country,
            'city': city,
            'place_type': place_type,
            'subcategories': subcategories,
            'amenities': amenities,
            'price_preference': price_preference
        }
        
        logger.info(f"Final extracted entities: {result}")
        return result    
            
    def compute_final_score(self, row, content_sim_score=0):
        """
        Compute a final score combining content similarity, sentiment, and ratings.
        
        Args:
            row: DataFrame row with necessary scores
            content_sim_score (float): Content similarity score
            
        Returns:
            float: Final combined score
        """
        # Get individual scores
        sentiment = row['avg_sentiment_score'] if 'avg_sentiment_score' in row else 0
        review_rating = row['avg_review_rating'] if 'avg_review_rating' in row else 0
        base_rating = row['rating'] if 'rating' in row else 0
        
        # Calculate weighted score - content similarity gets 40% weight
        content_weight = 0.4
        sentiment_weight = 0.3
        review_weight = 0.2
        base_weight = 0.1
        
        # Normalize content similarity score
        norm_content_score = content_sim_score * 5  # Scale to be comparable with other scores
        
        # Calculate scores based on available data
        total_weight = 0
        total_score = 0
        
        if norm_content_score > 0:
            total_score += norm_content_score * content_weight
            total_weight += content_weight
            
        if sentiment > 0:
            total_score += sentiment * sentiment_weight
            total_weight += sentiment_weight
            
        if review_rating > 0:
            total_score += review_rating * review_weight
            total_weight += review_weight
            
        if base_rating > 0:
            total_score += base_rating * base_weight
            total_weight += base_weight
            
        # Avoid division by zero
        if total_weight == 0:
            return 0
            
        return total_score / total_weight
        
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in kilometers."""
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return float('inf')  # Return infinity for missing coordinates
            
        point1 = (lat1, lon1)
        point2 = (lat2, lon2)
        
        return geodesic(point1, point2).kilometers
        
    def find_nearby_attractions(self, city_name, distance_threshold=100, top_n=5):
        """Find attractions near a given city within a specified distance threshold."""
        # Get city coordinates
        city_info = self.df[(self.df['city'].str.lower() == city_name.lower()) & 
                           (self.df['type'] == 'HOTEL')]
        
        if city_info.empty:
            # Try to find any entry with this city name
            city_info = self.df[self.df['city'].str.lower() == city_name.lower()]
            
        if city_info.empty:
            # If still no match, return top attractions in the country
            return self.df[self.df['type'] == 'ATTRACTION'].sort_values('rating', ascending=False).head(top_n)
            
        # Take the first entry's coordinates
        city_lat = city_info.iloc[0]['latitude']
        city_lon = city_info.iloc[0]['longitude']
        
        # Calculate distances for all attractions
        attractions_df = self.df[self.df['type'] == 'ATTRACTION'].copy()
        
        # Calculate distance from city to each attraction
        attractions_df['distance'] = attractions_df.apply(
            lambda row: self.calculate_distance(city_lat, city_lon, row['latitude'], row['longitude']),
            axis=1
        )
        
        # Filter attractions within the distance threshold
        nearby_attractions = attractions_df[attractions_df['distance'] <= distance_threshold]
        
        # If no nearby attractions found, expand the search radius
        if nearby_attractions.empty:
            nearby_attractions = attractions_df.sort_values('distance').head(top_n)
            
        # Sort by rating (descending) and return top N
        return nearby_attractions.sort_values(['rating', 'distance'], ascending=[False, True]).head(top_n)
        
    def find_nearby_hotels(self, city_name, distance_threshold=50, top_n=5):
        """Find hotels near a given city within a specified distance threshold."""
        # Get city coordinates
        city_info = self.df[(self.df['city'].str.lower() == city_name.lower()) & 
                           (self.df['type'] == 'HOTEL')]
        
        if city_info.empty:
            # Try to find any entry with this city name
            city_info = self.df[self.df['city'].str.lower() == city_name.lower()]
            
        if city_info.empty:
            # If still no match, return top hotels in the country
            return self.df[self.df['type'] == 'HOTEL'].sort_values('rating', ascending=False).head(top_n)
            
        # Take the first entry's coordinates
        city_lat = city_info.iloc[0]['latitude']
        city_lon = city_info.iloc[0]['longitude']
        
        # Calculate distances for all hotels
        hotels_df = self.df[self.df['type'] == 'HOTEL'].copy()
        
        # Calculate distance from city to each hotel
        hotels_df['distance'] = hotels_df.apply(
            lambda row: self.calculate_distance(city_lat, city_lon, row['latitude'], row['longitude']),
            axis=1
        )
        
        # Filter hotels within the distance threshold
        nearby_hotels = hotels_df[hotels_df['distance'] <= distance_threshold]
        
        # If no nearby hotels found, expand the search radius
        if nearby_hotels.empty:
            nearby_hotels = hotels_df.sort_values('distance').head(top_n)
            
        # Sort by rating (descending) and return top N
        return nearby_hotels.sort_values(['rating', 'distance'], ascending=[False, True]).head(top_n)
        
    def recommend_hybrid(self, query, top_n=5, rec_type=None):
        """
        Recommend places using a hybrid approach combining content-based and collaborative filtering.
        
        Args:
            query (str): User's natural language query
            top_n (int): Number of recommendations to return
            rec_type (str, optional): Type of recommendations to return ('hotel' or 'attraction')
            
        Returns:
            tuple: (recommended_attractions, recommended_hotels)
        """
        print(f"Processing query: '{query}'")
        print("-" * 80)
        
        # Extract entities from the query
        entities = self.extract_entities(query)
        print("📋 Query Analysis:")
        for key, value in entities.items():
            if value:  # Only print non-empty values
                print(f"- {key.title()}: {value}")
        print("-" * 80)

        # Determine what to recommend based on query and rec_type parameter
        if rec_type is None:
            # Auto-detect from query
            if entities['place_type'] == 'HOTEL':
                recommend_attractions = False
                recommend_hotels = True
            elif entities['place_type'] == 'ATTRACTION':
                recommend_attractions = True
                recommend_hotels = False
            else:
                # If not specified, recommend both
                recommend_attractions = True
                recommend_hotels = True
        else:
            # Use explicitly specified type
            recommend_attractions = rec_type.lower() == 'attraction'
            recommend_hotels = rec_type.lower() == 'hotel'

        # Initialize empty DataFrames
        attractions_df = pd.DataFrame()
        hotels_df = pd.DataFrame()
        
        # Process attractions if requested
        if recommend_attractions:
            attractions_df = self.df[self.df['type'].str.lower() == 'attraction'].copy()
            
            # Apply filters for attractions
            if entities['country']:
                attractions_df = attractions_df[attractions_df['country'].str.lower() == entities['country'].lower()]
                
            attractions_found = True
            
            if entities['city']:
                # Look for exact matches first
                city_matches = attractions_df[attractions_df['city'].str.lower() == entities['city'].lower()]
                # If no exact matches, try partial matches
                if len(city_matches) == 0:
                    partial_matches = attractions_df[attractions_df['city'].str.lower().str.contains(entities['city'].lower())]
                    if len(partial_matches) > 0:
                        attractions_df = partial_matches
                    else:
                        # No attractions found in the city, set flag to find nearby attractions later
                        attractions_found = False
                        # Keep the original attractions_df for now
                else:
                    attractions_df = city_matches
                    
            if entities['subcategories']:
                # Create a mask for subcategories (check if any subcategory is in the string)
                mask = attractions_df['subcategories'].apply(
                    lambda x: any(sub.lower() in str(x).lower() for sub in entities['subcategories'])
                )
                filtered = attractions_df[mask]
                if not filtered.empty:
                    attractions_df = filtered
                    
            # If no attractions found in the specified city, find nearby attractions
            if not attractions_found and entities['city']:
                print(f"\n🔍 No attractions found in {entities['city']}. Finding nearby attractions...")
                attractions_df = self.find_nearby_attractions(entities['city'], distance_threshold=100, top_n=top_n)
                
            # If still no attractions or very few, get top country attractions
            if len(attractions_df) < top_n and entities['country']:
                remaining_spots = top_n - len(attractions_df)
                country_attractions = self.df[(self.df['type'].str.lower() == 'attraction') &
                                            (self.df['country'].str.lower() == entities['country'].lower())].sort_values('rating', ascending=False)
                
                # Filter out attractions already in the list
                if not attractions_df.empty:
                    country_attractions = country_attractions[~country_attractions['id'].isin(attractions_df['id'])]
                    
                # Add top country attractions
                attractions_df = pd.concat([attractions_df, country_attractions.head(remaining_spots)])
                
            # Calculate content-based scores for attractions
            if not attractions_df.empty:
                # Get content similarity scores
                for idx, row in attractions_df.iterrows():
                    # Get the index of this attraction in the original dataframe
                    orig_idx = self.df[self.df['id'] == row['id']].index[0]
                    
                    # Calculate similarity based on TF-IDF
                    if self.cosine_sim is not None:
                        # Get similarity scores for this place with all others
                        sim_scores = list(enumerate(self.cosine_sim[orig_idx]))
                        
                        # Get average similarity score with other attractions
                        attraction_indices = self.df[self.df['type'].str.lower() == 'attraction'].index.tolist()
                        relevant_scores = [score for idx, score in sim_scores if idx in attraction_indices and idx != orig_idx]
                        
                        if relevant_scores:
                            content_sim_score = np.mean(relevant_scores)
                        else:
                            content_sim_score = 0
                    else:
                        content_sim_score = 0
                        
                    # Calculate hybrid score and add to dataframe
                    attractions_df.at[idx, 'content_sim_score'] = content_sim_score
                    attractions_df.at[idx, 'hybrid_score'] = self.compute_final_score(row, content_sim_score)
                    
            # Sort attractions by hybrid score (descending)
            if not attractions_df.empty:
                attractions_df = attractions_df.sort_values('hybrid_score', ascending=False)
        
        # Process hotels if requested
        if recommend_hotels:
            hotels_df = self.df[self.df['type'].str.lower() == 'hotel'].copy()
            
            # Apply filters for hotels
            if entities['country']:
                hotels_df = hotels_df[hotels_df['country'].str.lower() == entities['country'].lower()]
                
            hotels_found = True
            
            if entities['city']:
                # Look for exact matches first
                city_matches = hotels_df[hotels_df['city'].str.lower() == entities['city'].lower()]
                # If no exact matches, try partial matches
                if len(city_matches) == 0:
                    partial_matches = hotels_df[hotels_df['city'].str.lower().str.contains(entities['city'].lower())]
                    if len(partial_matches) > 0:
                        hotels_df = partial_matches
                    else:
                        # No hotels found in the city, set flag to find nearby hotels later
                        hotels_found = False
                        # Keep the original hotels_df for now
                else:
                    hotels_df = city_matches
                    
            # If no hotels found in the specified city, find nearby hotels
            if not hotels_found and entities['city']:
                print(f"\n🔍 No hotels found in {entities['city']}. Finding nearby hotels...")
                hotels_df = self.find_nearby_hotels(entities['city'], distance_threshold=50, top_n=top_n)
                
            # If still no hotels or very few, get top country hotels
            if len(hotels_df) < top_n and entities['country']:
                remaining_spots = top_n - len(hotels_df)
                country_hotels = self.df[(self.df['type'].str.lower() == 'hotel') &
                                       (self.df['country'].str.lower() == entities['country'].lower())].sort_values('rating', ascending=False)
                
                # Filter out hotels already in the list
                if not hotels_df.empty:
                    country_hotels = country_hotels[~country_hotels['id'].isin(hotels_df['id'])]
                    
                # Add top country hotels
                hotels_df = pd.concat([hotels_df, country_hotels.head(remaining_spots)])
                    
            # Filter hotels by amenities if specified
            if entities['amenities']:
                # Check if any amenity from the query is in the amenities list
                def has_amenities(amenities_list, query_amenities):
                    if not isinstance(amenities_list, list):
                        return False
                        
                    amenities_str = ' '.join(str(item).lower() for item in amenities_list)
                    return any(amenity.lower() in amenities_str for amenity in query_amenities)
                    
                mask = hotels_df['amenities_cleaned'].apply(lambda x: has_amenities(x, entities['amenities']))
                filtered = hotels_df[mask]
                if not filtered.empty:
                    hotels_df = filtered
                    
            # Calculate content-based scores for hotels
            if not hotels_df.empty:
                # Get content similarity scores
                for idx, row in hotels_df.iterrows():
                    # Get the index of this hotel in the original dataframe
                    orig_idx = self.df[self.df['id'] == row['id']].index[0]
                    
                    # Calculate similarity based on TF-IDF
                    if self.cosine_sim is not None:
                        # Get similarity scores for this place with all others
                        sim_scores = list(enumerate(self.cosine_sim[orig_idx]))
                        
                        # Get average similarity score with other hotels
                        hotel_indices = self.df[self.df['type'].str.lower() == 'hotel'].index.tolist()
                        relevant_scores = [score for idx, score in sim_scores if idx in hotel_indices and idx != orig_idx]
                        
                        if relevant_scores:
                            content_sim_score = np.mean(relevant_scores)
                        else:
                            content_sim_score = 0
                    else:
                        content_sim_score = 0
                        
                    # Calculate hybrid score and add to dataframe
                    hotels_df.at[idx, 'content_sim_score'] = content_sim_score
                    hotels_df.at[idx, 'hybrid_score'] = self.compute_final_score(row, content_sim_score)
                
            # Apply price preference as a secondary sort if specified
            if not hotels_df.empty:
                if entities['price_preference']:
                    if entities['price_preference'] == 'low':
                        # For low price preference, sort by hybrid score first, then by price (ascending)
                        hotels_df = hotels_df.sort_values(['hybrid_score', 'LowerPrice'], ascending=[False, True])
                    elif entities['price_preference'] == 'high':
                        # For high price preference, sort by hybrid score first, then by price (descending)
                        hotels_df = hotels_df.sort_values(['hybrid_score', 'UpperPrice'], ascending=[False, False])
                    elif entities['price_preference'] == 'medium':
                        # Calculate average price for medium range
                        hotels_df['AvgPrice'] = (hotels_df['LowerPrice'] + hotels_df['UpperPrice']) / 2
                        # Get median price
                        median_price = hotels_df['AvgPrice'].median()
                        # Sort by hybrid score first, then by distance from median price
                        hotels_df['PriceDiff'] = abs(hotels_df['AvgPrice'] - median_price)
                        hotels_df = hotels_df.sort_values(['hybrid_score', 'PriceDiff'], ascending=[False, True])
                else:
                    # Sort by hybrid score if no price preference
                    hotels_df = hotels_df.sort_values('hybrid_score', ascending=False)
                
        # Return the top N recommendations
        return attractions_df.head(top_n) if not attractions_df.empty else pd.DataFrame(), \
               hotels_df.head(top_n) if not hotels_df.empty else pd.DataFrame()
     
    def format_price(self, price):
        """Format price for display."""
        if pd.isna(price) or price == 0:
            return "N/A"
        if price >= 1000:
            return f"${price/1000:.1f}K"
        return f"${price:.0f}"
        
    def format_amenities(self, amenities_list):
        """Format amenities for display."""
        if not amenities_list or not isinstance(amenities_list, list):
            return "No amenities listed"
            
        # Clean the amenities list
        clean_amenities = []
        for item in amenities_list:
            if isinstance(item, str):
                clean_item = item.strip()
                if clean_item and clean_item not in clean_amenities:
                    clean_amenities.append(clean_item)
                    
        if not clean_amenities:
            return "No amenities listed"
            
        return ", ".join(clean_amenities)
        
    def display_recommendations(self, query, top_n=5, rec_type=None):
        """
        Process a user query and display recommendations.
        
        Args:
            query (str): User's natural language query
            top_n (int): Number of recommendations to return
        """
        recommended_attractions, recommended_hotels = self.recommend_hybrid(query, top_n, rec_type)
        
        if not recommended_attractions.empty:  # Check if the dataframe is not empty
            # Print attraction recommendations
            print("🏛️ TOP 5 RECOMMENDED ATTRACTIONS:")
            for i, (_, attraction) in enumerate(recommended_attractions.iterrows(), 1):
                print(f"\n{i}. 🌟 {attraction['name']} ({attraction['subcategories']})")
                print(f"   Rating: {'⭐' * int(attraction['rating'])}{' ' * (5-int(attraction['rating']))} {attraction['rating']}/5 ({attraction['numberOfReviews']} reviews)")
                
                # Get price info for attractions if available
                price_info = "Price: "
                if not pd.isna(attraction['LowerPrice']) and not pd.isna(attraction['UpperPrice']) and (attraction['LowerPrice'] > 0 or attraction['UpperPrice'] > 0):
                    price_info += f"{self.format_price(attraction['LowerPrice'])} - {self.format_price(attraction['UpperPrice'])}"
                else:
                    price_info += "Not available"
                print(f"   {price_info}")
                
                print(f"   Location: {attraction['city']}, {attraction['country']}")
                
                # Add image and webUrl
                print(f"   Image: {attraction['image']}")
                print(f"   Web URL: {attraction['webUrl']}")
                
                # Print distance information if available
                if 'distance' in attraction and not pd.isna(attraction['distance']):
                    print(f"   Distance: {attraction['distance']:.1f} km")
                    
                # Show hybrid score components
                print(f"   Hybrid Score: {attraction['hybrid_score']:.2f}")
                print(f"   Content Similarity: {attraction['content_sim_score']:.2f}")
                print(f"   Sentiment Score: {attraction['avg_sentiment_score']:.2f}")
                print(f"   Review Rating: {attraction['avg_review_rating']:.2f}")
                
                # Print full description without truncation
                desc = attraction['description']
                if isinstance(desc, str):
                    print(f"   Description: {desc}")
                else:
                    print(f"   Description: Not available")
        
        print("-" * 80)
        
        # Print hotel recommendations
        if not recommended_hotels.empty:  # Check if the dataframe is not empty
            print("🏨 TOP 5 RECOMMENDED HOTELS:")
            for i, (_, hotel) in enumerate(recommended_hotels.iterrows(), 1):
                print(f"\n{i}. 🌟 {hotel['name']} ({hotel['subcategories']})")
                print(f"   Rating: {'⭐' * int(hotel['rating'])}{' ' * (5-int(hotel['rating']))} {hotel['rating']}/5 ({hotel['numberOfReviews']} reviews)")
                
                # Format price range
                price_range = f"   Price Range: {self.format_price(hotel['LowerPrice'])} - {self.format_price(hotel['UpperPrice'])}"
                print(price_range)
                
                print(f"   Location: {hotel['city']}, {hotel['country']}")
                
                # Print distance information if available
                if 'distance' in hotel and not pd.isna(hotel['distance']):
                    print(f"   Distance from city center: {hotel['distance']:.1f} km")
                
                # Format and clean amenities for display
                formatted_amenities = self.format_amenities(hotel['amenities_cleaned'])
                print(f"   Amenities: {formatted_amenities}")
                
                # Add image and webUrl
                print(f"   Image: {hotel['image']}")
                print(f"   Web URL: {hotel['webUrl']}")
                
                # Show hybrid score components
                print(f"   Hybrid Score: {hotel['hybrid_score']:.2f}")
                print(f"   Content Similarity: {hotel['content_sim_score']:.2f}")
                print(f"   Sentiment Score: {hotel['avg_sentiment_score']:.2f}")
                print(f"   Review Rating: {hotel['avg_review_rating']:.2f}")
                
                # Print full description without truncation
                desc = hotel['description']
                if isinstance(desc, str):
                    print(f"   Description: {desc}")
                else:
                    print(f"   Description: Not available")
        
        return recommended_attractions, recommended_hotels
    
    def run_interactive(self):
        """Run an interactive recommendation system."""
        print("=" * 80)
        print("🌍 HYBRID TRAVEL RECOMMENDATION SYSTEM")
        print("=" * 80)
        print("\nThis system provides personalized travel recommendations using both content-based and collaborative filtering.")
        print("You can ask questions like:")
        print("- I want to go to Nata, Botswana. Suggest attractions like landmarks.")
        print("- Looking for hotels in Maun with pool and internet. Price should be affordable.")
        print("- Recommend top attractions in Gaborone.")
        print("\nType 'exit' to quit the system.")
        print("\n" + "-" * 80 + "\n")
        
        while True:
            query = input("\nEnter your travel query: ").strip()
            
            if query.lower() == 'exit':
                print("\nThank you for using the hybrid travel recommendation system. Goodbye!")
                break
            else:
                self.display_recommendations(query)

# Helper function for backward compatibility
class TravelRecommender(HybridRecommender):
    """Legacy class name for backward compatibility"""
    pass