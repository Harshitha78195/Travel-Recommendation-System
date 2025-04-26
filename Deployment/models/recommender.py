# -*- coding: utf-8 -*-
"""hybrid.py

This module implements a hybrid recommendation system that combines:
1. Content-based filtering (using TF-IDF and textual data)
2. Collaborative filtering (using sentiment and review scores)
"""

import pandas as pd
import numpy as np
import re
import nltk
import logging
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.distance import geodesic

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
    def __init__(self, places_df):
        """
        Initialize the recommender with places data that already includes sentiment scores.
        
        Args:
            places_df (DataFrame): DataFrame containing places information with sentiment scores
        """
        self.df = places_df
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        
        # Prepare the data
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare and clean the data for recommendation."""
        try:
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
                
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def _clean_amenities(self, amenities):
        """Clean amenities data."""
        try:
            if isinstance(amenities, list):
                return [item.strip() for item in amenities if isinstance(item, str)]
            elif isinstance(amenities, str):
                try:
                    cleaned = eval(amenities)
                    if isinstance(cleaned, list):
                        return [item.strip() for item in cleaned if isinstance(item, str)]
                except:
                    pass
            return []
        except Exception as e:
            logger.warning(f"Error cleaning amenities: {str(e)}")
            return []

    def _clean_text(self, text):
        """Clean text data."""
        try:
            if not isinstance(text, str):
                return ""
            text = text.lower()
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            logger.warning(f"Error cleaning text: {str(e)}")
            return ""

    def _create_tfidf_matrix(self):
        """Create TF-IDF matrix for content-based filtering."""
        try:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_features'])
            self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        except Exception as e:
            logger.error(f"Error creating TF-IDF matrix: {str(e)}")
            raise

    def extract_entities(self, query):
        """
        Extract entities from a user query.
        
        Args:
            query (str): User's natural language query
            
        Returns:
            dict: Extracted entities
        """
        if not isinstance(query, str) or not query.strip():
            return {
                'country': None,
                'city': None,
                'place_type': None,
                'subcategories': [],
                'amenities': [],
                'price_preference': None
            }

        try:
            query = query.lower()
            
            # Initialize variables to store extracted information
            entities = {
                'country': None,
                'city': None,
                'place_type': None,
                'subcategories': [],
                'amenities': [],
                'price_preference': None
            }
            
            # Extract country - get from actual dataset
            if hasattr(self.df, 'country'):
                countries = self.df['country'].str.lower().unique().tolist()
                for country_name in countries:
                    if isinstance(country_name, str) and country_name in query:
                        entities['country'] = country_name.title()
                        break
                        
            # Extract city - get from actual dataset
            if hasattr(self.df, 'city'):
                cities = self.df['city'].str.lower().unique().tolist()
                for city_name in cities:
                    if isinstance(city_name, str) and city_name in query:
                        entities['city'] = city_name.title()
                        break
                        
            # Extract place type
            place_type_keywords = {
                'HOTEL': ['hotel', 'lodging', 'accommodation', 'stay', 'room'],
                'ATTRACTION': ['attraction', 'landmark', 'sight', 'visit', 'see', 'tour']
            }
            
            for place_type, keywords in place_type_keywords.items():
                if any(word in query for word in keywords):
                    entities['place_type'] = place_type
                    break
                    
            # Extract subcategories
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
                'nature': 'Nature & Parks',
                'restaurant': 'Restaurants',
                'outdoor': 'Outdoor Activities'
            }
            
            for keyword, subcategory in subcategory_keywords.items():
                if keyword in query:
                    entities['subcategories'].append(subcategory)
                    
            # Extract amenities
            amenity_keywords = ['pool', 'internet', 'wifi', 'parking', 'restaurant', 
                               'bar', 'breakfast', 'air conditioning', 'gym', 
                               'fitness', 'spa', 'free', 'shuttle', 'airport', 'service']
            for amenity in amenity_keywords:
                if amenity in query:
                    entities['amenities'].append(amenity.title())
                    
            # Extract price preference
            price_preference_keywords = {
                'low': ['low price', 'cheap', 'affordable', 'budget', 'inexpensive'],
                'high': ['high price', 'luxury', 'expensive', 'premium', 'high end'],
                'medium': ['mid price', 'medium price', 'reasonable', 'average']
            }
            
            for preference, terms in price_preference_keywords.items():
                if any(term in query for term in terms):
                    entities['price_preference'] = preference
                    break
                    
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {
                'country': None,
                'city': None,
                'place_type': None,
                'subcategories': [],
                'amenities': [],
                'price_preference': None
            }

    def compute_final_score(self, row, content_sim_score=0):
        """
        Compute a final score combining content similarity, sentiment, and ratings.
        
        Args:
            row: DataFrame row with necessary scores
            content_sim_score (float): Content similarity score
            
        Returns:
            float: Final combined score
        """
        try:
            # Get individual scores
            sentiment = row.get('avg_sentiment_score', 0)
            review_rating = row.get('avg_review_rating', 0)
            base_rating = row.get('rating', 0)
            
            # Calculate weighted score
            weights = {
                'content': 0.4,
                'sentiment': 0.3,
                'review': 0.2,
                'base': 0.1
            }
            
            # Normalize content similarity score
            norm_content_score = content_sim_score * 5
            
            # Calculate weighted sum
            total_score = 0
            total_weight = 0
            
            if norm_content_score > 0:
                total_score += norm_content_score * weights['content']
                total_weight += weights['content']
                
            if sentiment > 0:
                total_score += sentiment * weights['sentiment']
                total_weight += weights['sentiment']
                
            if review_rating > 0:
                total_score += review_rating * weights['review']
                total_weight += weights['review']
                
            if base_rating > 0:
                total_score += base_rating * weights['base']
                total_weight += weights['base']
                
            return total_score / total_weight if total_weight > 0 else 0
            
        except Exception as e:
            logger.error(f"Error computing final score: {str(e)}")
            return 0

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in kilometers."""
        try:
            if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
                return float('inf')
                
            point1 = (lat1, lon1)
            point2 = (lat2, lon2)
            
            return geodesic(point1, point2).kilometers
        except Exception as e:
            logger.error(f"Error calculating distance: {str(e)}")
            return float('inf')

    def find_nearby_attractions(self, city_name, distance_threshold=100, top_n=5):
        """Find attractions near a given city within a specified distance threshold."""
        try:
            # Get city coordinates
            city_info = self.df[(self.df['city'].str.lower() == city_name.lower()) & 
                               (self.df['type'] == 'HOTEL')]
            
            if city_info.empty:
                city_info = self.df[self.df['city'].str.lower() == city_name.lower()]
                
            if city_info.empty:
                return self.df[self.df['type'] == 'ATTRACTION'].sort_values('rating', ascending=False).head(top_n)
                
            city_lat = city_info.iloc[0]['latitude']
            city_lon = city_info.iloc[0]['longitude']
            
            attractions_df = self.df[self.df['type'] == 'ATTRACTION'].copy()
            attractions_df['distance'] = attractions_df.apply(
                lambda row: self.calculate_distance(city_lat, city_lon, row['latitude'], row['longitude']),
                axis=1
            )
            
            nearby_attractions = attractions_df[attractions_df['distance'] <= distance_threshold]
            
            if nearby_attractions.empty:
                nearby_attractions = attractions_df.sort_values('distance').head(top_n)
                
            return nearby_attractions.sort_values(['rating', 'distance'], ascending=[False, True]).head(top_n)
            
        except Exception as e:
            logger.error(f"Error finding nearby attractions: {str(e)}")
            return pd.DataFrame()

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
        try:
            logger.info(f"Processing query: '{query}'")
            
            # Extract entities from the query
            entities = self.extract_entities(query)
            logger.info(f"Extracted entities: {entities}")

            # Determine recommendation types
            if rec_type is None:
                if entities['place_type'] == 'HOTEL':
                    recommend_attractions = False
                    recommend_hotels = True
                elif entities['place_type'] == 'ATTRACTION':
                    recommend_attractions = True
                    recommend_hotels = False
                else:
                    recommend_attractions = True
                    recommend_hotels = True
            else:
                recommend_attractions = rec_type.lower() == 'attraction'
                recommend_hotels = rec_type.lower() == 'hotel'

            # Process attractions if requested
            attractions_df = pd.DataFrame()
            if recommend_attractions:
                attractions_df = self._process_attractions(entities, top_n)

            # Process hotels if requested
            hotels_df = pd.DataFrame()
            if recommend_hotels:
                hotels_df = self._process_hotels(entities, top_n)

            logger.info(f"Found {len(attractions_df)} attractions and {len(hotels_df)} hotels")
            
            return attractions_df.head(top_n) if not attractions_df.empty else pd.DataFrame(), \
                   hotels_df.head(top_n) if not hotels_df.empty else pd.DataFrame()
                   
        except Exception as e:
            logger.error(f"Error in recommend_hybrid: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()

    def _process_attractions(self, entities, top_n):
        """Process and filter attractions based on extracted entities."""
        try:
            attractions_df = self.df[self.df['type'] == 'ATTRACTION'].copy()
            
            # Apply country filter
            if entities['country']:
                attractions_df = attractions_df[attractions_df['country'].str.lower() == entities['country'].lower()]
                
            attractions_found = True
            
            # Apply city filter
            if entities['city']:
                city_matches = attractions_df[attractions_df['city'].str.lower() == entities['city'].lower()]
                if len(city_matches) == 0:
                    partial_matches = attractions_df[attractions_df['city'].str.lower().str.contains(entities['city'].lower())]
                    if len(partial_matches) > 0:
                        attractions_df = partial_matches
                    else:
                        attractions_found = False
                else:
                    attractions_df = city_matches
                    
            # Apply subcategory filter
            if entities['subcategories']:
                mask = attractions_df['subcategories'].apply(
                    lambda x: any(sub.lower() in str(x).lower() for sub in entities['subcategories'])
                )
                filtered = attractions_df[mask]
                if not filtered.empty:
                    attractions_df = filtered
                    
            # Find nearby attractions if none found in specified city
            if not attractions_found and entities['city']:
                attractions_df = self.find_nearby_attractions(entities['city'], distance_threshold=100, top_n=top_n)
                
            # Get top country attractions if still not enough
            if len(attractions_df) < top_n and entities['country']:
                remaining_spots = top_n - len(attractions_df)
                country_attractions = self.df[(self.df['type'] == 'ATTRACTION') &
                                            (self.df['country'].str.lower() == entities['country'].lower())].sort_values('rating', ascending=False)
                
                if not attractions_df.empty:
                    country_attractions = country_attractions[~country_attractions['_id'].isin(attractions_df['_id'])]
                    
                attractions_df = pd.concat([attractions_df, country_attractions.head(remaining_spots)])
                
            # Calculate scores for attractions
            if not attractions_df.empty:
                for idx, row in attractions_df.iterrows():
                    orig_idx = self.df[self.df['_id'] == row['_id']].index[0]
                    
                    if self.cosine_sim is not None:
                        sim_scores = list(enumerate(self.cosine_sim[orig_idx]))
                        attraction_indices = self.df[self.df['type'] == 'ATTRACTION'].index.tolist()
                        relevant_scores = [score for idx, score in sim_scores if idx in attraction_indices and idx != orig_idx]
                        content_sim_score = np.mean(relevant_scores) if relevant_scores else 0
                    else:
                        content_sim_score = 0
                        
                    attractions_df.at[idx, 'content_sim_score'] = content_sim_score
                    attractions_df.at[idx, 'hybrid_score'] = self.compute_final_score(row, content_sim_score)
                    
                attractions_df = attractions_df.sort_values('hybrid_score', ascending=False)
            
            return attractions_df
            
        except Exception as e:
            logger.error(f"Error processing attractions: {str(e)}")
            return pd.DataFrame()

    def _process_hotels(self, entities, top_n):
        """Process and filter hotels based on extracted entities."""
        try:
            hotels_df = self.df[self.df['type'] == 'HOTEL'].copy()
            
            # Apply country filter
            if entities['country']:
                hotels_df = hotels_df[hotels_df['country'].str.lower() == entities['country'].lower()]
                
            # Apply city filter
            if entities['city']:
                city_matches = hotels_df[hotels_df['city'].str.lower() == entities['city'].lower()]
                if len(city_matches) == 0:
                    hotels_df = hotels_df[hotels_df['city'].str.lower().str.contains(entities['city'].lower())]
                else:
                    hotels_df = city_matches
                    
            # Apply amenities filter
            if entities['amenities']:
                def has_amenities(amenities_list, query_amenities):
                    if not isinstance(amenities_list, list):
                        return False
                    amenities_str = ' '.join(str(item).lower() for item in amenities_list)
                    return any(amenity.lower() in amenities_str for amenity in query_amenities)
                    
                mask = hotels_df['amenities_cleaned'].apply(lambda x: has_amenities(x, entities['amenities']))
                filtered = hotels_df[mask]
                if not filtered.empty:
                    hotels_df = filtered
                    
            # Calculate scores for hotels
            if not hotels_df.empty:
                for idx, row in hotels_df.iterrows():
                    orig_idx = self.df[self.df['_id'] == row['_id']].index[0]
                    
                    if self.cosine_sim is not None:
                        sim_scores = list(enumerate(self.cosine_sim[orig_idx]))
                        hotel_indices = self.df[self.df['type'] == 'HOTEL'].index.tolist()
                        relevant_scores = [score for idx, score in sim_scores if idx in hotel_indices and idx != orig_idx]
                        content_sim_score = np.mean(relevant_scores) if relevant_scores else 0
                    else:
                        content_sim_score = 0
                        
                    hotels_df.at[idx, 'content_sim_score'] = content_sim_score
                    hotels_df.at[idx, 'hybrid_score'] = self.compute_final_score(row, content_sim_score)
                
                # Apply price preference sorting
                if entities['price_preference']:
                    if entities['price_preference'] == 'low':
                        hotels_df = hotels_df.sort_values(['hybrid_score', 'LowerPrice'], ascending=[False, True])
                    elif entities['price_preference'] == 'high':
                        hotels_df = hotels_df.sort_values(['hybrid_score', 'UpperPrice'], ascending=[False, False])
                    elif entities['price_preference'] == 'medium':
                        hotels_df['AvgPrice'] = (hotels_df['LowerPrice'] + hotels_df['UpperPrice']) / 2
                        median_price = hotels_df['AvgPrice'].median()
                        hotels_df['PriceDiff'] = abs(hotels_df['AvgPrice'] - median_price)
                        hotels_df = hotels_df.sort_values(['hybrid_score', 'PriceDiff'], ascending=[False, True])
                else:
                    hotels_df = hotels_df.sort_values('hybrid_score', ascending=False)
            
            return hotels_df
            
        except Exception as e:
            logger.error(f"Error processing hotels: {str(e)}")
            return pd.DataFrame()