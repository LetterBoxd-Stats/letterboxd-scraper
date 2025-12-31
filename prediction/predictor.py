"""
predict_and_upload.py

Script to load XGBoost models from MongoDB, predict ratings and likes for all user-film pairs,
and upload predicted reviews to films collection with flags for existing interactions.
"""

import base64
import pickle
from pymongo import MongoClient
import logging
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Tuple, Any, Optional

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
load_dotenv()

# MongoDB configuration
DB_URI = os.getenv("DB_URI")
DB_NAME = os.getenv("DB_NAME")
FILMS_COLLECTION = os.getenv("DB_FILMS_COLLECTION")
USERS_COLLECTION = os.getenv("DB_USERS_COLLECTION")
MODELS_COLLECTION = os.getenv("DB_MODELS_COLLECTION")

# Sentinel value for missing data (matches training)
SENTINEL_MISSING = -1.0

# ──────────────────────────────────────────────
# DATABASE CONNECTION
# ──────────────────────────────────────────────
def connect_to_mongodb():
    """Establish connection to MongoDB"""
    logging.info("Connecting to MongoDB...")
    client = MongoClient(DB_URI)
    db = client[DB_NAME]
    return db

# ──────────────────────────────────────────────
# LOAD MODELS FROM MONGODB
# ──────────────────────────────────────────────
def load_xgboost_models(db):
    """Load XGBoost models from MongoDB collection"""
    models_col = db[MODELS_COLLECTION]
    model_doc = models_col.find_one({"name": "predictor"})
    
    if not model_doc:
        raise ValueError("No predictor model found in MongoDB")
    
    logging.info(f"Loading models (last updated: {model_doc.get('last_updated')})")
    
    # Load rating model
    if "rating_model_b64" not in model_doc:
        raise ValueError("Rating model not found in database")
    
    rating_model_bytes = base64.b64decode(model_doc["rating_model_b64"])
    rating_model = pickle.loads(rating_model_bytes)
    
    # Load feature columns for rating model
    feature_columns = model_doc.get("feature_columns", [])
    
    # Load like model if available
    like_model = None
    like_feature_names = []
    
    if model_doc.get("has_like_model", False) and "like_model_b64" in model_doc:
        like_model_bytes = base64.b64decode(model_doc["like_model_b64"])
        like_model = pickle.loads(like_model_bytes)
        
        # Try to get feature names from model or database
        if hasattr(like_model, 'feature_names_'):
            like_feature_names = like_model.feature_names_
        elif "like_feature_importance" in model_doc:
            like_feature_names = [item["feature"] for item in model_doc["like_feature_importance"]]
    
    # Load performance metrics
    rating_performance = model_doc.get("rating_performance", {})
    
    logging.info(f"Rating model loaded ({len(feature_columns)} features)")
    logging.info(f"Like model loaded: {like_model is not None}")
    
    return {
        "rating_model": rating_model,
        "like_model": like_model,
        "feature_columns": feature_columns,
        "like_feature_names": like_feature_names,
        "rating_performance": rating_performance,
        "model_type": model_doc.get("model_type", "xgboost")
    }

# ──────────────────────────────────────────────
# LOAD USERS AND FILMS DATA
# ──────────────────────────────────────────────
def load_users_and_films(db):
    """Load all users and films data from MongoDB"""
    users_col = db[USERS_COLLECTION]
    films_col = db[FILMS_COLLECTION]
    
    # Load users with their stats
    logging.info("Loading users data...")
    users = {}
    for user_doc in users_col.find({}, {"username": 1, "stats": 1}):
        username = user_doc.get("username")
        if username:
            users[username] = {
                "stats": user_doc.get("stats", {}),
                "existing_ratings": {},  # film_id -> rating
                "existing_likes": {},    # film_id -> liked (True/False)
                "existing_watches": set() # film_ids watched
            }
    
    # Load films with metadata and existing reviews/watches
    logging.info("Loading films data...")
    films = {}
    
    # We need to check all films for predictions
    for film_doc in films_col.find({}, {
        "film_id": 1,
        "film_title": 1,
        "avg_rating": 1,
        "like_ratio": 1,
        "metadata": 1,
        "reviews": 1,
        "watches": 1,
        "num_ratings": 1,
        "num_likes": 1,
        "num_watches": 1
    }):
        film_id = film_doc.get("film_id")
        if not film_id:
            continue
        
        # Process reviews to track existing user interactions
        existing_reviews = []
        for review in film_doc.get("reviews", []):
            user = review.get("user")
            rating = review.get("rating")
            is_liked = review.get("is_liked", False)
            
            if user and user in users and rating is not None:
                users[user]["existing_ratings"][film_id] = float(rating)
                users[user]["existing_likes"][film_id] = bool(is_liked)
                users[user]["existing_watches"].add(film_id)
                
                existing_reviews.append({
                    "user": user,
                    "rating": float(rating),
                    "is_liked": bool(is_liked),
                    "already_rated": True,
                    "already_watched": True
                })
        
        # Process watches (may have likes but no ratings)
        existing_watches = []
        for watch in film_doc.get("watches", []):
            user = watch.get("user")
            is_liked = watch.get("is_liked", False)
            
            if user and user in users:
                users[user]["existing_watches"].add(film_id)
                if film_id not in users[user]["existing_likes"]:
                    users[user]["existing_likes"][film_id] = bool(is_liked)
                
                # Only add as existing watch if not already in reviews
                if film_id not in users[user]["existing_ratings"]:
                    existing_watches.append({
                        "user": user,
                        "rating": None,
                        "is_liked": bool(is_liked),
                        "already_rated": False,
                        "already_watched": True
                    })
        
        # Store film data
        metadata = film_doc.get("metadata", {})
        films[film_id] = {
            "film_id": film_id,
            "film_title": film_doc.get("film_title", ""),
            "avg_rating": film_doc.get("avg_rating"),
            "like_ratio": film_doc.get("like_ratio", 0),
            "num_ratings": film_doc.get("num_ratings", 0),
            "num_likes": film_doc.get("num_likes", 0),
            "num_watches": film_doc.get("num_watches", 0),
            "metadata": metadata,
            "existing_reviews": existing_reviews,  # Reviews with ratings
            "existing_watches": existing_watches   # Watches without ratings
        }
    
    logging.info(f"Loaded {len(users)} users and {len(films)} films")
    return users, films

# ──────────────────────────────────────────────
# FEATURE ENGINEERING FOR PREDICTION
# ──────────────────────────────────────────────
def prepare_features_for_prediction(user_id: str, user_data: Dict, film_id: str, film_data: Dict) -> Dict:
    """Prepare feature vector for a user-film pair"""
    
    user_stats = user_data.get("stats", {})
    film_metadata = film_data.get("metadata", {})
    
    # Get film aggregates excluding current user (for new predictions)
    film_reviews = film_data.get("existing_reviews", [])
    film_watches = film_data.get("existing_watches", [])
    all_film_interactions = film_reviews + film_watches
    
    # Calculate film aggregates excluding current user
    other_interactions = [r for r in all_film_interactions if r.get("user") != user_id]
    
    if len(other_interactions) >= 1:
        other_ratings = [float(r["rating"]) for r in other_interactions if r.get("rating") is not None]
        film_avg_rating_excl_user = sum(other_ratings) / len(other_ratings) if other_ratings else film_data.get("avg_rating", SENTINEL_MISSING)
        
        other_likes = [r.get("is_liked", False) for r in other_interactions]
        film_like_ratio_excl_user = sum(1 for l in other_likes if l) / len(other_likes) if other_likes else film_data.get("like_ratio", SENTINEL_MISSING)
    else:
        film_avg_rating_excl_user = film_data.get("avg_rating", SENTINEL_MISSING)
        film_like_ratio_excl_user = film_data.get("like_ratio", SENTINEL_MISSING)
    
    # User statistics
    user_num_ratings = user_stats.get("num_ratings", 0)
    user_num_likes = user_stats.get("num_likes", 0)
    user_num_watches = user_stats.get("num_watches", 0)
    
    # User aggregates excluding current film if they've interacted with it
    user_has_rated = film_id in user_data.get("existing_ratings", {})
    user_has_watched = film_id in user_data.get("existing_watches", set())
    
    if user_has_rated:
        # Adjust counts for leave-one-out
        user_num_ratings_excl = max(0, user_num_ratings - 1)
        user_rating = user_data["existing_ratings"][film_id]
        user_avg_rating_excl_film = (
            (user_stats.get("avg_rating", 0) * user_num_ratings - user_rating)
            / user_num_ratings_excl if user_num_ratings_excl > 0 else SENTINEL_MISSING
        )
    else:
        user_avg_rating_excl_film = user_stats.get("avg_rating", SENTINEL_MISSING)
    
    if user_has_watched:
        user_num_watches_excl = max(0, user_num_watches - 1)
        user_is_liked = user_data.get("existing_likes", {}).get(film_id, False)
        user_num_likes_excl = max(0, user_num_likes - (1 if user_is_liked else 0))
        user_like_ratio_excl_film = user_num_likes_excl / user_num_watches_excl if user_num_watches_excl > 0 else SENTINEL_MISSING
    else:
        user_like_ratio_excl_film = user_num_likes / user_num_watches if user_num_watches > 0 else SENTINEL_MISSING
    
    # Genre compatibility features
    film_genres = film_metadata.get("genres", [])
    genre_stats = user_stats.get("genre_stats", {})
    
    genre_ratings = []
    genre_like_ratios = []
    genre_counts = []
    
    for genre in film_genres:
        if genre in genre_stats:
            genre_data = genre_stats[genre]
            genre_rating = genre_data.get("avg_rating")
            genre_count = genre_data.get("count", 0)
            genre_like_ratio = genre_data.get("like_ratio")
            
            if genre_rating is not None:
                genre_ratings.append(genre_rating)
                genre_counts.append(genre_count)
            
            if genre_like_ratio is not None:
                genre_like_ratios.append(genre_like_ratio)
    
    # Genre features for rating prediction
    if genre_ratings:
        max_genre_rating = max(genre_ratings)
        min_genre_rating = min(genre_ratings)
        avg_genre_rating = sum(genre_ratings) / len(genre_ratings)
        total_genre_watches = sum(genre_counts)
    else:
        max_genre_rating = SENTINEL_MISSING
        min_genre_rating = SENTINEL_MISSING
        avg_genre_rating = SENTINEL_MISSING
        total_genre_watches = SENTINEL_MISSING
    
    # Genre feature for like prediction
    avg_genre_like_ratio = sum(genre_like_ratios) / len(genre_like_ratios) if genre_like_ratios else SENTINEL_MISSING
    
    # Build features dictionary
    features = {
        # User-level
        "user_avg_rating": user_avg_rating_excl_film,
        "user_stdev_rating": user_stats.get("stdev_rating", SENTINEL_MISSING),
        "user_like_ratio": user_like_ratio_excl_film,
        
        # Film-level
        "film_avg_rating": film_avg_rating_excl_user,
        "film_like_ratio": film_like_ratio_excl_user,
        "film_num_ratings": film_data.get("num_ratings", 0),
        "film_num_watches": film_data.get("num_watches", 0),
        "film_letterboxd_avg": film_metadata.get("avg_rating", 3.0),
        "film_runtime": film_metadata.get("runtime", 0),
        "film_year": film_metadata.get("year", 0),
        
        # Genre compatibility
        "max_genre_rating": max_genre_rating,
        "min_genre_rating": min_genre_rating,
        "avg_genre_rating": avg_genre_rating,
        "total_genre_watches": total_genre_watches,
        "avg_genre_like_ratio": avg_genre_like_ratio
    }
    
    return features

# ──────────────────────────────────────────────
# MAKE PREDICTIONS
# ──────────────────────────────────────────────
def predict_for_all_users(users: Dict, films: Dict, models: Dict) -> Dict:
    """Predict ratings and likes for all user-film combinations"""
    
    rating_model = models["rating_model"]
    like_model = models["like_model"]
    feature_columns = models["feature_columns"]
    like_feature_names = models["like_feature_names"]
    
    predictions = {}
    total_predictions = 0
    
    logging.info("Starting predictions for all user-film pairs...")
    
    for film_id, film_data in films.items():
        film_predictions = []
        
        for user_id, user_data in users.items():
            # Check if user already has a rating or watch for this film
            already_rated = film_id in user_data.get("existing_ratings", {})
            already_watched = film_id in user_data.get("existing_watches", set())
            
            if already_rated:
                # Use existing rating and like
                rating = user_data["existing_ratings"][film_id]
                is_liked = user_data.get("existing_likes", {}).get(film_id, False)
                
                prediction_entry = {
                    "user": user_id,
                    "predicted_rating": float(rating),
                    "predicted_like_probability": 1.0 if is_liked else 0.0,
                    "predicted_like": bool(is_liked),
                    "already_rated": True,
                    "already_watched": True,
                    "prediction_timestamp": datetime.utcnow(),
                    "is_prediction": False  # This is actual data, not prediction
                }
                
            elif already_watched:
                # User watched but didn't rate - use existing like status
                is_liked = user_data.get("existing_likes", {}).get(film_id, False)
                
                # Still predict rating since user hasn't rated
                features = prepare_features_for_prediction(user_id, user_data, film_id, film_data)
                rating_features = [features.get(col, 0) for col in feature_columns]
                
                try:
                    predicted_rating = float(rating_model.predict([rating_features])[0])
                except Exception as e:
                    logging.warning(f"Rating prediction failed for user {user_id}, film {film_id}: {e}")
                    predicted_rating = features.get("user_avg_rating", 3.0)
                
                prediction_entry = {
                    "user": user_id,
                    "predicted_rating": predicted_rating,
                    "predicted_like_probability": 1.0 if is_liked else 0.0,
                    "predicted_like": bool(is_liked),
                    "already_rated": False,
                    "already_watched": True,
                    "prediction_timestamp": datetime.utcnow(),
                    "is_prediction": True  # Rating is predicted, like is actual
                }
                
            else:
                # No existing interaction - predict both rating and like
                features = prepare_features_for_prediction(user_id, user_data, film_id, film_data)
                
                # Predict rating
                rating_features = [features.get(col, 0) for col in feature_columns]
                
                try:
                    predicted_rating = float(rating_model.predict([rating_features])[0])
                except Exception as e:
                    logging.warning(f"Rating prediction failed for user {user_id}, film {film_id}: {e}")
                    predicted_rating = features.get("user_avg_rating", 3.0)
                
                # Predict like probability
                predicted_like = False
                predicted_like_probability = 0.0
                
                if like_model and like_feature_names:
                    try:
                        like_features = [
                            features.get("user_like_ratio", 0),
                            user_data.get("stats", {}).get("mean_abs_diff", 0),  # user_rating_consistency
                            features.get("film_avg_rating", 0),
                            features.get("film_like_ratio", 0),
                            features.get("film_num_ratings", 0),
                            features.get("film_letterboxd_avg", 0),
                            features.get("film_runtime", 0),
                            features.get("film_year", 0),
                            features.get("avg_genre_like_ratio", 0),
                            features.get("total_genre_watches", 0)
                        ]
                        
                        predicted_like_probability = float(like_model.predict_proba([like_features])[0][1])
                        predicted_like = predicted_like_probability >= 0.5
                    except Exception as e:
                        logging.warning(f"Like prediction failed for user {user_id}, film {film_id}: {e}")
                        # Fallback: use user's average like ratio
                        predicted_like_probability = features.get("user_like_ratio", 0.5)
                        predicted_like = predicted_like_probability >= 0.5
                else:
                    # Fallback if no like model
                    predicted_like_probability = features.get("user_like_ratio", 0.5)
                    predicted_like = predicted_like_probability >= 0.5
                
                prediction_entry = {
                    "user": user_id,
                    "predicted_rating": predicted_rating,
                    "predicted_like_probability": predicted_like_probability,
                    "predicted_like": bool(predicted_like),
                    "already_rated": False,
                    "already_watched": False,
                    "prediction_timestamp": datetime.utcnow(),
                    "is_prediction": True  # Both rating and like are predicted
                }
            
            film_predictions.append(prediction_entry)
            total_predictions += 1
        
        predictions[film_id] = film_predictions
    
    logging.info(f"Generated {total_predictions:,} predictions across {len(films)} films")
    return predictions

# ──────────────────────────────────────────────
# UPLOAD PREDICTIONS TO MONGODB
# ──────────────────────────────────────────────
def upload_predictions_to_mongodb(db, predictions: Dict, batch_size: int = 100):
    """Upload predicted reviews to films collection"""
    films_col = db[FILMS_COLLECTION]
    total_uploaded = 0
    
    logging.info(f"Uploading predictions to MongoDB (batch size: {batch_size})...")
    
    for film_id, film_predictions in predictions.items():
        # Prepare the predicted_reviews array
        predicted_reviews = []
        
        for pred in film_predictions:
            predicted_review = {
                "user": pred["user"],
                "predicted_rating": pred["predicted_rating"],
                "predicted_like_probability": pred["predicted_like_probability"],
                "predicted_like": pred["predicted_like"],
                "already_rated": pred["already_rated"],
                "already_watched": pred["already_watched"],
                "prediction_timestamp": pred["prediction_timestamp"],
                "is_prediction": pred["is_prediction"]
            }
            predicted_reviews.append(predicted_review)
        
        # Update film document with predicted_reviews
        result = films_col.update_one(
            {"film_id": film_id},
            {
                "$set": {
                    "predicted_reviews": predicted_reviews,
                    "predicted_reviews_last_updated": datetime.utcnow(),
                    "num_predicted_reviews": len(predicted_reviews)
                }
            }
        )
        
        if result.modified_count > 0:
            total_uploaded += 1
        
        # Log progress for large collections
        if total_uploaded % batch_size == 0:
            logging.info(f"Uploaded predictions for {total_uploaded} films...")
    
    logging.info(f"✓ Uploaded predictions for {total_uploaded} films to MongoDB")
    return total_uploaded

# ──────────────────────────────────────────────
# 📊 GENERATE PREDICTION SUMMARY
# ──────────────────────────────────────────────
def generate_prediction_summary(predictions: Dict, users: Dict, films: Dict) -> Dict:
    """Generate summary statistics about the predictions"""
    
    total_predictions = 0
    total_predicted_ratings = 0
    total_predicted_likes = 0
    already_rated_count = 0
    already_watched_count = 0
    fully_predicted_count = 0
    
    rating_values = []
    like_probabilities = []
    
    for film_id, film_predictions in predictions.items():
        for pred in film_predictions:
            total_predictions += 1
            rating_values.append(pred["predicted_rating"])
            like_probabilities.append(pred["predicted_like_probability"])
            
            if pred["already_rated"]:
                already_rated_count += 1
            if pred["already_watched"]:
                already_watched_count += 1
            if pred["is_prediction"] and not pred["already_rated"] and not pred["already_watched"]:
                fully_predicted_count += 1
            
            if pred["predicted_like"]:
                total_predicted_likes += 1
    
    summary = {
        "total_predictions": total_predictions,
        "total_users": len(users),
        "total_films": len(films),
        "already_rated_count": already_rated_count,
        "already_watched_count": already_watched_count,
        "fully_predicted_count": fully_predicted_count,
        "rating_statistics": {
            "mean": np.mean(rating_values) if rating_values else 0,
            "std": np.std(rating_values) if rating_values else 0,
            "min": np.min(rating_values) if rating_values else 0,
            "max": np.max(rating_values) if rating_values else 0,
            "median": np.median(rating_values) if rating_values else 0
        },
        "like_statistics": {
            "total_predicted_likes": total_predicted_likes,
            "like_probability_mean": np.mean(like_probabilities) if like_probabilities else 0,
            "like_probability_std": np.std(like_probabilities) if like_probabilities else 0,
            "predicted_like_ratio": total_predicted_likes / total_predictions if total_predictions > 0 else 0
        },
        "prediction_timestamp": datetime.utcnow()
    }
    
    logging.info("\n" + "="*80)
    logging.info("📊 PREDICTION SUMMARY")
    logging.info("="*80)
    logging.info(f"Total predictions: {summary['total_predictions']:,}")
    logging.info(f"Users: {summary['total_users']:,}, Films: {summary['total_films']:,}")
    logging.info(f"Already rated: {summary['already_rated_count']:,}")
    logging.info(f"Already watched: {summary['already_watched_count']:,}")
    logging.info(f"Fully predicted (new): {summary['fully_predicted_count']:,}")
    logging.info(f"\n📈 Rating Statistics:")
    logging.info(f"  Mean: {summary['rating_statistics']['mean']:.3f}")
    logging.info(f"  Std: {summary['rating_statistics']['std']:.3f}")
    logging.info(f"  Range: [{summary['rating_statistics']['min']:.2f}, {summary['rating_statistics']['max']:.2f}]")
    logging.info(f"\n❤️  Like Statistics:")
    logging.info(f"  Predicted likes: {summary['like_statistics']['total_predicted_likes']:,}")
    logging.info(f"  Like ratio: {summary['like_statistics']['predicted_like_ratio']:.3f}")
    logging.info(f"  Like probability mean: {summary['like_statistics']['like_probability_mean']:.3f}")
    logging.info("="*80)
    
    return summary

# ──────────────────────────────────────────────
# 🚀 MAIN EXECUTION PIPELINE
# ──────────────────────────────────────────────
def main():
    """Main execution pipeline"""
    logging.info("Starting prediction and upload pipeline...")
    
    try:
        # Connect to MongoDB
        db = connect_to_mongodb()
        
        # Load trained models
        models = load_xgboost_models(db)
        
        # Load users and films data
        users, films = load_users_and_films(db)
        
        if not users or not films:
            logging.error("No users or films found in database")
            return
        
        # Generate predictions for all user-film pairs
        predictions = predict_for_all_users(users, films, models)
        
        # Generate summary statistics
        summary = generate_prediction_summary(predictions, users, films)
        
        # Upload predictions to MongoDB
        uploaded_count = upload_predictions_to_mongodb(db, predictions, batch_size=50)
        
        # Save summary to a separate collection
        db["prediction_summaries"].insert_one(summary)
        
        logging.info(f"✅ Prediction pipeline completed successfully!")
        logging.info(f"   Uploaded: {uploaded_count} films with predictions")
        logging.info(f"   Summary saved to 'prediction_summaries' collection")
        
    except Exception as e:
        logging.error(f"❌ Prediction pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()