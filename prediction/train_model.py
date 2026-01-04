"""
train_model.py

Enhanced version using XGBoost with personalized correlation-based features.
Now incorporates user-specific correlation coefficients for runtime, year, and letterboxd preferences.
"""

import base64
from pymongo import MongoClient
import logging
from dotenv import load_dotenv
import pandas as pd
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from collections import defaultdict
import xgboost as xgb

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# MongoDB configuration
DB_URI = os.getenv("DB_URI")
DB_NAME = os.getenv("DB_NAME")
FILMS_COLLECTION = os.getenv("DB_FILMS_COLLECTION")
USERS_COLLECTION = os.getenv("DB_USERS_COLLECTION")
MODELS_COLLECTION = os.getenv("DB_MODELS_COLLECTION")

# ──────────────────────────────────────────────
# CONNECT TO DATABASE
# ──────────────────────────────────────────────
logging.info("Connecting to MongoDB...")
client = MongoClient(DB_URI)
db = client[DB_NAME]
films_col = db[FILMS_COLLECTION]
users_col = db[USERS_COLLECTION]

# ──────────────────────────────────────────────
# EXTRACT ENHANCED TRAINING DATA WITH CORRELATIONS
# ──────────────────────────────────────────────
def extract_enhanced_training_data(users_collection=users_col, films_collection=films_col):
    """Extract comprehensive training data with user stats, film metadata, and correlation data"""
    rating_records = []
    like_records = []
    
    # Pre-load all users data for quick access
    logging.info("Loading user statistics with correlation data...")
    users_data = {}
    for user in users_collection.find():
        username = user.get("username")
        users_data[username] = {
            'stats': user.get("stats", {}),
            'correlation_stats': user.get("stats", {}).get("correlation_stats", {})
        }
    
    # Process films with ANY interactions (reviews OR watches)
    cursor = films_collection.find(
        {"$or": [
            {"reviews": {"$exists": True, "$ne": []}}, 
            {"watches": {"$exists": True, "$ne": []}}
        ]},
        {"film_id": 1, "film_title": 1, "avg_rating": 1, "like_ratio": 1, 
         "metadata": 1, "reviews": 1, "watches": 1, "num_ratings": 1, "num_likes": 1, "num_watches": 1}
    )

    for film in cursor:
        film_id = film.get("film_id")
        film_reviews = film.get("reviews", [])
        film_watches = film.get("watches", [])
        
        # Handle films with no reviews (only watches)
        film_has_reviews = len(film_reviews) > 0
        
        # Use fallback for films with no reviews
        film_avg_rating = film.get("avg_rating")
        if film_avg_rating is None:
            film_avg_rating = film.get("metadata", {}).get("avg_rating", 3.0)  # Use Letterboxd average as fallback
        
        film_like_ratio = film.get("like_ratio", 0)
        film_num_ratings = film.get("num_ratings", 0)
        film_num_likes = film.get("num_likes", 0)
        film_num_watches = film.get("num_watches", 0)
        metadata = film.get("metadata", {})

        # Film metadata
        film_genres = metadata.get("genres", [])
        film_runtime = metadata.get("runtime", 0)
        film_year = metadata.get("year", 0)
        film_letterboxd_avg = metadata.get("avg_rating", 3.0)  # Always available from metadata

        # Process REVIEWS (have ratings and potentially likes)
        for review in film_reviews:
            user = review.get("user")
            rating = review.get("rating")
            is_liked = review.get("is_liked", False)

            if user and rating and user in users_data:
                user_data = users_data[user]
                process_interaction(
                    user, user_data, film_id, film_reviews, film_watches,
                    film_avg_rating, film_like_ratio, film_num_ratings, film_num_likes, film_num_watches,
                    film_genres, film_runtime, film_year, film_letterboxd_avg,
                    rating, is_liked, True,  # is_review=True
                    rating_records, like_records
                )

        # Process WATCHES (have likes but no ratings) - ONLY for like prediction
        for watch in film_watches:
            user = watch.get("user")
            is_liked = watch.get("is_liked", False)

            if user and user in users_data:
                user_data = users_data[user]
                process_interaction(
                    user, user_data, film_id, film_reviews, film_watches,
                    film_avg_rating, film_like_ratio, film_num_ratings, film_num_likes, film_num_watches,
                    film_genres, film_runtime, film_year, film_letterboxd_avg,
                    None, is_liked, False,  # is_review=False, no rating
                    rating_records, like_records
                )
    
    logging.info(f"Extracted {len(rating_records)} rating records and {len(like_records)} like records")
    return rating_records, like_records, users_data

def process_interaction(user, user_data, film_id, film_reviews, film_watches,
                       film_avg_rating, film_like_ratio, film_num_ratings, film_num_likes, film_num_watches,
                       film_genres, film_runtime, film_year, film_letterboxd_avg,
                       rating, is_liked, is_review, rating_records, like_records):
    """Process a single user-film interaction with correlation-based features"""
    
    user_stats = user_data['stats']
    correlation_stats = user_data['correlation_stats']
    
    # Combine all interactions for aggregate calculations
    all_film_interactions = film_reviews + film_watches
    
    # --- Compute leave-one-out film aggregates ---
    other_interactions = [r for r in all_film_interactions if r.get("user") != user]
    
    # Use sentinel values when data is insufficient
    SENTINEL_MISSING = -1.0
    
    if len(other_interactions) >= 2:
        # For ratings: only use reviews with ratings
        other_ratings = [float(r["rating"]) for r in other_interactions if r.get("rating") is not None]
        film_avg_rating_excl_user = sum(other_ratings) / len(other_ratings) if other_ratings else SENTINEL_MISSING
        
        # For likes: use all interactions (reviews + watches)
        other_likes = [r.get("is_liked", False) for r in other_interactions]
        film_like_ratio_excl_user = sum(1 for l in other_likes if l) / len(other_likes)
    else:
        film_avg_rating_excl_user = SENTINEL_MISSING
        film_like_ratio_excl_user = SENTINEL_MISSING

    # --- Compute leave-one-out user aggregates ---
    user_num_ratings_full = user_stats.get("num_ratings", 1)
    user_num_likes_full = user_stats.get("num_likes", 0)
    user_num_watches_full = user_stats.get("num_watches", 1)

    # Adjust counts excluding current interaction
    user_num_ratings_excl = max(1, user_num_ratings_full - (1 if is_review else 0))
    user_num_likes_excl = max(0, user_num_likes_full - (1 if is_liked else 0))
    user_num_watches_excl = max(1, user_num_watches_full - 1)

    if user_num_ratings_excl > 1 and is_review:
        user_avg_rating_excl_film = (
            (user_stats.get("avg_rating", 0) * user_num_ratings_full - float(rating))
            / user_num_ratings_excl
        )
    else:
        user_avg_rating_excl_film = user_stats.get("avg_rating", 3.0) if user_num_ratings_excl > 0 else SENTINEL_MISSING
    
    user_like_ratio_excl_film = user_num_likes_excl / user_num_watches_excl if user_num_watches_excl > 0 else SENTINEL_MISSING

    # --- Correlation-based features ---
    # Get user's correlation coefficients (default to 0 if not available)
    def get_corr_value(corr_dict, key):
        """Safely extract correlation value from correlation stats"""
        if not corr_dict:
            return 0.0
        data = corr_dict.get(key)
        if isinstance(data, dict):
            value = data.get('correlation')
            return value if value is not None else 0.0
        return 0.0

    runtime_rating_corr = get_corr_value(correlation_stats, 'runtime_vs_rating')
    year_rating_corr = get_corr_value(correlation_stats, 'year_vs_rating')
    letterboxd_rating_corr = get_corr_value(correlation_stats, 'letterboxd_vs_rating')
    runtime_like_corr = get_corr_value(correlation_stats, 'runtime_vs_like')
    year_like_corr = get_corr_value(correlation_stats, 'year_vs_like')
    letterboxd_like_corr = get_corr_value(correlation_stats, 'letterboxd_vs_like')
    rating_like_corr = get_corr_value(correlation_stats, 'rating_vs_like')

    # --- Base feature set with correlation-based adjustments ---
    base_features = {
        "user_id": user,
        "film_id": film_id,
        "rating": float(rating) if rating else SENTINEL_MISSING,
        "is_review": is_review,  # Track if this is a review or watch

        # User-level
        "user_avg_rating": user_avg_rating_excl_film,
        "user_stdev_rating": user_stats.get("stdev_rating", SENTINEL_MISSING),
        "user_like_ratio": user_like_ratio_excl_film,
        "user_num_likes": user_num_likes_excl,
        "user_num_watches": user_num_watches_excl,

        # Film-level
        "film_avg_rating": film_avg_rating_excl_user,
        "film_like_ratio": film_like_ratio_excl_user,
        "film_num_ratings": film_num_ratings,
        "film_num_watches": film_num_watches,
        "film_letterboxd_avg": film_letterboxd_avg,
        "film_runtime": film_runtime,
        "film_year": film_year,

        # Correlation coefficients
        "user_runtime_rating_corr": runtime_rating_corr,
        "user_year_rating_corr": year_rating_corr,
        "user_letterboxd_rating_corr": letterboxd_rating_corr,
        "user_runtime_like_corr": runtime_like_corr,
        "user_year_like_corr": year_like_corr,
        "user_letterboxd_like_corr": letterboxd_like_corr,
        "user_rating_like_corr": rating_like_corr,

        # Correlation-weighted features (personalized)
        "runtime_weighted_by_corr": film_runtime * (1 + runtime_rating_corr) if film_runtime else 0,
        "year_weighted_by_corr": film_year * (1 + year_rating_corr) if film_year else 0,
        "letterboxd_weighted_by_corr": (film_letterboxd_avg or 0) * (1 + letterboxd_rating_corr),

        # Genre compatibility
        **get_genre_compatibility_features(user_stats, film_genres),
    }

    # --- Rating model: only use reviews (with actual ratings) ---
    if is_review and rating and user_avg_rating_excl_film != SENTINEL_MISSING and film_avg_rating_excl_user != SENTINEL_MISSING:
        rating_records.append(base_features.copy())

    # --- Like model: use both reviews AND watches ---
    if is_liked is not None and user_like_ratio_excl_film != SENTINEL_MISSING:
        like_record = base_features.copy()
        like_record["is_liked"] = bool(is_liked)
        
        # Add correlation-weighted features for like prediction
        like_record["runtime_weighted_by_like_corr"] = (film_runtime or 0) * (1 + runtime_like_corr)
        like_record["year_weighted_by_like_corr"] = (film_year or 0) * (1 + year_like_corr)
        like_record["letterboxd_weighted_by_like_corr"] = (film_letterboxd_avg or 0) * (1 + letterboxd_like_corr)
        
        like_records.append(like_record)

def get_genre_compatibility_features(user_stats, film_genres):
    """Calculate genre compatibility features between user and film"""
    genre_stats = user_stats.get("genre_stats", {})
    
    features = {}
    genre_ratings = []      # For rating prediction
    genre_like_ratios = []  # For like prediction
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
    
    # Use -1 as sentinel value for missing data (outside normal range)
    SENTINEL_MISSING = -1.0
    
    # Genre compatibility metrics for RATING prediction
    if genre_ratings:
        features["max_genre_rating"] = max(genre_ratings)
        features["min_genre_rating"] = min(genre_ratings)
        features["avg_genre_rating"] = sum(genre_ratings) / len(genre_ratings)
        features["total_genre_watches"] = sum(genre_counts)
    else:
        # Use sentinel values instead of defaults
        features.update({
            "max_genre_rating": SENTINEL_MISSING,
            "min_genre_rating": SENTINEL_MISSING,
            "avg_genre_rating": SENTINEL_MISSING,
            "total_genre_watches": SENTINEL_MISSING,
        })
    
    # Genre compatibility metrics for LIKE prediction
    if genre_like_ratios:
        features["avg_genre_like_ratio"] = sum(genre_like_ratios) / len(genre_like_ratios)
    else:
        features["avg_genre_like_ratio"] = SENTINEL_MISSING
    
    return features

# ──────────────────────────────────────────────
# TRAIN XGBOOST RATING PREDICTION MODEL (WITH CORRELATIONS)
# ──────────────────────────────────────────────
def train_xgboost_rating_model(rating_records):
    """Train XGBoost model for rating prediction with correlation-based features"""
    if not rating_records:
        raise ValueError("No rating records found for training")
    
    ratings_df = pd.DataFrame(rating_records)
    
    # Define feature columns (excluding target and IDs)
    # Now includes correlation-based personalized features
    feature_columns = [
        # Core user behavior
        'user_avg_rating', 'user_stdev_rating',
        
        # Film characteristics  
        'film_avg_rating', 'film_num_ratings', 'film_letterboxd_avg',
        'film_runtime', 'film_year',
        
        # User correlation coefficients
        'user_runtime_rating_corr', 'user_year_rating_corr', 'user_letterboxd_rating_corr',
        
        # Correlation-weighted personalized features
        'runtime_weighted_by_corr', 'year_weighted_by_corr', 'letterboxd_weighted_by_corr',
        
        # Genre rating patterns
        'max_genre_rating', 'min_genre_rating', 'avg_genre_rating', 'total_genre_watches'
    ]
    
    # Prepare features and target
    X = ratings_df[feature_columns].fillna(0)
    y = ratings_df['rating']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info(f"\nRATING MODEL TRAINING DATA:")
    logging.info(f"Training samples: {len(X_train):,}")
    logging.info(f"Test samples: {len(X_test):,}")
    logging.info(f"Target mean: {y_train.mean():.3f}, std: {y_train.std():.3f}")
    logging.info(f"Target range: [{y_train.min():.1f}, {y_train.max():.1f}]")
    logging.info(f"Features: {len(feature_columns)} (including correlation-based features)")
    
    # Log correlation feature statistics
    corr_features = ['user_runtime_rating_corr', 'user_year_rating_corr', 'user_letterboxd_rating_corr']
    for feature in corr_features:
        if feature in X_train.columns:
            logging.info(f"  {feature}: mean={X_train[feature].mean():.3f}, std={X_train[feature].std():.3f}")
    
    # Train XGBoost model with parameters for better spread
    logging.info("Training XGBoost rating prediction model with correlation features...")
    
    rating_model = xgb.XGBRegressor(
        # Model architecture
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        
        # Regularization (lower for more spread)
        reg_alpha=0.1,      # L1 regularization
        reg_lambda=0.1,     # L2 regularization (LOWER for more spread)
        
        # Tree construction
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        
        # Training
        random_state=42,
        early_stopping_rounds=20,
        eval_metric='rmse'
    )
    
    # Train with validation set
    rating_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = rating_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    logging.info(f"\nRATING MODEL PERFORMANCE:")
    logging.info(f"RMSE: {rmse:.3f}")
    logging.info(f"MSE: {mse:.3f}")
    logging.info(f"Predictions - Mean: {y_pred.mean():.3f}, Std: {y_pred.std():.3f}")
    logging.info(f"Predictions range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    logging.info(f"Spread ratio: {y_pred.std() / y_train.std():.2f}")
    
    # Feature importance
    importance_scores = rating_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    logging.info(f"\nRATING FEATURE IMPORTANCE (with correlations):")
    for idx, row in feature_importance_df.iterrows():
        logging.info(f"  {row['feature']:.<35} {row['importance']:.4f}")
    
    # Analyze correlation feature importance
    corr_importance = feature_importance_df[feature_importance_df['feature'].str.contains('corr')]
    if not corr_importance.empty:
        logging.info(f"\nCorrelation feature importance summary:")
        for _, row in corr_importance.iterrows():
            logging.info(f"  {row['feature']:.<35} {row['importance']:.4f}")
    
    return {
        'model': rating_model,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance_df,
        'performance': {
            'rmse': rmse,
            'mse': mse,
            'pred_mean': y_pred.mean(),
            'pred_std': y_pred.std(),
            'train_std': y_train.std()
        }
    }

# ──────────────────────────────────────────────
# TRAIN XGBOOST LIKE PREDICTION MODEL (WITH CORRELATIONS)
# ──────────────────────────────────────────────
def train_xgboost_like_model(like_records, rating_model, users_data):
    """Train XGBoost classifier for like prediction with correlation-based features"""
    if not like_records:
        logging.warning("No like data found. Using fallback threshold-based approach.")
        return None
    
    like_df = pd.DataFrame(like_records)
    
    # Prepare features for like prediction
    features = []
    labels = []
    
    # Feature names for like prediction with correlations
    like_feature_names = [
        # User like behavior
        'user_like_ratio', 'user_rating_consistency',
        'user_rating_like_corr',  # New: How consistent is user's rating with liking?
        
        # User correlation coefficients for likes
        'user_runtime_like_corr', 'user_year_like_corr', 'user_letterboxd_like_corr',
        
        # Film like characteristics
        'film_avg_rating', 'film_like_ratio', 'film_num_ratings',
        'film_letterboxd_avg', 'film_runtime', 'film_year',
        
        # Correlation-weighted personalized features
        'runtime_weighted_by_like_corr', 'year_weighted_by_like_corr', 'letterboxd_weighted_by_like_corr',
        
        # Genre like patterns
        'avg_genre_like_ratio', 'total_genre_watches'
    ]
    
    for _, record in like_df.iterrows():
        user_id = record["user_id"]
        user_data = users_data.get(user_id, {})
        user_stats = user_data.get('stats', {})
        
        # Create feature vector for like prediction with correlation features
        feature_vector = [
            record['user_like_ratio'],                          # user_like_ratio
            user_stats.get("mean_abs_diff", 0),                 # user_rating_consistency
            record.get('user_rating_like_corr', 0),             # user_rating_like_corr
            record.get('user_runtime_like_corr', 0),            # user_runtime_like_corr
            record.get('user_year_like_corr', 0),               # user_year_like_corr
            record.get('user_letterboxd_like_corr', 0),         # user_letterboxd_like_corr
            record['film_avg_rating'],                          # film_avg_rating
            record['film_like_ratio'],                          # film_like_ratio
            record['film_num_ratings'],                         # film_num_ratings
            record['film_letterboxd_avg'],                      # film_letterboxd_avg
            record['film_runtime'],                             # film_runtime
            record['film_year'],                                # film_year
            record.get('runtime_weighted_by_like_corr', 0),     # runtime_weighted_by_like_corr
            record.get('year_weighted_by_like_corr', 0),        # year_weighted_by_like_corr
            record.get('letterboxd_weighted_by_like_corr', 0),  # letterboxd_weighted_by_like_corr
            record['avg_genre_like_ratio'],                     # avg_genre_like_ratio
            record['total_genre_watches']                       # total_genre_watches
        ]
        
        features.append(feature_vector)
        labels.append(record['is_liked'])
    
    # Prepare data
    X = np.array(features)
    y = np.array(labels)
    
    if len(np.unique(y)) < 2:
        logging.warning("Insufficient like/dislike data for training classifier")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logging.info(f"\nLIKE MODEL TRAINING DATA:")
    logging.info(f"Training samples: {len(X_train):,} (Likes: {y_train.sum():,}, Ratio: {y_train.mean():.3f})")
    logging.info(f"Test samples: {len(X_test):,} (Likes: {y_test.sum():,}, Ratio: {y_test.mean():.3f})")
    logging.info(f"Features: {len(like_feature_names)} (including correlation-based features)")
    
    # Train XGBoost classifier
    logging.info("Training XGBoost like prediction model with correlation features...")
    
    like_model = xgb.XGBClassifier(
        # Model architecture
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        
        # Regularization
        reg_alpha=0.1,
        reg_lambda=0.1,
        
        # Tree construction
        min_child_weight=5,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        
        # For classification
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        early_stopping_rounds=20,
        use_label_encoder=False
    )
    
    # Train with validation set
    like_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = like_model.predict(X_test)
    y_pred_proba = like_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"\nLIKE MODEL PERFORMANCE:")
    logging.info(f"Accuracy: {accuracy:.3f}")
    logging.info(f"Prediction distribution:")
    logging.info(f"  Like probability mean: {y_pred_proba.mean():.3f}")
    logging.info(f"  Like probability std: {y_pred_proba.std():.3f}")
    logging.info(f"  Like probability range: [{y_pred_proba.min():.3f}, {y_pred_proba.max():.3f}]")
    
    # Feature importance
    like_importance_scores = like_model.feature_importances_
    like_importance_df = pd.DataFrame({
        'feature': like_feature_names,
        'importance': like_importance_scores
    }).sort_values('importance', ascending=False)
    
    logging.info(f"\nLIKE FEATURE IMPORTANCE (with correlations):")
    for idx, row in like_importance_df.iterrows():
        logging.info(f"  {row['feature']:.<40} {row['importance']:.4f}")
    
    # Analyze correlation feature importance
    corr_importance = like_importance_df[like_importance_df['feature'].str.contains('corr')]
    if not corr_importance.empty:
        logging.info(f"\nCorrelation feature importance for like prediction:")
        for _, row in corr_importance.iterrows():
            logging.info(f"  {row['feature']:.<40} {row['importance']:.4f}")
    
    # Store feature names for future use
    like_model.feature_names_ = like_feature_names
    like_model.feature_importance_df_ = like_importance_df
    
    return like_model

# ──────────────────────────────────────────────
# TRAINING SUMMARY
# ──────────────────────────────────────────────
def print_training_summary(rating_models, like_model, rating_records, like_records, users_data):
    """Print comprehensive training summary with correlation insights"""
    
    logging.info("\n" + "="*80)
    logging.info("XGBOOST MODEL TRAINING SUMMARY (WITH CORRELATIONS)")
    logging.info("="*80)
    
    # Dataset statistics
    logging.info(f"\nDATASET STATISTICS:")
    logging.info(f"   • Rating records: {len(rating_records):,}")
    logging.info(f"   • Like records: {len(like_records):,}")
    like_count = len([r for r in like_records if r['is_liked']])
    dislike_count = len([r for r in like_records if not r['is_liked']])
    logging.info(f"   • Like/Dislike ratio: {like_count:,}/{dislike_count:,}")
    
    # Analyze correlation data availability
    total_users = len(users_data)
    users_with_corr = 0
    for user_data in users_data.values():
        corr_stats = user_data.get('correlation_stats', {})
        if any(corr.get('correlation') is not None for corr in corr_stats.values()):
            users_with_corr += 1
    
    logging.info(f"   • Users with correlation data: {users_with_corr:,}/{total_users:,} ({users_with_corr/total_users*100:.1f}%)")
    
    # Rating model insights
    if 'performance' in rating_models:
        perf = rating_models['performance']
        logging.info(f"\nRATING PREDICTION MODEL:")
        logging.info(f"   • RMSE: {perf['rmse']:.3f}")
        logging.info(f"   • Prediction spread: {perf['pred_std']:.3f} (target: {perf['train_std']:.3f})")
        logging.info(f"   • Spread ratio: {perf['pred_std'] / perf['train_std']:.2f}")
        logging.info(f"   • Features: {len(rating_models['feature_columns'])} (including {sum(1 for f in rating_models['feature_columns'] if 'corr' in f)} correlation features)")
        
        # Top features
        top_features = rating_models['feature_importance'].head(5)
        logging.info(f"   • Top 5 features:")
        for _, row in top_features.iterrows():
            logging.info(f"     - {row['feature']}: {row['importance']:.3f}")
    
    # Like model insights
    if like_model and hasattr(like_model, 'feature_importance_df_'):
        # Calculate accuracy for the summary
        like_df = pd.DataFrame(like_records)
        like_feature_names = like_model.feature_names_
        
        # Prepare features for accuracy calculation
        features = []
        labels = []
        
        for _, record in like_df.iterrows():
            user_id = record["user_id"]
            user_data = users_data.get(user_id, {})
            user_stats = user_data.get('stats', {})
            
            feature_vector = [
                record['user_like_ratio'],
                user_stats.get("mean_abs_diff", 0),
                record.get('user_rating_like_corr', 0),
                record.get('user_runtime_like_corr', 0),
                record.get('user_year_like_corr', 0),
                record.get('user_letterboxd_like_corr', 0),
                record['film_avg_rating'],
                record['film_like_ratio'],
                record['film_num_ratings'],
                record['film_letterboxd_avg'],
                record['film_runtime'],
                record['film_year'],
                record.get('runtime_weighted_by_like_corr', 0),
                record.get('year_weighted_by_like_corr', 0),
                record.get('letterboxd_weighted_by_like_corr', 0),
                record['avg_genre_like_ratio'],
                record['total_genre_watches']
            ]
            features.append(feature_vector)
            labels.append(record['is_liked'])
        
        # Calculate accuracy
        X = np.array(features)
        y = np.array(labels)
        y_pred = like_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        logging.info(f"\nLIKE PREDICTION MODEL:")
        logging.info(f"   • Accuracy: {accuracy:.3f}")
        logging.info(f"   • Features: {len(like_model.feature_names_)} (including {sum(1 for f in like_model.feature_names_ if 'corr' in f)} correlation features)")
        top_like_features = like_model.feature_importance_df_.head(5)
        logging.info(f"   • Top 5 features:")
        for _, row in top_like_features.iterrows():
            logging.info(f"     - {row['feature']}: {row['importance']:.3f}")
    
    # Key findings
    logging.info(f"\nKEY FINDINGS:")
    if 'performance' in rating_models:
        spread_ratio = rating_models['performance']['pred_std'] / rating_models['performance']['train_std']
        if spread_ratio > 0.8:
            logging.info(f"   Good prediction spread achieved ({spread_ratio:.2f})")
        else:
            logging.info(f"   Predictions still somewhat centralized ({spread_ratio:.2f})")
    
    if like_model:
        logging.info(f"   Like prediction model trained successfully with correlation features")
    else:
        logging.info(f"   Using fallback for like predictions")
    
    logging.info(f"   Correlation features integrated for personalized predictions")
    
    logging.info("="*80)

# ──────────────────────────────────────────────
# SAVE MODELS
# ──────────────────────────────────────────────
def save_xgboost_models(rating_models, like_model, models_collection):
    """Save XGBoost models to MongoDB with proper type conversion"""
    model_data = {
        "name": "predictor",
        "last_updated": datetime.utcnow(),
        "model_type": "xgboost_with_correlations",
        "feature_columns": rating_models['feature_columns'],
        "rating_performance": {
            'rmse': float(rating_models['performance']['rmse']),
            'mse': float(rating_models['performance']['mse']),
            'pred_mean': float(rating_models['performance']['pred_mean']),
            'pred_std': float(rating_models['performance']['pred_std']),
            'train_std': float(rating_models['performance']['train_std'])
        }
    }
    
    # Save rating model
    rating_model_bytes = pickle.dumps(rating_models['model'])
    model_data["rating_model_b64"] = base64.b64encode(rating_model_bytes).decode("utf-8")
    
    # Save feature importance with proper type conversion
    rating_importance_list = []
    for _, row in rating_models['feature_importance'].iterrows():
        rating_importance_list.append({
            'feature': row['feature'],
            'importance': float(row['importance'])  # Convert numpy float to Python float
        })
    model_data["rating_feature_importance"] = rating_importance_list
    
    # Save like model if available
    if like_model:
        like_model_bytes = pickle.dumps(like_model)
        model_data["like_model_b64"] = base64.b64encode(like_model_bytes).decode("utf-8")
        
        # Convert like feature importance
        like_importance_list = []
        for _, row in like_model.feature_importance_df_.iterrows():
            like_importance_list.append({
                'feature': row['feature'],
                'importance': float(row['importance'])  # Convert numpy float to Python float
            })
        model_data["like_feature_importance"] = like_importance_list
        model_data["has_like_model"] = True
        model_data["like_feature_names"] = like_model.feature_names_
    else:
        model_data["has_like_model"] = False
        model_data["like_feature_importance"] = []
        model_data["like_feature_names"] = []
        logging.info("Using fallback like prediction (threshold-based)")
    
    # Convert all numpy values in the model_data recursively
    model_data = convert_numpy_types(model_data)
    
    models_collection.replace_one(
        {"name": "predictor"},
        model_data,
        upsert=True
    )
    
    logging.info("Models with correlation features saved to MongoDB successfully")

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for MongoDB serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ──────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────
def main():
    logging.info("Starting XGBoost model training pipeline with correlation features...")
    
    try:
        # Extract data with correlation stats
        rating_records, like_records, users_data = extract_enhanced_training_data()
        logging.info(f"Extracted {len(rating_records)} rating records and {len(like_records)} like records")
        
        if not rating_records:
            logging.error("No rating records found. Exiting.")
            return
        
        # Train XGBoost rating model with correlation features
        rating_models = train_xgboost_rating_model(rating_records)
        
        # Train XGBoost like model with correlation features
        like_model = train_xgboost_like_model(like_records, rating_models, users_data)
        
        # Print comprehensive summary
        print_training_summary(rating_models, like_model, rating_records, like_records, users_data)
        
        # Save models
        models_col = db[MODELS_COLLECTION]
        save_xgboost_models(rating_models, like_model, models_col)
        
        logging.info("XGBoost model training with correlation features completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()