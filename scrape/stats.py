from collections import defaultdict
from dotenv import load_dotenv
import logging
import os
from pymongo import MongoClient
import statistics
import numpy as np
from scipy.stats import pearsonr, spearmanr

def compute_film_stats(db, films_collection_name):
    # ... (unchanged - same as original)
    films_collection = db[films_collection_name]
    films = list(films_collection.find({}))

    logging.info("Computing film statistics...")

    for film in films:
        reviews = film.get('reviews', [])
        watches = film.get('watches', [])

        num_ratings = len(reviews)
        num_watches = len(watches) + num_ratings

        # Sum ratings values
        ratings = [r['rating'] for r in reviews if 'rating' in r]
        total_rating = sum(ratings)

        # Count likes
        num_likes = sum(1 for r in reviews if r.get('is_liked'))
        num_likes += sum(1 for w in watches if w.get('is_liked'))

        # Calculate averages and ratios
        avg_rating = (total_rating / num_ratings) if num_ratings > 0 else None
        like_ratio = (num_likes / num_watches) if num_watches > 0 else None
        
        # Calculate standard deviation
        stdev_rating = statistics.stdev(ratings) if len(ratings) > 1 else None

        # Update the film in the DB
        films_collection.update_one(
            {'film_id': film['film_id']},
            {'$set': {
                'num_ratings': num_ratings,
                'avg_rating': avg_rating,
                'stdev_rating': stdev_rating,
                'num_likes': num_likes,
                'num_watches': num_watches,
                'like_ratio': like_ratio
            }}
        )

    logging.info("Film statistics updated successfully.")

def compute_user_stats(db, users_collection_name, films_collection_name):
    films_collection = db[films_collection_name]
    users_collection = db[users_collection_name]
    users = list(users_collection.find({}))

    logging.info("Preloading films from DB...")
    films = {film['film_id']: film for film in films_collection.find({})}
    logging.info(f"Loaded {len(films)} films into memory.")

    # Get genres from environment variable
    letterboxd_genres = os.getenv('LETTERBOXD_GENRES', '')
    all_genres = [genre.strip() for genre in letterboxd_genres.split(',')] if letterboxd_genres else []
    
    if not all_genres:
        logging.warning("LETTERBOXD_GENRES environment variable not found or empty, using default genres")
        all_genres = [
            "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
            "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
            "Romance", "Science Fiction", "Thriller", "War", "Western", "TV Movie"
        ]

    logging.info("Computing user statistics...")

    for user in users:
        reviews = user.get('reviews', [])
        watches = user.get('watches', [])
        
        # Combine reviews and watches for film interactions
        all_interactions = reviews + watches

        # Counts
        num_ratings = len(reviews)
        num_watches = num_ratings + len(watches)

        # Extract ratings (only from reviews)
        ratings = [r['rating'] for r in reviews if 'rating' in r]

        # Likes (from both reviews and watches)
        num_likes = sum(1 for r in reviews if r.get('is_liked'))
        num_likes += sum(1 for w in watches if w.get('is_liked'))

        # Averages (only from reviews with ratings)
        avg_rating = (sum(ratings) / num_ratings) if num_ratings > 0 else None
        like_ratio = (num_likes / num_watches) if num_watches > 0 else None

        # Rating distribution (0.5 → 5.0 in 0.5 steps) - only from reviews
        rating_distr = {str(i): ratings.count(i) for i in [x * 0.5 for x in range(1, 11)]}

        # Median, Mode, Stdev - only from reviews with ratings
        median_rating = statistics.median(ratings) if ratings else None
        try:
            mode_rating = statistics.mode(ratings) if ratings else None
        except statistics.StatisticsError:
            mode_rating = None
        stdev_rating = statistics.stdev(ratings) if len(ratings) > 1 else None

        # Pairwise diffs vs. group - only from reviews with ratings
        diffs = []
        abs_diffs = []

        # Pairwise agreement dict: {other_username: [diffs]}
        pairwise_diffs = defaultdict(list)
        pairwise_abs_diffs = defaultdict(list)

        # Genre statistics - from ALL interactions (reviews + watches)
        genre_counts = {genre: 0 for genre in all_genres}
        genre_ratings = {genre: [] for genre in all_genres}  # Only from reviews with ratings
        genre_likes = {genre: 0 for genre in all_genres}    # Likes per genre from all interactions
        genre_interactions = {genre: 0 for genre in all_genres}  # Total interactions per genre
        
        # NEW: Data collection for correlation coefficients
        runtime_rating_pairs = []      # For runtime vs rating correlation
        runtime_like_pairs = []        # For runtime vs like correlation (boolean)
        year_rating_pairs = []         # For release year vs rating correlation
        year_like_pairs = []           # For release year vs like correlation (boolean)
        letterboxd_rating_pairs = []   # For letterboxd avg vs rating correlation
        letterboxd_like_pairs = []     # For letterboxd avg vs like correlation
        rating_like_pairs = []         # For rating vs like correlation (for reviews only)
        
        total_runtime = 0
        total_years = 0
        count_with_runtime = 0
        count_with_year = 0

        # Process ALL interactions for genre, runtime, and year stats
        for interaction in all_interactions:
            film = films.get(interaction['film_id'])
            if not film:
                continue

            # Genre statistics - count for all interactions
            if 'metadata' in film:
                for genre in film['metadata'].get('genres', []):
                    if genre in genre_counts:
                        genre_counts[genre] += 1
                        genre_interactions[genre] += 1
                        
                        # Track likes for this genre
                        if interaction.get('is_liked'):
                            genre_likes[genre] += 1
                            
                        # Only add rating if this is a review with a rating
                        if 'rating' in interaction:
                            genre_ratings[genre].append(interaction['rating'])
                
                # Runtime and year - for all interactions
                year = film['metadata'].get('year')
                runtime = film['metadata'].get('runtime')
                letterboxd_avg = film['metadata'].get('avg_rating')
                
                # Collect data for correlation analysis
                if 'rating' in interaction:
                    rating = interaction['rating']
                    
                    # Runtime vs rating
                    if runtime:
                        runtime_rating_pairs.append((runtime, rating))
                    
                    # Year vs rating
                    if year:
                        year_rating_pairs.append((year, rating))
                    
                    # Letterboxd avg vs rating
                    if letterboxd_avg:
                        letterboxd_rating_pairs.append((letterboxd_avg, rating))
                    
                    # Rating vs like (for reviews with likes)
                    if 'is_liked' in interaction:
                        rating_like_pairs.append((rating, 1.0 if interaction['is_liked'] else 0.0))
                
                # Like correlations (for all interactions)
                if 'is_liked' in interaction:
                    like_val = 1.0 if interaction['is_liked'] else 0.0
                    
                    # Runtime vs like
                    if runtime:
                        runtime_like_pairs.append((runtime, like_val))
                    
                    # Year vs like
                    if year:
                        year_like_pairs.append((year, like_val))
                    
                    # Letterboxd avg vs like
                    if letterboxd_avg:
                        letterboxd_like_pairs.append((letterboxd_avg, like_val))
                
                # Track averages for overall stats
                if year:
                    total_years += year
                    count_with_year += 1
                if runtime:
                    total_runtime += runtime
                    count_with_runtime += 1

        # NEW: Calculate correlation coefficients
        correlation_stats = {}
        
        # Helper function to calculate correlation with proper handling
        def calculate_correlation(pairs, method='pearson'):
            """Calculate correlation coefficient for paired data"""
            if len(pairs) < 2:
                return {'correlation': None, 'p_value': None, 'sample_size': 0}
            
            x_vals, y_vals = zip(*pairs)
            
            try:
                if method == 'pearson':
                    corr, p_value = pearsonr(x_vals, y_vals)
                elif method == 'spearman':
                    corr, p_value = spearmanr(x_vals, y_vals)
                else:
                    return {'correlation': None, 'p_value': None, 'sample_size': len(pairs)}
                
                # Round to reasonable precision
                return {
                    'correlation': round(float(corr), 4),
                    'p_value': round(float(p_value), 4),
                    'sample_size': len(pairs),
                    'interpretation': interpret_correlation(corr)
                }
            except Exception as e:
                logging.debug(f"Correlation calculation error: {e}")
                return {'correlation': None, 'p_value': None, 'sample_size': len(pairs)}
        
        # 1. Runtime vs Rating correlation
        correlation_stats['runtime_vs_rating'] = calculate_correlation(runtime_rating_pairs, 'pearson')
        
        # 2. Runtime vs Like correlation (point-biserial correlation)
        correlation_stats['runtime_vs_like'] = calculate_correlation(runtime_like_pairs, 'pearson')
        
        # 3. Year vs Rating correlation
        correlation_stats['year_vs_rating'] = calculate_correlation(year_rating_pairs, 'pearson')
        
        # 4. Year vs Like correlation
        correlation_stats['year_vs_like'] = calculate_correlation(year_like_pairs, 'pearson')
        
        # 5. Letterboxd Avg vs Rating correlation
        correlation_stats['letterboxd_vs_rating'] = calculate_correlation(letterboxd_rating_pairs, 'pearson')
        
        # 6. Letterboxd Avg vs Like correlation
        correlation_stats['letterboxd_vs_like'] = calculate_correlation(letterboxd_like_pairs, 'pearson')
        
        # 7. Rating vs Like correlation (for reviews with both)
        correlation_stats['rating_vs_like'] = calculate_correlation(rating_like_pairs, 'pearson')
        
        # 8. Additional meaningful correlations
        
        # Calculate average rating per genre for user
        genre_avg_ratings = []
        genre_avg_likes = []
        for genre in all_genres:
            if genre_ratings[genre]:
                avg_genre_rating = statistics.mean(genre_ratings[genre])
                genre_avg_ratings.append(avg_genre_rating)
                
                # Genre like ratio
                genre_like_ratio = genre_likes[genre] / genre_interactions[genre] if genre_interactions[genre] > 0 else 0
                genre_avg_likes.append(genre_like_ratio)
        
        # 9. Average rating vs genre like ratio correlation (across genres)
        if len(genre_avg_ratings) >= 2 and len(genre_avg_likes) >= 2:
            genre_corr_pairs = list(zip(genre_avg_ratings, genre_avg_likes))
            correlation_stats['genre_rating_vs_like'] = calculate_correlation(genre_corr_pairs, 'pearson')
        else:
            correlation_stats['genre_rating_vs_like'] = {'correlation': None, 'p_value': None, 'sample_size': 0}
        
        # 10. Time-based correlations if we had watch dates
        # (Could be added if you track when users watched films)
        
        # Calculate genre percentages and average ratings
        genre_stats = {}
        for genre in all_genres:
            count = genre_counts[genre]
            percentage = (count / num_watches * 100) if num_watches > 0 else 0
            avg_genre_rating = statistics.mean(genre_ratings[genre]) if genre_ratings[genre] else None
            genre_stddev = statistics.stdev(genre_ratings[genre]) if len(genre_ratings[genre]) > 1 else None
            
            # Calculate genre like statistics
            genre_num_likes = genre_likes[genre]
            genre_like_ratio = (genre_num_likes / genre_interactions[genre]) if genre_interactions[genre] > 0 else None
            
            genre_stats[genre] = {
                'count': count,
                'percentage': percentage,
                'avg_rating': avg_genre_rating,
                'stddev': genre_stddev,
                'num_likes': genre_num_likes,
                'like_ratio': genre_like_ratio
            }

        mean_diff = (sum(diffs) / len(diffs)) if diffs else None
        mean_abs_diff = (sum(abs_diffs) / len(abs_diffs)) if abs_diffs else None

        # Summarize pairwise agreements
        agreement_stats = {}
        for other, difflist in pairwise_abs_diffs.items():
            mean_diff_with_other = (sum(pairwise_diffs[other]) / len(pairwise_diffs[other])) if pairwise_diffs[other] else None
            mean_abs_diff_with_other = (sum(difflist) / len(difflist)) if difflist else None
            agreement_stats[other] = {
                'mean_diff': mean_diff_with_other,
                'mean_abs_diff': mean_abs_diff_with_other,
                'num_shared': len(difflist)  # how many films both rated
            }

        # Calculate average movie length and year
        avg_runtime = (total_runtime / count_with_runtime) if count_with_runtime > 0 else None
        avg_year_watched = (total_years / count_with_year) if count_with_year > 0 else None

        # Update user stats in DB with NEW correlation stats
        users_collection.update_one(
            {'username': user['username']},
            {'$set': {
                'stats': {
                    'num_watches': num_watches,
                    'num_ratings': num_ratings,
                    'avg_rating': avg_rating,
                    'median_rating': median_rating,
                    'mode_rating': mode_rating,
                    'stdev_rating': stdev_rating,
                    'mean_diff': mean_diff,
                    'mean_abs_diff': mean_abs_diff,
                    'pairwise_agreement': agreement_stats,
                    'rating_distribution': rating_distr,
                    'num_likes': num_likes,
                    'like_ratio': like_ratio,
                    'genre_stats': genre_stats,
                    'avg_runtime': avg_runtime,
                    'total_runtime': total_runtime if total_runtime > 0 else None,
                    'avg_year_watched': avg_year_watched,
                    'correlation_stats': correlation_stats  # NEW: Added correlation statistics
                }
            }}
        )
    
    logging.info("User statistics updated successfully with correlation coefficients.")

def interpret_correlation(corr_value):
    """Interpret the correlation coefficient value"""
    if corr_value is None:
        return "Insufficient data"
    
    abs_corr = abs(corr_value)
    if abs_corr >= 0.7:
        return "Strong"
    elif abs_corr >= 0.5:
        return "Moderate"
    elif abs_corr >= 0.3:
        return "Weak"
    elif abs_corr >= 0.1:
        return "Very weak"
    else:
        return "Negligible"

def compute_superlatives(db, users_collection_name, films_collection_name, superlatives_collection_name):
    # ... (unchanged - same as original, but with added superlatives for correlations)
    users_collection = db[users_collection_name]
    films_collection = db[films_collection_name]
    superlatives_collection = db[superlatives_collection_name]
    
    logging.info("Computing superlatives...")
    
    # Clear existing superlatives
    superlatives_collection.delete_many({})
    
    # Get all users with stats
    users = list(users_collection.find({"stats": {"$exists": True}}))
    films = list(films_collection.find({"num_ratings": {"$gte": 3}}))  # Only films with at least 3 ratings
    
    # Initialize categories - ADDED correlation superlatives
    categories = {
        "User Superlatives": [],
        "Film Superlatives": [], 
        "Genre Superlatives": [],
        "Genre Preference Superlatives": [],
        "Correlation Superlatives": []  # NEW: For correlation-based superlatives
    }
    
    # ... (all existing superlative calculations remain the same)
    # User Superlatives
    # 1. Positive Polly (highest average rating)
    positive_users = sorted([u for u in users if u['stats'].get('avg_rating') is not None], 
                           key=lambda x: x['stats']['avg_rating'], reverse=True)
    categories["User Superlatives"].append({
        "name": "Positive Polly",
        "description": "User with the highest average rating",
        "first": [positive_users[0]['username']] if positive_users else [],
        "first_value": positive_users[0]['stats']['avg_rating'] if positive_users else None,
        "second": [positive_users[1]['username']] if len(positive_users) > 1 else [],
        "second_value": positive_users[1]['stats']['avg_rating'] if len(positive_users) > 1 else None,
        "third": [positive_users[2]['username']] if len(positive_users) > 2 else [],
        "third_value": positive_users[2]['stats']['avg_rating'] if len(positive_users) > 2 else None
    })
    
    # ... (continue with all existing superlatives)
    
    # NEW: Correlation Superlatives
    logging.info("Computing correlation superlatives...")
    
    # Helper to get correlation value safely
    def get_corr_value(user, corr_key):
        stats = user.get('stats', {})
        corr_stats = stats.get('correlation_stats', {}).get(corr_key, {})
        return corr_stats.get('correlation')
    
    # 1. Runtime-Rating Enthusiast (strongest positive correlation)
    runtime_rating_users = []
    for user in users:
        corr = get_corr_value(user, 'runtime_vs_rating')
        if corr is not None:
            runtime_rating_users.append({
                'username': user['username'],
                'correlation': corr,
                'sample_size': user['stats'].get('correlation_stats', {}).get('runtime_vs_rating', {}).get('sample_size', 0)
            })
    
    runtime_rating_users = sorted(runtime_rating_users, key=lambda x: x['correlation'], reverse=True)
    categories["Correlation Superlatives"].append({
        "name": "Epic Film Lover",
        "description": "User with the strongest positive correlation between runtime and rating (likes longer films)",
        "first": [runtime_rating_users[0]['username']] if runtime_rating_users else [],
        "first_value": runtime_rating_users[0]['correlation'] if runtime_rating_users else None,
        "second": [runtime_rating_users[1]['username']] if len(runtime_rating_users) > 1 else [],
        "second_value": runtime_rating_users[1]['correlation'] if len(runtime_rating_users) > 1 else None,
        "third": [runtime_rating_users[2]['username']] if len(runtime_rating_users) > 2 else [],
        "third_value": runtime_rating_users[2]['correlation'] if len(runtime_rating_users) > 2 else None
    })
    
    # 2. Short Film Lover (strongest negative correlation)
    runtime_rating_negative = sorted(runtime_rating_users, key=lambda x: x['correlation'])
    categories["Correlation Superlatives"].append({
        "name": "Short Film Lover",
        "description": "User with the strongest negative correlation between runtime and rating (prefers shorter films)",
        "first": [runtime_rating_negative[0]['username']] if runtime_rating_negative else [],
        "first_value": runtime_rating_negative[0]['correlation'] if runtime_rating_negative else None,
        "second": [runtime_rating_negative[1]['username']] if len(runtime_rating_negative) > 1 else [],
        "second_value": runtime_rating_negative[1]['correlation'] if len(runtime_rating_negative) > 1 else None,
        "third": [runtime_rating_negative[2]['username']] if len(runtime_rating_negative) > 2 else [],
        "third_value": runtime_rating_negative[2]['correlation'] if len(runtime_rating_negative) > 2 else None
    })
    
    # 3. Classic Film Lover (strongest positive year-rating correlation)
    year_rating_users = []
    for user in users:
        corr = get_corr_value(user, 'year_vs_rating')
        if corr is not None:
            year_rating_users.append({
                'username': user['username'],
                'correlation': corr,
                'sample_size': user['stats'].get('correlation_stats', {}).get('year_vs_rating', {}).get('sample_size', 0)
            })
    
    year_rating_users = sorted(year_rating_users, key=lambda x: x['correlation'])
    categories["Correlation Superlatives"].append({
        "name": "Classic Film Lover",
        "description": "User with the strongest negative correlation between release year and rating (prefers older films)",
        "first": [year_rating_users[0]['username']] if year_rating_users else [],
        "first_value": year_rating_users[0]['correlation'] if year_rating_users else None,
        "second": [year_rating_users[1]['username']] if len(year_rating_users) > 1 else [],
        "second_value": year_rating_users[1]['correlation'] if len(year_rating_users) > 1 else None,
        "third": [year_rating_users[2]['username']] if len(year_rating_users) > 2 else [],
        "third_value": year_rating_users[2]['correlation'] if len(year_rating_users) > 2 else None
    })
    
    # 4. Modernist (strongest positive year-rating correlation)
    year_rating_positive = sorted(year_rating_users, key=lambda x: x['correlation'], reverse=True)
    categories["Correlation Superlatives"].append({
        "name": "Modern Film Lover",
        "description": "User with the strongest positive correlation between release year and rating (prefers newer films)",
        "first": [year_rating_positive[0]['username']] if year_rating_positive else [],
        "first_value": year_rating_positive[0]['correlation'] if year_rating_positive else None,
        "second": [year_rating_positive[1]['username']] if len(year_rating_positive) > 1 else [],
        "second_value": year_rating_positive[1]['correlation'] if len(year_rating_positive) > 1 else None,
        "third": [year_rating_positive[2]['username']] if len(year_rating_positive) > 2 else [],
        "third_value": year_rating_positive[2]['correlation'] if len(year_rating_positive) > 2 else None
    })
    
    # 5. Letterboxd Agreer (strongest positive letterboxd-rating correlation)
    letterboxd_users = []
    for user in users:
        corr = get_corr_value(user, 'letterboxd_vs_rating')
        if corr is not None:
            letterboxd_users.append({
                'username': user['username'],
                'correlation': corr,
                'sample_size': user['stats'].get('correlation_stats', {}).get('letterboxd_vs_rating', {}).get('sample_size', 0)
            })
    
    letterboxd_users = sorted(letterboxd_users, key=lambda x: x['correlation'], reverse=True)
    categories["Correlation Superlatives"].append({
        "name": "Mainstream Tastes",
        "description": "User with the strongest positive correlation between Letterboxd average and their rating (agrees with consensus)",
        "first": [letterboxd_users[0]['username']] if letterboxd_users else [],
        "first_value": letterboxd_users[0]['correlation'] if letterboxd_users else None,
        "second": [letterboxd_users[1]['username']] if len(letterboxd_users) > 1 else [],
        "second_value": letterboxd_users[1]['correlation'] if len(letterboxd_users) > 1 else None,
        "third": [letterboxd_users[2]['username']] if len(letterboxd_users) > 2 else [],
        "third_value": letterboxd_users[2]['correlation'] if len(letterboxd_users) > 2 else None
    })
    
    # 6. Contrarian (strongest negative letterboxd-rating correlation)
    letterboxd_negative = sorted(letterboxd_users, key=lambda x: x['correlation'])
    categories["Correlation Superlatives"].append({
        "name": "Contrarian Tastes",
        "description": "User with the strongest negative correlation between Letterboxd average and their rating (disagrees with consensus)",
        "first": [letterboxd_negative[0]['username']] if letterboxd_negative else [],
        "first_value": letterboxd_negative[0]['correlation'] if letterboxd_negative else None,
        "second": [letterboxd_negative[1]['username']] if len(letterboxd_negative) > 1 else [],
        "second_value": letterboxd_negative[1]['correlation'] if len(letterboxd_negative) > 1 else None,
        "third": [letterboxd_negative[2]['username']] if len(letterboxd_negative) > 2 else [],
        "third_value": letterboxd_negative[2]['correlation'] if len(letterboxd_negative) > 2 else None
    })
    
    # 7. Consistent Liker (strongest rating-like correlation)
    rating_like_users = []
    for user in users:
        corr = get_corr_value(user, 'rating_vs_like')
        if corr is not None:
            rating_like_users.append({
                'username': user['username'],
                'correlation': corr,
                'sample_size': user['stats'].get('correlation_stats', {}).get('rating_vs_like', {}).get('sample_size', 0)
            })
    
    rating_like_users = sorted(rating_like_users, key=lambda x: x['correlation'], reverse=True)
    categories["Correlation Superlatives"].append({
        "name": "Predictable Liker",
        "description": "User with the strongest positive correlation between rating and liking (always likes what they rate highly)",
        "first": [rating_like_users[0]['username']] if rating_like_users else [],
        "first_value": rating_like_users[0]['correlation'] if rating_like_users else None,
        "second": [rating_like_users[1]['username']] if len(rating_like_users) > 1 else [],
        "second_value": rating_like_users[1]['correlation'] if len(rating_like_users) > 1 else None,
        "third": [rating_like_users[2]['username']] if len(rating_like_users) > 2 else [],
        "third_value": rating_like_users[2]['correlation'] if len(rating_like_users) > 2 else None
    })
    
    # 8. Unpredictable Liker (weakest rating-like correlation)
    rating_like_weak = sorted(rating_like_users, key=lambda x: abs(x['correlation']))
    categories["Correlation Superlatives"].append({
        "name": "Unpredictable Liker",
        "description": "User with the weakest correlation between rating and liking (liking doesn't always match rating)",
        "first": [rating_like_weak[0]['username']] if rating_like_weak else [],
        "first_value": rating_like_weak[0]['correlation'] if rating_like_weak else None,
        "second": [rating_like_weak[1]['username']] if len(rating_like_weak) > 1 else [],
        "second_value": rating_like_weak[1]['correlation'] if len(rating_like_weak) > 1 else None,
        "third": [rating_like_weak[2]['username']] if len(rating_like_weak) > 2 else [],
        "third_value": rating_like_weak[2]['correlation'] if len(rating_like_weak) > 2 else None
    })
    
    # Handle ties for all superlatives (including new correlation ones)
    for category_name, category_superlatives in categories.items():
        for superlative in category_superlatives:
            handle_ties(superlative, users, films)
    
    # Insert all categories into the database
    for category_name, category_superlatives in categories.items():
        if category_superlatives:  # Only insert categories that have superlatives
            superlatives_collection.insert_one({
                "category": category_name,
                "superlatives": category_superlatives
            })
    
    total_superlatives = sum(len(superlatives) for superlatives in categories.values())
    logging.info(f"Computed {total_superlatives} superlatives across {len(categories)} categories and saved to database.")

def handle_ties(superlative, users, films):
    """Handle ties for all positions in a superlative"""
    if not superlative['first'] or superlative['first_value'] is None:
        return
    
    # Determine if this is a user or film superlative
    is_user_superlative = 'username' in (superlative['first'][0] if superlative['first'] else '')
    
    # Handle first place ties
    first_value = superlative['first_value']
    all_first = find_all_with_value(superlative['name'], first_value, users, films, is_user_superlative)
    
    if len(all_first) > 1:
        superlative['first'] = all_first
        # If exactly 2 films tied for first: clear second place but keep third
        if len(all_first) == 2:
            superlative['second'] = []
            superlative['second_value'] = None
        # If 3+ films tied for first: clear both second and third places
        elif len(all_first) > 2:
            superlative['second'] = []
            superlative['second_value'] = None
            superlative['third'] = []
            superlative['third_value'] = None
    
    # Handle second place ties (only if first place has a single winner)
    if (superlative['second'] and superlative['second_value'] is not None and 
        len(superlative['first']) == 1):
        second_value = superlative['second_value']
        all_second = find_all_with_value(superlative['name'], second_value, users, films, is_user_superlative)
        # Remove any that are already in first place
        all_second = [c for c in all_second if c not in superlative['first']]
        
        if len(all_second) > 1:
            superlative['second'] = all_second
            # Clear third place since second place is now a tie
            superlative['third'] = []
            superlative['third_value'] = None
    
    # Handle third place ties (allowed if: first place single winner + second place single winner, OR first place 2-way tie)
    if (superlative['third'] and superlative['third_value'] is not None and 
        (len(superlative['first']) == 1 and len(superlative['second']) == 1) or  # single winners for 1st and 2nd
        (len(superlative['first']) == 2 and not superlative['second'])):  # 2-way tie for 1st, no 2nd place
        third_value = superlative['third_value']
        all_third = find_all_with_value(superlative['name'], third_value, users, films, is_user_superlative)
        # Remove any that are already in first or second place
        all_third = [c for c in all_third if c not in superlative['first'] and c not in superlative['second']]
        
        if len(all_third) > 1:
            superlative['third'] = all_third

def find_all_with_value(superlative_name, value, users, films, is_user_superlative):
    """Find all users or films with the given value for the superlative"""
    if is_user_superlative:
        return [user['username'] for user in users if get_user_value(user, superlative_name) == value]
    else:
        return [film['film_title'] for film in films if get_film_value(film, superlative_name) == value]

def find_next_unique_value(superlative_name, current_value, users, films, is_user_superlative, reverse=False):
    """Find the next unique value after the current value"""
    if is_user_superlative:
        all_values = sorted(set([get_user_value(user, superlative_name) for user in users if get_user_value(user, superlative_name) is not None]), reverse=reverse)
    else:
        all_values = sorted(set([get_film_value(film, superlative_name) for film in films if get_film_value(film, superlative_name) is not None]), reverse=reverse)
    
    try:
        current_index = all_values.index(current_value)
        if current_index + 1 < len(all_values):
            return all_values[current_index + 1]
    except ValueError:
        pass
    
    return None

def is_high_value_better(superlative_name):
    """Determine if higher values are better for this superlative"""
    high_value_better = [
        "Positive Polly", "Positive Polly (Comparative)", "Best Attention Span", 
        "Modernist", "Best Movie", "Most Underrated Movie", "Most Polarizing Movie",
        "Critic", "Film Junkie"
    ]
    return superlative_name in high_value_better

def get_user_value(user, superlative_name):
    """Helper function to get the appropriate value for a user based on superlative name"""
    stats = user.get('stats', {})
    if superlative_name == "Positive Polly":
        return stats.get('avg_rating')
    elif superlative_name == "Positive Polly (Comparative)":
        return stats.get('mean_diff')
    elif superlative_name == "Negative Nelly":
        return stats.get('avg_rating')
    elif superlative_name == "Negative Nelly (Comparative)":
        return stats.get('mean_diff')
    elif superlative_name == "Best Attention Span":
        return stats.get('avg_runtime')
    elif superlative_name == "TikTok Brain":
        return stats.get('avg_runtime')
    elif superlative_name == "Unc":
        return stats.get('avg_year_watched')
    elif superlative_name == "Modernist":
        return stats.get('avg_year_watched')
    elif superlative_name == "Critic":
        return stats.get('num_ratings')
    elif superlative_name == "Film Junkie":
        return stats.get('num_watches')
    elif superlative_name == "BFFs" or superlative_name == "Enemies":
        # These are handled separately in the pairs logic
        return None
    return None

def get_film_value(film, superlative_name):
    """Helper function to get the appropriate value for a film based on superlative name"""
    if superlative_name == "Best Movie":
        return film.get('avg_rating')
    elif superlative_name == "Worst Movie":
        return film.get('avg_rating')
    elif superlative_name == "Most Polarizing Movie":
        return film.get('stdev_rating')
    elif superlative_name == "Most Underrated Movie" or superlative_name == "Most Overrated Movie":
        # These are handled separately with diff calculation
        if film.get('avg_rating') is not None and film.get('metadata') is not None and film['metadata'].get('avg_rating') is not None:
            return film['avg_rating'] - film['metadata']['avg_rating']
    return None

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load environment variables
    load_dotenv()

    # MongoDB configuration
    logging.info("Connecting to MongoDB...")
    mongo_uri = os.getenv('DB_URI')
    db_name = os.getenv('DB_NAME')
    users_collection_name = os.getenv('DB_USERS_COLLECTION')
    films_collection_name = os.getenv('DB_FILMS_COLLECTION')
    superlatives_collection_name = os.getenv('DB_SUPERLATIVES_COLLECTION')
    client = MongoClient(mongo_uri)
    db = client[db_name]
    logging.info("Connected to MongoDB")

    # Compute statistics
    compute_film_stats(db, films_collection_name)
    compute_user_stats(db, users_collection_name, films_collection_name)
    compute_superlatives(db, users_collection_name, films_collection_name, superlatives_collection_name)

if __name__ == "__main__":
    main()