import pandas as pd
from pandas import DataFrame, Series
from typing import Iterable
import numpy as np

import logging

logger = logging.getLogger(__name__)


def create_user_profiles(
    user_ratings: DataFrame, movies_df: DataFrame, features: Iterable
):
    """
    Create weighted feature profiles for all users in user_ratings

    Parameters:
    - user_ratings: DataFrame with columns [userId, movieId, norm_rating]
                   (assumes ratings are already normalized to [0,1])
    - movies_df: DataFrame containing movie features
    - features: List of feature column names to use

    Returns:
    Dictionary containing:
    - profiles: DataFrame with users as index, features as columns
    - counts: Series with number of ratings per user
    """
    # Merge ratings with movie features
    features = list(features)

    merged_df = user_ratings.merge(
        movies_df[["movieId"] + features], on="movieId", how="left"
    )

    # Calculate user profiles (weighted average of movie features by normalized rating)
    user_profiles = merged_df.groupby("userId").apply(
        lambda x: pd.Series(
            {feature: (x[feature] * x["norm_rating"]).mean() for feature in features}
        )
    )

    # Get rating counts per user
    user_counts = user_ratings.groupby("userId").size()

    return user_profiles, user_counts


# Getting the Average_profile:
# user_profiles.mean()


def get_recommendations(
    user_id: int,
    user_profiles: DataFrame,
    user_counts: DataFrame | Series,
    movies_df: DataFrame,
    features: Iterable,
    exclude_movies=None,
    profile_regularize: int = 3,
):
    """
    Get movie recommendations for a user with similarity scores

    Parameters:
    - user_id: ID of the user to get recommendations for
    - user_profiles: DataFrame of user profiles (from create_user_profiles)
    - user_counts: Series of rating counts per user
    - movies_df: DataFrame containing movie features
    - features: List of feature columns to use
    - exclude_movies: list of movieIds to exclude
    - profile_regularize: Threshold for profile regularization

    Returns:
    DataFrame with all movies and their similarity scores, sorted by similarity
    """
    # Get average profile for regularization/cold start
    average_profile = user_profiles.mean()

    # Get user's profile (or average if user doesn't exist)
    if user_id in user_profiles.index:
        user_profile = user_profiles.loc[user_id]

        # Apply regularization if needed
        if profile_regularize > 0:
            user_count = user_counts.loc[user_id]
            if user_count < profile_regularize:
                r = 1 / (profile_regularize - user_count)
                user_profile = r * user_profile + (1 - r) * average_profile
    else:
        user_profile = average_profile

    # Get movie features
    movie_features = movies_df[features]

    # Calculate similarities
    similarities = cosine_similarity(movie_features, user_profile.values.reshape(1, -1))

    # Create recommendations dataframe
    recommendations = DataFrame(
        {"movieId": movies_df["movieId"], "similarity_score": similarities.flatten()}
    ).sort_values("similarity_score", ascending=False)

    # Exclude watched movies if specified
    if exclude_movies is not None:
        recommendations = recommendations[
            ~recommendations["movieId"].isin(exclude_movies)
        ]

    return recommendations


def get_all_recommendations(
    user_profiles: DataFrame,
    user_counts: DataFrame,
    movies_df: DataFrame,
    ratings_train: DataFrame,
    ratings_test: DataFrame,
    features: Iterable,
) -> DataFrame:
    """
    Get recommendations for all users in test set

    Parameters:
    - user_profiles: DataFrame of user profiles
    - user_counts: DataFrame of rating counts per user
    - movies_df: DataFrame containing movie features
    - ratings_train: DataFrame of training ratings
    - ratings_test: DataFrame of test ratings
    - features: List of feature columns to use

    Returns:
    DataFrame with columns [userId, movieId, similarity_score]
    containing recommendations for all users
    """
    # Get all unique users in test set
    test_users = ratings_test["userId"].unique()

    all_recommendations = []

    for user_id in test_users:
        # Get movies user has already rated in training set (to exclude)
        user_rated = set(ratings_train[ratings_train["userId"] == user_id]["movieId"])

        # Get recommendations for this user
        user_recommendations = get_recommendations(
            user_id,
            user_profiles,
            user_counts,
            movies_df,
            features,
            exclude_movies=user_rated,
        )

        # Add user_id to recommendations
        user_recommendations["userId"] = user_id

        all_recommendations.append(user_recommendations)

    # Combine all recommendations
    return pd.concat(all_recommendations, ignore_index=True)


def cosine_similarity(features, query):
    """
    Calculate cosine similarity between a feature matrix and a query vector

    Parameters:
    - features: DataFrame/array of shape (n_samples, n_features)
    - query: Array-like of shape (1, n_features) or (n_features,)

    Returns:
    Array of shape (n_samples,) containing similarity scores
    """
    # Convert pandas objects to numpy arrays if necessary
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(query, pd.Series):
        query = query.values

    # Ensure query is 2D
    if query.ndim == 1:
        query = query.reshape(1, -1)

    # Calculate dot product
    dot_product = np.dot(features, query.T)

    # Calculate norms
    feature_norms = np.sqrt(np.sum(features**2, axis=1))
    query_norm = np.sqrt(np.sum(query**2))

    # Calculate similarity
    similarities = dot_product / (feature_norms.reshape(-1, 1) * query_norm)

    return similarities.flatten()
