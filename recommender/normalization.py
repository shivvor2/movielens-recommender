from pandas import DataFrame
import numpy as np


def normalize_user_ratings(group, user_stats):
    """
    Z-score normalize ratings for a single user group

    Parameters:
    - group: DataFrame containing ratings for one user
    - user_stats: DataFrame with ['mean', 'std'] columns indexed by userId

    Returns:
    - DataFrame with normalized ratings for the user
    """
    user_mean = user_stats.loc[group.name, "mean"]
    user_std = user_stats.loc[group.name, "std"]

    if user_std == 0:
        # If user has no variance in ratings, center around mean
        group["norm_rating"] = 0
    else:
        # Z-score normalization
        group["norm_rating"] = (group["rating"] - user_mean) / user_std
    return group


def normalize_ratings(ratings_df: DataFrame):
    """
    Normalize ratings considering user rating behavior

    Parameters:
    - ratings_df: DataFrame with columns [userId, movieId, rating] where rating is already in [0,1]

    Returns:
    - DataFrame with normalized ratings centered around user means
    """
    # Calculate user means and standard deviations
    user_stats = ratings_df.groupby("userId")["rating"].agg(["mean", "std"]).fillna(0)

    # Apply normalization
    normalized_df = ratings_df.groupby("userId").apply(
        lambda x: normalize_user_ratings(x, user_stats)
    )

    # Reset index to remove the multi-index created by groupby
    normalized_df = normalized_df.reset_index(drop=True)

    # Scale to [-1, 1] range
    max_abs = abs(normalized_df["norm_rating"]).max()
    normalized_df["norm_rating"] = normalized_df["norm_rating"] / max_abs

    return normalized_df


def normalize_movie_features(movies_df, features):
    """
    Simple L2 normalization for movie features
    """
    normalized_df = movies_df.copy()

    # Normalize each movie vector to unit length
    feature_matrix = normalized_df[features].values
    norms = np.sqrt((feature_matrix**2).sum(axis=1))
    norms[norms == 0] = 1  # Prevent division by zero

    for feature in features:
        normalized_df[feature] = normalized_df[feature] / norms

    return normalized_df
