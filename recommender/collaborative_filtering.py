from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Dict, Tuple


class CollaborativeFilter:
    def __init__(self, n_components: int = 20, n_iter: int = 10):
        self.n_components = n_components
        self.n_iter = n_iter
        self.svd = TruncatedSVD(n_components=n_components, n_iter=n_iter)

        self.user_mapper = None
        self.movie_mapper = None
        self.user_inv_mapper = None
        self.movie_inv_mapper = None
        self.X = None
        self.user_features = None  # U * Σ
        self.movie_features = None  # V

    def _create_mappers(self, df: DataFrame) -> Tuple[Dict, Dict, Dict, Dict]:
        """Create mapping dictionaries for users and movies"""
        M = df["userId"].nunique()
        N = df["movieId"].nunique()

        user_mapper = dict(zip(np.unique(df["userId"]), list(range(M))))
        movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(N))))

        user_inv_mapper = dict(zip(list(range(M)), np.unique(df["userId"])))
        movie_inv_mapper = dict(zip(list(range(N)), np.unique(df["movieId"])))

        return user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper

    def _create_sparse_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """Create sparse matrix from ratings dataframe"""
        user_index = [self.user_mapper[i] for i in df["userId"]]
        item_index = [self.movie_mapper[i] for i in df["movieId"]]

        return csr_matrix(
            (df["rating"], (user_index, item_index)),
            shape=(len(self.user_mapper), len(self.movie_mapper)),
        )

    def fit(self, ratings_train: pd.DataFrame):
        """Fit the collaborative filtering model"""
        # Create mappers
        (
            self.user_mapper,
            self.movie_mapper,
            self.user_inv_mapper,
            self.movie_inv_mapper,
        ) = self._create_mappers(ratings_train)

        # Create and decompose matrix
        self.X = self._create_sparse_matrix(ratings_train)

        # Perform SVD
        self.movie_features = self.svd.fit_transform(self.X.T)  # V

        # Calculate user features (U * Σ)
        self.user_features = self.X.dot(self.movie_features)  # U * Σ

        return self

    def predict_ratings(self, user_id: int) -> np.ndarray:
        """Predict ratings for all movies for a given user"""
        if user_id not in self.user_mapper:
            # Handle cold start: return average ratings
            return np.mean(self.X.toarray(), axis=0)
        else:
            # Get user's latent features and predict ratings
            user_idx = self.user_mapper[user_id]
            user_vector = self.user_features[user_idx]
            return user_vector.dot(self.movie_features.T)

    def get_recommendations(
        self, user_id: int, exclude_movies: set = None
    ) -> pd.DataFrame:
        """Get recommendations using predicted ratings"""
        scores = self.predict_ratings(user_id)

        # Create recommendations dataframe
        recommendations = pd.DataFrame(
            {
                "movieId": [self.movie_inv_mapper[i] for i in range(len(scores))],
                "similarity_score": scores,
            }
        )

        # Exclude watched movies if specified
        if exclude_movies is not None:
            recommendations = recommendations[
                ~recommendations["movieId"].isin(exclude_movies)
            ]

        return recommendations.sort_values("similarity_score", ascending=False)


def get_all_recommendations(
    cf_model: CollaborativeFilter, ratings_train: DataFrame, ratings_test: DataFrame
) -> DataFrame:
    """
    Get recommendations for all users in test set using collaborative filtering

    Returns:
    DataFrame with columns [userId, movieId, similarity_score]
    """
    all_recommendations = []

    for user_id in ratings_test["userId"].unique():
        # Get movies user has already rated
        user_rated = set(ratings_train[ratings_train["userId"] == user_id]["movieId"])

        # Get recommendations
        user_recs = cf_model.get_recommendations(user_id, exclude_movies=user_rated)
        user_recs["userId"] = user_id

        all_recommendations.append(user_recs)

    return pd.concat(all_recommendations, ignore_index=True)
