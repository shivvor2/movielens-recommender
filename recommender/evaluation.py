from typing import Iterable
from pandas import DataFrame, Series
import numpy as np
from .collaborative_filtering import CollaborativeFilter


def evaluation(
    all_recommendations: DataFrame, ratings_test: DataFrame, k_values: Iterable
) -> DataFrame:
    """
    Evaluate recommendations across multiple k values

    Parameters:
    - all_recommendations: DataFrame with columns [userId, movieId, similarity_score]
    - ratings_test: DataFrame of test ratings
    - k_values: List of k values to evaluate

    Returns:
    DataFrame with evaluation metrics for each k value
    """
    # Initialize confusion matrices for each k
    confusion_matrices = {k: {"TP": 0, "FP": 0, "FN": 0} for k in k_values}

    # Get all unique users in test set
    test_users = ratings_test["userId"].unique()

    for user_id in test_users:
        # Get user's test ratings
        user_test = ratings_test[ratings_test["userId"] == user_id]

        # Get relevant movies from test set (ground truth positives)
        actual_relevant = set(user_test[user_test["relevant"] == 1]["movieId"])

        # Get user's recommendations
        user_recs = all_recommendations[all_recommendations["userId"] == user_id]

        # Evaluate for each k
        for k in k_values:
            # Get top-k recommendations
            top_k = set(user_recs.nlargest(k, "similarity_score")["movieId"])

            # Update confusion matrix
            confusion_matrices[k]["TP"] += len(top_k & actual_relevant)
            confusion_matrices[k]["FP"] += len(top_k - actual_relevant)
            confusion_matrices[k]["FN"] += len(actual_relevant - top_k)

    # Calculate metrics for each k
    results = []
    for k in k_values:
        cm = confusion_matrices[k]
        precision = cm["TP"] / (cm["TP"] + cm["FP"]) if (cm["TP"] + cm["FP"]) > 0 else 0
        recall = cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results.append(
            {
                "k": k,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": cm["TP"],
                "false_positives": cm["FP"],
                "false_negatives": cm["FN"],
            }
        )

    return DataFrame(results)


def count_recommendations(recommendations_df: DataFrame, k: int = 5) -> Series:
    """
    Count how many times each movie appears in top-k recommendations

    Parameters:
    - recommendations_df: DataFrame with columns [userId, movieId, similarity_score]
    - k: number of top recommendations to consider

    Returns:
    - Series with movieId as index and recommendation count as values
    """
    # Get top k recommendations for each user
    top_k_recs = (
        recommendations_df.groupby("userId")
        .apply(lambda x: x.nlargest(k, "similarity_score"))
        .reset_index(drop=True)
    )

    # Count occurrences
    rec_counts = top_k_recs["movieId"].value_counts()

    return rec_counts


def evaluate_rating_predictions(
    cf_model: CollaborativeFilter,
    ratings_test: DataFrame,
    metrics: list = ["rmse", "mae"],
) -> dict:
    """
    Evaluate rating predictions using specified metrics

    Parameters:
    - cf_model: Fitted CollaborativeFilter model
    - ratings_test: Test ratings DataFrame
    - metrics: List of metrics to compute ('rmse', 'mae')

    Returns:
    Dictionary with computed metrics
    """
    actual_ratings = []
    predicted_ratings = []

    for user_id in ratings_test["userId"].unique():
        # Get actual ratings for this user
        user_ratings = ratings_test[ratings_test["userId"] == user_id]

        # Get predicted ratings
        predictions = cf_model.predict_ratings(user_id)

        for _, row in user_ratings.iterrows():
            movie_idx = cf_model.movie_mapper.get(row["movieId"])
            if movie_idx is not None:  # ensure movie exists in training data
                actual_ratings.append(row["rating"])
                predicted_ratings.append(predictions[movie_idx])

    results = {}

    if "rmse" in metrics:
        results["rmse"] = np.sqrt(
            np.mean((np.array(actual_ratings) - np.array(predicted_ratings)) ** 2)
        ) / len(ratings_test)
    if "mae" in metrics:
        results["mae"] = np.mean(
            np.abs(np.array(actual_ratings) - np.array(predicted_ratings))
        ) / len(ratings_test)

    return DataFrame([results])
