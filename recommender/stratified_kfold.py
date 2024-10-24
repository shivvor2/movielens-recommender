from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame
from typing import List


def stratified_kfold_split(
    df: DataFrame,
    target_column: str,
    random_state: int,
    n_splits: int = 5,
    shuffle: bool = False,
) -> List[DataFrame]:
    """
    Perform stratified k-fold split on a pandas DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        target_column (str): Name of the target column for stratification
        n_splits (int): Number of folds (default: 5)
        shuffle (bool): Whether to shuffle the data (default: False)
        random_state (int): Random seed for reproducibility (default: 0)

    Returns:
        List[pd.DataFrame]: List of k DataFrames, where k = n_splits
    """
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    # Get target values
    y = df[target_column]

    # List to store the k DataFrames
    split_dfs = []

    # Perform the split
    for _, test_idx in skf.split(df, y):
        # Create a DataFrame for this fold using the test indices
        fold_df = df.iloc[test_idx].copy()
        split_dfs.append(fold_df)

    return split_dfs


# Example usage:
# Assuming you have a DataFrame 'df' with a target column 'target'
"""
# Create sample data
df = pd.DataFrame({
    'feature1': range(100),
    'feature2': range(100),
    'target': [0, 1] * 50
})

# Get the splits
splits = stratified_kfold_split(df, 'target', n_splits=5)

# Now splits is a list of 5 DataFrames
for i, split_df in enumerate(splits):
    print(f"Split {i+1} shape:", split_df.shape)
"""
