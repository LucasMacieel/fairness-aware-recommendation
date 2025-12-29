import pandas as pd
import os


def load_data(filepath):
    """
    Loads MovieLens 1M data from the specified filepath.
    Uses :: as separator for ML-1M (ratings.dat).
    """
    column_names = ["user_id", "item_id", "rating", "timestamp"]
    sep = "::"
    df = pd.read_csv(filepath, sep=sep, names=column_names, engine="python")
    return df


def split_train_test_stratified(df, test_ratio=0.2, seed=42):
    """
    Splits the dataframe into train and test sets, stratified by user.

    Note: Uses pandas random_state which is independent of numpy.random.seed().
    The seed=42 ensures reproducibility within this function regardless of
    external random state. For full reproducibility, ensure this function
    is called before any randomization that might affect pandas internals.
    """
    # Group by user and sample
    # Using specific column selection or include_groups=False avoids FutureWarning
    # We capture indices to ensure we get full rows back
    test_indices = (
        df.groupby("user_id", group_keys=False)
        .apply(
            lambda x: x.sample(frac=test_ratio, random_state=seed), include_groups=False
        )
        .index
    )

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    return train_df, test_df


def split_train_val_test_stratified(df, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    Splits the dataframe into train, validation, and test sets, stratified by user.

    Uses random ordering (not temporal) as recommended for general recommendation tasks.
    Each user will have interactions in all three sets proportionally.

    Args:
        df: DataFrame with user_id, item_id, rating columns
        val_ratio: Fraction of data for validation (default: 0.2)
        test_ratio: Fraction of data for test (default: 0.2)
        seed: Random seed for reproducibility

    Returns:
        train_df, val_df, test_df: Three DataFrames with stratified splits

    Note: Uses pandas random_state which is independent of numpy.random.seed().
    Default split: 60% train, 20% validation, 20% test
    """
    # First, separate out the test set
    test_indices = (
        df.groupby("user_id", group_keys=False)
        .apply(
            lambda x: x.sample(frac=test_ratio, random_state=seed), include_groups=False
        )
        .index
    )

    test_df = df.loc[test_indices]
    remaining_df = df.drop(test_indices)

    # Now split remaining data into train and validation
    # Adjust val_ratio to account for already removed test data
    # If original was 60/20/20, remaining is 80%, and we want 20% of original = 25% of remaining
    adjusted_val_ratio = val_ratio / (1 - test_ratio)

    val_indices = (
        remaining_df.groupby("user_id", group_keys=False)
        .apply(
            lambda x: x.sample(frac=adjusted_val_ratio, random_state=seed + 1),
            include_groups=False,
        )
        .index
    )

    val_df = remaining_df.loc[val_indices]
    train_df = remaining_df.drop(val_indices)

    print(
        f"Dataset Split: {len(train_df)} train ({len(train_df) / len(df) * 100:.1f}%), "
        f"{len(val_df)} val ({len(val_df) / len(df) * 100:.1f}%), "
        f"{len(test_df)} test ({len(test_df) / len(df) * 100:.1f}%)"
    )

    return train_df, val_df, test_df


def create_aligned_matrices(train_df, test_df):
    """
    Creates aligned user-item matrices for train and test sets.
    Ensures both matrices have the same shapes (Users x Items),
    filling missing entries with 0.
    """
    # Get all unique users and items from both sets
    unique_users = sorted(
        set(train_df["user_id"].unique()) | set(test_df["user_id"].unique())
    )
    unique_items = sorted(
        set(train_df["item_id"].unique()) | set(test_df["item_id"].unique())
    )

    # Use pivot_table to handle duplicates if any (though we shouldn't have them)
    # Reindex to ensure alignment
    train_matrix = (
        train_df.pivot(index="user_id", columns="item_id", values="rating")
        .reindex(index=unique_users, columns=unique_items, fill_value=0)
        .fillna(0)
    )

    test_matrix = (
        test_df.pivot(index="user_id", columns="item_id", values="rating")
        .reindex(index=unique_users, columns=unique_items, fill_value=0)
        .fillna(0)
    )

    return (
        train_matrix.values,
        test_matrix.values,
        unique_users,
        unique_items,
    )


def create_aligned_matrices_3way(train_df, val_df, test_df):
    """
    Creates aligned user-item matrices for train, validation, and test sets.
    Ensures all three matrices have the same shapes (Users x Items),
    filling missing entries with 0.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame

    Returns:
        train_matrix, val_matrix, test_matrix, user_ids, item_ids
    """
    # Get all unique users and items from all three sets
    unique_users = sorted(
        set(train_df["user_id"].unique())
        | set(val_df["user_id"].unique())
        | set(test_df["user_id"].unique())
    )
    unique_items = sorted(
        set(train_df["item_id"].unique())
        | set(val_df["item_id"].unique())
        | set(test_df["item_id"].unique())
    )

    # Create aligned matrices for each split
    train_matrix = (
        train_df.pivot(index="user_id", columns="item_id", values="rating")
        .reindex(index=unique_users, columns=unique_items, fill_value=0)
        .fillna(0)
    )

    val_matrix = (
        val_df.pivot(index="user_id", columns="item_id", values="rating")
        .reindex(index=unique_users, columns=unique_items, fill_value=0)
        .fillna(0)
    )

    test_matrix = (
        test_df.pivot(index="user_id", columns="item_id", values="rating")
        .reindex(index=unique_users, columns=unique_items, fill_value=0)
        .fillna(0)
    )

    return (
        train_matrix.values,
        val_matrix.values,
        test_matrix.values,
        unique_users,
        unique_items,
    )


def get_activity_group_map(train_df, user_ids, top_percentile=0.05):
    """
    Categorizes users into 'active' (top 5%) or 'inactive' (bottom 95%)
    based on their total interaction count in the training data.

    Args:
        train_df: Training DataFrame with 'user_id' column
        user_ids: List of all user IDs in the dataset
        top_percentile: Fraction of users to classify as 'active' (default: 0.05)

    Returns:
        dict: {user_id: 'active'/'inactive'}
    """
    # Count interactions per user
    user_counts = train_df.groupby("user_id").size()

    # Rank users by interaction count (descending) and determine threshold
    ranked = user_counts.sort_values(ascending=False)
    cutoff_idx = max(1, int(len(ranked) * top_percentile))
    active_users = set(ranked.head(cutoff_idx).index)

    # Build group map for all user_ids
    activity_map = {
        uid: "active" if uid in active_users else "inactive" for uid in user_ids
    }

    print(
        f"Activity Groups: {sum(1 for v in activity_map.values() if v == 'active')} active, "
        f"{sum(1 for v in activity_map.values() if v == 'inactive')} inactive"
    )

    return activity_map


def _find_data_file(subfolder, filename):
    """
    Generic helper to locate data files.
    Searches in:
      - data/<subfolder>/<filename>
      - ../data/<subfolder>/<filename>
      - data/<filename> (fallback)
    """
    possible_paths = [
        os.path.join("data", subfolder, filename),
        os.path.join("..", "data", subfolder, filename),
        os.path.join("data", filename),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def get_movielens_1m_data_path():
    """Locates the data file in data/ml-1m."""
    return _find_data_file("ml-1m", "ratings.dat")


def get_movielens_1m_data_numpy():
    """
    Returns ML-1M user-item matrix.
    """
    data_path = get_movielens_1m_data_path()
    if not data_path:
        raise FileNotFoundError("ML-1M data file not found.")

    df = load_data(data_path)
    train_df, test_df = split_train_test_stratified(df)
    return create_aligned_matrices(train_df, test_df)


def main():
    print("Testing ML-1M Data Loading...")
    try:
        # Load data and test the 3-way split
        data_path = get_movielens_1m_data_path()
        df = load_data(data_path)
        train_df, val_df, test_df = split_train_val_test_stratified(df)
        train_mat, val_mat, test_mat, u, i = create_aligned_matrices_3way(
            train_df, val_df, test_df
        )
        print(f"ML-1M Train Shape: {train_mat.shape}")
        print(f"ML-1M Val Shape: {val_mat.shape}")
        print(f"ML-1M Test Shape: {test_mat.shape}")
        activity_map = get_activity_group_map(train_df, u)
        print(f"ML-1M Activity Map Size: {len(activity_map)}")
    except Exception as e:
        print(e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
