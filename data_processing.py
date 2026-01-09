import os
import pandas as pd
from surprise import Dataset, Reader


# --- Helper Functions for Dataset Loading ---


def _find_file(filepath, alt_paths, dataset_name):
    """Find dataset file, trying alternate paths if primary not found."""
    if os.path.exists(filepath):
        return filepath
    for alt in alt_paths:
        if os.path.exists(alt):
            return alt
    raise FileNotFoundError(f"{dataset_name} dataset not found at {filepath}")


def _apply_kcore_filter(df, min_interactions, dataset_name):
    """Apply iterative k-core filtering until convergence."""
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)
        # Filter users with at least min_interactions
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_interactions].index
        df = df[df["user_id"].isin(valid_users)]
        # Filter items with at least min_interactions
        item_counts = df["item_id"].value_counts()
        valid_items = item_counts[item_counts >= min_interactions].index
        df = df[df["item_id"].isin(valid_items)]
    print(
        f"{dataset_name}: After {min_interactions}-core filtering: "
        f"{len(df)} ratings, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items"
    )
    return df


def _deduplicate_ratings(df, dataset_name):
    """Handle duplicate (user_id, item_id) pairs by taking mean rating."""
    duplicates = df.duplicated(subset=["user_id", "item_id"], keep=False).sum()
    if duplicates > 0:
        print(
            f"{dataset_name}: Found {duplicates} duplicate entries, aggregating with mean..."
        )
        df = df.groupby(["user_id", "item_id"], as_index=False)["rating"].mean()
        print(f"{dataset_name}: After deduplication: {len(df)} unique ratings")
    return df


def _standardize_columns(df):
    """Convert user_id, item_id to string and rating to float."""
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = df["rating"].astype(float)
    return df


def load_movielens_1m_surprise():
    """
    Load MovieLens 1M dataset using Surprise's built-in loader.

    Returns:
        surprise.Dataset: Surprise dataset object with ML-1M data
        pd.DataFrame: Full dataset as DataFrame for compatibility with activity_map
    """
    # Load the built-in ML-1M dataset
    data = Dataset.load_builtin("ml-1m")

    # Also get it as a DataFrame for downstream processing
    # Surprise stores the data internally, we need to extract it
    df = pd.DataFrame(
        data.raw_ratings, columns=["user_id", "item_id", "rating", "timestamp"]
    )

    # Convert types to match existing pipeline expectations
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["rating"] = df["rating"].astype(float)

    return data, df


def load_book_crossing(filepath="data/book_crossing.csv", min_interactions=5):
    """
    Load Book-Crossing dataset from local CSV file with k-core filtering.

    Args:
        filepath: Path to the book_crossing.csv file
        min_interactions: Minimum interactions per user AND item (k-core filtering)

    Returns:
        pd.DataFrame: DataFrame with user_id, item_id, rating columns
    """
    alt_paths = [
        os.path.join("data", "book_crossing.csv"),
        os.path.join("..", "data", "book_crossing.csv"),
    ]
    filepath = _find_file(filepath, alt_paths, "Book-Crossing")

    # Load CSV with semicolon separator and rename columns
    df = pd.read_csv(filepath, sep=";", encoding="latin-1")
    df = df.rename(
        columns={"User-ID": "user_id", "ISBN": "item_id", "Book-Rating": "rating"}
    )
    if "Rating" in df.columns:
        df = df.rename(columns={"Rating": "rating"})

    df = _standardize_columns(df)

    # Filter out implicit ratings (rating = 0)
    df = df[df["rating"] > 0]
    print(f"Book-Crossing: {len(df)} explicit ratings (filtered out rating=0)")

    return _apply_kcore_filter(df, min_interactions, "Book-Crossing")


def load_digital_music(filepath="data/digital_music.csv", min_interactions=5):
    """
    Load Amazon Digital Music dataset from local CSV file with k-core filtering.

    Args:
        filepath: Path to the digital_music.csv file
        min_interactions: Minimum interactions per user AND item (k-core filtering)

    Returns:
        pd.DataFrame: DataFrame with user_id, item_id, rating columns
    """
    alt_paths = [
        os.path.join("data", "digital_music.csv"),
        os.path.join("..", "data", "digital_music.csv"),
    ]
    filepath = _find_file(filepath, alt_paths, "Digital Music")

    df = pd.read_csv(filepath)
    df = _standardize_columns(df)
    print(f"Digital Music: {len(df)} total ratings loaded")

    df = _deduplicate_ratings(df, "Digital Music")
    return _apply_kcore_filter(df, min_interactions, "Digital Music")


def load_video_games(filepath="data/video_games.csv", min_interactions=5):
    """
    Load Amazon Video Games dataset from local CSV file with k-core filtering.

    Args:
        filepath: Path to the video_games.csv file
        min_interactions: Minimum interactions per user AND item (k-core filtering)

    Returns:
        pd.DataFrame: DataFrame with user_id, item_id, rating columns
    """
    alt_paths = [
        os.path.join("data", "video_games.csv"),
        os.path.join("..", "data", "video_games.csv"),
    ]
    filepath = _find_file(filepath, alt_paths, "Video Games")

    df = pd.read_csv(filepath)
    df = _standardize_columns(df)
    print(f"Video Games: {len(df)} total ratings loaded")

    df = _deduplicate_ratings(df, "Video Games")
    return _apply_kcore_filter(df, min_interactions, "Video Games")


def df_to_surprise_trainset(df, rating_scale=(1, 5)):
    """
    Convert a pandas DataFrame to a Surprise Trainset.

    Args:
        df: DataFrame with user_id, item_id, rating columns
        rating_scale: Tuple of (min_rating, max_rating)

    Returns:
        surprise.Trainset: Trainset object for training
    """
    reader = Reader(rating_scale=rating_scale)
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)
    return data.build_full_trainset()


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


def main():
    """Test the Surprise-based ML-1M data loading."""
    print("Testing ML-1M Data Loading (Surprise)...")
    try:
        _, df = load_movielens_1m_surprise()
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
