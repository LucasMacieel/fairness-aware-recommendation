import pandas as pd
import os


def load_data(filepath):
    """
    Loads MovieLens data from the specified filepath.
    Supports both ML-1M (ratings.dat, ::) and ML-100k (u.data, \t).
    """
    column_names = ["user_id", "item_id", "rating", "timestamp"]

    # Check for ML-100k (u.data) vs ML-1M (ratings.dat) based on filename or extension
    if filepath.endswith("u.data"):
        sep = "\t"
    else:
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


def get_movielens_100k_gender_map():
    """
    Parses ml-100k/u.user for gender.
    Format: user id | age | gender | occupation | zip code
    """
    data_path = get_movielens_100k_data_path()
    if not data_path:
        return {}

    base_dir = os.path.dirname(data_path)
    user_file = os.path.join(base_dir, "u.user")
    gender_map = {}

    if os.path.exists(user_file):
        try:
            with open(user_file, "r", encoding="ISO-8859-1") as f:
                for line in f:
                    parts = line.strip().split("|")
                    if len(parts) >= 3:
                        gender_map[int(parts[0])] = parts[2]
        except Exception:
            pass

    return gender_map


def get_movielens_1m_gender_map():
    """
    Parses ml-1m/users.dat for gender.
    Format: UserID::Gender::Age::Occupation::Zip-code
    """
    data_path = get_movielens_1m_data_path()
    if not data_path:
        return {}

    base_dir = os.path.dirname(data_path)
    user_file = os.path.join(base_dir, "users.dat")
    gender_map = {}

    if os.path.exists(user_file):
        try:
            with open(user_file, "r", encoding="ISO-8859-1") as f:
                for line in f:
                    parts = line.strip().split("::")
                    if len(parts) >= 2:
                        gender_map[int(parts[0])] = parts[1]
        except Exception:
            pass

    return gender_map


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


def get_movielens_100k_data_path():
    """Locates the data file in data/ml-100k."""
    return _find_data_file("ml-100k", "u.data")


def get_movielens_1m_data_path():
    """Locates the data file in data/ml-1m."""
    return _find_data_file("ml-1m", "ratings.dat")


def get_movielens_100k_data_numpy():
    """
    Returns ML-100k user-item matrix.
    """
    data_path = get_movielens_100k_data_path()
    if not data_path:
        raise FileNotFoundError("ML-100k data file not found.")

    df = load_data(data_path)
    train_df, test_df = split_train_test_stratified(df)
    return create_aligned_matrices(train_df, test_df)


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


def get_electronics_data_path():
    """Locates the data file in data/electronics."""
    return _find_data_file("electronics", "df_electronics.csv")


def get_electronics_data_numpy():
    """
    Returns the user-item rating matrix as a numpy array from data/electronics.
    """
    data_path = get_electronics_data_path()
    if not data_path:
        raise FileNotFoundError("Electronics data file not found.")

    df = pd.read_csv(data_path)
    # Ensure columns exist (they match our convention)
    # Filter only necessary columns to avoid overhead
    df = df[["user_id", "item_id", "rating"]]

    # Filter to 5000 random users to avoid OOM, but maintain diversity
    unique_users = df["user_id"].unique()
    if len(unique_users) > 5000:
        import numpy as np

        # Use local RandomState to avoid affecting global random state
        rng = np.random.RandomState(42)
        selected_users = rng.choice(unique_users, size=5000, replace=False)
    else:
        selected_users = unique_users

    df = df[df["user_id"].isin(selected_users)]

    # We can reuse the generic create_user_item_matrix if it fits
    # But we need to handle duplicates if any. Let's strictly drop duplicates.
    df = df.drop_duplicates(subset=["user_id", "item_id"])

    train_df, test_df = split_train_test_stratified(df)
    return create_aligned_matrices(train_df, test_df)


def get_electronics_gender_map():
    """
    Parses data/electronics/df_electronics.csv to create a mapping of user_id to gender.
    The column is 'user_attr'.

    Note: This function applies the same user filtering as get_electronics_data_numpy()
    to ensure consistency between the data matrix and gender map.

    Returns:
        dict: {user_id: gender (M/F)}
    """
    data_path = get_electronics_data_path()
    if not data_path:
        return {}

    df = pd.read_csv(data_path, usecols=["user_id", "user_attr"])

    # Apply the same user filtering as get_electronics_data_numpy()
    # This ensures the gender map only contains users that are in the filtered dataset
    unique_users = df["user_id"].unique()
    if len(unique_users) > 5000:
        import numpy as np

        # Use the SAME local RandomState(42) as get_electronics_data_numpy
        rng = np.random.RandomState(42)
        selected_users = rng.choice(unique_users, size=5000, replace=False)
        df = df[df["user_id"].isin(selected_users)]

    # Drop rows with missing gender
    df = df.dropna(subset=["user_attr"])

    gender_map = {}
    # Iterate. Note: A user might appear multiple times. We take the first valid one or all should be same.
    # Let's drop duplicates on user_id to speed up
    df_unique = df.drop_duplicates(subset=["user_id"])

    for _, row in df_unique.iterrows():
        uid = row["user_id"]
        g = str(row["user_attr"]).strip().lower()
        if g == "female":
            gender_map[uid] = "F"
        elif g == "male":
            gender_map[uid] = "M"

    return gender_map


def main():
    print("Testing ML-100k Data Loading...")
    try:
        mat, test, u, i = get_movielens_100k_data_numpy()
        print(f"ML-100k Train Shape: {mat.shape}, Test Shape: {test.shape}")
        g = get_movielens_100k_gender_map()
        print(f"ML-100k Gender Map Size: {len(g)}")
    except Exception as e:
        print(e)

    print("\nTesting ML-1M Data Loading...")
    try:
        mat, test, u, i = get_movielens_1m_data_numpy()
        print(f"ML-1M Train Shape: {mat.shape}, Test Shape: {test.shape}")
        g = get_movielens_1m_gender_map()
        print(f"ML-1M Gender Map Size: {len(g)}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
