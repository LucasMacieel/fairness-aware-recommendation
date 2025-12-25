import pandas as pd
import os


def load_data(filepath):
    """
    Loads MovieLens 1M data from the specified filepath.
    Expected format: UserID::MovieID::Rating::Timestamp
    """
    column_names = ["user_id", "item_id", "rating", "timestamp"]
    df = pd.read_csv(filepath, sep="::", names=column_names, engine="python")
    return df


def create_user_item_matrix(df):
    """
    Converts the ratings DataFrame into a user-item matrix.
    Rows: Users
    Columns: Items
    Values: Ratings
    Missing values are filled with 0.
    """
    # Create pivot table
    user_item_matrix = df.pivot(index="user_id", columns="item_id", values="rating")

    # Fill missing values with 0
    user_item_matrix_filled = user_item_matrix.fillna(0)

    return user_item_matrix_filled


def get_movielens_gender_map():
    """
    Parses users.dat to create a mapping of user_id to gender.
    Returns:
        dict: {user_id: gender (M/F)}
    """
    data_path = get_movielens_data_path()
    if not data_path:
        return {}

    base_dir = os.path.dirname(data_path)
    user_file_path_1m = os.path.join(base_dir, "users.dat")

    gender_map = {}

    if os.path.exists(user_file_path_1m):
        with open(user_file_path_1m, "r", encoding="ISO-8859-1") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) >= 2:
                    user_id = int(parts[0])
                    gender = parts[1]
                    gender_map[user_id] = gender

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


def get_movielens_data_path():
    """Locates the data file in data/movielens."""
    return _find_data_file("movielens", "ratings.dat")


def get_movielens_data_numpy():
    """
    Returns the user-item matrix as a numpy array.
    Also returns the user_ids and item_ids for mapping back.
    """
    data_path = get_movielens_data_path()
    if not data_path:
        raise FileNotFoundError("MovieLens data file not found.")

    df = load_data(data_path)
    matrix_df = create_user_item_matrix(df)

    return matrix_df.values, matrix_df.index.tolist(), matrix_df.columns.tolist()


def get_post_data_path(filename):
    """Locates files in the data/post directory."""
    return _find_data_file("post", filename)


def get_post_data_numpy():
    """
    Returns the user-post interaction matrix as a numpy array from data/post.
    Implicit feedback: 1 for viewed, 0 for not viewed.
    """
    view_path = get_post_data_path("view_data.csv")
    if not view_path:
        raise FileNotFoundError("View data file not found.")

    # Load view data
    # format: user_id, post_id, time_stamp
    df_views = pd.read_csv(view_path)

    # We treat any view as a 'positive' interaction (1)
    df_views["rating"] = 1

    # Create pivot table
    # Since users can view multiple times, we just want to know if they viewed it at least once.
    # drop_duplicates handles multiple views of same post by same user
    df_views_unique = df_views[["user_id", "post_id", "rating"]].drop_duplicates()

    # Pivot: Users as rows, Posts as columns
    matrix_df = df_views_unique.pivot(
        index="user_id", columns="post_id", values="rating"
    )

    # Fill missing with 0
    matrix_df = matrix_df.fillna(0)

    return matrix_df.values, matrix_df.index.tolist(), matrix_df.columns.tolist()


def get_post_gender_map():
    """
    Parses data/post/user_data.csv to create a mapping of user_id to gender.
    Returns:
        dict: {user_id: gender (M/F)}
    """
    user_path = get_post_data_path("user_data.csv")
    if not user_path:
        return {}

    df_users = pd.read_csv(user_path)

    # Standardize gender to M/F
    gender_map = {}
    for _, row in df_users.iterrows():
        uid = row["user_id"]
        gender_str = str(row["gender"]).lower()
        if gender_str == "male":
            gender_map[uid] = "M"
        elif gender_str == "female":
            gender_map[uid] = "F"
        else:
            gender_map[uid] = "Unknown"

    return gender_map


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

    # Filter to top N active users to avoid OOM (1M+ users in original file)
    # Target roughly 5000 users for manageable matrix size
    top_users = df["user_id"].value_counts().head(5000).index
    df = df[df["user_id"].isin(top_users)]

    # We can reuse the generic create_user_item_matrix if it fits
    # But we need to handle duplicates if any. Let's strictly drop duplicates.
    df = df.drop_duplicates(subset=["user_id", "item_id"])

    matrix_df = create_user_item_matrix(df)

    return matrix_df.values, matrix_df.index.tolist(), matrix_df.columns.tolist()


def get_electronics_gender_map():
    """
    Parses data/electronics/df_electronics.csv to create a mapping of user_id to gender.
    The column is 'user_attr'.
    Returns:
        dict: {user_id: gender (M/F)}
    """
    data_path = get_electronics_data_path()
    if not data_path:
        return {}

    df = pd.read_csv(data_path, usecols=["user_id", "user_attr"])

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
    data_path = get_movielens_data_path()

    if not data_path:
        print(f"Error: Data file not found.")
        return

    print(f"Loading data from: {data_path}")
    df = load_data(data_path)
    print(f"Data loaded. Shape: {df.shape}")
    print(df.head())

    print("\nCreating user-item matrix...")
    matrix_df = create_user_item_matrix(df)
    print(f"Matrix created. Shape: {matrix_df.shape}")
    print("\nMatrix Head (first 5 rows and columns):")
    print(matrix_df.iloc[:5, :5])

    # Optional: verifying uniqueness
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()
    print(f"\nUnique Users: {n_users}")
    print(f"Unique Items: {n_items}")

    if matrix_df.shape == (n_users, n_items):
        print("Success: Matrix dimensions match unique user and item counts.")
    else:
        print("Warning: Matrix dimensions do not match unique counts.")

    # Verify numpy conversion
    print("\nVerifying Numpy Conversion:")
    mat_np, u_ids, i_ids = get_movielens_data_numpy()
    print(f"Numpy matrix shape: {mat_np.shape}")
    print(f"Type: {type(mat_np)}")


if __name__ == "__main__":
    main()
