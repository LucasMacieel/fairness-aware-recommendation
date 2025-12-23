import numpy as np
from scipy.sparse.linalg import svds
from data_processing import (
    get_movielens_data_numpy,
    get_movielens_gender_map,
    get_post_data_numpy,
    get_post_gender_map,
    get_sushi_data_numpy,
    get_sushi_gender_map,
)


def perform_svd(matrix, k=20):
    """
    Performs Singular Value Decomposition on the user-item matrix.

    Args:
        matrix (np.ndarray): User-Item interaction matrix.
        k (int): Number of latent factors to keep.

    Returns:
        prediction_matrix (np.ndarray): The reconstructed matrix with predicted ratings.
    """
    # Normalize by user mean ratings
    user_ratings_mean = np.mean(matrix, axis=1)
    matrix_demeaned = matrix - user_ratings_mean.reshape(-1, 1)

    # Perform SVD
    # U: User features, sigma: Singular values, Vt: Item features
    # Check if k is valid for the matrix size
    k = min(k, min(matrix.shape) - 1)

    U, sigma, Vt = svds(matrix_demeaned, k=k)

    # Convert sigma to diagonal matrix
    sigma = np.diag(sigma)

    # Reconstruct matrix
    all_user_predicted_ratings = np.dot(
        np.dot(U, sigma), Vt
    ) + user_ratings_mean.reshape(-1, 1)

    return all_user_predicted_ratings


def dcg_at_k(r, k):
    """
    Score is discounted cumulative gain (DCG) at rank k
    r: Relevance scores (list or numpy array) in rank order
    k: Number of results to consider
    """
    r = np.asarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0


def ndcg_at_k(r, k):
    """
    Score is normalized discounted cumulative gain (NDCG) at rank k
    r: Relevance scores (list or numpy array) in rank order
    k: Number of results to consider
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max


def calculate_ndcg_scores(original_matrix, prediction_matrix, k=20):
    """
    Calculates NDCG scores for all users.
    Returns a list of scores corresponding to user indices.
    """
    ndcg_scores = []

    for user_idx in range(original_matrix.shape[0]):
        actual_ratings = original_matrix[user_idx]
        predicted_ratings = prediction_matrix[user_idx]

        item_pairs = list(zip(predicted_ratings, actual_ratings))
        item_pairs.sort(key=lambda x: x[0], reverse=True)
        relevance_ordered = [x[1] for x in item_pairs]

        score = ndcg_at_k(relevance_ordered, k)
        ndcg_scores.append(score)

    return ndcg_scores


def calculate_mean_ndcg(ndcg_scores):
    return np.mean(ndcg_scores)


def calculate_unfairness_gap(ndcg_scores, user_ids, gender_map):
    """
    Calculates the Unfairness Gap between Male and Female users using NDCG.
    Gap = |Avg(NDCG_M) - Avg(NDCG_F)|
    """
    male_scores = []
    female_scores = []

    for idx, user_id in enumerate(user_ids):
        # NDCG scores map directly to user_ids via index
        score = ndcg_scores[idx]
        gender = gender_map.get(user_id, "Unknown")

        if gender == "M":
            male_scores.append(score)
        elif gender == "F":
            female_scores.append(score)

    avg_male = np.mean(male_scores) if male_scores else 0.0
    avg_female = np.mean(female_scores) if female_scores else 0.0

    print(
        f"Fairness Analysis: Male Avg: {avg_male:.4f} (n={len(male_scores)}), Female Avg: {avg_female:.4f} (n={len(female_scores)})"
    )

    gap = abs(avg_male - avg_female)
    return gap


def calculate_item_coverage(prediction_matrix, original_matrix, item_ids, k=10):
    """
    Calculates the Item Coverage.
    Formula: | U_{u in U} L_N(u) |
    (Cardinality of the set of unique items recommended across all users).
    """
    unique_recommended_items = set()
    num_users = original_matrix.shape[0]

    for user_idx in range(num_users):
        user_predictions = prediction_matrix[user_idx]
        user_original_ratings = original_matrix[user_idx]

        # Filter unrated
        unrated_indices = np.where(user_original_ratings == 0)[0]
        unrated_scores = user_predictions[unrated_indices]

        if len(unrated_scores) == 0:
            continue

        # Get top K indices directly
        if len(unrated_scores) >= k:
            top_indices = np.argsort(unrated_scores)[::-1][:k]
            top_original_indices = unrated_indices[top_indices]
        else:
            top_original_indices = unrated_indices

        # Add item IDs to the set
        for idx in top_original_indices:
            unique_recommended_items.add(item_ids[idx])

    coverage_count = len(unique_recommended_items)
    total_items = len(item_ids)
    coverage_ratio = coverage_count / total_items if total_items > 0 else 0.0

    print(f"Item Coverage (Count): {coverage_count}")
    print(f"Item Coverage (Ratio): {coverage_ratio:.2%}")

    return coverage_count, coverage_ratio


def run_pipeline(matrix, user_ids, item_ids, gender_map, dataset_name):
    print(f"\nProcessing {dataset_name} Dataset...")
    print(f"Original Matrix Shape: {matrix.shape}")

    # Check sparsity or if we have enough data for k
    k_factors = min(matrix.shape) - 1
    k_factors = 20 if k_factors > 20 else k_factors

    print(f"Performing SVD with k={k_factors}...")
    prediction_matrix = perform_svd(matrix, k=k_factors)
    print("SVD Complete.")

    results = {
        "Dataset": dataset_name,
        "Users": matrix.shape[0],
        "Items": matrix.shape[1],
    }

    # Evaluation
    k_ndcg = 10
    print(f"Evaluating Recommender (NDCG@{k_ndcg})...")

    ndcg_scores = calculate_ndcg_scores(matrix, prediction_matrix, k=k_ndcg)
    mean_ndcg = calculate_mean_ndcg(ndcg_scores)
    print(f"Mean NDCG@{k_ndcg}: {mean_ndcg:.4f}")
    results[f"NDCG@{k_ndcg}"] = mean_ndcg

    # Fairness Evaluation
    if gender_map:
        unfairness_gap = calculate_unfairness_gap(ndcg_scores, user_ids, gender_map)
        print(f"Unfairness Gap (Gender): {unfairness_gap:.4f}")
        results["Unfairness Gap"] = unfairness_gap
    else:
        print("Warning: Could not load gender map. Skipping fairness analysis.")
        results["Unfairness Gap"] = "N/A"

    # Item Coverage Evaluation
    cov_count, cov_ratio = calculate_item_coverage(
        prediction_matrix, matrix, item_ids, k=10
    )
    # Use Coverage as the main metric
    results["Item Coverage"] = cov_ratio

    return results


def main():
    datasets = []

    # 1. MovieLens
    try:
        datasets.append(
            {
                "name": "MovieLens",
                "loader": get_movielens_data_numpy,
                "gender_loader": get_movielens_gender_map,
            }
        )
    except Exception as e:
        print(f"Error configuring MovieLens: {e}")

    # 2. Post Data
    try:
        datasets.append(
            {
                "name": "Post Data",
                "loader": get_post_data_numpy,
                "gender_loader": get_post_gender_map,
            }
        )
    except Exception as e:
        print(f"Error configuring Post Data: {e}")

    # 3. Sushi Data
    try:
        datasets.append(
            {
                "name": "Sushi Data",
                "loader": get_sushi_data_numpy,
                "gender_loader": get_sushi_gender_map,
            }
        )
    except Exception as e:
        print(f"Error configuring Sushi Data: {e}")

    all_results = []

    for ds in datasets:
        try:
            print(f"\n--- Loading {ds['name']} ---")
            matrix, user_ids, item_ids = ds["loader"]()
            gender_map = ds["gender_loader"]()

            result = run_pipeline(matrix, user_ids, item_ids, gender_map, ds["name"])
            all_results.append(result)

        except FileNotFoundError:
            print(f"Skipping {ds['name']}: Data file not found.")
        except Exception as e:
            print(f"Error processing {ds['name']}: {e}")

    # Print Comparison Table
    print("\n\n" + "=" * 90)
    print(f"{'DATASET COMPARISON':^90}")
    print("=" * 90)

    # Define headers
    headers = [
        "Dataset",
        "Users",
        "Items",
        "NDCG@10",
        "Unfairness Gap",
        "Item Coverage",
    ]

    # Print headers
    header_row = f"{headers[0]:<15} | {headers[1]:<8} | {headers[2]:<8} | {headers[3]:<10} | {headers[4]:<15} | {headers[5]:<15}"
    print(header_row)
    print("-" * len(header_row))

    # Print rows
    for res in all_results:
        ug_str = (
            f"{res['Unfairness Gap']:.4f}"
            if isinstance(res["Unfairness Gap"], float)
            else str(res["Unfairness Gap"])
        )
        # Format item coverage as decimal 0-1
        cov_str = f"{res['Item Coverage']:.4f}"

        row = f"{res['Dataset']:<15} | {res['Users']:<8} | {res['Items']:<8} | {res[f'NDCG@10']:.4f}     | {ug_str:<15} | {cov_str:<15}"
        print(row)
    print("=" * 90)


if __name__ == "__main__":
    main()
