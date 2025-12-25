import numpy as np


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


def calculate_gender_gap(ndcg_scores, user_ids, gender_map):
    """
    Calculates the Gender Gap between Male and Female users using NDCG.
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

    gender_gap = abs(avg_male - avg_female)
    return gender_gap


def calculate_item_coverage(
    prediction_matrix, original_matrix, item_ids, k=10, filter_rated=True
):
    """
    Calculates the Item Coverage.
    Formula: | U_{u in U} L_N(u) |
    (Cardinality of the set of unique items recommended across all users).

    filter_rated: If True, only considers items with 0 rating in original_matrix.
                  If False, considers all items (Oracular/Reconstruction setting).
    """
    unique_recommended_items = set()
    num_users = original_matrix.shape[0]

    for user_idx in range(num_users):
        user_predictions = prediction_matrix[user_idx]
        user_original_ratings = original_matrix[user_idx]

        if filter_rated:
            # Filter unrated
            unrated_indices = np.where(user_original_ratings == 0)[0]
            candidate_scores = user_predictions[unrated_indices]
            original_indices_map = unrated_indices
        else:
            # Consider all items
            candidate_scores = user_predictions
            original_indices_map = np.arange(len(user_predictions))

        if len(candidate_scores) == 0:
            continue

        # Get top K indices directly
        if len(candidate_scores) >= k:
            top_local_indices = np.argsort(candidate_scores)[::-1][:k]
            top_original_indices = original_indices_map[top_local_indices]
        else:
            top_original_indices = original_indices_map

        # Add item IDs to the set
        for idx in top_original_indices:
            unique_recommended_items.add(item_ids[idx])

    coverage_count = len(unique_recommended_items)
    total_items = len(item_ids)
    coverage_ratio = coverage_count / total_items if total_items > 0 else 0.0

    print(f"Item Coverage (Count): {coverage_count}")
    print(f"Item Coverage (Ratio): {coverage_ratio:.2%}")

    return coverage_count, coverage_ratio


def get_user_ideal_dcg(original_matrix, k):
    """
    Pre-calculates the Ideal DCG (IDCG) for all users based on their ground truth ratings.
    Used for normalizing DCG in the Genetic Algorithm to ensure consistency with Baseline NDCG.
    """
    idcg_scores = []
    for user_idx in range(original_matrix.shape[0]):
        actual_ratings = original_matrix[user_idx]
        # Calculate IDCG based on the best possible ordering of ALL items (Global Ideal)
        best_possible_dcg = dcg_at_k(sorted(actual_ratings, reverse=True), k)
        idcg_scores.append(best_possible_dcg)
    return np.array(idcg_scores)
