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

    Use this function when you have the original user_ids mapping (e.g., baseline evaluation).
    For GA/NSGA-II internal fitness with index-based mapping, use calculate_gender_gap_indexed().

    Args:
        ndcg_scores: list of NDCG scores corresponding to user indices
        user_ids: list of original user IDs (maps index -> user_id)
        gender_map: dict {user_id: 'M'/'F'/'Unknown'}
    """
    male_scores = []
    female_scores = []

    for idx, user_id in enumerate(user_ids):
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


def calculate_gender_gap_indexed(ndcg_scores, gender_map):
    """
    Calculates the Gender Gap using index-based gender_map (for GA/NSGA-II internal fitness).

    Use this function during GA/NSGA-II optimization where gender_map is pre-converted
    to index-based format. For baseline/test evaluation with original user IDs,
    use calculate_gender_gap().

    Args:
        ndcg_scores: list of NDCG scores indexed by user position
        gender_map: dict {user_index: 'M'/'F'/'Unknown'}
    """
    male_scores = []
    female_scores = []

    for idx, score in enumerate(ndcg_scores):
        gender = gender_map.get(idx, "Unknown")
        if gender == "M":
            male_scores.append(score)
        elif gender == "F":
            female_scores.append(score)

    avg_male = np.mean(male_scores) if male_scores else 0.0
    avg_female = np.mean(female_scores) if female_scores else 0.0

    return abs(avg_male - avg_female)


def calculate_item_coverage_simple(recs_indices, item_ids):
    """
    Calculate item coverage ratio from recommendation indices.

    Args:
        recs_indices: array of shape (num_users, k) with item indices
        item_ids: list of all item IDs in the dataset (must be unique)

    Returns:
        float: coverage ratio (unique recommended items / total items)

    Raises:
        AssertionError: if item_ids contains duplicates
    """
    # Validate that item_ids has no duplicates to ensure correct coverage calculation
    assert len(item_ids) == len(set(item_ids)), (
        f"item_ids must be unique. Found {len(item_ids)} items but only {len(set(item_ids))} unique."
    )

    unique_items = set()
    for u in range(recs_indices.shape[0]):
        for idx in recs_indices[u]:
            unique_items.add(item_ids[idx])
    return len(unique_items) / len(item_ids) if len(item_ids) > 0 else 0.0


def calculate_user_ndcg_scores(recs_indices, ground_truth_matrix, idcg_values):
    """
    Calculate NDCG scores for each user based on their recommendations.

    This is the centralized function for NDCG calculation used by both
    baseline evaluation and GA/NSGA-II evaluation to ensure consistency.

    Args:
        recs_indices: array of shape (num_users, k) with recommended item indices
        ground_truth_matrix: array of shape (num_users, num_items) with true ratings
        idcg_values: array of pre-calculated IDCG values per user

    Returns:
        list: NDCG scores for each user
    """
    ndcg_scores = []
    num_users = recs_indices.shape[0]

    for u in range(num_users):
        rec_inds = recs_indices[u]
        true_ratings = ground_truth_matrix[u, rec_inds]

        r = np.asarray(true_ratings, dtype=float)
        if r.size:
            dcg = dcg_at_k(r, len(r))
            idcg = idcg_values[u]
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)
        else:
            ndcg_scores.append(0.0)

    return ndcg_scores


def get_user_ideal_dcg(original_matrix, k):
    """
    Pre-calculates the Ideal DCG (IDCG) for all users based on their ground truth ratings.
    Used for normalizing DCG in the Genetic Algorithm to ensure consistency with Baseline NDCG.

    Note: Users with no positive ratings in ground truth will have IDCG=0.
    The GA/NSGA-II fitness functions handle this by returning NDCG=0 for such users.
    """
    idcg_scores = []
    for user_idx in range(original_matrix.shape[0]):
        actual_ratings = original_matrix[user_idx]
        # Calculate IDCG based on the best possible ordering of ALL items (Global Ideal)
        best_possible_dcg = dcg_at_k(sorted(actual_ratings, reverse=True), k)
        idcg_scores.append(best_possible_dcg)
    return np.array(idcg_scores)
