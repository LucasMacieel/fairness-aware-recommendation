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


def calculate_activity_gap(ndcg_scores, activity_map, user_ids=None, verbose=True):
    """
    Calculates the Activity Gap between Active and Inactive users using NDCG.
    Gap = |Avg(NDCG_active) - Avg(NDCG_inactive)|

    This unified function handles both:
    - Original user_id mapping (when user_ids is provided)
    - Index-based mapping (when user_ids is None, activity_map uses indices as keys)

    Args:
        ndcg_scores: list of NDCG scores corresponding to user indices
        activity_map: dict mapping to 'active'/'inactive'. Keys are:
            - user_ids (original IDs) if user_ids parameter is provided
            - user indices (0, 1, 2, ...) if user_ids is None
        user_ids: Optional list of original user IDs (maps index -> user_id).
            If None, activity_map is assumed to use indices as keys.
        verbose: If True, print fairness analysis summary

    Returns:
        float: Absolute difference between active and inactive group average NDCG
    """
    active_scores = []
    inactive_scores = []

    for idx, score in enumerate(ndcg_scores):
        # Use user_id as key if provided, otherwise use index
        key = user_ids[idx] if user_ids is not None else idx
        group = activity_map.get(key, "inactive")

        if group == "active":
            active_scores.append(score)
        else:
            inactive_scores.append(score)

    avg_active = np.mean(active_scores) if active_scores else 0.0
    avg_inactive = np.mean(inactive_scores) if inactive_scores else 0.0

    if verbose:
        print(
            f"Fairness Analysis: Active Avg: {avg_active:.4f} (n={len(active_scores)}), "
            f"Inactive Avg: {avg_inactive:.4f} (n={len(inactive_scores)})"
        )

    return abs(avg_active - avg_inactive)


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
