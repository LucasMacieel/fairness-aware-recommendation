import numpy as np
import warnings

# --- Module-Level Constants ---
# Single source of truth for default weights used by GA/NSGA-II
DEFAULT_WEIGHTS = {"mdcg": 1.0, "activity_gap": 1.0, "item_coverage": 1.0}


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


# NOTE: ndcg_at_k and calculate_ndcg_scores were removed as they are unused.
# The pipeline uses calculate_user_ndcg_scores with pre-calculated IDCG instead,
# ensuring consistent normalization between baseline and GA/NSGA-II evaluations.


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

    Raises:
        ValueError: If user_ids length doesn't match ndcg_scores length
    """
    # Validate user_ids length if provided
    if user_ids is not None and len(user_ids) != len(ndcg_scores):
        raise ValueError(
            f"user_ids length ({len(user_ids)}) must match ndcg_scores length ({len(ndcg_scores)})"
        )

    ndcg_arr = np.asarray(ndcg_scores)
    num_users = len(ndcg_arr)

    # Build boolean mask for active users (vectorized lookup)
    if user_ids is not None:
        keys = user_ids
    else:
        keys = range(num_users)

    active_mask = np.array([activity_map.get(k, "inactive") == "active" for k in keys])
    missing_users_count = sum(1 for k in keys if k not in activity_map)

    active_scores = ndcg_arr[active_mask]
    inactive_scores = ndcg_arr[~active_mask]

    # Warn if any users were not found in activity_map
    if missing_users_count > 0:
        warnings.warn(
            f"{missing_users_count} users not found in activity_map, defaulted to 'inactive'. "
            "This may indicate a key mismatch between user_ids and activity_map.",
            UserWarning,
        )

    avg_active = np.mean(active_scores) if len(active_scores) > 0 else 0.0
    avg_inactive = np.mean(inactive_scores) if len(inactive_scores) > 0 else 0.0

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

    # Vectorized: flatten all indices and count unique
    unique_indices = np.unique(recs_indices.flatten())
    return len(unique_indices) / len(item_ids) if len(item_ids) > 0 else 0.0


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


def get_user_ideal_dcg_from_candidates(ground_truth_matrix, candidate_lists, top_k):
    """
    Calculates the Ideal DCG (IDCG) for each user based ONLY on items in their candidate pool.

    This is the appropriate IDCG calculation for re-ranking evaluation:
    - The algorithm can only recommend from the candidate pool
    - IDCG should reflect the BEST possible ordering of candidate items
    - This gives realistic NDCG scores that measure re-ranking quality

    Args:
        ground_truth_matrix: array of shape (num_users, num_items) with true ratings
        candidate_lists: array of shape (num_users, candidate_size) with item indices
        top_k: Number of items to consider for IDCG

    Returns:
        array: IDCG value for each user (based on their candidate pool)

    Example:
        If a user's candidates have ratings [5, 0, 3, 4, 0, ...], the IDCG is
        computed from sorted([5, 4, 3, 0, 0, ...])[:k], not from ALL items globally.
    """
    num_users = ground_truth_matrix.shape[0]
    idcg_scores = []

    for user_idx in range(num_users):
        # Get ratings for ONLY the candidate items
        candidate_indices = candidate_lists[user_idx]
        candidate_ratings = ground_truth_matrix[user_idx, candidate_indices]

        # IDCG = DCG of the best possible ordering of candidate items
        best_possible_dcg = dcg_at_k(sorted(candidate_ratings, reverse=True), top_k)
        idcg_scores.append(best_possible_dcg)

    return np.array(idcg_scores)


def compute_weighted_score(mdcg, activity_gap, item_coverage, weights):
    """
    Compute the combined weighted fitness score.

    This is the single source of truth for the weighted scoring formula:
    score = (w_mdcg * MDCG) - (w_gap * Gap) + (w_cov * Coverage)

    - MDCG and Coverage are ADDED (higher is better)
    - Activity Gap is SUBTRACTED (lower is better)

    Args:
        mdcg: Mean normalized DCG score
        activity_gap: Absolute difference in NDCG between active/inactive groups
        item_coverage: Ratio of unique items recommended
        weights: dict with keys 'mdcg', 'activity_gap', 'item_coverage'

    Returns:
        float: Combined weighted score
    """
    return (
        (weights["mdcg"] * mdcg)
        - (weights["activity_gap"] * activity_gap)
        + (weights["item_coverage"] * item_coverage)
    )
