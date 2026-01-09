from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
import warnings

# --- Module-Level Constants ---
# Single source of truth for default weights used by GA/NSGA-II
DEFAULT_WEIGHTS: dict[str, float] = {
    "mdcg": 1.0,
    "activity_gap": 1.0,
    "item_entropy": 1.0,
}


def dcg_at_k(r: ArrayLike, k: int) -> float:
    """
    Score is discounted cumulative gain (DCG) at rank k
    r: Relevance scores (list or numpy array) in rank order
    k: Number of results to consider
    """
    r = np.asarray(r)[:k]
    if r.size:
        return float(np.sum(r / np.log2(np.arange(2, r.size + 2))))
    return 0.0


# NOTE: ndcg_at_k and calculate_ndcg_scores were removed as they are unused.
# The pipeline uses calculate_user_ndcg_scores with pre-calculated IDCG instead,
# ensuring consistent normalization between baseline and GA/NSGA-II evaluations.


def calculate_activity_gap(
    ndcg_scores: list[float] | NDArray[np.floating],
    activity_map: dict[Any, str],
    user_ids: list[Any] | None = None,
    verbose: bool = True,
) -> float:
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

    return float(abs(avg_active - avg_inactive))


def calculate_shannon_entropy(
    recs_indices: NDArray[np.integer], num_items: int
) -> float:
    """
    Calculate normalized Shannon entropy of item recommendations.

    Shannon entropy measures how evenly distributed recommendations are across items.
    Higher entropy = more diverse/uniform distribution (better for long-tail exposure).
    Lower entropy = concentration on fewer items (popularity bias).

    Args:
        recs_indices: array of shape (num_users, k) with item indices
        num_items: total number of items in the catalog

    Returns:
        float: normalized entropy in [0, 1] where 1 = perfectly uniform distribution
    """
    # Flatten and count occurrences of each item
    flat_recs = recs_indices.flatten()
    item_counts = np.bincount(flat_recs, minlength=num_items)

    # Filter to only recommended items (non-zero counts)
    non_zero_counts = item_counts[item_counts > 0]

    if len(non_zero_counts) <= 1:
        return 0.0  # No diversity if 0 or 1 unique items

    # Calculate probabilities
    total_recs = len(flat_recs)
    probabilities = non_zero_counts / total_recs

    # Shannon entropy: H = -sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))

    # Normalize by maximum possible entropy (uniform over entire catalog)
    max_entropy = np.log2(num_items)

    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def calculate_user_ndcg_scores(
    recs_indices: NDArray[np.integer],
    ground_truth_matrix: NDArray[np.floating],
    idcg_values: NDArray[np.floating] | list[float],
) -> list[float]:
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
    num_users, top_k = recs_indices.shape

    # Vectorized rating lookup using advanced indexing
    user_indices = np.arange(num_users)[:, np.newaxis]
    true_ratings = ground_truth_matrix[user_indices, recs_indices]

    # Precompute discount factors (shared for all users)
    discount = np.log2(np.arange(2, top_k + 2))

    # Compute DCG for all users: sum(rating / discount) per user
    dcg_values = np.sum(true_ratings / discount, axis=1)

    # Compute NDCG with safe division (where IDCG is 0, NDCG is 0)
    idcg_arr = np.asarray(idcg_values)
    ndcg_scores = np.divide(
        dcg_values,
        idcg_arr,
        out=np.zeros_like(dcg_values, dtype=float),
        where=idcg_arr > 0,
    )

    return ndcg_scores.tolist()


def get_user_ideal_dcg_from_candidates(
    ground_truth_matrix: NDArray[np.floating],
    candidate_lists: NDArray[np.integer],
    top_k: int,
) -> NDArray[np.floating]:
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


def compute_weighted_score(
    mdcg: float,
    activity_gap: float,
    item_entropy: float,
    weights: dict[str, float],
) -> float:
    """
    Compute the combined weighted fitness score.

    This is the single source of truth for the weighted scoring formula:
    score = (w_mdcg * MDCG) - (w_gap * Gap) + (w_entropy * Entropy)

    - MDCG and Entropy are ADDED (higher is better)
    - Activity Gap is SUBTRACTED (lower is better)

    Args:
        mdcg: Mean normalized DCG score
        activity_gap: Absolute difference in NDCG between active/inactive groups
        item_entropy: Normalized Shannon entropy of item recommendations [0, 1]
        weights: dict with keys 'mdcg', 'activity_gap', 'item_entropy'

    Returns:
        float: Combined weighted score
    """
    return (
        (weights["mdcg"] * mdcg)
        - (weights["activity_gap"] * activity_gap)
        + (weights["item_entropy"] * item_entropy)
    )
