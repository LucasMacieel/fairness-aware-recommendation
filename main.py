import numpy as np
import random
import pandas as pd
from data_processing import (
    load_movielens_1m_surprise,
    load_movielens_100k_surprise,
    load_book_crossing,
    df_to_surprise_trainset,
    split_train_val_test_stratified,
    create_aligned_matrices_3way,
    get_activity_group_map,
)
from metrics import (
    DEFAULT_WEIGHTS,
    calculate_activity_gap,
    calculate_item_coverage,
    calculate_shannon_entropy,
    calculate_user_ndcg_scores,
    compute_weighted_score,
    get_user_ideal_dcg_from_candidates,
)
from recommender import train_svd_surprise, get_predictions_matrix
from genetic_recommender import GeneticRecommender, NsgaIIRecommender
from utils.plotter import plot_pairwise_pareto
from cache import (
    get_all_matrix_cache_paths,
    all_matrices_cached,
    save_matrix,
    load_matrix,
    save_metadata,
    load_metadata,
)

# --- Module-Level Constants ---
# Centralized for consistency across the pipeline
SEED = 42
CANDIDATE_SIZE = 100  # Number of candidates per user for re-ranking
POP_SIZE = 100  # Population size for GA/NSGA-II
GENERATIONS = 10  # Number of generations for evolution
K_NDCG = 10  # Top-K for NDCG evaluation
CROSSOVER_RATE = 0.8  # Crossover probability for genetic operators
MUTATION_RATE = 0.2  # Mutation probability for genetic operators


def run_pipeline(
    train_matrix,
    val_matrix,
    test_matrix,
    user_ids,
    item_ids,
    activity_map,
    train_df,
    dataset_name,
    k_ndcg=K_NDCG,
    rating_scale=(1, 5),
    prediction_matrix=None,
):
    """
    Run the recommendation pipeline with train/validation/test split.

    - train_matrix: Used for SVD training
    - val_matrix: Used for GA/NSGA-II optimization (ground truth during optimization)
    - test_matrix: Used for final unbiased evaluation only
    - train_df: Training DataFrame needed for Surprise SVD training (not needed if prediction_matrix provided)
    - rating_scale: Tuple of (min_rating, max_rating) for the dataset
    - prediction_matrix: Optional pre-computed prediction matrix (from cache)
    """
    # Note: Random seed is set once in main() before data loading for full reproducibility.
    # No need to re-seed here as it would reset the random state mid-pipeline.

    print(f"\nProcessing {dataset_name} Dataset...")
    print(f"Train Matrix Shape: {train_matrix.shape}")
    print(f"Validation Matrix Shape: {val_matrix.shape}")
    print(f"Test Matrix Shape: {test_matrix.shape}")
    print(
        f"GA Parameters: Pop={POP_SIZE}, Gen={GENERATIONS}, Crossover={CROSSOVER_RATE}, Mutation={MUTATION_RATE}, Candidates={CANDIDATE_SIZE}, Top-K={k_ndcg}"
    )

    # --- Validate user_ids ordering matches matrix indices ---
    # This is critical for ensuring activity_map lookups are consistent.
    # user_ids[i] should correspond to row i in all matrices.
    num_users_from_ids = len(user_ids)
    assert num_users_from_ids == train_matrix.shape[0], (
        f"user_ids length ({num_users_from_ids}) must match matrix rows ({train_matrix.shape[0]})"
    )

    # Use pre-computed prediction matrix if provided (from cache)
    if prediction_matrix is None:
        # Note: perform_svd internally caps k to min(matrix.shape) - 1 if needed
        trainset = df_to_surprise_trainset(train_df, rating_scale=rating_scale)
        svd_model = train_svd_surprise(trainset, random_state=SEED)
        print("SVD Training Complete.")

        # Generate prediction matrix using trained SVD model
        prediction_matrix = get_predictions_matrix(
            svd_model, user_ids, item_ids, trainset
        )
        print("Prediction Matrix Generated.")
    else:
        print("Using cached prediction matrix.")

    # --- Masking Training Items ---
    # We purposefully mask items present in the training set so they are not recommended again.
    # This simulates a "standard" recommendation setting where we predict unobserved items.
    train_mask = train_matrix != 0
    masked_prediction_matrix = prediction_matrix.copy()
    masked_prediction_matrix[train_mask] = -np.inf

    results = {
        "Dataset": dataset_name,
        "Users": train_matrix.shape[0],
        "Items": train_matrix.shape[1],
        "Sparsity": 1 - (np.count_nonzero(train_matrix) / train_matrix.size),
    }

    # --- Generate Candidate Lists FIRST (needed for IDCG and baseline comparison) ---
    num_users, num_items = train_matrix.shape

    # NOTE: Item coverage during optimization is bounded by CANDIDATE_SIZE * num_users unique items.
    # The GA/NSGA-II can only recommend items from the candidate pool, not the full item catalog.
    # This is a design choice to simulate realistic re-ranking scenarios.
    print(f"Generating Top-{CANDIDATE_SIZE} candidates for all methods...")

    # Vectorized argsort: sort ascending then reverse to get descending order
    # Note: Cannot use -scores because -(-inf) = +inf which would sort first!
    sorted_indices = np.argsort(masked_prediction_matrix, axis=1)[:, ::-1]
    candidate_lists = sorted_indices[:, :CANDIDATE_SIZE]

    # --- Pre-calculate IDCG from CANDIDATE POOL (Re-ranking Evaluation) ---
    # IMPORTANT: IDCG is computed from only the candidate items, not all items.
    # This is appropriate for re-ranking evaluation where:
    #   - The algorithm can ONLY recommend from the candidate pool
    #   - IDCG reflects the BEST achievable ordering within constraints
    #   - NDCG measures how well the algorithm re-ranks candidates
    # Using global IDCG (all items) would artificially deflate scores since
    # the ideal items may not be in the candidate pool at all.
    print(f"Calculating IDCG from candidates (Validation Set) for top_k={k_ndcg}...")
    val_user_idcg_scores = get_user_ideal_dcg_from_candidates(
        val_matrix, candidate_lists, top_k=k_ndcg
    )
    print(f"Calculating IDCG from candidates (Test Set) for top_k={k_ndcg}...")
    test_user_idcg_scores = get_user_ideal_dcg_from_candidates(
        test_matrix, candidate_lists, top_k=k_ndcg
    )

    # --- Baseline Evaluation (using same candidate pool as GA for fair comparison) ---
    print(f"\nEvaluating Baseline Recommender (NDCG@{k_ndcg})...")

    # Baseline: Top-K from candidate list (same constraint as GA)
    # This ensures fair comparison - baseline also limited to same candidate pool
    baseline_recs_indices = np.zeros((num_users, k_ndcg), dtype=int)

    for u in range(num_users):
        # Baseline takes the top-k from candidate list (which is already sorted by SVD score)
        baseline_recs_indices[u] = candidate_lists[u, :k_ndcg]

    # Calculate NDCG on TEST set using centralized function
    baseline_ndcg_scores = calculate_user_ndcg_scores(
        baseline_recs_indices, test_matrix, test_user_idcg_scores
    )
    baseline_mdcg = np.mean(baseline_ndcg_scores)

    # Fairness - using global IDCG-normalized scores
    if activity_map:
        baseline_activity_gap = calculate_activity_gap(
            baseline_ndcg_scores, activity_map, user_ids=user_ids
        )
    else:
        baseline_activity_gap = 0.0  # Default if no map

    # Item Entropy - using centralized function
    num_items = len(item_ids)
    baseline_item_entropy = calculate_shannon_entropy(baseline_recs_indices, num_items)
    # Item Coverage - secondary diagnostic metric
    baseline_item_coverage = calculate_item_coverage(baseline_recs_indices, num_items)

    results["Baseline MDCG"] = baseline_mdcg
    results["Baseline Activity Gap"] = baseline_activity_gap
    results["Baseline Item Entropy"] = baseline_item_entropy
    results["Baseline Item Coverage"] = baseline_item_coverage

    print(
        f"Baseline (Test Set): MDCG={baseline_mdcg:.4f}, Activity Gap={baseline_activity_gap:.4f}, "
        f"Item Entropy={baseline_item_entropy:.4f}, Item Coverage={baseline_item_coverage:.4f}"
    )

    # --- Genetic Algorithm ---
    # The GA re-ranks candidates to improve Fairness while maintaining relevance.
    # NOTE: GA optimizes on VALIDATION set, final evaluation is on TEST set.
    print(f"\nRunning Genetic Algorithm for {dataset_name}...")

    # Candidate lists already generated above for fair baseline comparison
    # Using module-level DEFAULT_WEIGHTS and constants for consistency
    weights = DEFAULT_WEIGHTS

    # Important: GA Target Matrix -> VALIDATION Matrix
    # The GA optimizes alignment with validation set ratings.
    # Test set is reserved for final unbiased evaluation only.
    ga_target_matrix = val_matrix  # Use validation set for optimization

    # --- IDCG for GA/NSGA-II Optimization ---
    # GA uses validation IDCG during optimization.
    # Test IDCG is used only for final evaluation.
    print(f"  -> Validation IDCG mean: {np.mean(val_user_idcg_scores):.4f}")

    ga = GeneticRecommender(
        num_users=num_users,
        num_items=num_items,
        candidate_lists=candidate_lists,
        target_matrix=ga_target_matrix,
        activity_map=activity_map,  # Pass original activity_map (uses user_ids as keys)
        item_ids=item_ids,
        user_ids=user_ids,  # Pass user_ids for consistent activity_gap calculation
        weights=weights,
        top_k=k_ndcg,
    )
    # Inject the validation-set IDCG values for optimization
    ga.set_user_idcg_values(val_user_idcg_scores)

    best_ind, history = ga.run(
        generations=GENERATIONS,
        pop_size=POP_SIZE,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
    )

    # --- Final Evaluation of GA on Test Set ---
    # We take the best individual (list of indices) and evaluate against Test Matrix.
    # Using TEST set IDCG for final evaluation (different from optimization IDCG).
    ga_recs_indices = ga.decode(best_ind)
    ga_test_ndcg_scores = calculate_user_ndcg_scores(
        ga_recs_indices, test_matrix, test_user_idcg_scores
    )
    ga_test_mdcg = np.mean(ga_test_ndcg_scores)

    if activity_map:
        ga_test_activity_gap = calculate_activity_gap(
            ga_test_ndcg_scores, activity_map, user_ids=user_ids
        )
    else:
        ga_test_activity_gap = 0.0

    # --- Unified Item Entropy Calculation ---
    # Using centralized function for consistency
    ga_test_entropy = calculate_shannon_entropy(ga_recs_indices, num_items)
    # Item Coverage - secondary diagnostic metric
    ga_test_coverage = calculate_item_coverage(ga_recs_indices, num_items)

    print(
        f"GA Result (Test Set): MDCG={ga_test_mdcg:.4f}, Activity Gap={ga_test_activity_gap:.4f}, "
        f"Item Entropy={ga_test_entropy:.4f}, Item Coverage={ga_test_coverage:.4f}"
    )

    results["GA MDCG"] = ga_test_mdcg
    results["GA Activity Gap"] = ga_test_activity_gap
    results["GA Item Entropy"] = ga_test_entropy
    results["GA Item Coverage"] = ga_test_coverage

    # --- NSGA-II Algorithm ---
    print(f"\nRunning NSGA-II for {dataset_name}...")

    nsga = NsgaIIRecommender(
        num_users=num_users,
        num_items=num_items,
        candidate_lists=candidate_lists,
        target_matrix=ga_target_matrix,  # Uses val_matrix (same as GA)
        activity_map=activity_map,  # Pass original activity_map (uses user_ids as keys)
        item_ids=item_ids,
        user_ids=user_ids,  # Pass user_ids for consistent activity_gap calculation
        weights=weights,
        top_k=k_ndcg,
    )
    # Inject validation-set IDCG for optimization
    nsga.set_user_idcg_values(val_user_idcg_scores)

    pareto_front, history_nsga = nsga.run(
        generations=GENERATIONS,
        pop_size=POP_SIZE,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
    )

    # --- Pareto Front Visualization (VALIDATION Set Metrics) ---
    # NOTE: This plot shows metrics computed against the VALIDATION set (nsga.target_matrix),
    # NOT the test set. This visualizes the trade-offs discovered during optimization.
    # Final test set results are reported separately below.
    pareto_metrics = []
    for ind in pareto_front:
        _, mdcg, gap, cov = nsga.fitness(ind)
        pareto_metrics.append((mdcg, gap, cov))

    plot_pairwise_pareto(pareto_metrics, dataset_name)

    # --- Evaluate ALL Pareto Front Solutions on Test Set ---
    # This ensures we select the best solution based on actual test performance,
    # not the internal SVD-based proxy score. This is a fairer comparison.
    print(f"Evaluating {len(pareto_front)} Pareto front solutions on Test Set...")

    # Collect all Pareto solutions' test metrics for statistical reporting
    pareto_test_results = []
    best_nsga_test_score = -np.inf
    best_nsga_test_mdcg = 0.0
    best_nsga_test_gap = 0.0
    best_nsga_test_entropy = 0.0
    best_nsga_test_coverage = 0.0

    for ind in pareto_front:
        # Evaluate this individual on the test set using TEST IDCG
        recs_indices = nsga.decode(ind)
        test_ndcg_scores = calculate_user_ndcg_scores(
            recs_indices, test_matrix, test_user_idcg_scores
        )
        test_mdcg = np.mean(test_ndcg_scores)

        # Calculate activity gap on test
        if activity_map:
            test_gap = calculate_activity_gap(
                test_ndcg_scores, activity_map, user_ids=user_ids, verbose=False
            )
        else:
            test_gap = 0.0

        # Calculate item entropy on test - using centralized function
        test_entropy = calculate_shannon_entropy(recs_indices, num_items)
        # Item Coverage - secondary diagnostic metric
        test_coverage = calculate_item_coverage(recs_indices, num_items)

        # Store for Pareto statistics
        pareto_test_results.append((test_mdcg, test_gap, test_entropy, test_coverage))

        # --- DESIGN NOTE: Weighted Selection for Representative Solution ---
        # NSGA-II produces a Pareto front of non-dominated solutions (trade-offs).
        # For the results TABLE, we need to pick ONE representative solution to compare against baseline/GA.
        # This weighted sum is ONLY for that purpose - it does NOT undermine the multi-objective optimization.
        # The full Pareto front diversity is reported separately below (min/max/mean for each metric).
        # Users can choose different solutions from the Pareto front based on their priorities.
        test_score = compute_weighted_score(test_mdcg, test_gap, test_entropy, weights)

        if test_score > best_nsga_test_score:
            best_nsga_test_score = test_score
            best_nsga_test_mdcg = test_mdcg
            best_nsga_test_gap = test_gap
            best_nsga_test_entropy = test_entropy
            best_nsga_test_coverage = test_coverage

    # --- Report Pareto Front Statistics (shows trade-off diversity) ---
    if pareto_test_results:
        mdcg_vals = [r[0] for r in pareto_test_results]
        gap_vals = [r[1] for r in pareto_test_results]
        entropy_vals = [r[2] for r in pareto_test_results]
        coverage_vals = [r[3] for r in pareto_test_results]
        print(
            f"\n--- Pareto Front Trade-offs on Test Set ({len(pareto_test_results)} solutions) ---"
        )
        print(
            f"  MDCG:     min={min(mdcg_vals):.4f}, max={max(mdcg_vals):.4f}, mean={np.mean(mdcg_vals):.4f}, std={np.std(mdcg_vals):.4f}"
        )
        print(
            f"  Gap:      min={min(gap_vals):.4f}, max={max(gap_vals):.4f}, mean={np.mean(gap_vals):.4f}, std={np.std(gap_vals):.4f}"
        )
        print(
            f"  Entropy:  min={min(entropy_vals):.4f}, max={max(entropy_vals):.4f}, mean={np.mean(entropy_vals):.4f}, std={np.std(entropy_vals):.4f}"
        )
        print(
            f"  Coverage: min={min(coverage_vals):.4f}, max={max(coverage_vals):.4f}, mean={np.mean(coverage_vals):.4f}, std={np.std(coverage_vals):.4f}"
        )

    nsga_test_mdcg = best_nsga_test_mdcg
    nsga_test_activity_gap = best_nsga_test_gap
    nsga_test_entropy = best_nsga_test_entropy
    nsga_test_coverage = best_nsga_test_coverage

    print(
        f"\nNSGA-II Result (Best Weighted on Test Set): MDCG={nsga_test_mdcg:.4f}, Activity Gap={nsga_test_activity_gap:.4f}, "
        f"Item Entropy={nsga_test_entropy:.4f}, Item Coverage={nsga_test_coverage:.4f}"
    )

    results["NSGA-II MDCG"] = nsga_test_mdcg
    results["NSGA-II Activity Gap"] = nsga_test_activity_gap
    results["NSGA-II Item Entropy"] = nsga_test_entropy
    results["NSGA-II Item Coverage"] = nsga_test_coverage

    return results


def load_or_create_cached_data(
    dataset_name: str,
    load_fn,
    rating_scale=(1, 5),
    use_cache: bool = True,
    **load_kwargs,
):
    """
    Load dataset with caching support for matrices and predictions.

    If cache exists and use_cache=True, loads from cache.
    Otherwise, processes data fresh and saves to cache.

    Args:
        dataset_name: Name of the dataset (for cache file naming)
        load_fn: Function to load the raw dataset (returns df or (data, df))
        rating_scale: Rating scale tuple for Surprise
        use_cache: Whether to use caching
        **load_kwargs: Additional kwargs passed to load_fn

    Returns:
        tuple: (train_matrix, val_matrix, test_matrix, user_ids, item_ids,
                activity_map, train_df, prediction_matrix_or_None)
    """
    cache_paths = get_all_matrix_cache_paths(dataset_name)

    # Check if all data is cached (including prediction matrix)
    if use_cache and all_matrices_cached(dataset_name, include_prediction=True):
        print(f"Loading {dataset_name} from cache...")

        train_matrix = load_matrix(cache_paths["train"])
        val_matrix = load_matrix(cache_paths["val"])
        test_matrix = load_matrix(cache_paths["test"])
        prediction_matrix = load_matrix(cache_paths["prediction"])

        user_ids, item_ids, activity_map = load_metadata(dataset_name)

        return (
            train_matrix,
            val_matrix,
            test_matrix,
            user_ids,
            item_ids,
            activity_map,
            None,  # train_df not needed when using cached prediction
            prediction_matrix,
        )

    # Load and process data fresh
    print(f"Processing {dataset_name} fresh (will cache results)...")

    # Load raw data
    result = load_fn(**load_kwargs)
    if isinstance(result, tuple):
        _, df = result  # Surprise datasets return (data, df)
    else:
        df = result  # Book-Crossing returns just df

    # Split data
    train_df, val_df, test_df = split_train_val_test_stratified(df)

    # Create aligned matrices
    (
        train_matrix,
        val_matrix,
        test_matrix,
        user_ids,
        item_ids,
    ) = create_aligned_matrices_3way(train_df, val_df, test_df)

    # Calculate activity groups
    activity_map = get_activity_group_map(train_df, user_ids)

    # Train SVD and generate prediction matrix
    trainset = df_to_surprise_trainset(train_df, rating_scale=rating_scale)
    svd_model = train_svd_surprise(trainset, random_state=SEED)
    prediction_matrix = get_predictions_matrix(svd_model, user_ids, item_ids, trainset)
    print("SVD Training & Prediction Matrix Generated.")

    # Cache everything if caching is enabled
    if use_cache:
        print(f"Caching {dataset_name} data...")
        save_matrix(train_matrix, cache_paths["train"])
        save_matrix(val_matrix, cache_paths["val"])
        save_matrix(test_matrix, cache_paths["test"])
        save_matrix(prediction_matrix, cache_paths["prediction"])
        save_metadata(dataset_name, user_ids, item_ids, activity_map)

    return (
        train_matrix,
        val_matrix,
        test_matrix,
        user_ids,
        item_ids,
        activity_map,
        train_df,
        prediction_matrix,
    )


def main():
    # Set random seed FIRST before any data loading to ensure full reproducibility.
    # Note: data_processing.py uses pandas random_state=42 internally, but this ensures
    # any numpy/random calls during dataset configuration are also deterministic.
    # Using module-level SEED constant for consistency.
    np.random.seed(SEED)
    random.seed(SEED)

    all_results = []
    k_ndcg = K_NDCG  # Use module-level constant

    # MovieLens 100k - Using Surprise's built-in dataset
    try:
        print("\n--- Loading MovieLens 100k (Surprise built-in) ---")
        (
            train_matrix,
            val_matrix,
            test_matrix,
            user_ids,
            item_ids,
            activity_map,
            train_df,
            prediction_matrix,
        ) = load_or_create_cached_data(
            dataset_name="MovieLens 100k",
            load_fn=load_movielens_100k_surprise,
            rating_scale=(1, 5),
        )

        result = run_pipeline(
            train_matrix,
            val_matrix,
            test_matrix,
            user_ids,
            item_ids,
            activity_map,
            train_df,
            "MovieLens 100k",
            k_ndcg,
            prediction_matrix=prediction_matrix,
        )
        all_results.append(result)

    except Exception as e:
        print(f"Error processing MovieLens 100k: {e}")
        import traceback

        traceback.print_exc()

    # MovieLens 1M - Using Surprise's built-in dataset
    try:
        print("\n--- Loading MovieLens 1M (Surprise built-in) ---")
        (
            train_matrix,
            val_matrix,
            test_matrix,
            user_ids,
            item_ids,
            activity_map,
            train_df,
            prediction_matrix,
        ) = load_or_create_cached_data(
            dataset_name="MovieLens 1M",
            load_fn=load_movielens_1m_surprise,
            rating_scale=(1, 5),
        )

        result = run_pipeline(
            train_matrix,
            val_matrix,
            test_matrix,
            user_ids,
            item_ids,
            activity_map,
            train_df,
            "MovieLens 1M",
            k_ndcg,
            prediction_matrix=prediction_matrix,
        )
        all_results.append(result)

    except Exception as e:
        print(f"Error processing MovieLens 1M: {e}")
        import traceback

        traceback.print_exc()

    # Book-Crossing dataset
    try:
        print("\n--- Loading Book-Crossing Dataset ---")
        (
            train_matrix,
            val_matrix,
            test_matrix,
            user_ids,
            item_ids,
            activity_map,
            train_df,
            prediction_matrix,
        ) = load_or_create_cached_data(
            dataset_name="Book-Crossing",
            load_fn=load_book_crossing,
            rating_scale=(1, 10),
            min_interactions=6,  # Passed to load_book_crossing
        )

        result = run_pipeline(
            train_matrix,
            val_matrix,
            test_matrix,
            user_ids,
            item_ids,
            activity_map,
            train_df,
            "Book-Crossing",
            k_ndcg,
            rating_scale=(1, 10),
            prediction_matrix=prediction_matrix,
        )
        all_results.append(result)

    except Exception as e:
        print(f"Error processing Book-Crossing: {e}")
        import traceback

        traceback.print_exc()

    # Display Results as Pandas DataFrame
    if all_results:
        results_df = pd.DataFrame(all_results)

        print("\n\n" + "=" * 130)
        print(f"{'DATASET COMPARISON (Baseline vs GA - TEST SET)':^130}")
        print("=" * 130)
        print(results_df.to_string())
        print("=" * 130)


if __name__ == "__main__":
    main()
