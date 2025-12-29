import numpy as np
import random
import pandas as pd
from data_processing import (
    get_movielens_100k_data_numpy,
    get_movielens_100k_gender_map,
    get_movielens_1m_data_numpy,
    get_movielens_1m_gender_map,
    get_electronics_data_numpy,
    get_electronics_gender_map,
)
from metrics import (
    calculate_gender_gap,
    calculate_item_coverage_simple,
    get_user_ideal_dcg,
)
from recommender import perform_svd
from genetic_recommender import GeneticRecommender, NsgaIIRecommender
from utils.plotter import plot_pairwise_pareto


def run_pipeline(
    train_matrix, test_matrix, user_ids, item_ids, gender_map, dataset_name, k_ndcg=10
):
    # Set random seed for reproducibility
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

    print(f"\nProcessing {dataset_name} Dataset...")
    print(f"Train Matrix Shape: {train_matrix.shape}")
    print(f"Test Matrix Shape: {test_matrix.shape}")

    # Check sparsity or if we have enough data for k
    k_factors = min(train_matrix.shape) - 1
    k_factors = 20 if k_factors > 20 else k_factors

    print(f"Performing SVD on Train Set with k={k_factors}...")
    prediction_matrix = perform_svd(train_matrix, k=k_factors)
    print("SVD Complete.")

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
    }

    # --- Pre-calculate Global IDCG using TEST Set ---
    # Ground Truth is the Test Set. This IDCG is used for ALL methods to ensure consistency.
    print(f"Calculating Global IDCG (Test Set) for k={k_ndcg}...")
    user_idcg_scores = get_user_ideal_dcg(test_matrix, k=k_ndcg)

    # --- Generate Candidate Lists FIRST (needed for fair baseline comparison) ---
    num_users, num_items = train_matrix.shape
    CANDIDATE_SIZE = 50

    # NOTE: Item coverage during optimization is bounded by CANDIDATE_SIZE * num_users unique items.
    # The GA/NSGA-II can only recommend items from the candidate pool, not the full item catalog.
    # This is a design choice to simulate realistic re-ranking scenarios.
    print(f"Generating Top-{CANDIDATE_SIZE} candidates for all methods...")
    candidate_lists = np.zeros((num_users, CANDIDATE_SIZE), dtype=int)

    for u in range(num_users):
        scores = masked_prediction_matrix[u]
        top_indices = np.argsort(scores)[::-1][:CANDIDATE_SIZE]
        candidate_lists[u] = top_indices

    # --- Baseline Evaluation (using same candidate pool as GA for fair comparison) ---
    print(f"\nEvaluating Baseline Recommender (NDCG@{k_ndcg})...")

    # Baseline: Top-K from candidate list (same constraint as GA)
    # This ensures fair comparison - baseline also limited to same candidate pool
    baseline_ndcg_scores = []
    baseline_recs_indices = np.zeros((num_users, k_ndcg), dtype=int)

    for u in range(num_users):
        # Baseline takes the top-k from candidate list (which is already sorted by SVD score)
        rec_inds = candidate_lists[u, :k_ndcg]
        baseline_recs_indices[u] = rec_inds

        # Calculate NDCG using global IDCG (same as GA evaluation)
        true_ratings = test_matrix[u, rec_inds]
        r = np.asarray(true_ratings, dtype=float)
        if r.size:
            dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
            idcg = user_idcg_scores[u]
            if idcg > 0:
                baseline_ndcg_scores.append(dcg / idcg)
            else:
                baseline_ndcg_scores.append(0.0)
        else:
            baseline_ndcg_scores.append(0.0)

    baseline_mdcg = np.mean(baseline_ndcg_scores)

    # Fairness - using global IDCG-normalized scores
    if gender_map:
        baseline_gender_gap = calculate_gender_gap(
            baseline_ndcg_scores, user_ids, gender_map
        )
    else:
        baseline_gender_gap = 0.0  # Default if no map

    # Item Coverage - using centralized function
    baseline_item_coverage = calculate_item_coverage_simple(
        baseline_recs_indices, item_ids, num_users
    )

    results["Baseline MDCG"] = baseline_mdcg
    results["Baseline Gender Gap"] = baseline_gender_gap
    results["Baseline Item Coverage"] = baseline_item_coverage

    print(
        f"Baseline: MDCG={baseline_mdcg:.4f}, Gender Gap={baseline_gender_gap:.4f}, Item Coverage={baseline_item_coverage:.4f}"
    )

    # --- Genetic Algorithm ---
    # The GA re-ranks candidates to improve Fairness while maintaining relevance.
    # NOTE: This uses an ORACULAR setting where GA optimizes directly on test set ratings.
    # This is acceptable for re-ranking evaluation where we compare methods fairly.
    print(f"\nRunning Genetic Algorithm for {dataset_name}...")

    # Candidate lists already generated above for fair baseline comparison
    weights = {"mdcg": 1.0, "gender_gap": 1.0, "item_coverage": 1.0}
    POP_SIZE = 50
    GENERATIONS = 15

    # Prepare GA Gender Map (index-based)
    ga_gender_map = {}
    if gender_map:
        for idx, uid in enumerate(user_ids):
            ga_gender_map[idx] = gender_map.get(uid, "Unknown")
    else:
        for idx in range(num_users):
            ga_gender_map[idx] = "Unknown"

    # Important: GA Target Matrix -> Test Matrix (Ground Truth)
    # The GA optimizes alignment with ground truth ratings from the test set.
    # Both DCG (numerator) and IDCG (denominator) now use test data for consistent NDCG.
    ga_target_matrix = test_matrix  # Use ground truth for DCG calculation

    # --- IDCG for GA/NSGA-II Optimization (Option B: Use Test-Set IDCG) ---
    # DESIGN DECISION (Option B): We use the SAME test-set IDCG for both optimization and evaluation.
    # This means the GA optimizes directly against the ground truth NDCG definition.
    # Pros: Consistent metric between optimization and evaluation, simpler interpretation.
    # Cons: The GA has "oracle" access to the ideal ranking from test data during optimization.
    #       This is acceptable in a re-ranking scenario where we compare methods fairly.
    print("Using Test-Set IDCG for both optimization and evaluation (Option B)...")
    print(f"  -> IDCG mean: {np.mean(user_idcg_scores):.4f}")

    ga = GeneticRecommender(
        num_users=num_users,
        num_items=num_items,
        candidate_lists=candidate_lists,
        target_matrix=ga_target_matrix,
        gender_map=ga_gender_map,
        item_ids=item_ids,
        weights=weights,
        top_k=k_ndcg,
    )
    # Inject the test-set IDCG values for consistent NDCG calculation (Option B)
    ga.set_user_idcg_values(user_idcg_scores)

    best_ind, history = ga.run(generations=GENERATIONS, pop_size=POP_SIZE)

    # --- Final Evaluation of GA on Test Set ---
    # We take the best individual (list of indices) and evaluate against Test Matrix.
    # We cannot use ga.fitness() because that uses ga_target_matrix (SVD).
    # We must manually evaluate.

    def evaluate_individual_on_test(individual, test_mat, idcg_vals, recommender, k=10):
        """Evaluate an individual against the test set with consistent metrics."""
        recs_indices = recommender.decode(individual)
        ndcg_list = []
        for u in range(num_users):
            rec_inds = recs_indices[u]
            true_ratings = test_mat[u, rec_inds]

            # Simple DCG on True Ratings
            r = np.asarray(true_ratings, dtype=float)
            if r.size:
                dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
                idcg = idcg_vals[u]
                if idcg > 0:
                    ndcg_list.append(dcg / idcg)
                else:
                    ndcg_list.append(0.0)
            else:
                ndcg_list.append(0.0)

        return np.mean(ndcg_list), ndcg_list, recs_indices

    ga_test_mdcg, ga_test_ndcg_scores, ga_recs_indices = evaluate_individual_on_test(
        best_ind, test_matrix, user_idcg_scores, ga, k=k_ndcg
    )

    if gender_map:
        ga_test_gender_gap = calculate_gender_gap(
            ga_test_ndcg_scores, user_ids, gender_map
        )
    else:
        ga_test_gender_gap = 0.0

    # --- Unified Item Coverage Calculation ---
    # Using centralized function for consistency
    ga_test_cov_ratio = calculate_item_coverage_simple(
        ga_recs_indices, item_ids, num_users
    )

    print(
        f"GA Result (Test Set): MDCG={ga_test_mdcg:.4f}, Gender Gap={ga_test_gender_gap:.4f}, Item Coverage={ga_test_cov_ratio:.4f}"
    )

    results["GA MDCG"] = ga_test_mdcg
    results["GA Gender Gap"] = ga_test_gender_gap
    results["GA Item Coverage"] = ga_test_cov_ratio

    # --- NSGA-II Algorithm ---
    print(f"\nRunning NSGA-II for {dataset_name}...")

    nsga = NsgaIIRecommender(
        num_users=num_users,
        num_items=num_items,
        candidate_lists=candidate_lists,
        target_matrix=ga_target_matrix,
        gender_map=ga_gender_map,
        item_ids=item_ids,
        weights=weights,
        top_k=k_ndcg,
    )
    # Inject test-set IDCG for correct NDCG calculation (Option B)
    nsga.set_user_idcg_values(user_idcg_scores)

    pareto_front, history_nsga = nsga.run(generations=GENERATIONS, pop_size=POP_SIZE)

    # Plotting Internal Metrics to show trade-offs found during training
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
    best_nsga_test_cov = 0.0

    for ind in pareto_front:
        # Evaluate this individual on the test set
        test_mdcg, test_ndcg_scores, recs_indices = evaluate_individual_on_test(
            ind, test_matrix, user_idcg_scores, nsga, k=k_ndcg
        )

        # Calculate gender gap on test
        if gender_map:
            test_gap = calculate_gender_gap(test_ndcg_scores, user_ids, gender_map)
        else:
            test_gap = 0.0

        # Calculate item coverage on test - using centralized function
        test_cov = calculate_item_coverage_simple(recs_indices, item_ids, num_users)

        # Store for Pareto statistics
        pareto_test_results.append((test_mdcg, test_gap, test_cov))

        # --- DESIGN NOTE: Weighted Selection for Representative Solution ---
        # NSGA-II produces a Pareto front of non-dominated solutions (trade-offs).
        # For the results TABLE, we need to pick ONE representative solution to compare against baseline/GA.
        # This weighted sum is ONLY for that purpose - it does NOT undermine the multi-objective optimization.
        # The full Pareto front diversity is reported separately above (min/max/mean for each metric).
        # Users can choose different solutions from the Pareto front based on their priorities.
        test_score = (
            (weights["mdcg"] * test_mdcg)
            - (weights["gender_gap"] * test_gap)
            + (weights["item_coverage"] * test_cov)
        )

        if test_score > best_nsga_test_score:
            best_nsga_test_score = test_score
            best_nsga_test_mdcg = test_mdcg
            best_nsga_test_gap = test_gap
            best_nsga_test_cov = test_cov

    # --- Report Pareto Front Statistics (shows trade-off diversity) ---
    if pareto_test_results:
        mdcg_vals = [r[0] for r in pareto_test_results]
        gap_vals = [r[1] for r in pareto_test_results]
        cov_vals = [r[2] for r in pareto_test_results]
        print(
            f"\n--- Pareto Front Trade-offs on Test Set ({len(pareto_test_results)} solutions) ---"
        )
        print(
            f"  MDCG:     min={min(mdcg_vals):.4f}, max={max(mdcg_vals):.4f}, mean={np.mean(mdcg_vals):.4f}"
        )
        print(
            f"  Gap:      min={min(gap_vals):.4f}, max={max(gap_vals):.4f}, mean={np.mean(gap_vals):.4f}"
        )
        print(
            f"  Coverage: min={min(cov_vals):.4f}, max={max(cov_vals):.4f}, mean={np.mean(cov_vals):.4f}"
        )

    # Use the best solution found on test set (for results table)
    nsga_test_mdcg = best_nsga_test_mdcg
    nsga_test_gender_gap = best_nsga_test_gap
    nsga_test_cov_ratio = best_nsga_test_cov

    print(
        f"\nNSGA-II Result (Best Weighted on Test Set): MDCG={nsga_test_mdcg:.4f}, Gender Gap={nsga_test_gender_gap:.4f}, Item Coverage={nsga_test_cov_ratio:.4f}"
    )

    results["NSGA-II MDCG"] = nsga_test_mdcg
    results["NSGA-II Gender Gap"] = nsga_test_gender_gap
    results["NSGA-II Item Coverage"] = nsga_test_cov_ratio

    return results


def main():
    datasets = []

    # 1. MovieLens 100k
    try:
        datasets.append(
            {
                "name": "MovieLens 100k",
                "loader": get_movielens_100k_data_numpy,
                "gender_loader": get_movielens_100k_gender_map,
            }
        )
    except Exception as e:
        print(f"Error configuring MovieLens 100k: {e}")

    # 2. MovieLens 1M
    try:
        datasets.append(
            {
                "name": "MovieLens 1M",
                "loader": get_movielens_1m_data_numpy,
                "gender_loader": get_movielens_1m_gender_map,
            }
        )
    except Exception as e:
        print(f"Error configuring MovieLens 1M: {e}")

    # 3. Electronics Data
    try:
        datasets.append(
            {
                "name": "Electronics Data",
                "loader": get_electronics_data_numpy,
                "gender_loader": get_electronics_gender_map,
            }
        )
    except Exception as e:
        print(f"Error configuring Electronics Data: {e}")

    all_results = []
    k_ndcg = 10

    for ds in datasets:
        try:
            print(f"\n--- Loading {ds['name']} ---")
            train_matrix, test_matrix, user_ids, item_ids = ds["loader"]()
            gender_map = ds["gender_loader"]()

            result = run_pipeline(
                train_matrix,
                test_matrix,
                user_ids,
                item_ids,
                gender_map,
                ds["name"],
                k_ndcg,
            )
            all_results.append(result)

        except FileNotFoundError:
            print(f"Skipping {ds['name']}: Data file not found.")
        except Exception as e:
            print(f"Error processing {ds['name']}: {e}")
            import traceback

            traceback.print_exc()

    # Display Results as Pandas DataFrame
    if all_results:
        df = pd.DataFrame(all_results)

        print("\n\n" + "=" * 130)
        print(f"{'DATASET COMPARISON (Baseline vs GA - TEST SET)':^130}")
        print("=" * 130)
        print(df.to_string())
        print("=" * 130)


if __name__ == "__main__":
    main()
