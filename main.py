import numpy as np
import random
import pandas as pd
from data_processing import (
    get_movielens_data_numpy,
    get_movielens_gender_map,
    get_post_data_numpy,
    get_post_gender_map,
    get_electronics_data_numpy,
    get_electronics_gender_map,
)
from metrics import (
    calculate_ndcg_scores,
    calculate_mdcg,
    calculate_gender_gap,
    calculate_item_coverage,
)
from recommender import perform_svd
from genetic_recommender import GeneticRecommender, NsgaIIRecommender
from utils.plotter import plot_pairwise_pareto


def run_pipeline(matrix, user_ids, item_ids, gender_map, dataset_name, k_ndcg=10):
    # Set random seed for reproducibility
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)

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

    # --- Baseline Evaluation ---
    print(f"\nEvaluating Baseline Recommender (NDCG@{k_ndcg})...")

    baseline_ndcg_scores = calculate_ndcg_scores(matrix, prediction_matrix, k=k_ndcg)
    baseline_mdcg = calculate_mdcg(baseline_ndcg_scores)

    # Fairness
    if gender_map:
        baseline_gender_gap = calculate_gender_gap(
            baseline_ndcg_scores, user_ids, gender_map
        )
    else:
        baseline_gender_gap = 0.0  # Default if no map

    # Item Coverage
    # calculate_item_coverage expects prediction_matrix to select top K.
    baseline_cov_count, baseline_item_coverage = calculate_item_coverage(
        prediction_matrix, matrix, item_ids, k=k_ndcg
    )

    results["Baseline MDCG"] = baseline_mdcg
    results["Baseline Gender Gap"] = baseline_gender_gap
    results["Baseline Item Coverage"] = baseline_item_coverage

    print(
        f"Baseline: MDCG={baseline_mdcg:.4f}, Gender Gap={baseline_gender_gap:.4f}, Item Coverage={baseline_item_coverage:.4f}"
    )

    # --- Genetic Algorithm ---
    print(f"\nRunning Genetic Algorithm for {dataset_name}...")

    num_users, num_items = matrix.shape
    CANDIDATE_SIZE = 100
    weights = {"mdcg": 1.0, "gender_gap": 1.0, "item_coverage": 1.0}
    POP_SIZE = 10
    GENERATIONS = 5

    # Generate Candidate Lists
    print(f"Generating Top-{CANDIDATE_SIZE} candidates...")
    candidate_lists = np.zeros((num_users, CANDIDATE_SIZE), dtype=int)

    for u in range(num_users):
        scores = prediction_matrix[u]
        # Note: We do NOT filter rated items to allow ground truth access (Oracular Re-ranking setting for offline eval)
        top_indices = np.argsort(scores)[::-1][:CANDIDATE_SIZE]
        candidate_lists[u] = top_indices

    # Prepare GA Gender Map (index-based)
    ga_gender_map = {}
    if gender_map:
        for idx, uid in enumerate(user_ids):
            ga_gender_map[idx] = gender_map.get(uid, "Unknown")
    else:
        for idx in range(num_users):
            ga_gender_map[idx] = "Unknown"

    ga = GeneticRecommender(
        num_users=num_users,
        num_items=num_items,
        candidate_lists=candidate_lists,
        target_matrix=matrix,
        gender_map=ga_gender_map,
        item_ids=range(
            num_users
        ),  # Not heavily used inside GA other than map keying if implemented differently
        weights=weights,
        top_k=k_ndcg,
    )

    best_ind, history = ga.run(generations=GENERATIONS, pop_size=POP_SIZE)

    final_score, ga_mdcg, ga_gender_gap, ga_item_coverage = ga.fitness(best_ind)

    print(
        f"GA Result: MDCG={ga_mdcg:.4f}, Gender Gap={ga_gender_gap:.4f}, Item Coverage={ga_item_coverage:.4f}"
    )

    results["GA MDCG"] = ga_mdcg
    results["GA Gender Gap"] = ga_gender_gap
    results["GA Item Coverage"] = ga_item_coverage

    # --- NSGA-II Algorithm ---
    print(f"\nRunning NSGA-II for {dataset_name}...")

    nsga = NsgaIIRecommender(
        num_users=num_users,
        num_items=num_items,
        candidate_lists=candidate_lists,
        target_matrix=matrix,
        gender_map=ga_gender_map,
        item_ids=range(num_users),
        weights=weights,
        top_k=k_ndcg,
    )

    # NSGA-II returns a population (Pareto front)
    pareto_front, history_nsga = nsga.run(generations=GENERATIONS, pop_size=POP_SIZE)

    # Extract metrics for plotting
    pareto_metrics = []
    for ind in pareto_front:
        _, mdcg, gap, cov = nsga.fitness(ind)
        pareto_metrics.append((mdcg, gap, cov))

    plot_pairwise_pareto(pareto_metrics, dataset_name)

    # Select best solution from Pareto front based on the same weighted score for fair comparison
    best_nsga_score = -np.inf
    best_nsga_ind = None
    best_nsga_metrics = None

    for ind in pareto_front:
        score, mdcg, gap, cov = nsga.fitness(ind)
        if score > best_nsga_score:
            best_nsga_score = score
            best_nsga_ind = ind
            best_nsga_metrics = (score, mdcg, gap, cov)

    print(
        f"NSGA-II Result (Best Weighted): MDCG={best_nsga_metrics[1]:.4f}, Gender Gap={best_nsga_metrics[2]:.4f}, Item Coverage={best_nsga_metrics[3]:.4f}"
    )

    results["NSGA-II MDCG"] = best_nsga_metrics[1]
    results["NSGA-II Gender Gap"] = best_nsga_metrics[2]
    results["NSGA-II Item Coverage"] = best_nsga_metrics[3]

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

    # 4. Electronics Data
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
            matrix, user_ids, item_ids = ds["loader"]()
            gender_map = ds["gender_loader"]()

            result = run_pipeline(
                matrix, user_ids, item_ids, gender_map, ds["name"], k_ndcg
            )
            all_results.append(result)

        except FileNotFoundError:
            print(f"Skipping {ds['name']}: Data file not found.")
    # Display Results as Pandas DataFrame
    if all_results:
        df = pd.DataFrame(all_results)

        print("\n\n" + "=" * 130)
        print(f"{'DATASET COMPARISON (Baseline vs GA)':^130}")
        print("=" * 130)
        print(df.to_string())
        print("=" * 130)


if __name__ == "__main__":
    main()
