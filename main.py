
import numpy as np
import random
from data_processing import (
    get_movielens_data_numpy,
    get_movielens_gender_map,
    get_post_data_numpy,
    get_post_gender_map,
    get_sushi_data_numpy,
    get_sushi_gender_map,
)
from metrics import (
    calculate_ndcg_scores,
    calculate_mean_ndcg,
    calculate_unfairness_gap,
    calculate_item_coverage
)
from recommender import perform_svd
from genetic_recommender import GeneticRecommender

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def run_pipeline(matrix, user_ids, item_ids, gender_map, dataset_name, k_ndcg=10):
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
    baseline_mean_ndcg = calculate_mean_ndcg(baseline_ndcg_scores)
    
    # Fairness
    if gender_map:
        baseline_gap = calculate_unfairness_gap(baseline_ndcg_scores, user_ids, gender_map)
    else:
        baseline_gap = 0.0 # Default if no map
        
    # Item Coverage
    # calculate_item_coverage expects prediction_matrix to select top K.
    baseline_cov_count, baseline_cov_ratio = calculate_item_coverage(
        prediction_matrix, matrix, item_ids, k=k_ndcg
    )

    results["Baseline NDCG"] = baseline_mean_ndcg
    results["Baseline Gap"] = baseline_gap
    results["Baseline Coverage"] = baseline_cov_ratio
    
    print(f"Baseline: NDCG={baseline_mean_ndcg:.4f}, Gap={baseline_gap:.4f}, Cov={baseline_cov_ratio:.4f}")

    # --- Genetic Algorithm ---
    print(f"\nRunning Genetic Algorithm for {dataset_name}...")
    
    num_users, num_items = matrix.shape
    CANDIDATE_SIZE = 100
    weights = {'ndcg': 1.0, 'gap': 1.0, 'coverage': 1.0}
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
            ga_gender_map[idx] = gender_map.get(uid, 'Unknown')
    else:
        for idx in range(num_users):
            ga_gender_map[idx] = 'Unknown'

    ga = GeneticRecommender(
        num_users=num_users, 
        num_items=num_items, 
        candidate_lists=candidate_lists, 
        target_matrix=matrix, 
        gender_map=ga_gender_map, 
        item_ids=range(num_users), # Not heavily used inside GA other than map keying if implemented differently
        weights=weights,
        top_k=k_ndcg
    )
    
    best_ind, history = ga.run(generations=GENERATIONS, pop_size=POP_SIZE)
    
    final_score, ga_ndcg, ga_gap, ga_cov = ga.fitness(best_ind)
    
    print(f"GA Result: NDCG={ga_ndcg:.4f}, Gap={ga_gap:.4f}, Cov={ga_cov:.4f}")
    
    results["GA NDCG"] = ga_ndcg
    results["GA Gap"] = ga_gap
    results["GA Coverage"] = ga_cov
    
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
    k_ndcg = 10

    for ds in datasets:
        try:
            print(f"\n--- Loading {ds['name']} ---")
            matrix, user_ids, item_ids = ds["loader"]()
            gender_map = ds["gender_loader"]()

            result = run_pipeline(matrix, user_ids, item_ids, gender_map, ds["name"], k_ndcg)
            all_results.append(result)

        except FileNotFoundError:
            print(f"Skipping {ds['name']}: Data file not found.")
        except Exception as e:
            print(f"Error processing {ds['name']}: {e}")

    # Print Comparison Table
    print("\n\n" + "=" * 120)
    print(f"{'DATASET COMPARISON (Baseline vs GA)':^120}")
    print("=" * 120)

    # Define headers
    headers = [
        "Dataset",
        "NDCG (Base)", "NDCG (GA)",
        "Gap (Base)", "Gap (GA)",
        "Cov (Base)", "Cov (GA)"
    ]

    # Print headers
    # Dataset | NDCG B | NDCG G | Gap B | Gap G | Cov B | Cov G
    header_row = f"{headers[0]:<15} | {headers[1]:<10} | {headers[2]:<10} | {headers[3]:<10} | {headers[4]:<10} | {headers[5]:<10} | {headers[6]:<10}"
    print(header_row)
    print("-" * len(header_row))

    # Print rows
    for res in all_results:
        # Format metrics
        row = (
            f"{res['Dataset']:<15} | "
            f"{res['Baseline NDCG']:.4f}     | {res['GA NDCG']:.4f}     | "
            f"{res['Baseline Gap']:.4f}     | {res['GA Gap']:.4f}     | "
            f"{res['Baseline Coverage']:.4f}     | {res['GA Coverage']:.4f}"
        )
        print(row)
    print("=" * 120)


if __name__ == "__main__":
    main()
