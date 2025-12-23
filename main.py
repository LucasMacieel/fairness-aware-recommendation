
import numpy as np
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

def run_pipeline(matrix, user_ids, item_ids, gender_map, dataset_name, k_ndcg=5):
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
        prediction_matrix, matrix, item_ids, k=k_ndcg
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
    k_ndcg = 5

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
    print("\n\n" + "=" * 90)
    print(f"{'DATASET COMPARISON':^90}")
    print("=" * 90)

    # Define headers
    headers = [
        "Dataset",
        "Users",
        "Items",
        f"NDCG@{k_ndcg}",
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

        row = f"{res['Dataset']:<15} | {res['Users']:<8} | {res['Items']:<8} | {res[f'NDCG@{k_ndcg}']:.4f}     | {ug_str:<15} | {cov_str:<15}"
        print(row)
    print("=" * 90)


if __name__ == "__main__":
    main()
