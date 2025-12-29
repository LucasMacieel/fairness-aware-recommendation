import matplotlib.pyplot as plt
import os
import itertools


def plot_pairwise_pareto(solutions, dataset_name, output_dir="plots"):
    """
    Plots pairwise comparisons of objectives for the NSGA-II Pareto front.

    solutions: list of tuples (mdcg, activity_gap, item_coverage)
    dataset_name: str, name of the dataset for titling/saving
    output_dir: str, directory to save the plots
    """
    if not solutions:
        print("No solutions to plot.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    objectives = {
        "MDCG": [s[0] for s in solutions],
        "Activity Gap": [s[1] for s in solutions],
        "Item Coverage": [s[2] for s in solutions],
    }

    obj_names = list(objectives.keys())
    pairs = list(itertools.combinations(obj_names, 2))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Pareto Front Pairwise Objectives - {dataset_name}", fontsize=16)

    for i, (x_name, y_name) in enumerate(pairs):
        ax = axes[i]
        x_vals = objectives[x_name]
        y_vals = objectives[y_name]

        ax.scatter(x_vals, y_vals, c="blue", alpha=0.6, edgecolors="black")
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_title(f"{x_name} vs {y_name}")
        ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"{dataset_name.replace(' ', '_').lower()}_pareto_pairwise.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()

    print(f"Saved pairwise Pareto plots to {filepath}")
