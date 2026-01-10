# Fairness-Aware Recommendation System

A research project implementing fairness-aware collaborative filtering using genetic algorithms (GA) and multi-objective optimization (NSGA-II) to balance recommendation accuracy with user fairness and item diversity.

## 🎯 Overview

This project addresses the **activity bias problem** in recommender systems, where highly active users often receive better recommendations than less active users. By combining Singular Value Decomposition (SVD) predictions with evolutionary optimization, the system re-ranks recommendations to achieve a better trade-off between:

- **Accuracy** (NDCG - Normalized Discounted Cumulative Gain)
- **User Fairness** (Activity Gap - difference between active and inactive user groups)
- **Diversity** (Shannon Entropy - distribution of items across recommendations)

## 📊 Datasets

The system supports multiple recommendation datasets:

| Dataset | Rating Scale | Min Interactions |
|---------|--------------|------------------|
| MovieLens 1M | 1-5 | Default |
| Book-Crossing | 1-10 | 6 (k-core) |
| Amazon Video Games | 1-5 | 10 (k-core) |
| Amazon Digital Music | 1-5 | 6 (k-core) |

Data files should be placed in the `data/` directory:
- `book_crossing.csv`
- `digital_music.csv`
- `video_games.csv`

## 🏗️ Architecture

```
fairness-aware-recommendation/
├── main.py                  # Main pipeline orchestrator
├── data_processing.py       # Dataset loading, splitting, and preprocessing
├── recommender.py           # SVD training and prediction matrix generation
├── genetic_recommender.py   # GA and NSGA-II optimization algorithms
├── metrics.py               # Evaluation metrics (NDCG, Activity Gap, Entropy)
├── cache.py                 # Caching system for matrices and metadata
├── process.py               # Utility scripts for dataset conversion
├── utils/
│   └── plotter.py           # Pareto front visualization
├── data/                    # Dataset files (CSV)
├── cache/                   # Cached matrices and metadata
└── plots/                   # Generated visualization outputs
```

## 🔧 Installation

### Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
- numpy
- pandas
- matplotlib
- scikit-surprise

### Python Version
Python 3.8+ recommended

## 🚀 Usage

### Running the Full Pipeline

```bash
python main.py
```

This will:
1. Load and preprocess all configured datasets
2. Train SVD models for rating prediction
3. Generate candidate recommendation lists
4. Run Genetic Algorithm optimization
5. Run NSGA-II multi-objective optimization
6. Evaluate all methods on held-out test data
7. Generate Pareto front visualizations
8. Output comparison tables

### Pipeline Configuration

Key parameters in `main.py`:

```python
SEED = 42              # Random seed for reproducibility
CANDIDATE_SIZE = 100   # Candidates per user for re-ranking
POP_SIZE = 100         # Population size for GA/NSGA-II
GENERATIONS = 10       # Number of evolutionary generations
K_NDCG = 10            # Top-K for evaluation
CROSSOVER_RATE = 0.8   # Genetic crossover probability
MUTATION_RATE = 0.2    # Genetic mutation probability
```

## 📈 Methodology

### 1. Data Processing
- **K-Core Filtering**: Ensures minimum interactions per user AND item
- **Stratified Split**: 60% train / 20% validation / 20% test (user-level preservation)
- **Matrix Alignment**: Consistent user-item matrices across all splits

### 2. SVD Prediction
- Uses [scikit-surprise](https://surpriselib.com/) for matrix factorization
- Generates a dense prediction matrix for all user-item pairs
- Training items are masked to prevent re-recommendation

### 3. Candidate Generation
- Top-100 candidates per user based on SVD predictions
- Shared candidate pool across all methods for fair comparison

### 4. Optimization Algorithms

#### Genetic Algorithm (GA)
- **Weighted single-objective** optimization
- Fitness: `w_mdcg * MDCG - w_gap * Activity_Gap + w_entropy * Entropy`
- Operators: Order Crossover (OX), Insert Mutation, Tournament Selection

#### NSGA-II
- **True multi-objective** optimization using Pareto dominance
- No weights during optimization—discovers trade-off surface
- Returns Pareto front of non-dominated solutions
- Final solution selected using weighted criteria for comparison

### 5. Evaluation
- **Candidate-constrained IDCG**: Normalization based on achievable ideal within candidate pool
- **Test set evaluation**: All methods evaluated on held-out test data
- **Pareto front statistics**: Min/max/mean/std for all objectives

## 📐 Metrics

| Metric | Description | Optimization Goal |
|--------|-------------|-------------------|
| **MDCG** | Mean Discounted Cumulative Gain | Maximize |
| **Activity Gap** | \|Avg(NDCG_active) - Avg(NDCG_inactive)\| | Minimize |
| **Shannon Entropy** | Normalized entropy of item distribution | Maximize |

### Activity Classification
- **Active Users**: Top 5% by interaction count
- **Inactive Users**: Bottom 95% by interaction count

## 🎨 Visualizations

The system generates Pareto front visualizations showing trade-offs between objectives:
- MDCG vs Activity Gap
- MDCG vs Item Entropy
- Activity Gap vs Item Entropy

Plots are saved to the `plots/` directory.

## ⚡ Caching

The system automatically caches:
- Train/Validation/Test matrices
- SVD prediction matrices
- User and item ID mappings
- Activity group classifications

Cache files are stored in `cache/` as `.npy` and `.json` files.

To force a fresh run, delete the cache files or set `use_cache=False`.

## 📝 Output Example

```
=== DATASET COMPARISON (Baseline vs GA - TEST SET) ===
Dataset        Users   Items   Density  Baseline MDCG  GA MDCG  NSGA-II MDCG  ...
MovieLens 1M   6040    3706    0.0447   0.3521         0.3498   0.3512        ...
Book-Crossing  2531    8234    0.0124   0.2845         0.2812   0.2831        ...
...
```

## 🔬 Research Context

This project implements concepts from fairness-aware recommendation research, specifically addressing:
- **Consumer-side fairness**: Ensuring equitable recommendation quality across user groups
- **Multi-objective optimization**: Balancing competing goals without a priori preference specification
- **Re-ranking approaches**: Post-processing optimization of initial recommendation lists

## 📄 License

This project is for research and educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.