import numpy as np
import copy
from metrics import (
    DEFAULT_WEIGHTS,
    calculate_activity_gap,
    calculate_item_coverage_simple,
    compute_weighted_score,
)


class GeneticRecommender:
    def __init__(
        self,
        num_users,
        num_items,
        candidate_lists,
        target_matrix,
        activity_map,
        item_ids,
        user_ids=None,
        weights=None,
        top_k=10,
    ):
        """
        candidate_lists: np.array of shape (num_users, M), where M > k.
                         Contains item INDICES (not IDs) sorted by predicted score.
        target_matrix: np.array of shape (num_users, num_items). Ground truth ratings
                       from the validation set (used during GA/NSGA-II optimization).
                       Final evaluation is performed separately on the held-out test set.
        activity_map: dict mapping user_id -> 'active'/'inactive' (uses user_ids as keys)
        user_ids: list of original user IDs (index -> user_id mapping). Required for
                  consistent activity_gap calculation with main.py evaluation.
        weights: dict with keys 'mdcg', 'activity_gap', 'item_coverage'.
                 For GA: Used directly in fitness optimization (weighted sum).
                 For NSGA-II: Only used post-hoc to select representative solution from Pareto front.
                 Defaults to DEFAULT_WEIGHTS from metrics.py.
        """
        if weights is None:
            weights = (
                DEFAULT_WEIGHTS.copy()
            )  # Use centralized default, copy to avoid mutation

        self.num_users = num_users
        self.num_items = num_items
        self.candidate_lists = candidate_lists
        self.target_matrix = target_matrix
        self.activity_map = activity_map
        self.item_ids = item_ids
        self.user_ids = user_ids
        self.weights = weights
        self.top_k = top_k
        self.candidate_len = candidate_lists.shape[1]
        self.user_idcg_values = None

    def set_user_idcg_values(self, idcg_values):
        self.user_idcg_values = idcg_values

    def initialize_population(self, pop_size):
        """
        Individual representation: np.array of shape (num_users, candidate_len).
        Each row is a permutation of the candidate items for that user.
        The top-k of this permutation represents the recommendation.
        """
        population = []

        # 1. Greedy individual (original order)
        population.append(self.candidate_lists.copy())

        # 2. Random permutations
        for _ in range(pop_size - 1):
            individual = self.candidate_lists.copy()
            for u in range(self.num_users):
                np.random.shuffle(individual[u])
            population.append(individual)

        return population

    def decode(self, individual):
        """
        Extracts the Top-K items for each user from the individual.
        Returns: matrix of shape (num_users, top_k) containing item indices.
        """
        return individual[:, : self.top_k]

    def fitness(self, individual):
        """
        Calculates fitness based on objective functions.

        OPTIMIZATION SETTING:
        - DCG numerator: Computed from target_matrix (validation set during optimization)
        - IDCG denominator: Pre-calculated from target_matrix via set_user_idcg_values()

        This ensures the GA optimizes on validation set ground-truth relevance,
        while final evaluation is performed separately on the held-out test set.
        Both baseline and GA/NSGA-II use the same IDCG normalization approach.
        """
        # 1. Decode to get recommendations
        recs_indices = self.decode(individual)
        num_users = self.num_users
        top_k = self.top_k

        # Vectorized DCG computation:
        # Get all true ratings at once using advanced indexing
        user_indices = np.arange(num_users)[:, np.newaxis]  # (num_users, 1)
        true_ratings = self.target_matrix[
            user_indices, recs_indices
        ]  # (num_users, top_k)

        # Precompute discount factors once (shared for all users)
        discount = np.log2(np.arange(2, top_k + 2))  # (top_k,)

        # Compute DCG for all users: sum(rating / discount) per user
        dcg_values = np.sum(true_ratings / discount, axis=1)  # (num_users,)

        # Compute NDCG using pre-calculated IDCG
        if self.user_idcg_values is None:
            raise ValueError(
                "user_idcg_values must be set via set_user_idcg_values() before calling fitness(). "
                "Without pre-calculated IDCG, NDCG normalization would be incorrect."
            )

        # Avoid division by zero: where IDCG is 0, NDCG is 0
        idcg = self.user_idcg_values
        ndcg_scores = np.where(idcg > 0, dcg_values / idcg, 0.0)

        mean_mdcg = np.mean(ndcg_scores)

        # 2. Activity Gap - using centralized function with user_ids for consistency with main.py
        activity_gap = calculate_activity_gap(
            ndcg_scores, self.activity_map, user_ids=self.user_ids, verbose=False
        )

        # 3. Item Coverage - using centralized function for consistency
        item_coverage = calculate_item_coverage_simple(recs_indices, self.item_ids)

        # Use centralized weighted score function for consistency with main.py evaluation
        score = compute_weighted_score(
            mean_mdcg, activity_gap, item_coverage, self.weights
        )

        return score, mean_mdcg, activity_gap, item_coverage

    def crossover(self, parent1, parent2):
        """
        Partially Mapped Crossover (PMX).
        Applied individually to each user's permutation.
        """
        child1 = np.empty_like(parent1)
        child2 = np.empty_like(parent2)

        for u in range(self.num_users):
            c1_row, c2_row = self._pmx_1d(parent1[u], parent2[u])
            child1[u] = c1_row
            child2[u] = c2_row

        return child1, child2

    def _pmx_1d(self, p1, p2):
        """
        Helper for PMX on a single permutation (1D array).
        """
        size = len(p1)
        # Select two random cut points
        a = np.random.randint(0, size)
        b = np.random.randint(0, size)
        start, end = min(a, b), max(a, b) + 1  # Slice end is exclusive

        # Initialize children
        c1 = np.full(size, -1, dtype=p1.dtype)
        c2 = np.full(size, -1, dtype=p2.dtype)

        # Copy segments
        c1[start:end] = p1[start:end]
        c2[start:end] = p2[start:end]

        p1_seg_set = set(p1[start:end])
        p2_seg_set = set(p2[start:end])

        # Let's map value -> index in segment P1
        p1_seg_map = {val: idx for idx, val in enumerate(p1[start:end])}
        # p2_seg_map = {val: idx for idx, val in enumerate(p2[start:end])} # relative index 0..len

        # Fill Child 1
        for i in range(size):
            if start <= i < end:
                continue

            candidate = p2[i]

            while candidate in p1_seg_set:
                rel_idx = p1_seg_map[candidate]
                candidate = p2[start + rel_idx]

            c1[i] = candidate

        # Fill Child 2 (Symmetric)
        p2_seg_map = {val: idx for idx, val in enumerate(p2[start:end])}

        for i in range(size):
            if start <= i < end:
                continue

            candidate = p1[i]
            while candidate in p2_seg_set:
                rel_idx = p2_seg_map[candidate]
                candidate = p1[start + rel_idx]

            c2[i] = candidate

        return c1, c2

    def mutate(self, individual, mutation_rate=0.01):
        """
        Swap Mutation.
        For each user, with prob mutation_rate, swap two items in their list.
        """

        # Optimization: Only mutate some users
        num_mutations = int(self.num_users * mutation_rate)
        if num_mutations == 0:
            return individual

        users_to_mutate = np.random.choice(self.num_users, num_mutations, replace=False)

        for u in users_to_mutate:
            # Select a position within top-k to swap out
            idx1 = np.random.randint(0, self.top_k)

            # Select a different position from the candidate list to swap in
            # Ensure idx2 != idx1 to avoid no-op swaps
            # We pick from [top_k, candidate_len) to always bring in a new item
            if self.candidate_len > self.top_k:
                idx2 = np.random.randint(self.top_k, self.candidate_len)
            else:
                # Edge case: candidate_len == top_k, pick any different index
                idx2 = (idx1 + 1) % self.candidate_len

            individual[u, idx1], individual[u, idx2] = (
                individual[u, idx2],
                individual[u, idx1],
            )

        return individual

    def select(self, population, fitnesses):
        """
        Tournament Selection
        """
        selected = []
        pop_len = len(population)
        tournament_size = 3

        for _ in range(pop_len):
            # Pick random competitors
            inds = np.random.randint(0, pop_len, tournament_size)
            best_idx = inds[0]
            best_fit = fitnesses[best_idx][0]  # fitness is tuple (score, ...)

            for i in inds[1:]:
                if fitnesses[i][0] > best_fit:
                    best_fit = fitnesses[i][0]
                    best_idx = i

            selected.append(population[best_idx])

        return selected

    def run(self, generations=10, pop_size=20, crossover_rate=0.8, mutation_rate=0.1):
        population = self.initialize_population(pop_size)

        best_overall = None
        best_score = -np.inf

        history = []

        print(f"Starting Evolution: Pop={pop_size}, Gens={generations}")

        for gen in range(generations):
            # Evaluate
            fitness_results = [self.fitness(ind) for ind in population]
            scores = [f[0] for f in fitness_results]

            # Stats
            max_score = np.max(scores)
            best_idx = np.argmax(scores)

            current_best_ind = population[best_idx]
            current_best_metrics = fitness_results[
                best_idx
            ]  # (score, mdcg, activity_gap, item_coverage)

            if max_score > best_score:
                best_score = max_score
                best_overall = copy.deepcopy(current_best_ind)

            history.append(current_best_metrics)

            print(
                f"Gen {gen}: Best Score={max_score:.4f}, MDCG={current_best_metrics[1]:.4f}, Activity Gap={current_best_metrics[2]:.4f}, Item Coverage={current_best_metrics[3]:.4f}"
            )

            # Selection
            selected_pop = self.select(population, fitness_results)

            # Reproduction
            next_pop = []

            # Elitism: keep best
            next_pop.append(copy.deepcopy(population[best_idx]))

            while len(next_pop) < pop_size:
                p1 = selected_pop[np.random.randint(len(selected_pop))]
                p2 = selected_pop[np.random.randint(len(selected_pop))]

                if np.random.rand() < crossover_rate:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = self.mutate(c1, mutation_rate)
                c2 = self.mutate(c2, mutation_rate)

                next_pop.append(c1)
                if len(next_pop) < pop_size:
                    next_pop.append(c2)

            population = next_pop

        return best_overall, history


class NsgaIIRecommender(GeneticRecommender):
    """
    NSGA-II Multi-Objective Recommender.

    Unlike GeneticRecommender, this class performs true multi-objective optimization
    using Pareto dominance. The 'weights' parameter is NOT used during optimization;
    instead, it's only used post-hoc to select a single representative solution
    from the Pareto front for comparison in results tables.

    The Pareto front itself contains all non-dominated trade-off solutions.
    """

    def __init__(
        self,
        num_users,
        num_items,
        candidate_lists,
        target_matrix,
        activity_map,
        item_ids,
        user_ids=None,
        weights=None,
        top_k=10,
    ):
        super().__init__(
            num_users,
            num_items,
            candidate_lists,
            target_matrix,
            activity_map,
            item_ids,
            user_ids,
            weights,
            top_k,
        )

    def fast_non_dominated_sort(self, population_metrics):
        """
        Sorts the population into fronts.
        population_metrics: list of tuples (score, mdcg, activity_gap, item_coverage)
        We want to:
         - Maximize MDCG -> Minimize -MDCG
         - Minimize Activity Gap
         - Maximize Item Coverage -> Minimize -Item Coverage
        """
        pop_size = len(population_metrics)
        S = [[] for _ in range(pop_size)]
        n = [0] * pop_size
        rank = [0] * pop_size
        fronts = [[]]

        # Convert to minimization problem
        objectives = []
        for m in population_metrics:
            # m = (score, mdcg, activity_gap, item_coverage)
            # Obj 1: -MDCG
            # Obj 2: Activity Gap
            # Obj 3: -Item Coverage
            objectives.append([-m[1], m[2], -m[3]])

        objectives = np.array(objectives)

        for p in range(pop_size):
            p_obj = objectives[p]
            for q in range(pop_size):
                if p == q:
                    continue
                q_obj = objectives[q]

                # Check domination
                # p dominates q if p <= q for all obj AND p < q for at least one
                diff = p_obj - q_obj
                if np.all(diff <= 0) and np.any(diff < 0):
                    S[p].append(q)
                elif np.all(diff >= 0) and np.any(diff > 0):
                    n[p] += 1

            if n[p] == 0:
                rank[p] = 0
                fronts[0].append(p)

        i = 0
        while i < len(fronts):
            current_front = fronts[i]
            if not current_front:
                break

            next_front = []
            for p in current_front:
                for q in S[p]:
                    n[q] -= 1
                    if n[q] == 0:
                        rank[q] = i + 1
                        next_front.append(q)

            if next_front:
                fronts.append(next_front)
            i += 1

        return fronts, rank

    def calculate_crowding_distance(self, front, population_metrics):
        """
        Calculates crowding distance for individuals in a front.
        """
        front_size = len(front)
        distances = {idx: 0.0 for idx in front}

        if front_size == 0:
            return distances

        # We need actual values for crowding distance calculation, unscaled
        metrics_array = np.array([[m[1], m[2], m[3]] for m in population_metrics])

        # For each objective
        for m in range(3):  # 3 objectives
            # Sort front by this objective
            sorted_front = sorted(front, key=lambda x: metrics_array[x][m])

            distances[sorted_front[0]] = np.inf
            distances[sorted_front[-1]] = np.inf

            obj_min = metrics_array[sorted_front[0]][m]
            obj_max = metrics_array[sorted_front[-1]][m]

            if obj_max - obj_min == 0:
                continue

            for i in range(1, front_size - 1):
                distances[sorted_front[i]] += (
                    metrics_array[sorted_front[i + 1]][m]
                    - metrics_array[sorted_front[i - 1]][m]
                ) / (obj_max - obj_min)

        return distances

    def nsga_selection(self, population, ranks, crowding_dists):
        """
        Binary Tournament Selection based on Rank and Crowding Distance
        """
        selected = []
        pop_len = len(population)

        for _ in range(pop_len):
            idx1 = np.random.randint(0, pop_len)
            idx2 = np.random.randint(0, pop_len)

            # Crowded Comparison Operator
            # Prefer lower rank
            if ranks[idx1] < ranks[idx2]:
                winner = idx1
            elif ranks[idx2] < ranks[idx1]:
                winner = idx2
            else:
                # If ranks equal, prefer higher crowding distance
                if crowding_dists[idx1] > crowding_dists[idx2]:
                    winner = idx1
                else:
                    winner = idx2

            selected.append(population[winner])

        return selected

    def run(self, generations=10, pop_size=20, crossover_rate=0.8, mutation_rate=0.1):
        print(f"Starting NSGA-II Evolution: Pop={pop_size}, Gens={generations}")

        population = self.initialize_population(pop_size)

        # Initial Evaluate
        fitness_results = [self.fitness(ind) for ind in population]

        # Initial Sort
        fronts, ranks = self.fast_non_dominated_sort(fitness_results)

        # Calculate Crowding Distance for global tracking (needed for selection)
        crowding_dists = {}
        for front in fronts:
            cd = self.calculate_crowding_distance(front, fitness_results)
            crowding_dists.update(cd)

        history = []

        for gen in range(generations):
            # 1. Selection
            # We select parents to create offspring
            mating_pool = self.nsga_selection(population, ranks, crowding_dists)

            # 2. Reproduction to create Q_t
            offspring_pop = []
            while len(offspring_pop) < pop_size:
                p1 = mating_pool[np.random.randint(len(mating_pool))]
                p2 = mating_pool[np.random.randint(len(mating_pool))]

                if np.random.rand() < crossover_rate:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = self.mutate(c1, mutation_rate)
                c2 = self.mutate(c2, mutation_rate)

                offspring_pop.append(c1)
                if len(offspring_pop) < pop_size:
                    offspring_pop.append(c2)

            # 3. Combine R_t = P_t + Q_t
            combined_pop = population + offspring_pop

            # Evaluate ONLY offspring (parents already have fitness_results from previous gen)
            offspring_fitness = [self.fitness(ind) for ind in offspring_pop]
            combined_fitness = fitness_results + offspring_fitness

            # 4. Non-Dominated Sort of R_t
            fronts, ranks_combined = self.fast_non_dominated_sort(combined_fitness)

            # 5. Create next P_{t+1}
            next_pop = []
            next_fitness = []
            i = 0

            while len(next_pop) + len(fronts[i]) <= pop_size:
                # Add entire front
                for idx in fronts[i]:
                    next_pop.append(combined_pop[idx])
                    next_fitness.append(combined_fitness[idx])
                i += 1
                if i >= len(fronts):
                    break

            # If we still need individuals, sort the current front by crowding distance
            if len(next_pop) < pop_size and i < len(fronts):
                current_front = fronts[i]
                cd = self.calculate_crowding_distance(current_front, combined_fitness)

                # Sort indices by CD descending
                current_front_sorted = sorted(
                    current_front, key=lambda x: cd[x], reverse=True
                )

                fill_count = pop_size - len(next_pop)
                for j in range(fill_count):
                    idx = current_front_sorted[j]
                    next_pop.append(combined_pop[idx])
                    next_fitness.append(combined_fitness[idx])

            population = next_pop
            fitness_results = next_fitness

            # Re-rank for next selection (optional if we re-calc at top of loop, but let's update state)
            fronts, ranks = self.fast_non_dominated_sort(fitness_results)
            crowding_dists = {}
            for front in fronts:
                cd = self.calculate_crowding_distance(front, fitness_results)
                crowding_dists.update(cd)

            # Log best "weighted score" individual for tracking
            scores = [f[0] for f in fitness_results]
            best_idx = np.argmax(scores)
            current_best_metrics = fitness_results[best_idx]
            history.append(current_best_metrics)

            print(
                f"Gen {gen} (NSGA-II): Best Weighted Score={current_best_metrics[0]:.4f}, MDCG={current_best_metrics[1]:.4f}, Gap={current_best_metrics[2]:.4f}, Cov={current_best_metrics[3]:.4f}"
            )

        # Return the Pareto Front (Rank 0)
        pareto_front_indices = fronts[0]
        pareto_population = [population[i] for i in pareto_front_indices]

        return pareto_population, history
