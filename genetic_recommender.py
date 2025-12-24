import numpy as np
import copy

class GeneticRecommender:
    def __init__(self, num_users, num_items, candidate_lists, target_matrix, 
                 gender_map, item_ids,
                 weights={'mdcg': 1.0, 'gender_gap': 1.0, 'item_coverage': 1.0},
                 top_k=10):
        """
        candidate_lists: np.array of shape (num_users, M), where M > k.
                         Contains item INDICES (not IDs) sorted by predicted score.
        target_matrix: np.array of shape (num_users, num_items). Ground truth.
        """
        self.num_users = num_users
        self.num_items = num_items
        self.candidate_lists = candidate_lists
        self.target_matrix = target_matrix
        self.gender_map = gender_map
        self.item_ids = item_ids
        self.weights = weights
        self.top_k = top_k
        self.candidate_len = candidate_lists.shape[1]

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
        return individual[:, :self.top_k]

    def fitness(self, individual):
        """
        Calculates fitness based on objective functions.
        """
        # 1. Decode to get recommendations
        recs_indices = self.decode(individual)
        
        ndcg_scores = []
        for u in range(self.num_users):
            rec_inds = recs_indices[u]
            # Get true ratings for these items
            true_ratings = self.target_matrix[u, rec_inds]
            
            relevance = true_ratings # These are the true ratings of the K items in their current order.

            
            # Let's calculate DCG directly here for speed.
            
            r = np.asarray(relevance, dtype=float)
            if r.size:
                dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
                # Calculate IDCG
                r_sorted = np.sort(r)[::-1]
                idcg = np.sum(r_sorted / np.log2(np.arange(2, r_sorted.size + 2)))
                if idcg > 0:
                    ndcg = dcg / idcg
                else:
                    ndcg = 0.0
            else:
                ndcg = 0.0
            
            ndcg_scores.append(ndcg)

        mean_mdcg = np.mean(ndcg_scores)
        
        # 2. Gender Gap
        # We need the actual NDCG/DCG values to calculate the gap.
        male_scores = []
        female_scores = []
        for idx in range(self.num_users):
            g = self.gender_map.get(idx, 'Unknown') # Assuming mapped to 0..N indices or dict keys.
            if g == 'M':
                male_scores.append(ndcg_scores[idx])
            elif g == 'F':
                female_scores.append(ndcg_scores[idx])
        
        avg_m = np.mean(male_scores) if male_scores else 0
        avg_f = np.mean(female_scores) if female_scores else 0
        gender_gap = abs(avg_m - avg_f)
        
        # 3. Item Coverage
        # Count unique item indices in the top K of all users
        unique_items = set(recs_indices.flatten())
        cov_count = len(unique_items)
        item_coverage = cov_count / self.num_items
        
        score = (self.weights['mdcg'] * mean_mdcg) - \
                (self.weights['gender_gap'] * gender_gap) + \
                (self.weights['item_coverage'] * item_coverage)
                
        return score, mean_mdcg, gender_gap, item_coverage

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
        start, end = min(a, b), max(a, b) + 1 # Slice end is exclusive
        
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
            idx1 = np.random.randint(0, self.top_k)
            idx2 = np.random.randint(0, self.candidate_len)
            
            individual[u, idx1], individual[u, idx2] = individual[u, idx2], individual[u, idx1]

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
            best_fit = fitnesses[best_idx][0] # fitness is tuple (score, ...)
            
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
            current_best_metrics = fitness_results[best_idx] #(score, mdcg, gender_gap, item_coverage)
            
            if max_score > best_score:
                best_score = max_score
                best_overall = copy.deepcopy(current_best_ind)
            
            history.append(current_best_metrics)
            
            print(f"Gen {gen}: Best Score={max_score:.4f}, MDCG={current_best_metrics[1]:.4f}, Gender Gap={current_best_metrics[2]:.4f}, Item Coverage={current_best_metrics[3]:.4f}")
            
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
    def __init__(self, num_users, num_items, candidate_lists, target_matrix, 
                 gender_map, item_ids,
                 weights={'mdcg': 1.0, 'gender_gap': 1.0, 'item_coverage': 1.0},
                 top_k=10):
        super().__init__(num_users, num_items, candidate_lists, target_matrix, 
                 gender_map, item_ids, weights, top_k)

    def fast_non_dominated_sort(self, population_metrics):
        """
        Sorts the population into fronts.
        population_metrics: list of tuples (score, mdcg, gender_gap, item_coverage)
        We want to:
         - Maximize MDCG -> Minimize -MDCG
         - Minimize Gender Gap
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
            # m = (score, mdcg, gender_gap, item_coverage)
            # Obj 1: -MDCG
            # Obj 2: Gender Gap
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
        l = len(front)
        distances = {idx: 0.0 for idx in front}
        
        if l == 0:
            return distances
            
        # We need actual values for crowding distance calculation, unscaled
        metrics_array = np.array([ [m[1], m[2], m[3]] for m in population_metrics ])
        
        # For each objective
        for m in range(3): # 3 objectives
            # Sort front by this objective
            sorted_front = sorted(front, key=lambda x: metrics_array[x][m])
            
            distances[sorted_front[0]] = np.inf
            distances[sorted_front[-1]] = np.inf
            
            obj_min = metrics_array[sorted_front[0]][m]
            obj_max = metrics_array[sorted_front[-1]][m]
            
            if obj_max - obj_min == 0:
                continue
                
            for i in range(1, l - 1):
                distances[sorted_front[i]] += (metrics_array[sorted_front[i+1]][m] - metrics_array[sorted_front[i-1]][m]) / (obj_max - obj_min)
                
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
            
            # Evaluate all
            combined_fitness = [self.fitness(ind) for ind in combined_pop]
            
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
                current_front_sorted = sorted(current_front, key=lambda x: cd[x], reverse=True)
                
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
            
            print(f"Gen {gen} (NSGA-II): Best Weighted Score={current_best_metrics[0]:.4f}, MDCG={current_best_metrics[1]:.4f}, Gap={current_best_metrics[2]:.4f}, Cov={current_best_metrics[3]:.4f}")

        # Return the Pareto Front (Rank 0)
        pareto_front_indices = fronts[0]
        pareto_population = [population[i] for i in pareto_front_indices]
        
        return pareto_population, history
