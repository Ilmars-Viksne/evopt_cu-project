import torch
from evopt.backends.gpu.interfaces.base import VectorizedSelector

class VectorizedTournamentSelection(VectorizedSelector):
    def __init__(self, tournament_size: int = 2):
        self.tournament_size = tournament_size

    def select(self, population: torch.Tensor, objectives: torch.Tensor, violations: torch.Tensor) -> torch.Tensor:
        """
        Assumes Single-Objective optimization. If multiple objectives are provided,
        it uses the first one.
        """
        N = population.shape[0]
        # We assume objectives and violations are already combined if necessary
        # or we just use the first objective.

        # Here we assume objectives[:, 0] is the fitness to minimize
        # and violations are already handled via penalty or passed as combined fitness.

        indices = torch.randint(0, N, (N, self.tournament_size), device=population.device)

        fitness = objectives[:, 0:1]
        participant_fitness = fitness[indices].squeeze(-1) # (N, tournament_size)

        winner_in_tournament = torch.argmin(participant_fitness, dim=1) # (N,)
        winner_indices = indices[torch.arange(N), winner_in_tournament] # (N,)

        return population[winner_indices]

class VectorizedRankCrowdingTournamentSelection(VectorizedSelector):
    """
    Tournament selection for MOO based on Rank and Crowding Distance.
    """
    def __init__(self, tournament_size: int = 2):
        self.tournament_size = tournament_size

    def select(self, population: torch.Tensor, objectives: torch.Tensor, violations: torch.Tensor) -> torch.Tensor:
        # We need ranks and crowding distances
        # This selector expects ranks and crowding distances to be passed or pre-calculated.
        # For simplicity in this interface, we'll re-calculate here or expect them in a certain format.
        # But wait, the select interface only takes objectives and violations.

        N = population.shape[0]
        fronts = fast_non_dominated_sort(objectives, violations)

        ranks = torch.zeros(N, device=population.device, dtype=torch.long)
        crowding_distances = torch.zeros(N, device=population.device)

        for i, front in enumerate(fronts):
            ranks[front] = i
            crowding_distances[front] = compute_crowding_distance(objectives[front])

        indices = torch.randint(0, N, (N, self.tournament_size), device=population.device)

        p_ranks = ranks[indices] # (N, tournament_size)
        p_distances = crowding_distances[indices] # (N, tournament_size)

        # Winner is one with smaller rank. If ranks equal, larger distance.
        # We can combine into a single score: score = rank - distance / (max_dist + 1)
        # But easier to just compare.

        winners = []
        for i in range(N):
            idx1 = indices[i, 0]
            idx2 = indices[i, 1]

            if ranks[idx1] < ranks[idx2]:
                winners.append(idx1)
            elif ranks[idx2] < ranks[idx1]:
                winners.append(idx2)
            else:
                if crowding_distances[idx1] > crowding_distances[idx2]:
                    winners.append(idx1)
                else:
                    winners.append(idx2)

        return population[torch.tensor(winners, device=population.device)]

def compute_dominance_matrix(objectives: torch.Tensor):
    """
    Computes a dominance matrix where matrix[i, j] is True if i dominates j.
    objectives: (N, M)
    """
    N, M = objectives.shape
    obj_i = objectives.unsqueeze(1).expand(N, N, M)
    obj_j = objectives.unsqueeze(0).expand(N, N, M)

    is_not_worse = torch.all(obj_i <= obj_j, dim=2)
    is_better = torch.any(obj_i < obj_j, dim=2)

    return is_not_worse & is_better

def fast_non_dominated_sort(objectives: torch.Tensor, violations: torch.Tensor = None):
    """
    Vectorized implementation of non-dominated sorting.
    Returns a list of fronts (indices).
    """
    N = objectives.shape[0]

    if violations is not None and violations.shape[1] > 0:
        total_violation = torch.sum(violations, dim=1)
        v_i = total_violation.unsqueeze(1)
        v_j = total_violation.unsqueeze(0)

        feasible = total_violation <= 0
        f_i = feasible.unsqueeze(1)
        f_j = feasible.unsqueeze(0)

        dom1 = f_i & (~f_j)
        dom2 = (~f_i) & (~f_j) & (v_i < v_j)
        obj_dom = compute_dominance_matrix(objectives)
        dom3 = f_i & f_j & obj_dom

        dom_matrix = dom1 | dom2 | dom3
    else:
        dom_matrix = compute_dominance_matrix(objectives)

    domination_counts = torch.sum(dom_matrix, dim=0)

    fronts = []
    remaining_indices = torch.arange(N, device=objectives.device)

    # Pre-calculated dom_matrix is N_total x N_total
    # In each step we find those with 0 count in the current set

    while len(remaining_indices) > 0:
        # subset of dom_matrix for remaining
        # current_dom_matrix[i, j] is True if remaining[i] dominates remaining[j]
        current_dom_matrix = dom_matrix[remaining_indices][:, remaining_indices]
        # current_counts[j] is how many of remaining dominate remaining[j]
        current_counts = torch.sum(current_dom_matrix, dim=0)

        front_mask = (current_counts == 0)
        front_indices = remaining_indices[front_mask]

        if len(front_indices) == 0:
            break

        fronts.append(front_indices)
        remaining_indices = remaining_indices[~front_mask]

    return fronts

def compute_crowding_distance(objectives: torch.Tensor):
    """
    Computes crowding distance for a set of individuals.
    objectives: (N, M)
    """
    N, M = objectives.shape
    if N == 0:
        return torch.tensor([], device=objectives.device)
    if N <= 2:
        return torch.full((N,), float('inf'), device=objectives.device)

    distances = torch.zeros(N, device=objectives.device)

    for m in range(M):
        obj_m = objectives[:, m]
        sorted_indices = torch.argsort(obj_m)
        sorted_obj = obj_m[sorted_indices]

        distances[sorted_indices[0]] = float('inf')
        distances[sorted_indices[-1]] = float('inf')

        range_m = sorted_obj[-1] - sorted_obj[0]
        if range_m > 0:
            distances[sorted_indices[1:-1]] += (sorted_obj[2:] - sorted_obj[:-2]) / range_m

    return distances

class VectorizedNSGA2SurvivorSelection(VectorizedSelector):
    """NSGA-II Survivor Selection: Truncate P + Q to N."""
    def select(self, population: torch.Tensor, objectives: torch.Tensor, violations: torch.Tensor) -> torch.Tensor:
        # This assumes population is P + Q (size 2N)
        # We need to return size N
        N = population.shape[0] // 2
        fronts = fast_non_dominated_sort(objectives, violations)

        selected_indices = []
        for front in fronts:
            if len(selected_indices) + len(front) <= N:
                selected_indices.extend(front.tolist())
            else:
                remaining = N - len(selected_indices)
                distances = compute_crowding_distance(objectives[front])
                sort_idx = torch.argsort(distances, descending=True)
                selected_indices.extend(front[sort_idx[:remaining]].tolist())
                break

        return population[selected_indices]
