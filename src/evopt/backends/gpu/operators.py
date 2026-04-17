import torch
from evopt.backends.gpu.interfaces.base import VectorizedOperator

class VectorizedGaussianMutation(VectorizedOperator):
    def __init__(self, mutation_rate: float, sigma: float, search_min: float, search_max: float):
        self.mutation_rate = mutation_rate
        self.sigma = sigma
        self.search_min = search_min
        self.search_max = search_max

    def __call__(self, population_tensor: torch.Tensor) -> torch.Tensor:
        mask = torch.rand(population_tensor.shape, device=population_tensor.device) < self.mutation_rate
        noise = torch.randn(population_tensor.shape, device=population_tensor.device) * self.sigma
        mutated = population_tensor + mask * noise
        return torch.clamp(mutated, self.search_min, self.search_max)

class VectorizedSBX(VectorizedOperator):
    """Vectorized Simulated Binary Crossover (SBX)."""
    def __init__(self, crossover_rate: float, eta: float = 20.0, search_min: float = 0.0, search_max: float = 1.0):
        self.crossover_rate = crossover_rate
        self.eta = eta
        self.search_min = search_min
        self.search_max = search_max

    def __call__(self, population_tensor: torch.Tensor) -> torch.Tensor:
        N, D = population_tensor.shape
        is_odd = (N % 2 != 0)

        if is_odd:
            work_pop = population_tensor[:-1]
            last_ind = population_tensor[-1:]
        else:
            work_pop = population_tensor

        N_work = work_pop.shape[0]
        parent1 = work_pop[0::2]
        parent2 = work_pop[1::2]

        do_crossover = torch.rand(N_work // 2, 1, device=population_tensor.device) < self.crossover_rate

        u = torch.rand((N_work // 2, D), device=population_tensor.device)
        beta = torch.where(u <= 0.5,
                           (2.0 * u) ** (1.0 / (self.eta + 1.0)),
                           (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1.0)))

        child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
        child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)

        # Apply crossover mask
        child1 = torch.where(do_crossover, child1, parent1)
        child2 = torch.where(do_crossover, child2, parent2)

        new_work_pop = torch.empty_like(work_pop)
        new_work_pop[0::2] = child1
        new_work_pop[1::2] = child2

        if is_odd:
            new_population = torch.cat([new_work_pop, last_ind], dim=0)
        else:
            new_population = new_work_pop

        return torch.clamp(new_population, self.search_min, self.search_max)

class VectorizedAverageCrossover(VectorizedOperator):
    def __init__(self, crossover_rate: float, search_min: float, search_max: float):
        self.crossover_rate = crossover_rate
        self.search_min = search_min
        self.search_max = search_max

    def __call__(self, population_tensor: torch.Tensor) -> torch.Tensor:
        N, D = population_tensor.shape
        is_odd = (N % 2 != 0)

        if is_odd:
            work_pop = population_tensor[:-1]
            last_ind = population_tensor[-1:]
        else:
            work_pop = population_tensor

        N_work = work_pop.shape[0]
        parent1 = work_pop[0::2]
        parent2 = work_pop[1::2]

        do_crossover = torch.rand(N_work // 2, 1, device=population_tensor.device) < self.crossover_rate

        avg = (parent1 + parent2) / 2.0

        child1 = torch.where(do_crossover, avg, parent1)
        child2 = torch.where(do_crossover, avg, parent2)

        new_work_pop = torch.empty_like(work_pop)
        new_work_pop[0::2] = child1
        new_work_pop[1::2] = child2

        if is_odd:
            new_population = torch.cat([new_work_pop, last_ind], dim=0)
        else:
            new_population = new_work_pop

        return torch.clamp(new_population, self.search_min, self.search_max)
