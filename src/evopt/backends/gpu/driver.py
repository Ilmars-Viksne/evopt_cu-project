import torch
import time
import logging
from evopt.backends.gpu.interfaces.base import VectorizedEvaluator, VectorizedOperator, VectorizedSelector
from evopt.backends.gpu.logging import GPUStatsLogger
from evopt.backends.gpu.selection import (
    VectorizedTournamentSelection,
    VectorizedRankCrowdingTournamentSelection,
    VectorizedNSGA2SurvivorSelection
)

logger = logging.getLogger(__name__)

class GPUDriver:
    def __init__(
        self,
        evaluator: VectorizedEvaluator,
        mutation_op: VectorizedOperator,
        crossover_op: VectorizedOperator,
        mode: str = 'so', # 'so' or 'moo'
        constraint_strategy: str = 'penalty', # 'objective', 'penalty', 'strict'
        penalty_factor: float = 1e9,
        population_size: int = 100,
        generations: int = 100,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.evaluator = evaluator
        self.mutation_op = mutation_op
        self.crossover_op = crossover_op
        self.mode = mode.lower()
        self.constraint_strategy = constraint_strategy.lower()
        self.penalty_factor = penalty_factor
        self.population_size = population_size
        self.generations = generations
        self.device = torch.device(device)

        self.stats_logger = GPUStatsLogger(self.device)

        if self.mode == 'so':
            self.mating_selector = VectorizedTournamentSelection()
        else:
            self.mating_selector = VectorizedRankCrowdingTournamentSelection()
            self.survivor_selector = VectorizedNSGA2SurvivorSelection()

    def _prepare_fitness(self, objectives, violations):
        """
        Processes objectives and violations based on constraint_strategy.
        Returns (processed_objectives, processed_violations)
        """
        N, M = objectives.shape
        N, K = violations.shape

        if self.constraint_strategy == 'penalty':
            # Collapse violations into a single penalty added to objectives
            total_violation = torch.sum(violations, dim=1, keepdim=True)
            # Adjusted objectives: (N, M)
            adjusted_objectives = objectives + total_violation * self.penalty_factor
            return adjusted_objectives, torch.zeros((N, 0), device=self.device)

        elif self.constraint_strategy == 'objective':
            # Add total violation as an additional objective
            total_violation = torch.sum(violations, dim=1, keepdim=True)
            adjusted_objectives = torch.cat([objectives, total_violation], dim=1)
            return adjusted_objectives, torch.zeros((N, 0), device=self.device)

        elif self.constraint_strategy == 'strict':
            # Keep them separate, selection logic will handle "feasible always beats infeasible"
            return objectives, violations

        return objectives, violations

    def run(self, initial_population: torch.Tensor):
        population = initial_population.to(self.device)

        # Initial evaluation
        start_eval = time.time()
        objectives, violations = self.evaluator.evaluate(population)
        eval_time = time.time() - start_eval

        # Prepare initial fitness
        curr_obj, curr_viol = self._prepare_fitness(objectives, violations)

        for gen in range(self.generations):
            # 1. Logging
            if gen % 10 == 0 or gen == self.generations - 1:
                self.stats_logger.log_generation(gen, population, objectives, violations, eval_time)

            # 2. Mating Selection
            parents = self.mating_selector.select(population, curr_obj, curr_viol)

            # 3. Variation
            idx = torch.randperm(self.population_size, device=self.device)
            parents = parents[idx]
            offspring = self.crossover_op(parents)
            offspring = self.mutation_op(offspring)

            # 4. Evaluation of offspring
            start_eval = time.time()
            off_objectives, off_violations = self.evaluator.evaluate(offspring)
            eval_time = time.time() - start_eval

            # Prepare offspring fitness
            off_curr_obj, off_curr_viol = self._prepare_fitness(off_objectives, off_violations)

            # 5. Survivor Selection
            if self.mode == 'moo':
                # NSGA-II uses (P + Q) selection
                combined_pop = torch.cat([population, offspring], dim=0)
                combined_obj = torch.cat([curr_obj, off_curr_obj], dim=0)
                combined_viol = torch.cat([curr_viol, off_curr_viol], dim=0)

                population = self.survivor_selector.select(combined_pop, combined_obj, combined_viol)

                # Re-evaluate or filter existing scores for the selected population
                # For simplicity and correctness, we re-evaluate or use pre-calculated
                # Since we kept everything aligned, we can just re-evaluate here to get fresh obj/viol
                # but it's more efficient to track them.
                # Let's re-evaluate for now to ensure no alignment issues,
                # though in a real-world scenario we'd slice combined_obj/viol.
                objectives, violations = self.evaluator.evaluate(population)
                curr_obj, curr_viol = self._prepare_fitness(objectives, violations)
            else:
                # Generational SO or could implement Elitism
                # Let's implement simple elitism for SO: keep best overall
                # (Optional: user might prefer pure generational)
                population = offspring
                objectives = off_objectives
                violations = off_violations
                curr_obj = off_curr_obj
                curr_viol = off_curr_viol

        # Final result
        if self.mode == 'moo':
            from evopt.backends.gpu.selection import fast_non_dominated_sort
            fronts = fast_non_dominated_sort(objectives, violations)
            pareto_indices = fronts[0]
            return population[pareto_indices], objectives[pareto_indices], violations[pareto_indices]
        else:
            total_violation = torch.sum(violations, dim=1, keepdim=True)
            fitness = objectives[:, 0:1] + total_violation * self.penalty_factor
            best_idx = torch.argmin(fitness)
            return population[best_idx], objectives[best_idx], violations[best_idx]
