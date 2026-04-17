import logging
import argparse
import sys

def run_cpu():
    from evopt.backends.cpu.config import EvolutionConfig
    from evopt.backends.cpu.core.engine import EvolutionEngine
    from evopt.backends.cpu.implementations.fitness import PenalizedFitnessStrategy
    from evopt.backends.cpu.implementations.initialization import RandomInitializationStrategy
    from evopt.backends.cpu.implementations.crossover import AverageCrossoverStrategy
    from evopt.backends.cpu.implementations.mutation import GaussianMutationStrategy
    from evopt.backends.cpu.implementations.selection import TournamentSelectionStrategy

    logger = logging.getLogger(__name__)
    logger.info("Initializing evopt CPU backend...")

    config = EvolutionConfig()

    init_strategy = RandomInitializationStrategy(
        search_min=config.search_space_min, search_max=config.search_space_max
    )
    fitness_strategy = PenalizedFitnessStrategy(penalty_factor=config.penalty_factor)
    selection_strategy = TournamentSelectionStrategy(tournament_size=config.tournament_size)
    crossover_strategy = AverageCrossoverStrategy()
    mutation_strategy = GaussianMutationStrategy(
        search_min=config.search_space_min, search_max=config.search_space_max, sigma=config.mutation_sigma
    )

    engine = EvolutionEngine(
        initialization_strategy=init_strategy,
        fitness_strategy=fitness_strategy,
        selection_strategy=selection_strategy,
        crossover_strategy=crossover_strategy,
        mutation_strategy=mutation_strategy,
        population_size=config.population_size,
        crossover_rate=config.crossover_rate,
        mutation_rate=config.mutation_rate,
        generations=config.generations
    )

    result = engine.run()

    best_genes = result.best_individual.genes
    min_value = -result.best_fitness

    print("\n" + "="*50)
    print("EVOLUTIONARY OPTIMIZATION RESULTS (CPU)")
    print("="*50)
    print(f"Optimal Solution Found: x1 = {best_genes[0]:.6f}, x2 = {best_genes[1]:.6f}")
    print(f"Minimum Function Value: f(x1, x2) = {min_value:.6f}")
    print("="*50)

def run_gpu():
    try:
        import torch
        from evopt.backends.gpu.driver import GPUDriver
        from evopt.backends.gpu.operators import VectorizedGaussianMutation, VectorizedAverageCrossover
        from evopt.backends.gpu.interfaces.base import VectorizedEvaluator
    except ImportError:
        print("Error: PyTorch not found. Please install the GPU extra using 'pip install evopt[gpu]'.")
        sys.exit(1)

    class SimpleEvaluator(VectorizedEvaluator):
        def evaluate(self, population: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Default objective: f(x1, x2) = x1^2 + x2^2 (minimize)
            objectives = torch.sum(population**2, dim=1, keepdim=True)
            violations = torch.zeros((population.shape[0], 0), device=population.device)
            return objectives, violations

    logger = logging.getLogger(__name__)
    logger.info("Initializing evopt GPU backend...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluator = SimpleEvaluator()
    mutation_op = VectorizedGaussianMutation(mutation_rate=0.3, sigma=0.3, search_min=-3.0, search_max=3.0)
    crossover_op = VectorizedAverageCrossover(crossover_rate=0.8, search_min=-3.0, search_max=3.0)

    driver = GPUDriver(
        evaluator=evaluator,
        mutation_op=mutation_op,
        crossover_op=crossover_op,
        population_size=100,
        generations=100,
        device=str(device)
    )

    initial_pop = torch.rand((100, 2), device=device) * 6.0 - 3.0
    best_genes, best_obj, _ = driver.run(initial_pop)

    print("\n" + "="*50)
    print("EVOLUTIONARY OPTIMIZATION RESULTS (GPU)")
    print("="*50)
    print(f"Optimal Solution Found: x1 = {best_genes[0].item():.6f}, x2 = {best_genes[1].item():.6f}")
    print(f"Minimum Function Value: f(x1, x2) = {best_obj.item():.6f}")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="evopt CLI - Evolutionary Algorithm Optimizer")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu", help="Backend to use (default: cpu)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.backend == "cpu":
        run_cpu()
    else:
        run_gpu()

if __name__ == "__main__":
    main()
