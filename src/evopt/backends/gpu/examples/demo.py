import torch
import logging
from evopt.backends.gpu.driver import GPUDriver
from evopt.backends.gpu.examples.rocket import RocketVectorizedEvaluator
from evopt.backends.gpu.operators import VectorizedGaussianMutation, VectorizedSBX

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_rocket_optimization(mode='so'):
    logger.info(f"Starting Rocket Optimization in {mode.upper()} mode...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Constants
    n_steps = 13
    pop_size = 200
    generations = 50
    f_max = 200.0

    evaluator = RocketVectorizedEvaluator(
        n_steps=n_steps,
        dt=2.0,
        m_0=10.0,
        fuel_mass=8.0,
        g=9.81,
        k_d=0.04,
        k_m=0.001,
        device=device
    )

    mutation_op = VectorizedGaussianMutation(
        mutation_rate=0.4,
        sigma=20.0,
        search_min=0.0,
        search_max=f_max
    )

    crossover_op = VectorizedSBX(
        crossover_rate=0.85,
        eta=20.0,
        search_min=0.0,
        search_max=f_max
    )

    driver = GPUDriver(
        evaluator=evaluator,
        mutation_op=mutation_op,
        crossover_op=crossover_op,
        mode=mode,
        population_size=pop_size,
        generations=generations,
        device=device
    )

    initial_population = torch.rand((pop_size, n_steps), device=device) * f_max

    result = driver.run(initial_population)

    if mode == 'so':
        best_genes, best_obj, best_violations = result
        print("\n" + "="*50)
        print(f"🚀 ROCKET OPTIMIZATION RESULTS ({mode.upper()})")
        print("="*50)
        print(f"Best Fuel Consumed: {best_obj.item():.4f} kg")
        print(f"Total Violation: {torch.sum(best_violations).item():.4f}")
        print("-" * 50)
    else:
        population, objectives, violations = result
        print("\n" + "="*50)
        print(f"🚀 ROCKET OPTIMIZATION RESULTS ({mode.upper()})")
        print("="*50)
        print(f"Found {len(population)} individuals on the Pareto front.")
        print(f"Avg Fuel Consumed: {torch.mean(objectives[:, 0]).item():.4f} kg")
        print(f"Feasible individuals on front: {torch.sum(torch.sum(violations, dim=1) <= 0).item()}")
        print("-" * 50)

if __name__ == "__main__":
    run_rocket_optimization(mode='so')
    run_rocket_optimization(mode='moo')
