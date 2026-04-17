import torch
import logging
from evopt.backends.gpu.driver import GPUDriver
from evopt.backends.gpu.operators import VectorizedGaussianMutation, VectorizedAverageCrossover
from evopt.backends.gpu.examples.rocket import RocketVectorizedEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# --- Problem Constants ---
g = 9.81
k_d = 0.04
k_m = 0.001
F_max = 200.0
dt = 2.0
m_0 = 10.0
fuel_mass = 8.0
N_STEPS = 13
PENALTY_FACTOR = 10000.0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    evaluator = RocketVectorizedEvaluator(
        n_steps=N_STEPS,
        dt=dt,
        m_0=m_0,
        fuel_mass=fuel_mass,
        g=g,
        k_d=k_d,
        k_m=k_m,
        device=device
    )

    mutation_op = VectorizedGaussianMutation(mutation_rate=0.4, sigma=20.0, search_min=0.0, search_max=F_max)
    crossover_op = VectorizedAverageCrossover(crossover_rate=0.85, search_min=0.0, search_max=F_max)

    driver = GPUDriver(
        evaluator=evaluator,
        mutation_op=mutation_op,
        crossover_op=crossover_op,
        population_size=1000,
        generations=300,
        penalty_factor=PENALTY_FACTOR,
        device=str(device)
    )

    # Initial population
    initial_pop = torch.rand((1000, N_STEPS), device=device) * F_max

    logger.info("Starting GPU-accelerated Rocket Optimization...")
    best_genes, best_obj, best_viol = driver.run(initial_pop)

    print("\n" + "="*50)
    print("🚀 ROCKET OPTIMIZATION RESULTS (GPU)")
    print("="*50)
    print("Optimal Thrust Profile (N) per time step:")
    for i, thrust in enumerate(best_genes):
        print(f"  t={i*2:02d}s : {thrust.item():>6.2f} N")

    print("-" * 50)
    print("Final Flight Metrics:")
    print(f"  Fuel Consumed      : {best_obj.item():>8.2f} kg  (Available: 8.0)")
    print(f"  Total Violation    : {torch.sum(best_viol).item():>8.2f}")
    print("="*50)

if __name__ == "__main__":
    main()
