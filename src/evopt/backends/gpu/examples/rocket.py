import torch
from evopt.backends.gpu.interfaces.base import VectorizedEvaluator

class RocketVectorizedEvaluator(VectorizedEvaluator):
    def __init__(self, n_steps: int, dt: float, m_0: float, fuel_mass: float, g: float, k_d: float, k_m: float, device: torch.device):
        self.n_steps = n_steps
        self.dt = dt
        self.m_0 = m_0
        self.fuel_mass = fuel_mass
        self.g = g
        self.k_d = k_d
        self.k_m = k_m
        self.device = device

    def evaluate(self, population_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        population_tensor: (N, n_steps) - thrust profiles
        """
        N = population_tensor.shape[0]

        # Initialize states
        m = torch.full((N, self.n_steps), self.m_0, device=self.device)
        v = torch.zeros((N, self.n_steps), device=self.device)
        s = torch.zeros((N, self.n_steps), device=self.device)
        a = torch.zeros((N, self.n_steps), device=self.device)

        fuel_remaining = torch.full((N,), self.fuel_mass, device=self.device)

        # Initial acceleration
        # a = T/m - g
        # X[0] is initial thrust
        a[:, 0] = population_tensor[:, 0] / m[:, 0] - self.g
        ds0 = 0.5 * a[:, 0] * (self.dt ** 2)
        s[:, 0] = ds0

        for i in range(self.n_steps - 1):
            # Update velocity
            v[:, i+1] = v[:, i] + a[:, i] * self.dt

            # Fuel consumption
            fuel_consumed = self.k_m * torch.abs(population_tensor[:, i]) * self.dt
            actual_fuel_consumed = torch.minimum(fuel_consumed, fuel_remaining)
            fuel_remaining -= actual_fuel_consumed

            # Update mass
            m[:, i+1] = m[:, i] - actual_fuel_consumed

            # Current thrust (only if fuel remaining)
            # For simplicity, we assume if fuel_remaining > 0 at start of step, we can thrust
            current_thrust = torch.where(fuel_remaining > 0, population_tensor[:, i+1], torch.zeros_like(population_tensor[:, i+1]))

            # Update acceleration
            # a = T/m - g - k_d * v / m
            a[:, i+1] = current_thrust / m[:, i+1] - self.g - self.k_d * v[:, i+1] / m[:, i+1]

            # Update position
            ds = v[:, i+1] * self.dt + 0.5 * a[:, i+1] * (self.dt ** 2)
            s[:, i+1] = s[:, i] + ds

        # Objectives:
        # 1. Minimize fuel consumed (maximize remaining fuel)
        fuel_used = self.m_0 - m[:, -1]

        # Constraints:
        # Target Position: 500 to 550
        # Final Velocity: -1.0 to 1.0
        # Final Acceleration: -0.5 to 0.5

        final_s = s[:, -1]
        final_v = v[:, -1]
        final_a = a[:, -1]

        pos_violation_lower = torch.clamp(500.0 - final_s, min=0.0)
        pos_violation_upper = torch.clamp(final_s - 550.0, min=0.0)

        vel_violation_lower = torch.clamp(-1.0 - final_v, min=0.0)
        vel_violation_upper = torch.clamp(final_v - 1.0, min=0.0)

        acc_violation_lower = torch.clamp(-0.5 - final_a, min=0.0)
        acc_violation_upper = torch.clamp(final_a - 0.5, min=0.0)

        violations = torch.stack([
            pos_violation_lower, pos_violation_upper,
            vel_violation_lower, vel_violation_upper,
            acc_violation_lower, acc_violation_upper
        ], dim=1)

        # In SO mode, we might want to return (fuel_used, violations)
        # In MOO mode, we might want to return (fuel_used, violations) and treat violations as objectives
        # The GPUDriver will handle the mode switching.

        objectives = fuel_used.unsqueeze(1) # (N, 1)

        return objectives, violations
