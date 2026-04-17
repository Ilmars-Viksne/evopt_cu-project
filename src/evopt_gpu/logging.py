import torch
import time
import logging

logger = logging.getLogger(__name__)

class GPUStatsLogger:
    def __init__(self, device: torch.device):
        self.device = device
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.gen_times = []

    def log_generation(self, generation: int, population: torch.Tensor, objectives: torch.Tensor, violations: torch.Tensor, eval_time: float):
        current_time = time.time()
        gen_dt = current_time - self.prev_time
        self.prev_time = current_time

        pop_size = population.shape[0]
        ips = pop_size / gen_dt

        # Memory tracking
        if self.device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated(self.device) / 1024**2 # MB
            mem_reserved = torch.cuda.memory_reserved(self.device) / 1024**2 # MB
            mem_info = f" | Mem: {mem_allocated:.1f}/{mem_reserved:.1f}MB"
        else:
            mem_info = ""

        # Feasibility tracking
        total_violations = torch.sum(violations, dim=1)
        feasible_count = torch.sum(total_violations <= 0).item()
        feasibility_pct = (feasible_count / pop_size) * 100

        # Evolutionary quality (MOO)
        # For simplicity, we can count the number of individuals in the first Pareto front
        # if we have the info, but here we'll just log general stats.

        avg_obj = torch.mean(objectives, dim=0)
        obj_str = ", ".join([f"{x:.4f}" for x in avg_obj.tolist()])

        logger.info(
            f"Gen {generation:4d} | {ips:8.2f} ind/s | Eval: {eval_time*1000:6.2f}ms"
            f"{mem_info} | Feasible: {feasibility_pct:5.1f}% | Avg Obj: [{obj_str}]"
        )
