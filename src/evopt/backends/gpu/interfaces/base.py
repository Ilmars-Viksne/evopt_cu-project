from abc import ABC, abstractmethod
import torch

class VectorizedEvaluator(ABC):
    """Takes a batch of genomes and returns a batch of fitness scores and violations."""
    @abstractmethod
    def evaluate(self, population_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            population_tensor: (N, D) tensor where N is population size and D is number of genes.
        Returns:
            objectives: (N, M) tensor where M is number of objectives.
            violations: (N, K) tensor where K is number of constraints.
        """
        pass

class VectorizedOperator(ABC):
    """Performs variation (crossover/mutation) on the entire population tensor."""
    @abstractmethod
    def __call__(self, population_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            population_tensor: (N, D) tensor.
        Returns:
            modified_population: (N, D) tensor.
        """
        pass

class VectorizedSelector(ABC):
    """Ranks or selects individuals from the population based on fitness and constraints."""
    @abstractmethod
    def select(self, population: torch.Tensor, objectives: torch.Tensor, violations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            population: (N, D) tensor.
            objectives: (N, M) tensor.
            violations: (N, K) tensor.
        Returns:
            selected_population: (N, D) tensor.
        """
        pass
