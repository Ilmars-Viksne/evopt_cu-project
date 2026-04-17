import torch
from evopt_gpu.selection import compute_dominance_matrix, fast_non_dominated_sort

def test_dominance():
    # Minimization
    # A = [1, 2], B = [2, 3], C = [2, 2]
    # A dominates B
    # A dominates C
    # C dominates B
    objectives = torch.tensor([
        [1.0, 2.0],
        [2.0, 3.0],
        [2.0, 2.0]
    ])

    dom_matrix = compute_dominance_matrix(objectives)

    # matrix[i, j] is True if i dominates j
    assert dom_matrix[0, 1] == True
    assert dom_matrix[0, 2] == True
    assert dom_matrix[2, 1] == True
    assert dom_matrix[1, 0] == False
    assert dom_matrix[1, 2] == False
    assert dom_matrix[2, 0] == False
    assert dom_matrix[0, 0] == False

def test_non_dominated_sort():
    objectives = torch.tensor([
        [1.0, 2.0], # A (Front 0)
        [2.0, 1.0], # B (Front 0)
        [2.0, 3.0], # C (Front 1, dominated by A and B)
        [3.0, 2.0], # D (Front 1, dominated by A and B)
        [4.0, 4.0]  # E (Front 2)
    ])

    fronts = fast_non_dominated_sort(objectives)

    assert len(fronts) == 3
    # Front 0: A and B
    assert set(fronts[0].tolist()) == {0, 1}
    # Front 1: C and D
    assert set(fronts[1].tolist()) == {2, 3}
    # Front 2: E
    assert set(fronts[2].tolist()) == {4}

if __name__ == "__main__":
    test_dominance()
    test_non_dominated_sort()
    print("Selection tests passed!")
