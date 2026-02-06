"""
Baseline π-kernel demonstration.

Shows basic usage of the π-kernel with orthogonal projector families,
ACE safety projection, and contraction certificates.
"""

import numpy as np
from pikernel import (
    ProjectorFamily, PiIndexGrid, PiKernel,
    Ledger, slope_upper_bound, gap_lower_bound
)


def main():
    """Run baseline π-kernel demo."""
    print("=" * 70)
    print("π-kernel Baseline Demo")
    print("=" * 70)
    
    # 1. ATOM REGISTRATION — two commuting families over R^8
    print("\n1. Atom Registration")
    print("-" * 70)
    
    # Family A: Function type (attention vs. embedding)
    A = ProjectorFamily([[0, 2, 4, 6], [1, 3, 5, 7]], name="FunctionType")
    print(f"Family A ({A.name}): {A.numblocks} blocks, dim={A.dim}")
    
    # Family B: Memory region (low vs. high)
    B = ProjectorFamily([[0, 1, 2, 3], [4, 5, 6, 7]], name="MemoryRegion")
    print(f"Family B ({B.name}): {B.numblocks} blocks, dim={B.dim}")
    
    # Build π-atom grid
    grid = PiIndexGrid([A, B])
    piids = grid.piids
    m = len(piids)
    print(f"\nπ-atoms: {piids} ({m} atoms)")
    
    # Verify orthogonality
    max_defect = 0.0
    for i, pi1 in enumerate(piids):
        for pi2 in piids[i+1:]:
            defect = grid.orthogonality_defect(pi1, pi2)
            max_defect = max(max_defect, defect)
    print(f"Max orthogonality defect: {max_defect:.2e}")
    
    # 2. DYNAMICS — Lipschitz-bounded parameters
    print("\n2. Dynamics Parameters")
    print("-" * 70)
    
    alphas = {pi: 0.25 for pi in piids}
    weights = {pi: np.ones(8) for pi in piids}
    taus = {pi: 1.5 for pi in piids}
    
    # Coupling matrix (small off-diagonal coupling)
    K = 0.05 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    # Verify contraction
    slopeUB = slope_upper_bound(alphas, K)
    gapLB = gap_lower_bound(slopeUB)
    print(f"SlopeUB: {slopeUB:.6f}")
    print(f"GapLB: {gapLB:.6f}")
    print(f"Contractive: {gapLB > 0}")
    
    # 3. LEDGER
    print("\n3. Ledger Setup")
    print("-" * 70)
    ledger = Ledger()  # In-memory ledger
    print(f"Ledger initialized (in-memory)")
    
    # 4. KERNEL
    print("\n4. Kernel Initialization")
    print("-" * 70)
    kernel = PiKernel(grid, alphas, weights, taus, K, ledger=ledger)
    print(f"π-kernel initialized with {m} atoms")
    
    # 5. RUN
    print("\n5. Execution")
    print("-" * 70)
    
    # Random initial state
    np.random.seed(42)
    x = np.random.randn(8)
    print(f"Initial state: ||x|| = {np.linalg.norm(x):.6f}")
    
    # Run for 10 steps
    num_steps = 10
    for t in range(num_steps):
        out = kernel.step(x)
        x = out["xnew"]
        
        # Verify contraction
        assert out["GapLB"] > 0, f"Contraction violated at step {t}!"
        
        print(f"Step {t:2d}: SlopeUB={out['SlopeUB']:.4f}, "
              f"GapLB={out['GapLB']:.4f}, "
              f"touched={out['num_touched']}, "
              f"||x||={np.linalg.norm(x):.6f}")
    
    # 6. RESULTS
    print("\n6. Results")
    print("-" * 70)
    print(f"Total steps: {num_steps}")
    print(f"Ledger entries: {len(ledger)}")
    print(f"Final state norm: {np.linalg.norm(x):.6f}")
    
    # Verify recomposition
    recomp_error = grid.recomposition_error(x)
    print(f"Recomposition error: {recomp_error:.2e}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
