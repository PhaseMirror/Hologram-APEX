"""
Tests for pikernel kernel module.
"""

import pytest
import sys
import os

# Add pikernel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pikernel.kernel import PiKernel, DefaultProposer
from pikernel.projectors import ProjectorFamily, PiIndexGrid
from pikernel.ledger import Ledger
import numpy as np


# ============================================================================
# Test DefaultProposer
# ============================================================================

def test_default_proposer():
    """Test default proposer with damping."""
    proposer = DefaultProposer(damping=0.8)
    c = np.array([1.0, 2.0, 3.0])
    
    prop = proposer(c)
    
    np.testing.assert_array_almost_equal(prop, 0.8 * c)


def test_default_proposer_full_damping():
    """Test proposer with full damping (zeros output)."""
    proposer = DefaultProposer(damping=0.0)
    c = np.array([5.0, 10.0])
    
    prop = proposer(c)
    
    np.testing.assert_array_almost_equal(prop, np.zeros(2))


# ============================================================================
# Test PiKernel initialization
# ============================================================================

def test_pikernel_init():
    """Test PiKernel initialization."""
    A = ProjectorFamily([[0, 1], [2, 3]], "A")
    B = ProjectorFamily([[0, 2], [1, 3]], "B")
    grid = PiIndexGrid([A, B])
    
    piids = grid.piids
    m = len(piids)
    
    alphas = {pi: 0.25 for pi in piids}
    weights = {pi: np.ones(4) for pi in piids}
    taus = {pi: 1.5 for pi in piids}
    K = 0.05 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    kernel = PiKernel(grid, alphas, weights, taus, K)
    
    assert kernel.step_count == 0
    assert len(kernel.piids) == m


def test_pikernel_dimension_check():
    """Test that dimension mismatches raise errors."""
    A = ProjectorFamily([[0, 1], [2, 3]], "A")
    grid = PiIndexGrid([A])
    
    piids = grid.piids
    m = len(piids)
    
    alphas = {pi: 0.25 for pi in piids}
    weights = {pi: np.ones(4) for pi in piids}
    taus = {pi: 1.5 for pi in piids}
    
    # Wrong K dimension
    K_wrong = np.zeros((m+1, m+1))
    
    with pytest.raises(AssertionError):
        PiKernel(grid, alphas, weights, taus, K_wrong)


# ============================================================================
# Test PiKernel step
# ============================================================================

def test_pikernel_step_basic():
    """Test single kernel step."""
    A = ProjectorFamily([[0, 1], [2, 3]], "A")
    B = ProjectorFamily([[0, 2], [1, 3]], "B")
    grid = PiIndexGrid([A, B])
    
    piids = grid.piids
    m = len(piids)
    
    alphas = {pi: 0.25 for pi in piids}
    weights = {pi: np.ones(4) for pi in piids}
    taus = {pi: 2.0 for pi in piids}
    K = 0.05 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    kernel = PiKernel(grid, alphas, weights, taus, K)
    
    x = np.random.randn(4)
    result = kernel.step(x)
    
    assert "xnew" in result
    assert "SlopeUB" in result
    assert "GapLB" in result
    assert "touched" in result
    assert "num_touched" in result
    assert result["step"] == 0


def test_pikernel_step_contraction():
    """Test that GapLB > 0 for contractive system."""
    A = ProjectorFamily([[0, 1, 2, 3], [4, 5, 6, 7]], "A")
    grid = PiIndexGrid([A])
    
    piids = grid.piids
    m = len(piids)
    
    alphas = {pi: 0.2 for pi in piids}
    weights = {pi: np.ones(8) for pi in piids}
    taus = {pi: 1.5 for pi in piids}
    K = 0.03 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    kernel = PiKernel(grid, alphas, weights, taus, K)
    
    x = np.random.randn(8)
    result = kernel.step(x)
    
    assert result["GapLB"] > 0, "System should be contractive"
    assert result["SlopeUB"] < 1.0


def test_pikernel_step_with_ledger():
    """Test that ledger records touched atoms."""
    A = ProjectorFamily([[0, 1], [2, 3]], "A")
    grid = PiIndexGrid([A])
    
    piids = grid.piids
    m = len(piids)
    
    alphas = {pi: 0.5 for pi in piids}
    weights = {pi: np.ones(4) for pi in piids}
    taus = {pi: 0.5 for pi in piids}  # Small tau to force projection
    K = 0.05 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    ledger = Ledger()
    kernel = PiKernel(grid, alphas, weights, taus, K, ledger=ledger)
    
    x = np.random.randn(4) * 10  # Large initial state
    result = kernel.step(x)
    
    # Should have touched some atoms
    if result["num_touched"] > 0:
        assert len(ledger) > 0


# ============================================================================
# Test PiKernel run
# ============================================================================

def test_pikernel_run_multiple_steps():
    """Test running kernel for multiple steps."""
    A = ProjectorFamily([[0, 2, 4, 6], [1, 3, 5, 7]], "A")
    grid = PiIndexGrid([A])
    
    piids = grid.piids
    m = len(piids)
    
    alphas = {pi: 0.3 for pi in piids}
    weights = {pi: np.ones(8) for pi in piids}
    taus = {pi: 1.5 for pi in piids}
    K = 0.05 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    kernel = PiKernel(grid, alphas, weights, taus, K)
    
    x0 = np.random.randn(8)
    num_steps = 10
    
    result = kernel.run(x0, num_steps)
    
    assert "trajectory" in result
    assert "final_state" in result
    assert "touched_history" in result
    assert "slope_history" in result
    assert "gap_history" in result
    assert len(result["trajectory"]) == num_steps + 1  # Includes initial state


def test_pikernel_convergence():
    """Test that norm decreases (for contractive system)."""
    A = ProjectorFamily([[0, 1, 2, 3], [4, 5, 6, 7]], "A")
    grid = PiIndexGrid([A])
    
    piids = grid.piids
    m = len(piids)
    
    alphas = {pi: 0.5 for pi in piids}
    weights = {pi: np.ones(8) for pi in piids}
    taus = {pi: 1.0 for pi in piids}
    K = 0.05 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    kernel = PiKernel(grid, alphas, weights, taus, K)
    
    x0 = np.random.randn(8) * 10
    result = kernel.run(x0, num_steps=20)
    
    # Norm should decrease or stay bounded
    norms = [np.linalg.norm(x) for x in result["trajectory"]]
    
    # Final norm should be less than or equal to initial (with some tolerance)
    assert norms[-1] <= norms[0] + 1e-6


def test_pikernel_gap_always_positive():
    """Test that GapLB stays positive throughout run."""
    A = ProjectorFamily([[0, 1, 2, 3], [4, 5, 6, 7]], "A")
    grid = PiIndexGrid([A])
    
    piids = grid.piids
    m = len(piids)
    
    alphas = {pi: 0.2 for pi in piids}
    weights = {pi: np.ones(8) for pi in piids}
    taus = {pi: 1.5 for pi in piids}
    K = 0.03 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    kernel = PiKernel(grid, alphas, weights, taus, K)
    
    x0 = np.random.randn(8)
    result = kernel.run(x0, num_steps=10)
    
    # All gaps should be positive
    assert all(gap > 0 for gap in result["gap_history"])


# ============================================================================
# Integration test
# ============================================================================

def test_blueprint_example_kernel():
    """Test the full example from the blueprint."""
    # 8-dimensional space with two families
    A = ProjectorFamily([[0, 2, 4, 6], [1, 3, 5, 7]], name="FunctionType")
    B = ProjectorFamily([[0, 1, 2, 3], [4, 5, 6, 7]], name="MemoryRegion")
    grid = PiIndexGrid([A, B])
    
    piids = grid.piids
    m = len(piids)
    
    alphas = {pi: 0.25 for pi in piids}
    weights = {pi: np.ones(8) for pi in piids}
    taus = {pi: 1.5 for pi in piids}
    K = 0.05 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    ledger = Ledger()
    kernel = PiKernel(grid, alphas, weights, taus, K, ledger=ledger)
    
    np.random.seed(42)
    x = np.random.randn(8)
    
    # Run for 10 steps
    for t in range(10):
        out = kernel.step(x)
        x = out["xnew"]
        
        # Verify contraction at each step
        assert out["GapLB"] > 0, f"Contraction violated at step {t}"
    
    # Verify recomposition
    recomp_error = grid.recomposition_error(x)
    assert recomp_error < 1e-12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
