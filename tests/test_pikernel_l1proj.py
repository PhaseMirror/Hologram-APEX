"""
Tests for pikernel l1proj module.
"""

import pytest
import sys
import os

# Add pikernel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pikernel.l1proj import project_weighted_l1_ball, project_l1_ball_simple
import numpy as np


# ============================================================================
# Test project_weighted_l1_ball
# ============================================================================

def test_project_inside_ball():
    """Test projection of point already inside ball."""
    v = np.array([0.1, 0.2, 0.3])
    w = np.ones(3)
    tau = 1.0
    
    x, lam = project_weighted_l1_ball(v, w, tau)
    
    # Should return unchanged
    np.testing.assert_array_almost_equal(x, v)
    assert lam == 0.0


def test_project_outside_ball():
    """Test projection of point outside ball."""
    v = np.array([10.0, 10.0])
    w = np.ones(2)
    tau = 1.0
    
    x, lam = project_weighted_l1_ball(v, w, tau)
    
    # Projected point should be on boundary
    l1_norm = np.sum(w * np.abs(x))
    assert abs(l1_norm - tau) < 1e-10
    assert lam > 0


def test_project_weighted_asymmetric():
    """Test projection with asymmetric weights."""
    v = np.array([5.0, 5.0])
    w = np.array([1.0, 2.0])  # Second component has higher weight
    tau = 3.0
    
    x, lam = project_weighted_l1_ball(v, w, tau)
    
    # Check constraint satisfied
    weighted_l1 = np.sum(w * np.abs(x))
    assert abs(weighted_l1 - tau) < 1e-10
    
    # Second component should be shrunk more (higher weight)
    assert x[1] < x[0]


def test_project_zero_vector():
    """Test projection of zero vector."""
    v = np.zeros(3)
    w = np.ones(3)
    tau = 1.0
    
    x, lam = project_weighted_l1_ball(v, w, tau)
    
    np.testing.assert_array_almost_equal(x, v)
    assert lam == 0.0


def test_project_negative_values():
    """Test projection preserves signs."""
    v = np.array([5.0, -5.0, 3.0, -3.0])
    w = np.ones(4)
    tau = 4.0
    
    x, lam = project_weighted_l1_ball(v, w, tau)
    
    # Signs should be preserved
    assert np.all(np.sign(x) == np.sign(v))
    
    # Check constraint
    l1_norm = np.sum(w * np.abs(x))
    assert abs(l1_norm - tau) < 1e-10


def test_project_dimension_mismatch():
    """Test that mismatched dimensions raise error."""
    v = np.array([1.0, 2.0])
    w = np.array([1.0, 2.0, 3.0])
    tau = 1.0
    
    with pytest.raises(AssertionError):
        project_weighted_l1_ball(v, w, tau)


def test_project_convergence():
    """Test that bisection converges to high precision."""
    v = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    w = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    tau = 5.0
    
    x, lam = project_weighted_l1_ball(v, w, tau, atol=1e-14)
    
    weighted_l1 = np.sum(w * np.abs(x))
    assert abs(weighted_l1 - tau) < 1e-12


def test_project_large_dimension():
    """Test projection in higher dimension."""
    np.random.seed(42)
    n = 100
    v = np.random.randn(n) * 10
    w = np.ones(n)
    tau = 50.0
    
    x, lam = project_weighted_l1_ball(v, w, tau)
    
    l1_norm = np.sum(w * np.abs(x))
    assert abs(l1_norm - tau) < 1e-9
    assert lam > 0


# ============================================================================
# Test project_l1_ball_simple
# ============================================================================

def test_simple_projection():
    """Test simple (unweighted) projection."""
    v = np.array([5.0, 5.0, 5.0])
    tau = 6.0
    
    x = project_l1_ball_simple(v, tau)
    
    l1_norm = np.sum(np.abs(x))
    assert abs(l1_norm - tau) < 1e-10


def test_simple_inside():
    """Test simple projection when already inside."""
    v = np.array([0.5, 0.5])
    tau = 2.0
    
    x = project_l1_ball_simple(v, tau)
    
    np.testing.assert_array_almost_equal(x, v)


# ============================================================================
# Property-based tests
# ============================================================================

def test_projection_idempotence():
    """Test that projecting twice gives same result."""
    v = np.array([10.0, 8.0, 6.0])
    w = np.ones(3)
    tau = 5.0
    
    x1, _ = project_weighted_l1_ball(v, w, tau)
    x2, _ = project_weighted_l1_ball(x1, w, tau)
    
    np.testing.assert_array_almost_equal(x1, x2, decimal=10)


def test_projection_reduces_norm():
    """Test that projection doesn't increase norm."""
    v = np.array([10.0, 10.0])
    w = np.ones(2)
    tau = 1.0
    
    x, _ = project_weighted_l1_ball(v, w, tau)
    
    assert np.linalg.norm(x) <= np.linalg.norm(v)


def test_projection_monotone_in_tau():
    """Test that larger τ gives less shrinkage."""
    v = np.array([5.0, 5.0])
    w = np.ones(2)
    
    x1, _ = project_weighted_l1_ball(v, w, tau=2.0)
    x2, _ = project_weighted_l1_ball(v, w, tau=5.0)
    
    # Larger tau should give point closer to v
    assert np.linalg.norm(v - x2) <= np.linalg.norm(v - x1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
