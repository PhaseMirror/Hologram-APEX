"""
Tests for pikernel certificates module.
"""

import pytest
import sys
import os

# Add pikernel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pikernel.certificates import slope_upper_bound, gap_lower_bound, verify_contraction
import numpy as np


# ============================================================================
# Test slope_upper_bound
# ============================================================================

def test_slope_upper_bound_basic():
    """Test basic slope computation."""
    alphas = np.array([0.5, 0.5])
    K = np.array([[0.0, 0.1], [0.1, 0.0]])
    
    slopeUB = slope_upper_bound(alphas, K)
    
    # A = diag([0.5, 0.5]) + diag([0.5, 0.5]) @ |K|
    # Row sums: 0.5 + 0.5*0.1 = 0.55
    assert abs(slopeUB - 0.55) < 1e-10


def test_slope_upper_bound_dict_input():
    """Test slope with dict input for alphas."""
    alphas = {(0, 0): 0.3, (0, 1): 0.4}
    K = np.array([[0.0, 0.05], [0.05, 0.0]])
    
    slopeUB = slope_upper_bound(alphas, K)
    
    # Should work with dict input
    assert slopeUB > 0 and slopeUB < 1


def test_slope_upper_bound_zero_coupling():
    """Test slope with zero coupling matrix."""
    alphas = np.array([0.2, 0.3, 0.4])
    K = np.zeros((3, 3))
    
    slopeUB = slope_upper_bound(alphas, K)
    
    # A = diag(1 - alphas) since K=0
    # Max row sum = max(1 - alphas) = 1 - 0.2 = 0.8
    assert abs(slopeUB - 0.8) < 1e-10


def test_slope_upper_bound_identity():
    """Test slope with identity dynamics (alpha=0)."""
    alphas = np.zeros(3)
    K = 0.1 * np.ones((3, 3))
    np.fill_diagonal(K, 0.0)
    
    slopeUB = slope_upper_bound(alphas, K)
    
    # A = I, so slope = 1
    assert abs(slopeUB - 1.0) < 1e-10


def test_slope_upper_bound_contractive():
    """Test slope for a contractive system."""
    m = 4
    alphas = np.array([0.25] * m)
    K = 0.05 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    slopeUB = slope_upper_bound(alphas, K)
    
    # Should be strictly less than 1 for contraction
    assert slopeUB < 1.0


# ============================================================================
# Test gap_lower_bound
# ============================================================================

def test_gap_lower_bound_contractive():
    """Test gap for contractive system."""
    slopeUB = 0.85
    gapLB = gap_lower_bound(slopeUB)
    
    assert gapLB == 0.15
    assert gapLB > 0  # Positive gap indicates contraction


def test_gap_lower_bound_neutral():
    """Test gap for neutral system."""
    slopeUB = 1.0
    gapLB = gap_lower_bound(slopeUB)
    
    assert gapLB == 0.0


def test_gap_lower_bound_expansive():
    """Test gap for expansive system."""
    slopeUB = 1.1
    gapLB = gap_lower_bound(slopeUB)
    
    assert gapLB < 0  # Negative gap indicates expansion


# ============================================================================
# Test verify_contraction
# ============================================================================

def test_verify_contraction_pass():
    """Test contraction verification for a contractive system."""
    alphas = np.array([0.2, 0.2])
    K = 0.03 * np.ones((2, 2))
    np.fill_diagonal(K, 0.0)
    
    cert = verify_contraction(alphas, K)
    
    assert cert["is_contractive"] is True
    assert cert["gapLB"] > 0
    assert cert["slopeUB"] < 1.0
    assert cert["safety_margin"] > 0


def test_verify_contraction_fail():
    """Test contraction verification for a non-contractive system."""
    alphas = np.array([0.9, 0.9])
    K = 0.5 * np.ones((2, 2))
    np.fill_diagonal(K, 0.0)
    
    cert = verify_contraction(alphas, K)
    
    assert cert["is_contractive"] is False
    assert cert["slopeUB"] >= 1.0


def test_verify_contraction_certificate_fields():
    """Test that certificate contains all required fields."""
    alphas = np.array([0.25, 0.25])
    K = 0.05 * np.ones((2, 2))
    np.fill_diagonal(K, 0.0)
    
    cert = verify_contraction(alphas, K)
    
    assert "slopeUB" in cert
    assert "gapLB" in cert
    assert "is_contractive" in cert
    assert "safety_margin" in cert


# ============================================================================
# Integration tests
# ============================================================================

def test_blueprint_example():
    """Test the example from the blueprint."""
    # 4 atoms with α=0.25, K with 0.05 off-diagonal
    m = 4
    alphas = {pi: 0.25 for pi in [(0,0), (1,0), (0,1), (1,1)]}
    K = 0.05 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    slopeUB = slope_upper_bound(alphas, K)
    gapLB = gap_lower_bound(slopeUB)
    
    # Should be contractive
    assert gapLB > 0
    assert slopeUB < 0.9  # Blueprint requirement


def test_dimension_mismatch():
    """Test that mismatched dimensions raise error."""
    alphas = np.array([0.5, 0.5])
    K = np.array([[0.0, 0.1, 0.1], 
                   [0.1, 0.0, 0.1],
                   [0.1, 0.1, 0.0]])  # 3x3 but alphas has 2 elements
    
    with pytest.raises(AssertionError):
        slope_upper_bound(alphas, K)


def test_varying_alphas():
    """Test system with non-uniform mixing rates."""
    alphas = np.array([0.1, 0.2, 0.3, 0.4])
    K = 0.05 * np.ones((4, 4))
    np.fill_diagonal(K, 0.0)
    
    slopeUB = slope_upper_bound(alphas, K)
    gapLB = gap_lower_bound(slopeUB)
    
    # Should still be contractive with small K
    assert gapLB > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
