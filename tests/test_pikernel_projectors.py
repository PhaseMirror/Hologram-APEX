"""
Tests for pikernel projectors module.
"""

import pytest
import sys
import os

# Add pikernel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pikernel.projectors import ProjectorFamily, PiIndexGrid, intersect_many


# ============================================================================
# Test intersect_many
# ============================================================================

def test_intersect_many_empty():
    """Test intersection of empty list."""
    assert intersect_many([]) == []


def test_intersect_many_single():
    """Test intersection of single set."""
    assert intersect_many([[1, 2, 3]]) == [1, 2, 3]


def test_intersect_many_disjoint():
    """Test intersection of disjoint sets."""
    assert intersect_many([[1, 2], [3, 4]]) == []


def test_intersect_many_overlapping():
    """Test intersection of overlapping sets."""
    result = intersect_many([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    assert result == [3]


# ============================================================================
# Test ProjectorFamily
# ============================================================================

def test_projector_family_basic():
    """Test basic ProjectorFamily creation."""
    family = ProjectorFamily([[0, 1], [2, 3]], "Test")
    assert family.name == "Test"
    assert family.dim == 4
    assert family.numblocks == 2


def test_projector_family_disjoint_check():
    """Test that overlapping indices raise error."""
    with pytest.raises(AssertionError):
        ProjectorFamily([[0, 1], [1, 2]], "Overlapping")


def test_projector_family_blockindices():
    """Test blockindices method."""
    family = ProjectorFamily([[0, 2, 4], [1, 3, 5]], "Even-Odd")
    assert family.blockindices(0) == [0, 2, 4]
    assert family.blockindices(1) == [1, 3, 5]


def test_projector_family_project():
    """Test projection extraction."""
    import numpy as np
    family = ProjectorFamily([[0, 1], [2, 3]], "Pairs")
    x = np.array([1.0, 2.0, 3.0, 4.0])
    
    proj0 = family.project(x, 0)
    assert list(proj0) == [1.0, 2.0]
    
    proj1 = family.project(x, 1)
    assert list(proj1) == [3.0, 4.0]


def test_projector_family_embed():
    """Test embedding back into ambient space."""
    import numpy as np
    family = ProjectorFamily([[0, 1], [2, 3]], "Pairs")
    
    coeffs = np.array([5.0, 6.0])
    embedded = family.embed(coeffs, 0, 4)
    
    assert embedded[0] == 5.0
    assert embedded[1] == 6.0
    assert embedded[2] == 0.0
    assert embedded[3] == 0.0


# ============================================================================
# Test PiIndexGrid
# ============================================================================

def test_piindexgrid_basic():
    """Test basic PiIndexGrid creation."""
    A = ProjectorFamily([[0, 2], [1, 3]], "A")
    B = ProjectorFamily([[0, 1], [2, 3]], "B")
    
    grid = PiIndexGrid([A, B])
    assert grid.dim == 4
    assert len(grid.piids) == 4  # 2x2 = 4 atoms


def test_piindexgrid_indices():
    """Test π-atom index retrieval."""
    A = ProjectorFamily([[0, 2], [1, 3]], "A")
    B = ProjectorFamily([[0, 1], [2, 3]], "B")
    
    grid = PiIndexGrid([A, B])
    
    # Atom (0, 0): intersection of A[0]={0,2} and B[0]={0,1} = {0}
    assert grid.indices((0, 0)) == [0]
    
    # Atom (0, 1): intersection of A[0]={0,2} and B[1]={2,3} = {2}
    assert grid.indices((0, 1)) == [2]
    
    # Atom (1, 0): intersection of A[1]={1,3} and B[0]={0,1} = {1}
    assert grid.indices((1, 0)) == [1]
    
    # Atom (1, 1): intersection of A[1]={1,3} and B[1]={2,3} = {3}
    assert grid.indices((1, 1)) == [3]


def test_piindexgrid_project():
    """Test π-atom projection."""
    import numpy as np
    
    A = ProjectorFamily([[0, 2], [1, 3]], "A")
    B = ProjectorFamily([[0, 1], [2, 3]], "B")
    grid = PiIndexGrid([A, B])
    
    x = np.array([10.0, 20.0, 30.0, 40.0])
    
    # Extract each atom
    assert grid.project(x, (0, 0))[0] == 10.0
    assert grid.project(x, (1, 0))[0] == 20.0
    assert grid.project(x, (0, 1))[0] == 30.0
    assert grid.project(x, (1, 1))[0] == 40.0


def test_piindexgrid_orthogonality_defect():
    """Test orthogonality defect computation."""
    A = ProjectorFamily([[0, 2], [1, 3]], "A")
    B = ProjectorFamily([[0, 1], [2, 3]], "B")
    grid = PiIndexGrid([A, B])
    
    # Different atoms should have defect 0 (disjoint supports)
    assert grid.orthogonality_defect((0, 0), (1, 0)) == 0.0
    assert grid.orthogonality_defect((0, 0), (0, 1)) == 0.0
    
    # Same atom should have defect 1 (identity on support)
    assert grid.orthogonality_defect((0, 0), (0, 0)) == 1.0


def test_piindexgrid_recomposition_error():
    """Test recomposition error (should be zero for orthogonal families)."""
    import numpy as np
    
    A = ProjectorFamily([[0, 2], [1, 3]], "A")
    B = ProjectorFamily([[0, 1], [2, 3]], "B")
    grid = PiIndexGrid([A, B])
    
    x = np.array([1.0, 2.0, 3.0, 4.0])
    error = grid.recomposition_error(x)
    
    assert error < 1e-12  # Should be essentially zero


def test_piindexgrid_dimension_mismatch():
    """Test that families with different dimensions raise error."""
    A = ProjectorFamily([[0, 1]], "A")
    B = ProjectorFamily([[0, 1, 2]], "B")
    
    with pytest.raises(AssertionError):
        PiIndexGrid([A, B])


# ============================================================================
# Integration Tests
# ============================================================================

def test_example_8d_grid():
    """Test the 8D example from the blueprint."""
    A = ProjectorFamily([[0, 2, 4, 6], [1, 3, 5, 7]], name="FunctionType")
    B = ProjectorFamily([[0, 1, 2, 3], [4, 5, 6, 7]], name="MemoryRegion")
    
    grid = PiIndexGrid([A, B])
    
    # Should have 4 atoms
    assert len(grid.piids) == 4
    assert (0, 0) in grid.piids
    assert (0, 1) in grid.piids
    assert (1, 0) in grid.piids
    assert (1, 1) in grid.piids
    
    # Check atom sizes
    assert len(grid.indices((0, 0))) == 2  # {0, 2}
    assert len(grid.indices((0, 1))) == 2  # {4, 6}
    assert len(grid.indices((1, 0))) == 2  # {1, 3}
    assert len(grid.indices((1, 1))) == 2  # {5, 7}
    
    # Verify orthogonality
    for pi1 in grid.piids:
        for pi2 in grid.piids:
            if pi1 != pi2:
                assert grid.orthogonality_defect(pi1, pi2) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
