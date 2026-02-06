"""
Residue Number System (RNS) encode/decode for carry-free arithmetic.

RNS represents integers as tuples of residues modulo coprime bases.
This enables parallel lane-wise arithmetic without carry propagation.
"""

import numpy as np
from typing import List, Tuple


class RNSLane:
    """
    Single RNS lane with a modulus.
    
    Supports lane-wise arithmetic operations.
    """
    
    def __init__(self, modulus: int):
        """
        Initialize RNS lane.
        
        Args:
            modulus: Prime modulus for this lane
        """
        self.modulus = modulus
    
    def encode(self, x: int) -> int:
        """Encode integer as residue modulo this lane's modulus."""
        return x % self.modulus
    
    def add(self, a: int, b: int) -> int:
        """Lane-wise addition."""
        return (a + b) % self.modulus
    
    def mul(self, a: int, b: int) -> int:
        """Lane-wise multiplication."""
        return (a * b) % self.modulus
    
    def sub(self, a: int, b: int) -> int:
        """Lane-wise subtraction."""
        return (a - b) % self.modulus


def rns_encode(x: int, moduli: List[int]) -> List[int]:
    """
    Encode integer in RNS representation.
    
    Args:
        x: Integer to encode
        moduli: List of coprime moduli
    
    Returns:
        List of residues [x mod m_0, x mod m_1, ...]
    """
    return [x % m for m in moduli]


def rns_decode(residues: List[int], moduli: List[int]) -> int:
    """
    Decode RNS representation to integer using Chinese Remainder Theorem.
    
    Args:
        residues: List of residues
        moduli: List of coprime moduli
    
    Returns:
        Decoded integer (unique modulo product of moduli)
    """
    assert len(residues) == len(moduli), "Residues and moduli must have same length"
    
    # Compute M = product of all moduli
    M = 1
    for m in moduli:
        M *= m
    
    # CRT reconstruction
    x = 0
    for r_i, m_i in zip(residues, moduli):
        M_i = M // m_i
        # Find modular inverse of M_i modulo m_i
        y_i = pow(M_i, -1, m_i)  # Python 3.8+ modular inverse
        x += r_i * M_i * y_i
    
    return x % M


def rns_add(a_residues: List[int], b_residues: List[int], moduli: List[int]) -> List[int]:
    """
    RNS addition: add corresponding residues lane-wise.
    
    Args:
        a_residues: RNS representation of a
        b_residues: RNS representation of b
        moduli: List of moduli
    
    Returns:
        RNS representation of (a + b)
    """
    assert len(a_residues) == len(b_residues) == len(moduli)
    return [(a + b) % m for a, b, m in zip(a_residues, b_residues, moduli)]


def rns_mul(a_residues: List[int], b_residues: List[int], moduli: List[int]) -> List[int]:
    """
    RNS multiplication: multiply corresponding residues lane-wise.
    
    Args:
        a_residues: RNS representation of a
        b_residues: RNS representation of b
        moduli: List of moduli
    
    Returns:
        RNS representation of (a * b)
    """
    assert len(a_residues) == len(b_residues) == len(moduli)
    return [(a * b) % m for a, b, m in zip(a_residues, b_residues, moduli)]


def rns_lane_projector_family(moduli: List[int], name: str = "RNSLanes"):
    """
    Create a ProjectorFamily where each block corresponds to an RNS lane.
    
    This is useful when the ambient dimension is the total dynamic range
    and each lane handles a subset of that range.
    
    Args:
        moduli: List of RNS moduli
        name: Name for projector family
    
    Returns:
        ProjectorFamily with one block per lane
    """
    from .projectors import ProjectorFamily
    
    # Simple partitioning: divide ambient dimension equally among lanes
    num_lanes = len(moduli)
    # For demo purposes, assume ambient dimension is a multiple of num_lanes
    # In practice, this would be determined by the application
    
    # This is a placeholder - real implementation would depend on
    # how RNS lanes map to coordinate indices
    indexsets = []
    # Example: if we have 12 coordinates and 3 lanes, each gets 4 coordinates
    # This would be customized based on actual RNS structure
    
    return ProjectorFamily(indexsets, name)
