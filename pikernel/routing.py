"""
Semantic channel routing map.

Maps π-atoms to prime channels based on semantic function.
Provides aggregation functions for combining per-atom proposals into channel-level weights.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Callable


# Semantic prime channel map
CHANNEL_MAP = {
    2: "attention",       # Attention heads, query/key/value
    3: "embedding",       # Token embeddings, positional encoding
    5: "loss",            # Loss computation, gradient signals
    7: "normalization",   # LayerNorm, BatchNorm, RMSNorm
    11: "feedforward",    # MLP / FFN blocks
    13: "residual",       # Skip connections, residual streams
}


def default_route(pi_id: tuple, channel_primes: List[int] = None) -> int:
    """
    Default routing function: map π-atom to prime channel.
    
    Uses the first index of the π-tuple to determine the channel.
    
    Args:
        pi_id: π-atom identifier tuple (family_A_block, family_B_block, ...)
        channel_primes: List of available prime channels (default: keys of CHANNEL_MAP)
    
    Returns:
        Prime channel number
    """
    if channel_primes is None:
        channel_primes = sorted(CHANNEL_MAP.keys())
    
    # Use first family index to select channel
    fn_type = pi_id[0]
    return channel_primes[fn_type % len(channel_primes)]


def build_channel_blocks(grid, route_fn: Callable[[tuple], int]) -> Tuple[Dict[int, List[tuple]], Dict[int, List[int]]]:
    """
    Build channel block operators from routing map.
    
    Aggregates atoms sharing a channel into a single block operator B_p.
    
    Args:
        grid: PiIndexGrid instance
        route_fn: Function mapping pi_id → channel prime
    
    Returns:
        Tuple of:
        - channel_atoms: Dict mapping channel → list of π-atoms
        - channel_indices: Dict mapping channel → sorted list of coordinate indices
    """
    channel_atoms = defaultdict(list)
    
    # Assign each atom to its channel
    for pi in grid.piids:
        p = route_fn(pi)
        channel_atoms[p].append(pi)
    
    # Build coordinate index sets for each channel
    channel_indices = {}
    for p, atoms in channel_atoms.items():
        indices = []
        for pi in atoms:
            indices.extend(grid.indices(pi))
        # Remove duplicates and sort
        channel_indices[p] = sorted(set(indices))
    
    return dict(channel_atoms), channel_indices


def aggregate_to_channel(
    kappa_by_atom: Dict[tuple, float], 
    route_fn: Callable[[tuple], int],
    method: str = "max"
) -> Dict[int, float]:
    """
    Aggregate per-atom proposals to channel level.
    
    Monotone aggregators (max, energy-weighted) preserve safety margins:
    if κ_α,t ≤ 1 for all atoms, then w_p,t ≤ 1 for all channels.
    
    Args:
        kappa_by_atom: Dict mapping pi_id → local proposal
        route_fn: Function mapping pi_id → channel prime
        method: Aggregation method
            - "max": Maximum (preserves margins)
            - "mean": Arithmetic mean
            - "energy_weighted": Energy-weighted average
    
    Returns:
        Dict mapping channel → aggregated weight
    """
    channel_weights = defaultdict(list)
    
    # Group proposals by channel
    for pi, kappa in kappa_by_atom.items():
        p = route_fn(pi)
        channel_weights[p].append(kappa)
    
    # Aggregate within each channel
    what = {}
    for p, kappas in channel_weights.items():
        if method == "max":
            what[p] = max(kappas)  # Preserves safety margin
        elif method == "mean":
            what[p] = sum(kappas) / len(kappas)
        elif method == "energy_weighted":
            energies = [k**2 for k in kappas]
            total = sum(energies)
            what[p] = total**0.5 if total > 0 else 0.0
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    return what


def create_semantic_router(
    channel_map: Dict[int, str] = None
) -> Callable[[tuple], int]:
    """
    Create a semantic router based on a channel map.
    
    Args:
        channel_map: Dict mapping prime → semantic name (default: CHANNEL_MAP)
    
    Returns:
        Routing function
    """
    if channel_map is None:
        channel_map = CHANNEL_MAP
    
    channel_primes = sorted(channel_map.keys())
    
    def router(pi_id: tuple) -> int:
        return default_route(pi_id, channel_primes)
    
    return router
