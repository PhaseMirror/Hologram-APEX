"""
π-kernel: Projection-first kernel with ACE safety and PETC ledger.

The π-kernel is a strong monoidal functor F:(A,⊗)→(C,⊕) from the atom category 
(tensoring factors) to the prime-channel category (direct sums), equipping each 
channel with ACE-consistent weights.
"""

from .projectors import ProjectorFamily, PiIndexGrid
from .l1proj import project_weighted_l1_ball
from .certificates import slope_upper_bound, gap_lower_bound
from .kernel import PiKernel, DefaultProposer
from .ledger import Ledger
from .ledgerposeidon import PoseidonLedger
from .poseidon import poseidon_hash
from .spectral import eigh_projectors
from .rns import rns_encode, rns_decode, RNSLane
from .mub_audit import mub_drift_audit
from .routing import build_channel_blocks, aggregate_to_channel, CHANNEL_MAP
from .hologram_adapter import HologramAdapter

__all__ = [
    "ProjectorFamily",
    "PiIndexGrid",
    "project_weighted_l1_ball",
    "slope_upper_bound",
    "gap_lower_bound",
    "PiKernel",
    "DefaultProposer",
    "Ledger",
    "PoseidonLedger",
    "poseidon_hash",
    "eigh_projectors",
    "rns_encode",
    "rns_decode",
    "RNSLane",
    "mub_drift_audit",
    "build_channel_blocks",
    "aggregate_to_channel",
    "CHANNEL_MAP",
    "HologramAdapter",
]
