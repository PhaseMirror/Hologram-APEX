"""
Hologram vGPU ↔ π-kernel bridge.

Integrates the three new modules (routing, mub_audit, hologram_adapter) with
the existing π-kernel scaffold to create the Hologram-specific bridge.
"""

import numpy as np
from typing import Dict, List, Optional, Any

from .projectors import ProjectorFamily, PiIndexGrid
from .kernel import PiKernel, DefaultProposer
from .routing import build_channel_blocks, aggregate_to_channel, default_route, CHANNEL_MAP
from .mub_audit import mub_drift_audit
from .ledger import Ledger
from .ledgerposeidon import PoseidonLedger


class HologramAdapter:
    """
    Adapter connecting Hologram virtual GPU operators to Multiplicity Runtime.
    
    This class orchestrates:
    1. Atom registration from Hologram compute units
    2. Semantic routing to prime channels
    3. ACE-safe kernel updates
    4. MUB drift monitoring
    5. PETC ledger emission
    """
    
    def __init__(
        self,
        families: List[ProjectorFamily],
        alphas: Dict[tuple, float],
        weights: Dict[tuple, np.ndarray],
        taus: Dict[tuple, float],
        K: np.ndarray,
        use_poseidon: bool = False,
        ledger_path: Optional[str] = None,
        mub_threshold: float = 3.0,
        tau_shrink_factor: float = 0.9,
    ):
        """
        Initialize Hologram adapter.
        
        Args:
            families: List of ProjectorFamily instances
            alphas: Per-atom mixing rates
            weights: Per-atom ACE weights
            taus: Per-atom ℓ₁ budgets
            K: Coupling matrix
            use_poseidon: Use Poseidon (BN254) ledger instead of SHA-256
            ledger_path: Path for ledger persistence
            mub_threshold: MUB drift alarm threshold
            tau_shrink_factor: Factor to shrink τ on MUB alarm
        """
        # Build π-atom grid
        self.grid = PiIndexGrid(families)
        self.piids = self.grid.piids
        
        # Store parameters
        self.alphas = alphas
        self.weights = weights
        self.taus = taus
        self.K = K
        self.mub_threshold = mub_threshold
        self.tau_shrink_factor = tau_shrink_factor
        
        # Initialize ledger
        if use_poseidon:
            self.ledger = PoseidonLedger(ledger_path)
        else:
            self.ledger = Ledger(ledger_path)
        
        # Initialize kernel
        self.kernel = PiKernel(
            self.grid,
            self.alphas,
            self.weights,
            self.taus,
            self.K,
            ledger=self.ledger,
        )
        
        # Build channel routing
        self.channel_atoms, self.channel_indices = build_channel_blocks(
            self.grid, 
            default_route
        )
        
        # Statistics
        self.mub_alarms = 0
        self.total_steps = 0
    
    def step(self, x: np.ndarray, enable_mub_audit: bool = True) -> Dict[str, Any]:
        """
        Execute one step with MUB audit and adaptive τ.
        
        Args:
            x: Current state vector
            enable_mub_audit: Whether to run MUB drift audit
        
        Returns:
            Dictionary with step results and audit status
        """
        # Execute kernel step
        result = self.kernel.step(x)
        xnew = result["xnew"]
        
        # MUB drift audit
        audit_result = None
        if enable_mub_audit:
            audit_result = mub_drift_audit(xnew, threshold=self.mub_threshold)
            
            if audit_result["alarm"]:
                self.mub_alarms += 1
                # Shrink τ for all channels to increase conservatism
                for pi in self.piids:
                    self.taus[pi] *= self.tau_shrink_factor
                
                # Reinitialize kernel with new τ values
                self.kernel = PiKernel(
                    self.grid,
                    self.alphas,
                    self.weights,
                    self.taus,
                    self.K,
                    ledger=self.ledger,
                )
        
        self.total_steps += 1
        
        # Combine results
        return {
            **result,
            "audit": audit_result,
            "mub_alarms": self.mub_alarms,
            "total_steps": self.total_steps,
        }
    
    def run(
        self, 
        x0: np.ndarray, 
        num_steps: int,
        enable_mub_audit: bool = True
    ) -> Dict[str, Any]:
        """
        Run adapter for multiple steps.
        
        Args:
            x0: Initial state
            num_steps: Number of steps to run
            enable_mub_audit: Whether to enable MUB auditing
        
        Returns:
            Dictionary with trajectory and statistics
        """
        x = x0.copy()
        trajectory = [x.copy()]
        audit_history = []
        
        for _ in range(num_steps):
            result = self.step(x, enable_mub_audit=enable_mub_audit)
            x = result["xnew"]
            trajectory.append(x.copy())
            
            if result["audit"]:
                audit_history.append(result["audit"]["D_t"])
        
        return {
            "trajectory": trajectory,
            "final_state": x,
            "mub_alarms": self.mub_alarms,
            "total_steps": self.total_steps,
            "audit_history": audit_history,
            "ledger_entries": len(self.ledger),
        }
    
    def get_channel_state(self, x: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Extract per-channel state vectors.
        
        Args:
            x: Global state vector
        
        Returns:
            Dictionary mapping channel prime → channel state
        """
        channel_states = {}
        
        for channel, indices in self.channel_indices.items():
            channel_states[channel] = x[indices]
        
        return channel_states
    
    def get_channel_names(self) -> Dict[int, str]:
        """Get semantic names for channels."""
        return {p: CHANNEL_MAP.get(p, f"channel_{p}") 
                for p in self.channel_atoms.keys()}
