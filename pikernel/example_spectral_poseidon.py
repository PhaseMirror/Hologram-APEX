"""
Extended π-kernel demonstration with spectral bands and Poseidon ledger.

Shows advanced features:
- Spectral band projectors
- Poseidon (BN254) ledger
- MUB drift audit
- Hologram adapter integration
"""

import numpy as np
from pikernel import (
    ProjectorFamily, PiIndexGrid, PoseidonLedger,
    HologramAdapter, mub_drift_audit
)


def main():
    """Run extended π-kernel demo with spectral and Poseidon."""
    print("=" * 70)
    print("π-kernel Extended Demo: Spectral + Poseidon + MUB Audit")
    print("=" * 70)
    
    # 1. ATOM REGISTRATION WITH SPECTRAL STRUCTURE
    print("\n1. Atom Registration (Spectral)")
    print("-" * 70)
    
    # Use power-of-2 dimension for MUB audit (Hadamard transform)
    dim = 16
    
    # Family A: 4 blocks
    A = ProjectorFamily([
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15]
    ], name="SpectralBands")
    
    # Family B: 2 blocks
    B = ProjectorFamily([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15]
    ], name="MemoryHalves")
    
    print(f"Family A ({A.name}): {A.numblocks} blocks")
    print(f"Family B ({B.name}): {B.numblocks} blocks")
    
    families = [A, B]
    grid = PiIndexGrid(families)
    piids = grid.piids
    m = len(piids)
    print(f"\nπ-atoms: {m} atoms")
    
    # 2. DYNAMICS WITH TIGHTER COUPLING
    print("\n2. Dynamics Parameters")
    print("-" * 70)
    
    alphas = {pi: 0.2 for pi in piids}
    weights = {pi: np.ones(dim) for pi in piids}
    taus = {pi: 2.0 for pi in piids}
    
    # Coupling matrix
    K = 0.03 * np.ones((m, m))
    np.fill_diagonal(K, 0.0)
    
    # 3. HOLOGRAM ADAPTER WITH POSEIDON
    print("\n3. Hologram Adapter Initialization")
    print("-" * 70)
    
    adapter = HologramAdapter(
        families=families,
        alphas=alphas,
        weights=weights,
        taus=taus,
        K=K,
        use_poseidon=True,  # Use Poseidon instead of SHA-256
        ledger_path=None,   # In-memory
        mub_threshold=3.0,
        tau_shrink_factor=0.9
    )
    print("Hologram adapter initialized with Poseidon ledger")
    
    # Display channel routing
    channel_names = adapter.get_channel_names()
    print(f"\nChannel routing:")
    for channel, name in sorted(channel_names.items()):
        atoms = adapter.channel_atoms[channel]
        print(f"  Channel {channel} ({name}): {len(atoms)} atoms")
    
    # 4. RUN WITH MUB AUDIT
    print("\n4. Execution with MUB Audit")
    print("-" * 70)
    
    # Random initial state
    np.random.seed(42)
    x = np.random.randn(dim)
    print(f"Initial state: ||x|| = {np.linalg.norm(x):.6f}")
    
    # Run for 20 steps
    num_steps = 20
    x_current = x.copy()
    
    for t in range(num_steps):
        result = adapter.step(x_current, enable_mub_audit=True)
        x_current = result["xnew"]
        
        audit = result["audit"]
        alarm_marker = " [ALARM]" if audit["alarm"] else ""
        
        print(f"Step {t:2d}: "
              f"SlopeUB={result['SlopeUB']:.4f}, "
              f"GapLB={result['GapLB']:.4f}, "
              f"D_t={audit['D_t']:.4f}, "
              f"touched={result['num_touched']}"
              f"{alarm_marker}")
    
    # 5. RESULTS
    print("\n5. Results")
    print("-" * 70)
    print(f"Total steps: {num_steps}")
    print(f"MUB alarms: {adapter.mub_alarms}")
    print(f"Alarm rate: {adapter.mub_alarms / num_steps * 100:.1f}%")
    print(f"Ledger entries: {len(adapter.ledger)}")
    print(f"Final state norm: {np.linalg.norm(x_current):.6f}")
    
    # Sample ledger entry
    if len(adapter.ledger) > 0:
        print(f"\nSample ledger entry:")
        entry = adapter.ledger.entries[0]
        print(f"  Step: {entry.get('step', 'N/A')}")
        print(f"  π-atom: {entry.get('pi', 'N/A')}")
        print(f"  α: {entry.get('alpha', 'N/A'):.4f}")
        print(f"  τ: {entry.get('tau', 'N/A'):.4f}")
        print(f"  Digest: {entry.get('digest', 'N/A')[:16]}...")
        print(f"  Hash type: {entry.get('hash_type', 'sha256')}")
    
    # Final MUB audit
    print("\nFinal MUB Audit:")
    final_audit = mub_drift_audit(x_current, threshold=3.0)
    print(f"  D_t: {final_audit['D_t']:.4f}")
    print(f"  Max energy: {final_audit['max_energy']:.6f}")
    print(f"  Mean energy: {final_audit['mean_energy']:.6f}")
    print(f"  Alarm: {final_audit['alarm']}")
    
    # Channel-wise analysis
    print("\nChannel-wise State:")
    channel_states = adapter.get_channel_state(x_current)
    for channel in sorted(channel_states.keys()):
        state = channel_states[channel]
        name = channel_names[channel]
        print(f"  Channel {channel} ({name}): "
              f"dim={len(state)}, ||x||={np.linalg.norm(state):.4f}")
    
    print("\n" + "=" * 70)
    print("Extended demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
