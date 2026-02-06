# π-kernel: Kernel-Multiplicity Runtime Bridge

A projection-first kernel with ACE safety, contraction certificates, and PETC ledger for the Hologram-APEX Multiplicity Runtime.

## Overview

The π-kernel is a **strong monoidal functor** F:(A,⊗)→(C,⊕) from the atom category (tensoring factors) to the prime-channel category (direct sums), equipping each channel with ACE-consistent weights. It implements:

1. **Atom Registration**: Projector families and π-atoms via intersection
2. **Routing Map**: Semantic channel assignment and aggregation
3. **Adapter Operatorization**: Projection-first ACE update with contraction certificates
4. **MUB Drift Audit**: Energy concentration detection

## Module Structure

```
pikernel/
├── __init__.py              # Public API
├── projectors.py            # ProjectorFamily, PiIndexGrid, orthogonality
├── l1proj.py                # Weighted ℓ₁-ball projection (bisection)
├── certificates.py          # SlopeUB, GapLB contraction certificates
├── kernel.py                # PiKernel, projection-first update loop
├── ledger.py                # SHA-256 PETC ledger
├── ledgerposeidon.py        # BN254 Poseidon PETC ledger
├── poseidon.py              # Educational Poseidon hash
├── spectral.py              # Spectral band projectors
├── rns.py                   # Residue Number System
├── routing.py               # Semantic channel routing
├── mub_audit.py             # Walsh-Hadamard MUB drift detection
├── hologram_adapter.py      # Hologram vGPU ↔ π-kernel bridge
├── example.py               # Baseline demonstration
└── example_spectral_poseidon.py  # Extended demonstration
```

## Quick Start

### Basic Usage

```python
from pikernel import (
    ProjectorFamily, PiIndexGrid, PiKernel,
    Ledger, slope_upper_bound, gap_lower_bound
)
import numpy as np

# 1. Define projector families (orthogonal partitions)
A = ProjectorFamily([[0, 2, 4, 6], [1, 3, 5, 7]], name="FunctionType")
B = ProjectorFamily([[0, 1, 2, 3], [4, 5, 6, 7]], name="MemoryRegion")

# 2. Build π-atom grid
grid = PiIndexGrid([A, B])
piids = grid.piids  # [(0,0), (1,0), (0,1), (1,1)]

# 3. Configure dynamics
alphas = {pi: 0.25 for pi in piids}  # Mixing rates
weights = {pi: np.ones(8) for pi in piids}  # ACE weights
taus = {pi: 1.5 for pi in piids}  # ℓ₁ budgets

# Coupling matrix
m = len(piids)
K = 0.05 * np.ones((m, m))
np.fill_diagonal(K, 0.0)

# 4. Initialize kernel
ledger = Ledger()
kernel = PiKernel(grid, alphas, weights, taus, K, ledger=ledger)

# 5. Run
x = np.random.randn(8)
for t in range(10):
    result = kernel.step(x)
    x = result["xnew"]
    
    print(f"Step {t}: SlopeUB={result['SlopeUB']:.4f}, "
          f"GapLB={result['GapLB']:.4f}")
    
    assert result["GapLB"] > 0, "Contraction violated!"
```

### With Hologram Adapter

```python
from pikernel import HologramAdapter, ProjectorFamily
import numpy as np

# Define families for 16D space
A = ProjectorFamily([
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15]
], name="SpectralBands")

B = ProjectorFamily([
    [0, 1, 2, 3, 4, 5, 6, 7],
    [8, 9, 10, 11, 12, 13, 14, 15]
], name="MemoryHalves")

# Initialize adapter with Poseidon ledger
adapter = HologramAdapter(
    families=[A, B],
    alphas={(pi): 0.2 for pi in ...},  # Configure per-atom
    weights={(pi): np.ones(16) for pi in ...},
    taus={(pi): 2.0 for pi in ...},
    K=K,  # Coupling matrix
    use_poseidon=True,
    mub_threshold=3.0
)

# Run with MUB audit
x0 = np.random.randn(16)
result = adapter.run(x0, num_steps=20, enable_mub_audit=True)

print(f"MUB alarms: {result['mub_alarms']}")
print(f"Ledger entries: {result['ledger_entries']}")
```

## Mathematical Guarantees

### Contraction Certificate (Theorem 3.1)

For iteration matrix `A = diag(1-α) + diag(α)|K|`:
- **SlopeUB** = ||A||_∞ (max row sum)
- **GapLB** = 1 - SlopeUB

When **GapLB > 0**:
- Unique bounded trajectory
- Exponential convergence to fixed point
- Stable rollback/replay

### ACE Safety Set

Projection onto weighted ℓ₁ ball:
```
S_t = {w : Σ b_p w_p ≤ 1, w ≥ 0}
```

Projection via bisection soft-threshold:
```
x_i = sign(v_i) * max(|v_i| - λw_i, 0)
```

where λ is found such that `Σ w_i|x_i| = τ`.

### Orthogonality

For commuting projector families:
- π-atoms have **disjoint supports**
- Orthogonality defect δ < 10⁻⁸
- Recomposition error ||x - Σ P_π x|| < 10⁻¹²

## Gate Criteria (Day 7)

| Criterion | Target | Status |
|-----------|--------|--------|
| Zero crashes | 0 in 10K iterations | ✅ Tested |
| GapLB > 0 | 100% compliance | ✅ Verified |
| Orthogonality defect | δ < 10⁻⁸ | ✅ Verified |
| Recomposition error | < 10⁻¹² | ✅ Verified |
| Ledger hash stability | Identical for identical payloads | ✅ Verified |
| SlopeUB bound | ≤ 0.9 all steps | ✅ Verified |
| MUB false alarm rate | < 5% | ✅ Verified |

## Running Examples

```bash
# Baseline example
python pikernel/example.py

# Extended example with Poseidon and MUB audit
python pikernel/example_spectral_poseidon.py
```

## Running Tests

```bash
# All pikernel tests
pytest tests/test_pikernel_*.py -v

# Specific modules
pytest tests/test_pikernel_projectors.py -v
pytest tests/test_pikernel_l1proj.py -v
pytest tests/test_pikernel_certificates.py -v
pytest tests/test_pikernel_kernel.py -v
pytest tests/test_pikernel_integration.py -v
```

## API Reference

### Core Classes

**ProjectorFamily**: Orthogonal coordinate projector family
- `__init__(indexsets, name)`: Initialize from index sets
- `project(x, block_id)`: Extract block coefficients
- `embed(coeffs, block_id, dim)`: Embed back to ambient space

**PiIndexGrid**: Multi-family π-atom grid
- `__init__(families)`: Build from projector families
- `indices(pi_id)`: Get coordinate indices for π-atom
- `project(x, pi_id)`: Extract π-atom coefficients
- `orthogonality_defect(pi1, pi2)`: Compute overlap
- `recomposition_error(x)`: Verify completeness

**PiKernel**: Projection-first kernel
- `__init__(grid, alphas, weights, taus, K, ...)`: Initialize
- `step(x)`: Execute one kernel step
- `run(x0, num_steps)`: Run multiple steps

**HologramAdapter**: Complete Hologram bridge
- `__init__(families, alphas, weights, taus, K, ...)`: Initialize
- `step(x, enable_mub_audit)`: Step with MUB audit
- `run(x0, num_steps, enable_mub_audit)`: Run multiple steps
- `get_channel_state(x)`: Extract per-channel states

### Functions

**project_weighted_l1_ball(v, w, tau)**: ACE projection via bisection

**slope_upper_bound(alphas, K)**: Compute ||A||_∞

**gap_lower_bound(slopeUB)**: Compute 1 - SlopeUB

**mub_drift_audit(x, threshold)**: Walsh-Hadamard concentration audit

## References

- Blueprint: "Kernel-Multiplicity Runtime Bridge: Detailed Coding Blueprint (Phase 0A)"
- Mathematical foundation: Theorem 3.1 (contraction), Corollary 3.2 (local-to-global)
- Safety: ACE ℓ₁-budgeted projection with KKT certificate
- Audit: MUB drift detection via Walsh-Hadamard transform

## License

See repository LICENSE file.
