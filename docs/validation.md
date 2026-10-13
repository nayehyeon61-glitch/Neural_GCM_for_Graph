# Validation record

Environment: Python 3.12.14, CPU, JAX/JAXlib 0.11.1, dm-haiku 0.0.17,
NumPy 2.5.4, SciPy 1.18.1, Optax 0.2.8, NeuralGCM 1.2.3 at the commit
pinned in `pyproject.toml`.

## Automated tests

`pytest -q`: **48 passed** in 40.20 seconds, including the optional real-core
integration tests. Two upstream Optax deprecation warnings concern
`optax.global_norm`; they did not affect the results.

- 18 geometry cases: nested icosphere sizes, unit sphere, multiscale links,
  sparse connectivity coverage, longitude seam, polar geometry, quadrature.
- 15 model cases: JIT/vmap, finite derivatives, zero-FiLM identity relative to
  the same local parameters, gradient activation after a FiLM update, and
  nonlocal input influence.
- 8 integration cases: actual Gaussian coordinates/weights, incorrect but
  equal-shaped grid rejection, and gradients through a two-step primitive
  equations rollout, including mesh messages and the initial state.
- 7 training/loss cases: full rollout gradient, analytical weighted losses,
  pressure-coordinate streamfunction units and derivatives, and atomic
  rejection of nonfinite optimizer updates.

## Actual dynamical-core smoke

`python examples/neuralgcm_smoke.py` runs the upstream NeuralGCM
`DycoreWithPhysicsCorrector` and Dinosaur primitive equations with IMEX RK SIL3.
It uses a synthetic isothermal atmosphere on a 13×7 Gaussian grid, two sigma
levels, two 60-second physics steps, and a small nonzero FiLM initialization
to expose the whole GNN gradient immediately.

| Observation | Result |
| --- | ---: |
| Finite predicted trajectory | true |
| Parameter gradient norm | 0.01385025 |
| Synthetic loss before update | 0.53846139 |
| Synthetic loss after one Adam update | 0.53842074 |

The full-core test additionally checks nonzero gradients in grid-to-mesh,
mesh interaction, FiLM, and the initial state. The default model initialization
remains zero-FiLM; its delayed graph-gradient behavior is separately tested.

## Synthetic optimization

The shipped C configuration was run with:

```bash
python examples/train_toy.py --config configs/C_mesh.json --steps 30 --output runs/C_mesh_validation.json
```

This uses a 16×8 grid, width 32, four message-passing blocks, three predicted
steps, seed 0, four synthetic training trajectories and two separate evaluation
trajectories. All 30 optimizer updates were accepted.

| Metric | Result |
| --- | ---: |
| Initial training loss | 0.160856 |
| Final training loss | 0.00416250 |
| Final evaluation loss | 0.00707207 |

This establishes that the implementation can optimize the synthetic objective.
All four A/B/C/D CLI paths also completed two accepted updates on a smaller
8×4 grid with width 8 and one message-passing block. A/B/C started with exactly
the same local-network loss after copying the shared initial local weights.
It is not evidence of improvement over NeuralGCM, GraphCast, or a climate
baseline. No ERA5 experiment, trained weather checkpoint, long-range climate
stability evaluation, or GPU benchmark was performed in this implementation
session. Generated run outputs are excluded from source control by default.

The documented pinned-extra installation was also dependency-resolved with
`uv pip install --dry-run -e '.[test,neuralgcm]' -c constraints-tested.txt`.
