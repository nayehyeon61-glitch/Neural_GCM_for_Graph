# Research protocol

## Hypothesis

The differentiable primitive-equation core already transports information
globally. The question is whether conditioning unresolved local physics on
multiscale atmospheric context improves forecasting and circulation statistics.

Do not infer that adding a graph enforces closed circulation, conservation,
geostrophic balance, or rotation equivariance. No such guarantees are claimed.

## Controlled experiments

Run A/B/C/D with identical data splits, field normalization, rollout schedule,
optimizer budget and random seeds. Report both parameter count and wall time.
Compare multiple widths for B versus C to control for capacity; equal width does
not give equal parameter count. Also compare mesh local edges only against the
multiscale union to isolate coarse long-range edges.

Use at least several independent seeds before attributing a small improvement
to the architecture. Define metrics and validation-based model selection before
examining the held-out period.

## Training data contract

- Horizontal order inside the model is `(longitude, latitude)`; use explicit
  coordinates and verify ordering when converting ERA5 `(latitude, longitude)`.
- Gaussian nodes require Gaussian quadrature weights, not a uniform-latitude
  approximation. The NeuralGCM integration extracts them from the model grid.
- A column may combine temperature, humidity, wind, pressure, geopotential,
  static terrain/land mask and forcing features. Features and tendency units
  must match the original physics specification and nondimensional time.
- Train on causal input histories; future reference fields belong only in the
  supervised loss. Train-only normalization prevents temporal leakage.
- Jointly train the local parameterization and graph with gradients through the
  original dynamical core. Recompute context from each current rollout state.

## Stability and interpretation

Zero-initialized FiLM exactly suppresses the context branch's initial effect
for fixed local weights. The FiLM output head receives gradients first; deeper
graph weights usually start receiving gradients after that head changes.
Bounding FiLM modulation is a useful initialization/stability control, not a
proof of stable long climate simulations.

The initial integration is designed for a single complete horizontal grid on
one device. Distributed graph partitions and communication need separate work.
Increasing mesh refinement or running a graph at every physics update
can substantially increase runtime and memory; begin at refinement 1–2.
The supplied upstream corrector holds the learned tendency fixed during the
internal IMEX stages of one physics update, then recomputes it at the next
outer physics step.

## Circulation diagnostics

For northward wind on pressure levels, a useful Hadley diagnostic is

$$
\Psi(\phi,p) = \frac{2\pi a\cos\phi}{g}
\int_{p_{top}}^p [v](\phi,p')\,dp'.
$$

The pressure integration uses pascals, wind uses m/s, and the result is kg/s.
With pressure samples whose first level is above zero pressure, the diagnostic
starts at that supplied top level; it does not invent an unsampled top layer.
It is not directly a sigma-coordinate formula. Convert to physical pressure
levels with surface pressure and mask underground values appropriately.
The standalone diagnostic is not a continuity constraint and cannot ensure a
closed three-dimensional overturning circulation.

The synthetic D experiment uses a simple spatial-mean circulation proxy.
For actual D experiments, evaluate the physically dimensioned diagnostic from
predicted and observed pressure-level winds and normalize its loss independently
of field RMSE. Walker circulation requires a separate longitude–pressure
diagnostic, a stated latitude band, and vertical velocity consistency checks.

`neuralgcm_graph.losses.mass_streamfunction_mse` supplies this differentiable
Hadley diagnostic supervision directly:

```python
from neuralgcm_graph.losses import mass_streamfunction_mse

# v_pred / v_target: [batch, time, pressure, longitude, latitude], m/s
# pressure_pa: strictly increasing top-to-bottom physical pressure levels
loss_circulation = mass_streamfunction_mse(
    v_pred, v_target, pressure_pa, latitude_deg,
    scale_kg_s=training_reference_streamfunction_scale,
    latitude_weights=latitude_band_areas,
    longitude_weights=longitude_cell_widths,
)
loss = loss_forecast + circulation_weight * loss_circulation
```

Pressure samples receive equal loss weight in this implementation, while the
streamfunction itself uses trapezoidal pressure integration. Choose the loss
scale using the training partition. This loss does not perform pressure/sigma
conversion or handling of below-ground samples for the caller.

## Acceptance criteria

1. Finite gradients through more than one physical time step.
2. No unintended spatial transpose or disconnected data transfer.
3. Adapter initialization identity relative to the same local network.
4. Improvement on held-out leads and circulation statistics across seeds.
5. No unacceptable long-run drift or instability relative to A/B.

The supplied smoke tests address the first three criteria only.
