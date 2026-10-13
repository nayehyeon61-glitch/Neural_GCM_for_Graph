# Sources and scope

This project implements a new trainable, GraphCast-inspired context network.
It does not bundle either model's weights or substitute GraphCast's forecast
for a NeuralGCM physical time step.

| Source | Inspected revision | Role |
| --- | --- | --- |
| [NeuralGCM](https://github.com/neuralgcm/neuralgcm) | `56cfb4089a608413048fbc65d1ace60d5b1ae37d` | Actual `legacy.parameterizations.DivCurlNeuralParameterization`, nodal mapping, feature/tendency transformations and dynamical corrector |
| [Dinosaur](https://github.com/neuralgcm/dinosaur) | `56edc85cbfe7e695f6063d58180a0f0b684cc1ae` | Differentiable spherical primitive equations, sigma coordinates and IMEX integration |
| [Terrax](https://github.com/neuralgcm/terrax) | `aaaf72045d4a7fb152e89164386914cbe51f6794` | Transitive dependency of the inspected NeuralGCM release |
| [ECMWF ai-models-graphcast](https://github.com/ecmwf-lab/ai-models-graphcast) | `8250e3f2e68f7840de31f85e7ecbf59c25c467d9` | Reference inference / GRIB–xarray wrapper; not the mesh implementation |
| [GraphCast / WeatherNext](https://github.com/google-deepmind/weathernext) | `f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3` | Architectural reference for grid→mesh→multimesh→grid learned message passing |

GraphCast reference paths at that revision:

- `weathernext/weathernext1_graph/graphcast.py`
- `weathernext/utils/icosahedral_mesh.py`
- `weathernext/utils/legacy/grid_mesh_connectivity.py`
- `weathernext/utils/legacy/deep_typed_graph_net.py`

The original `google-deepmind/graphcast` URL currently redirects to WeatherNext.
The source code references have Apache-2.0 licenses. Model checkpoints have
separate terms; no checkpoints are distributed here. The geometry and graph
network in this repository are independent implementations of the techniques.

## Deliberate deviations from GraphCast

- Sparse nearest-neighbor grid↔mesh connectivity provides a small, reproducible
  prototype without a triangle-search dependency. Canonical GraphCast uses a
  radius graph for grid→mesh and containing-triangle vertices for mesh→grid.
- Input layout is **(channels, longitude, latitude)**, matching NeuralGCM;
  GraphCast dataset conventions commonly place latitude before longitude.
- The decoder produces context for a FiLM adapter; the final output is a learned
  physics tendency, not GraphCast's next weather state.
- Grid-cell area weights and normalized message aggregation are used for
  grid→mesh information aggregation.
- NeuralGCM's legacy Haiku interface is targeted explicitly. This is not a
  general adapter for arbitrary NNX/Flax checkpoints.

Do not install the ECMWF wrapper into this environment merely to get the graph
code: its pinned old Haiku and `graphcast` import layout are a different runtime.
