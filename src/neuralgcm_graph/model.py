"""Sparse global context for a local, column-wise physics parameterization.

The mesh supplies context to the learned physics; it does not replace the
resolved dynamical core. This is a newly trained research architecture, not a
loader for GraphCast or NeuralGCM pretrained parameter trees.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral

import haiku as hk
import jax
import jax.numpy as jnp

from neuralgcm_graph.geometry import EdgeSet, GraphGeometry


@dataclass(frozen=True)
class ModelConfig:
    """Architecture and bounded feature modulation settings.

    ``none`` is column-local physics; ``local`` adds grid-neighbor context;
    ``mesh`` uses grid-to-mesh, multiscale mesh processing, and mesh-to-grid.
    ``film_init_scale=0`` gives the exact column-local prediction initially.
    Consequently the context network starts receiving gradients only after the
    FiLM output weights have moved away from zero.
    """

    context_mode: str = "mesh"
    latent_size: int = 32
    message_passing_steps: int = 4
    film_scale: float = 0.1
    film_init_scale: float = 0.0

    def __post_init__(self) -> None:
        if self.context_mode not in ("mesh", "local", "none"):
            raise ValueError("context_mode must be 'mesh', 'local', or 'none'")
        for name in ("latent_size", "message_passing_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("film_scale", "film_init_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _aggregate(
    messages: jax.Array,
    receivers: jax.Array,
    node_count: int,
    weights: jax.Array | None = None,
) -> jax.Array:
    """Sparse incoming-message mean, optionally weighted by source-cell area.

    Nodes with no incoming edges receive zero. The operation creates an
    ``[node_count, latent_size]`` array, never a dense adjacency matrix.
    """
    receivers = jnp.asarray(receivers, dtype=jnp.int32)
    if weights is None:
        weights = jnp.ones((messages.shape[0],), dtype=messages.dtype)
    weights = jnp.asarray(weights, dtype=messages.dtype).reshape((-1, 1))
    numerator = jax.ops.segment_sum(
        messages * weights, receivers, num_segments=node_count
    )
    denominator = jax.ops.segment_sum(weights, receivers, num_segments=node_count)
    return numerator / jnp.maximum(denominator, jnp.finfo(messages.dtype).tiny)


class GraphConditionedPhysics(hk.Module):
    """Map ``[channels, longitude, latitude]`` features to local tendencies.

    Use this module inside :func:`haiku.transform`. Batch with an external
    ``jax.vmap`` over the transformed apply function. The immutable topology
    is captured in the transform closure, so connectivity remains static under
    JIT. Inputs should be normalized by the caller; physical units and tendency
    transforms belong to the surrounding dynamical model.

    The column encoder and physics decoder retain the same parameter names
    across all context modes. A ``none`` transform can therefore apply the
    parameter tree initialized by a ``mesh`` or ``local`` transform, allowing an
    exact comparison when FiLM is zero initialized.
    """

    def __init__(
        self,
        output_size: int,
        topology: GraphGeometry,
        config: ModelConfig = ModelConfig(),
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if isinstance(output_size, bool) or not isinstance(output_size, Integral) or output_size < 1:
            raise ValueError("output_size must be a positive integer")
        self.output_size = int(output_size)
        self.topology = topology
        self.config = config

    def _mlp(self, values: jax.Array, name: str) -> jax.Array:
        width = self.config.latent_size
        return hk.nets.MLP(
            (width, width), activation=jax.nn.silu, name=name
        )(values)

    def _interaction(
        self,
        nodes: jax.Array,
        edge_latents: jax.Array,
        edges: EdgeSet,
        name: str,
    ) -> tuple[jax.Array, jax.Array]:
        senders = jnp.asarray(edges.senders, dtype=jnp.int32)
        receivers = jnp.asarray(edges.receivers, dtype=jnp.int32)
        update = self._mlp(
            jnp.concatenate((edge_latents, nodes[senders], nodes[receivers]), axis=-1),
            name=f"{name}_edge",
        )
        edge_latents = edge_latents + update
        incoming = _aggregate(edge_latents, receivers, nodes.shape[0])
        nodes = nodes + self._mlp(
            jnp.concatenate((nodes, incoming), axis=-1), name=f"{name}_node"
        )
        return nodes, edge_latents

    def _mesh_context(self, columns: jax.Array) -> jax.Array:
        topology = self.topology
        grid_xyz = jnp.asarray(topology.grid_xyz, dtype=columns.dtype)
        mesh_xyz = jnp.asarray(topology.mesh_xyz, dtype=columns.dtype)
        mesh = self._mlp(mesh_xyz, "mesh_position_encoder")
        g2m = topology.grid_to_mesh
        senders = jnp.asarray(g2m.senders, dtype=jnp.int32)
        receivers = jnp.asarray(g2m.receivers, dtype=jnp.int32)
        messages = self._mlp(
            jnp.concatenate(
                (
                    columns[senders],
                    grid_xyz[senders],
                    mesh[receivers],
                    jnp.asarray(g2m.features, dtype=columns.dtype),
                ),
                axis=-1,
            ),
            "grid_to_mesh_edge",
        )
        area_weights = jnp.asarray(topology.grid_area_weights).reshape(-1)[senders]
        incoming = _aggregate(messages, receivers, mesh.shape[0], area_weights)
        mesh = mesh + self._mlp(
            jnp.concatenate((mesh, incoming), axis=-1), "grid_to_mesh_node"
        )
        mesh_edges = topology.mesh_to_mesh
        edge_latents = self._mlp(
            jnp.asarray(mesh_edges.features, dtype=columns.dtype), "mesh_edge_encoder"
        )
        for step in range(self.config.message_passing_steps):
            mesh, edge_latents = self._interaction(
                mesh, edge_latents, mesh_edges, f"mesh_interaction_{step}"
            )
        m2g = topology.mesh_to_grid
        senders = jnp.asarray(m2g.senders, dtype=jnp.int32)
        receivers = jnp.asarray(m2g.receivers, dtype=jnp.int32)
        messages = self._mlp(
            jnp.concatenate(
                (
                    mesh[senders],
                    columns[receivers],
                    jnp.asarray(m2g.features, dtype=columns.dtype),
                ),
                axis=-1,
            ),
            "mesh_to_grid_edge",
        )
        incoming = _aggregate(messages, receivers, columns.shape[0])
        return self._mlp(
            jnp.concatenate((columns, grid_xyz, incoming), axis=-1),
            "mesh_to_grid_node",
        )

    def _local_context(self, columns: jax.Array) -> jax.Array:
        edges = self.topology.grid_to_grid
        nodes = columns + self._mlp(
            jnp.asarray(self.topology.grid_xyz, dtype=columns.dtype),
            "local_position_encoder",
        )
        edge_latents = self._mlp(
            jnp.asarray(edges.features, dtype=columns.dtype), "local_edge_encoder"
        )
        for step in range(self.config.message_passing_steps):
            nodes, edge_latents = self._interaction(
                nodes, edge_latents, edges, f"local_interaction_{step}"
            )
        return nodes

    def __call__(self, features: jax.Array) -> jax.Array:
        features = jnp.asarray(features)
        if features.ndim != 3 or tuple(features.shape[-2:]) != tuple(self.topology.grid_shape):
            raise ValueError(
                "features must have shape (channels, nlon, nlat), with grid "
                f"{self.topology.grid_shape}; got {features.shape}"
            )
        if features.shape[0] < 1:
            raise ValueError("features must have at least one channel")
        features = features.astype(jnp.result_type(features.dtype, jnp.float32))
        columns = jnp.moveaxis(features, 0, -1).reshape((-1, features.shape[0]))
        latent = self._mlp(columns, "column_encoder")
        if self.config.context_mode != "none":
            if self.config.context_mode == "mesh":
                context = self._mesh_context(latent)
            else:
                context = self._local_context(latent)
            modulation = hk.Linear(
                2 * self.config.latent_size,
                w_init=hk.initializers.RandomNormal(stddev=self.config.film_init_scale),
                b_init=hk.initializers.Constant(0.0),
                name="film_modulation",
            )(context)
            gamma, beta = jnp.split(
                self.config.film_scale * jnp.tanh(modulation), 2, axis=-1
            )
            latent = (1.0 + gamma) * latent + beta
        tendencies = hk.nets.MLP(
            (self.config.latent_size, self.output_size),
            activation=jax.nn.silu,
            name="physics_decoder",
        )(latent)
        tendencies = tendencies.reshape((*self.topology.grid_shape, self.output_size))
        return jnp.moveaxis(tendencies, -1, 0)
