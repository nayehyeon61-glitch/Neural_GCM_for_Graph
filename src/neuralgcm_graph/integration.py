"""NeuralGCM's legacy Haiku parameterization hook.

Only the nodal learned-physics mapping is replaced. NeuralGCM still owns
feature preparation, tendency units, velocity-to-divergence/curl conversion,
spectral transforms, prediction masks, filters, and the dynamical core.

This module requires the optional ``neuralgcm`` dependencies. Geometry must be
built on the host before entering a Haiku transform or JAX tracing context.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import gin
from neuralgcm.legacy import mappings, parameterizations
import numpy as np

from neuralgcm_graph.geometry import GraphGeometry, build_graph
from neuralgcm_graph.model import GraphConditionedPhysics, ModelConfig


def _validate_graph_coordinates(coords: Any, topology: GraphGeometry) -> None:
    """Check host-side coordinate constants, including their flattened order."""
    horizontal = coords.horizontal
    expected_shape = tuple(horizontal.nodal_shape)
    if tuple(topology.grid_shape) != expected_shape:
        raise ValueError(
            f"Graph grid {topology.grid_shape} does not match the "
            f"NeuralGCM nodal grid {expected_shape}; preserve (lon, lat) order."
        )
    # These are NumPy constants captured with the graph, not model inputs or
    # traced JAX values. Validation therefore does not enter the compiled graph.
    expected_lon = np.rad2deg(np.asarray(horizontal.longitudes))
    expected_lat = np.rad2deg(np.asarray(horizontal.latitudes))
    actual_lon = np.asarray(topology.longitudes)
    actual_lat = np.asarray(topology.latitudes)
    if actual_lon.shape != expected_lon.shape or not np.all(
        np.abs((actual_lon - expected_lon + 180.0) % 360.0 - 180.0) <= 1e-5
    ):
        raise ValueError("Graph longitudes must match the NeuralGCM coordinates in the same order")
    if actual_lat.shape != expected_lat.shape or not np.allclose(
        actual_lat, expected_lat, rtol=0, atol=1e-5
    ):
        raise ValueError("Graph latitudes must match the NeuralGCM Gaussian coordinates in the same order")
    expected_weights = np.broadcast_to(
        np.asarray(horizontal.quadrature_weights), expected_shape
    )
    expected_weights = expected_weights / expected_weights.sum()
    actual_weights = np.asarray(topology.grid_area_weights)
    if actual_weights.shape != expected_shape or not np.allclose(
        actual_weights, expected_weights, rtol=1e-6, atol=1e-10
    ):
        raise ValueError("Graph area weights must match normalized NeuralGCM quadrature weights")


def graph_from_coords(coords: Any, *, mesh_level: int = 1, **options: Any) -> GraphGeometry:
    """Build a graph in Dinosaur's native ``(longitude, latitude)`` ordering.

    Dinosaur exposes longitude/latitude in radians and Gaussian quadrature
    weights. The geometry builder takes degrees. Supplying these exact weights
    avoids incorrectly treating the Gaussian grid as a regular latitude grid.
    Call once before model initialization, and reuse the resulting topology.
    """
    horizontal = coords.horizontal
    if "area_weights" in options:
        raise ValueError("graph_from_coords uses the dynamical core's quadrature weights")
    weights = np.asarray(horizontal.quadrature_weights)
    if weights.ndim == 2 and weights.shape[0] == 1:
        weights = weights[0]
    return build_graph(
        longitudes=np.rad2deg(np.asarray(horizontal.longitudes)),
        latitudes=np.rad2deg(np.asarray(horizontal.latitudes)),
        mesh_level=mesh_level,
        area_weights=weights,
        **options,
    )


@gin.register
class MeshConditionedNeuralParameterization(parameterizations.DivCurlNeuralParameterization):
    """Drop-in physics parameterization for NeuralGCM's Haiku step modules.

    ``modal_to_nodal_features_module`` and ``tendency_transform_module`` are
    deliberately required: production training must retain the original
    feature normalizers and physical tendency scaling. Identity transforms
    are useful only for a carefully scaled synthetic smoke test.

    ``config.context_mode`` selects the local-column, local-neighborhood, or
    multiscale-mesh ablation. This is a new trainable physics mapping; it does
    not automatically load the columns of an existing NeuralGCM checkpoint.

    The graph currently assumes the whole horizontal grid is present on each
    device. Do not use it with horizontal domain partitioning without adding
    graph communication between partitions.
    """

    def __init__(
        self,
        coords: Any,
        dt: float,
        physics_specs: Any,
        aux_features: Any,
        *,
        topology: GraphGeometry,
        modal_to_nodal_features_module: Callable,
        tendency_transform_module: Callable,
        config: ModelConfig = ModelConfig(),
        prediction_mask: Any = None,
        filter_module: Callable | None = None,
        name: str | None = None,
    ):
        _validate_graph_coordinates(coords, topology)
        tower_factory = functools.partial(
            GraphConditionedPhysics, topology=topology, config=config
        )
        nodal_mapping_module = functools.partial(
            mappings.NodalMapping, tower_factory=tower_factory
        )
        super().__init__(
            coords=coords,
            dt=dt,
            physics_specs=physics_specs,
            aux_features=aux_features,
            modal_to_nodal_features_module=modal_to_nodal_features_module,
            nodal_mapping_module=nodal_mapping_module,
            tendency_transform_module=tendency_transform_module,
            prediction_mask=prediction_mask,
            filter_module=filter_module,
            name=name,
        )
