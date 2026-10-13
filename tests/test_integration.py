"""Optional integration checks using the actual NeuralGCM/Dinosaur core."""

import dataclasses
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import pytest

pytest.importorskip("neuralgcm")
pytestmark = pytest.mark.integration

from neuralgcm_graph.integration import MeshConditionedNeuralParameterization, graph_from_coords


@pytest.fixture(scope="module")
def smoke():
    path = Path(__file__).parents[1] / "examples" / "neuralgcm_smoke.py"
    spec = importlib.util.spec_from_file_location("neuralgcm_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def problem(smoke):
    return smoke.make_problem(steps=2)


def test_gaussian_grid_geometry_and_quadrature(problem):
    _, _, _, coords, graph = problem
    assert graph.grid_shape == coords.horizontal.nodal_shape
    expected = np.asarray(coords.horizontal.quadrature_weights)
    np.testing.assert_allclose(graph.grid_area_weights, expected / expected.sum(), rtol=1e-6)
    lon, lat = np.meshgrid(coords.horizontal.longitudes, coords.horizontal.latitudes, indexing="ij")
    expected_xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
    np.testing.assert_allclose(graph.grid_xyz, expected_xyz.reshape(-1, 3), atol=1e-6)
    with pytest.raises(ValueError, match="quadrature weights"):
        graph_from_coords(coords, area_weights=np.ones(graph.grid_shape))


def test_two_step_real_dycore_rollout_has_trainable_gradient(smoke, problem):
    model, initial, params, coords, _ = problem
    loss = lambda p: smoke.rollout_loss(model, initial, coords, p)
    value, grads = jax.jit(jax.value_and_grad(loss))(params)
    assert np.isfinite(value)
    gradient_norm = sum(float(jnp.sum(jnp.square(x))) for x in jax.tree_util.tree_leaves(grads))
    assert gradient_norm > 0
    assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(grads))
    for module_fragment in ("grid_to_mesh_edge", "mesh_interaction", "film_modulation"):
        matching = [values for name, values in grads.items() if module_fragment in name]
        assert matching, f"Missing parameter group: {module_fragment}"
        group_norm = sum(float(jnp.sum(jnp.square(x))) for x in jax.tree_util.tree_leaves(matching))
        assert group_norm > 0, f"No multi-step gradient reached {module_fragment}"
    trajectory = model.apply(params, initial)
    assert trajectory.temperature_variation.shape[0] == 2
    assert all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(trajectory))
    assert not np.array_equal(trajectory.temperature_variation[0], trajectory.temperature_variation[1])
    # The objective must also differentiate through earlier states, not just
    # through the final tendency network.
    def initial_loss(temperature):
        state = dataclasses.replace(initial, temperature_variation=temperature)
        final = model.apply(params, state).temperature_variation[-1]
        return jnp.sum(jnp.square(final))

    state_grad = jax.jit(jax.grad(initial_loss))(initial.temperature_variation)
    assert np.all(np.isfinite(state_grad))
    assert np.linalg.norm(state_grad) > 0


def test_short_rollout_is_rejected(smoke):
    with pytest.raises(ValueError, match="at least two"):
        smoke.make_problem(steps=1)


@pytest.mark.parametrize("mismatch", ["latitude_order", "longitude_shift", "longitude_order", "latitude_grid", "quadrature"])
def test_same_shaped_coordinate_mismatch_is_rejected(mismatch):
    from dinosaur import coordinate_systems, sigma_coordinates, spherical_harmonic
    from neuralgcm.legacy import features, transforms

    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.with_wavenumbers(4),
        sigma_coordinates.SigmaCoordinates.equidistant(2),
    )
    topology = graph_from_coords(coords, mesh_level=0)
    if mismatch == "latitude_order":
        topology = dataclasses.replace(topology, latitudes=topology.latitudes[::-1])
        message = "latitudes"
    elif mismatch == "longitude_shift":
        topology = dataclasses.replace(topology, longitudes=topology.longitudes + 1.0)
        message = "longitudes"
    elif mismatch == "longitude_order":
        topology = dataclasses.replace(topology, longitudes=np.roll(topology.longitudes, 1))
        message = "longitudes"
    elif mismatch == "latitude_grid":
        topology = dataclasses.replace(topology, latitudes=np.linspace(-90, 90, len(topology.latitudes)))
        message = "latitudes"
    else:
        topology = dataclasses.replace(topology, grid_area_weights=np.ones(topology.grid_shape) / np.prod(topology.grid_shape))
        message = "quadrature weights"

    def construct():
        MeshConditionedNeuralParameterization(
            coords, 1.0, None, {}, topology=topology,
            modal_to_nodal_features_module=features.PrimitiveEquationsDiagnosticState,
            tendency_transform_module=transforms.IdentityTransform,
        )
        return jnp.zeros(())

    with pytest.raises(ValueError, match=message):
        hk.transform(construct).init(jax.random.PRNGKey(0))
