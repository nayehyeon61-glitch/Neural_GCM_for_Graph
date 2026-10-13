"""Behavioral tests for sparse conditioning and its zero-init training path."""

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neuralgcm_graph.geometry import build_graph
from neuralgcm_graph.model import GraphConditionedPhysics, ModelConfig


@pytest.fixture(scope="module")
def graph():
    return build_graph(
        np.arange(8) * 45.0, np.array([-60.0, -20.0, 20.0, 60.0]), mesh_level=0
    )


@pytest.fixture(scope="module")
def features():
    return jax.random.normal(jax.random.PRNGKey(3), (3, 8, 4))


def transformed(graph, mode="mesh", **kwargs):
    config = ModelConfig(
        context_mode=mode, latent_size=8, message_passing_steps=2, **kwargs
    )
    return hk.transform(
        lambda x: GraphConditionedPhysics(2, graph, config=config)(x)
    )


def all_finite(tree):
    return all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves(tree))


@pytest.mark.parametrize("mode", ["none", "local", "mesh"])
def test_shapes_jit_and_external_batch_vmap(graph, features, mode):
    model = transformed(graph, mode)
    params = model.init(jax.random.PRNGKey(0), features)
    result = jax.jit(model.apply)(params, None, features)
    assert result.shape == (2, 8, 4)
    assert all_finite(result)
    batch = jnp.stack((features, 2 * features), axis=0)
    batch_result = jax.jit(jax.vmap(model.apply, in_axes=(None, None, 0)))(
        params, None, batch
    )
    assert batch_result.shape == (2, 2, 8, 4)
    np.testing.assert_allclose(batch_result[0], result, atol=1e-6, rtol=1e-6)
    gradients = jax.jit(jax.grad(lambda p: jnp.mean(model.apply(p, None, features) ** 2)))(params)
    assert all_finite(gradients)


@pytest.mark.parametrize("mode", ["local", "mesh"])
def test_zero_initialized_conditioning_exactly_matches_local_physics(graph, features, mode):
    conditioned = transformed(graph, mode)
    baseline = transformed(graph, "none")
    params = conditioned.init(jax.random.PRNGKey(0), features)
    # Copying the local parameter groups is sufficient; unused groups are
    # deliberately omitted so changing hidden mode-specific names breaks this.
    local_params = {
        key: value
        for key, value in params.items()
        if "column_encoder" in key or "physics_decoder" in key
    }
    assert local_params
    np.testing.assert_array_equal(
        conditioned.apply(params, None, features),
        baseline.apply(local_params, None, features),
    )


def test_mesh_receives_gradients_after_first_film_update(graph, features):
    model = transformed(graph, "mesh")
    params = model.init(jax.random.PRNGKey(1), features)
    target = jnp.sin(features[:2]) + 0.2

    @jax.jit
    def gradient(parameters):
        return jax.grad(
            lambda p: jnp.mean((model.apply(p, None, features) - target) ** 2)
        )(parameters)

    first = gradient(params)
    context_keys = [
        key for key in params
        if not any(name in key for name in ("column_encoder", "physics_decoder", "film_modulation"))
    ]
    assert context_keys
    assert all(np.all(x == 0) for key in context_keys for x in first[key].values())
    film_keys = [key for key in params if "film_modulation" in key]
    assert any(np.any(first[key]["w"] != 0) for key in film_keys)
    updated = jax.tree_util.tree_map(lambda p, g: p - 0.1 * g, params, first)
    second = gradient(updated)
    assert all_finite(second)
    for component in ("grid_to_mesh_edge", "mesh_interaction_0", "mesh_to_grid_edge"):
        assert any(
            np.any(value != 0)
            for key in second if component in key
            for value in second[key].values()
        ), f"no training signal reaches {component} after the FiLM update"


def test_mesh_context_connects_distant_columns(graph, features):
    # Two mesh processing steps, with g2m/m2g links, reach across this coarse
    # sphere. The unconditioned local physics cannot see the opposite longitude.
    mesh = transformed(graph, "mesh", film_init_scale=0.2)
    params = mesh.init(jax.random.PRNGKey(8), features)
    sensitivity = jax.grad(lambda x: mesh.apply(params, None, x)[0, 0, 1])(features)
    assert np.any(np.abs(sensitivity[:, 4, 1]) > 1e-10)
    baseline = transformed(graph, "none")
    local_sensitivity = jax.grad(lambda x: baseline.apply(params, None, x)[0, 0, 1])(features)
    np.testing.assert_array_equal(local_sensitivity[:, 4, 1], 0)


@pytest.mark.parametrize("settings", [
    {"context_mode": "attention"},
    {"latent_size": 0},
    {"latent_size": 1.5},
    {"message_passing_steps": -1},
    {"film_scale": float("nan")},
    {"film_init_scale": -0.1},
])
def test_invalid_configuration_rejected(settings):
    with pytest.raises(ValueError):
        ModelConfig(**settings)


def test_shape_mismatch_rejected(graph, features):
    model = transformed(graph)
    with pytest.raises(ValueError, match="features must have shape"):
        model.init(jax.random.PRNGKey(1), jnp.swapaxes(features, -1, -2))


def test_numpy_output_size_from_neuralgcm_mapping(graph, features):
    model = hk.transform(
        lambda x: GraphConditionedPhysics(
            np.int64(2), graph, ModelConfig(context_mode="none", latent_size=8)
        )(x)
    )
    params = model.init(jax.random.PRNGKey(1), features)
    assert model.apply(params, None, features).shape == (2, 8, 4)
