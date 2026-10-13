"""Tests target gradient propagation, physical units, and failed-step recovery."""
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from neuralgcm_graph.losses import (
    area_weighted_mse, mass_streamfunction, mass_streamfunction_mse, zonal_mean_mse,
)
from neuralgcm_graph.training import autoregressive_rollout, init_train_state, make_train_step


def test_rollout_recomputes_context_and_differentiates_through_physical_step():
    # x_next = 2*x + theta*x; the derivative 3*(2+theta)^2 verifies all
    # three physical updates are in the gradient, not merely the last call.
    def final_state(theta):
        return autoregressive_rollout(theta, jnp.asarray(1.0),
                                      lambda p, x: p * x,
                                      lambda x, tendency, dt: 2 * x + dt * tendency,
                                      3)[-1]
    np.testing.assert_allclose(final_state(0.5), 2.5**3)
    np.testing.assert_allclose(jax.grad(final_state)(0.5), 3 * 2.5**2)


def test_transformed_haiku_apply_works_inside_rollout_scan():
    model = hk.without_apply_rng(hk.transform(lambda x: hk.Linear(1)(x)))
    initial = jnp.ones((2, 1))
    params = model.init(jax.random.PRNGKey(4), initial)
    loss = lambda p: jnp.sum(autoregressive_rollout(
        p, initial, model.apply, lambda x, tendency, dt: x + dt * tendency, 3))
    grads = jax.jit(jax.grad(loss))(params)
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(grads))
    assert float(optax.global_norm(grads)) > 0


def test_area_scale_and_time_weighting_have_expected_value():
    # Two leads, two fields, two longitude cells, one latitude.
    prediction = jnp.array([[[[1.0], [3.0]], [[2.0], [6.0]]],
                            [[[2.0], [6.0]], [[4.0], [12.0]]]])
    weights = jnp.array([[0.75], [0.25]])
    value = area_weighted_mse(prediction, jnp.zeros_like(prediction), weights,
                              field_scales=[1.0, 2.0], time_weights=[1.0, 3.0])
    # Each normalized field: lead 1 = 3, lead 2 = 12; weighted result 9.75.
    np.testing.assert_allclose(value, 9.75)
    assert np.isnan(area_weighted_mse(prediction, prediction, weights, [0.0, 1.0]))


def test_zonal_mean_proxy_is_distinct_from_full_field_error():
    wave = jnp.array([[[1.0, 1.0], [-1.0, -1.0]]])
    weights = jnp.ones((2, 2)) / 4
    np.testing.assert_allclose(zonal_mean_mse(wave, jnp.zeros_like(wave), weights), 0)
    np.testing.assert_allclose(area_weighted_mse(wave, jnp.zeros_like(wave), weights), 1)


def test_mass_streamfunction_trapezoid_pressure_units_and_boundary():
    # v proportional to pressure makes trapezoid exact. p=10kPa top boundary
    # is explicitly zero, not an assumed surface at p=0.
    pressure = np.array([10_000.0, 30_000.0, 80_000.0], dtype=np.float32)
    latitude = np.array([-60.0, 0.0, 60.0], dtype=np.float32)
    wind = jnp.broadcast_to(jnp.asarray(pressure / 10_000)[:, None, None], (3, 4, 3))
    actual = mass_streamfunction(wind, pressure, latitude)
    integral = (pressure**2 - pressure[0]**2) / 20_000
    expected = integral[:, None] * (2 * np.pi * 6_371_000 / 9.80665) * np.cos(np.deg2rad(latitude))
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    np.testing.assert_allclose(actual[0], 0)
    grad = jax.grad(lambda v: jnp.sum(mass_streamfunction(v, pressure, latitude)))(wind)
    assert np.isfinite(grad).all() and np.any(np.asarray(grad) != 0)
    with pytest.raises(ValueError, match="increase strictly"):
        mass_streamfunction(wind, pressure[::-1], latitude)


def test_nonfinite_update_preserves_parameters_and_optimizer_state():
    optimizer = optax.adam(0.1)
    state = init_train_state({"weight": jnp.asarray(1.0)}, optimizer)
    step = make_train_step(lambda p, target: (
        jnp.square(p["weight"] - target), {"target": target}), optimizer)
    updated, metrics = step(state, jnp.asarray(0.0))
    assert bool(metrics["update_accepted"])
    rejected, metrics = step(updated, jnp.asarray(jnp.nan))
    assert not bool(metrics["update_accepted"])
    for before, after in zip(jax.tree_util.tree_leaves(updated.params),
                             jax.tree_util.tree_leaves(rejected.params)):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(jax.tree_util.tree_leaves(updated.opt_state),
                             jax.tree_util.tree_leaves(rejected.opt_state)):
        np.testing.assert_array_equal(before, after)
    assert int(rejected.attempted_steps) == 2
    assert int(rejected.accepted_steps) == 1
    assert int(rejected.rejected_steps) == 1


def test_streamfunction_loss_has_physical_scale_and_gradient():
    pressure = np.array([0.0, 50_000.0], dtype=np.float32)
    latitude = np.array([-60.0, 0.0, 60.0], dtype=np.float32)
    scale = 2 * np.pi * 6_371_000 * 50_000 / 9.80665
    # Wind=amplitude at both levels; normalized bottom psi=amplitude*cos(lat).
    # cos latitude weights give E[cos(lat)^2]=0.625; two levels -> 0.3125.
    def loss(amplitude):
        wind = amplitude * jnp.ones((2, 4, 3))
        return mass_streamfunction_mse(wind, jnp.zeros_like(wind), pressure,
                                       latitude, scale_kg_s=scale)
    np.testing.assert_allclose(loss(2.0), 1.25, rtol=2e-6)
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(2.0), 1.25, rtol=2e-6)
