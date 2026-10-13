"""Differentiable rollout and guarded optimization independent of the solver.

The learned tendency function is a *transformed apply function*: do not create
Haiku modules inside this module's ``lax.scan``. Call ``hk.transform`` first and
pass a wrapper of its ``apply`` method. Gradients then cross every solver step
and the context is recomputed from each evolving predicted state.
"""
from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax

from .losses import area_weighted_mse, zonal_mean_mse


def autoregressive_rollout(params, initial_state, tendency_fn: Callable,
                           physical_step: Callable, num_steps: int, *, dt=1.0):
    """Return forecast states at steps 1..N, differentiating through the solver.

    ``tendency_fn(params, current_state)`` recomputes all local/mesh context.
    ``physical_step(current_state, learned_tendency, dt)`` owns the physical
    integration and unit conversions. It must be JAX differentiable. No state,
    tendency, or solver output is stop-gradiented or cached between steps.
    State/tendency may be pytrees, provided the solver returns the same structure
    and array shapes as ``initial_state``. For stochastic physics, carry its PRNG
    key explicitly in the state or extend the scan inputs; never reuse one key.
    """
    if not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive static integer")

    def advance(state, _):
        tendency = tendency_fn(params, state)
        next_state = physical_step(state, tendency, dt)
        return next_state, next_state

    _, predictions = jax.lax.scan(advance, initial_state, xs=None, length=num_steps)
    return predictions


def forecast_loss(predictions, targets, area_weights, field_scales, *,
                  time_weights=None, zonal_weight=0.0):
    """Return total loss and normalized components for field-array forecasts.

    ``zonal_weight`` adds the explicitly named zonal-mean diagnostic proxy. It
    does not imply conservation or closed meridional circulation. Fields follow
    ``[..., time, field, longitude, latitude]`` (batch axes are optional).
    """
    field_loss = area_weighted_mse(predictions, targets, area_weights,
                                   field_scales, time_weights)
    zonal_loss = zonal_mean_mse(predictions, targets, area_weights,
                              field_scales, time_weights)
    total = field_loss + zonal_weight * zonal_loss
    return total, {"field_mse": field_loss, "zonal_mean_mse": zonal_loss}


class TrainState(NamedTuple):
    params: Any
    opt_state: Any
    attempted_steps: jax.Array
    accepted_steps: jax.Array
    rejected_steps: jax.Array


def init_train_state(params, optimizer: optax.GradientTransformation) -> TrainState:
    zero = jnp.asarray(0, dtype=jnp.int32)
    return TrainState(params, optimizer.init(params), zero, zero, zero)


def _all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    checks = [jnp.all(jnp.isfinite(leaf)) for leaf in leaves]
    return jnp.all(jnp.stack(checks)) if checks else jnp.asarray(True)


def make_train_step(loss_fn: Callable, optimizer: optax.GradientTransformation):
    """Build a jitted update that atomically rejects nonfinite optimizer steps.

    ``loss_fn(params, batch)`` returns ``(scalar_loss, metrics_dict)``. Rejected
    updates preserve both parameters and optimizer moments/counters. Attempted
    and rejected counters still advance; callers can abort on repeated failures.
    """
    @jax.jit
    def train_step(state: TrainState, batch):
        (loss, auxiliary), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params, batch)
        updates, next_opt_state = optimizer.update(grads, state.opt_state, state.params)
        next_params = optax.apply_updates(state.params, updates)
        grad_norm = optax.global_norm(grads)
        accepted = (jnp.isfinite(loss) & _all_finite(auxiliary) & _all_finite(grads)
                    & jnp.isfinite(grad_norm) & _all_finite(next_params)
                    & _all_finite(next_opt_state))
        params, opt_state = jax.lax.cond(
            accepted, lambda _: (next_params, next_opt_state),
            lambda _: (state.params, state.opt_state), operand=None)
        new_state = TrainState(
            params, opt_state, state.attempted_steps + 1,
            state.accepted_steps + accepted.astype(jnp.int32),
            state.rejected_steps + (~accepted).astype(jnp.int32))
        metrics = {**auxiliary, "loss": loss, "grad_norm": grad_norm,
                   "update_accepted": accepted}
        return new_state, metrics

    return train_step
