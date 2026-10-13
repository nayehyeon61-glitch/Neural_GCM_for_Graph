"""CPU proof of two-step training through the real NeuralGCM dynamical core.

Run after installing the NeuralGCM extra::

    python examples/neuralgcm_smoke.py

This uses an isothermal synthetic atmosphere, not ERA5 or a pretrained model.
Its tiny grid, 60-second step and arbitrary tendency scale are smoke-test
settings only. Passing this test is not evidence of forecast skill or long-run
stability.
"""

from __future__ import annotations

import argparse
import json

from dinosaur import coordinate_systems, primitive_equations_states, scales
from dinosaur import sigma_coordinates, spherical_harmonic
import haiku as hk
import jax
import jax.numpy as jnp
from neuralgcm.legacy import correctors, equations, features, physics_specifications
import optax

from neuralgcm_graph.integration import MeshConditionedNeuralParameterization, graph_from_coords
from neuralgcm_graph.model import ModelConfig


class SmokeTendencyScale:
    """Arbitrary small nondimensional tendencies, exclusively for this test."""

    def __init__(self, coords, dt, physics_specs, aux_features):
        del coords, dt, physics_specs, aux_features

    def __call__(self, tendencies):
        return jax.tree_util.tree_map(lambda value: 1e-3 * value, tendencies)


def make_problem(*, steps: int = 2, context_mode: str = "mesh"):
    """Return initialized real-core model, state, parameters, and geometry."""
    if steps < 2:
        raise ValueError("The integration smoke test requires at least two steps")
    physics = physics_specifications.primitive_eq_specs_constructor()
    coords = coordinate_systems.CoordinateSystem(
        spherical_harmonic.Grid.with_wavenumbers(4),
        sigma_coordinates.SigmaCoordinates.equidistant(2),
    )
    initial_state_fn, aux = primitive_equations_states.isothermal_rest_atmosphere(
        coords, physics
    )
    initial = initial_state_fn(jax.random.PRNGKey(1))
    dt = physics.nondimensionalize(60 * scales.units.s)
    topology = graph_from_coords(coords, mesh_level=1)
    # Nonzero small FiLM initialization makes a first-pass gradient test able to
    # inspect the entire GNN path. Production defaults remain zero initialized.
    config = ModelConfig(
        context_mode=context_mode,
        latent_size=8,
        message_passing_steps=1,
        film_init_scale=0.01,
    )

    def forward(state):
        parameterization = MeshConditionedNeuralParameterization(
            coords, dt, physics, aux,
            topology=topology,
            config=config,
            modal_to_nodal_features_module=features.PrimitiveEquationsDiagnosticState,
            tendency_transform_module=SmokeTendencyScale,
        )
        corrector = correctors.DycoreWithPhysicsCorrector(
            coords, dt, physics, aux,
            dycore_equation_module=equations.PrimitiveEquations,
            dycore_substeps=1,
            checkpoint_explicit_terms=False,
        )

        def step(previous, unused):
            del unused
            # Recompute context from the current predicted state at every step.
            tendencies = parameterization(previous)
            following = corrector(previous, tendencies)
            return following, following

        _, trajectory = hk.scan(step, state, xs=None, length=steps)
        return trajectory

    model = hk.without_apply_rng(hk.transform(forward))
    params = model.init(jax.random.PRNGKey(7), initial)
    return model, initial, params, coords, topology


def rollout_loss(model, initial, coords, params):
    """Synthetic supervised objective over every predicted time step."""
    trajectory = model.apply(params, initial)
    temperature = coords.horizontal.to_nodal(trajectory.temperature_variation)
    target = 0.001 * jnp.cos(jnp.asarray(coords.horizontal.latitudes))[None, None, None, :]
    # Rescale this synthetic objective to make the gradient norm readable.
    return 1e6 * jnp.mean(jnp.square(temperature - target))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args(argv)
    model, initial, params, coords, topology = make_problem(steps=args.steps)
    loss = lambda weights: rollout_loss(model, initial, coords, weights)
    value, gradients = jax.jit(jax.value_and_grad(loss))(params)
    gradient_norm = optax.global_norm(gradients)
    optimizer = optax.adam(learning_rate=1e-3)
    updates, _ = optimizer.update(gradients, optimizer.init(params), params)
    updated = optax.apply_updates(params, updates)
    updated_value = jax.jit(loss)(updated)
    trajectory = model.apply(updated, initial)
    finite = all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(trajectory))
    if not finite or not bool(jnp.isfinite(value)) or not float(gradient_norm) > 0:
        raise RuntimeError("Real-core rollout or parameter gradient is invalid")
    print(json.dumps({
        "backend": jax.default_backend(),
        "dycore": "NeuralGCM legacy / Dinosaur PrimitiveEquations + IMEX RK SIL3",
        "grid_shape": list(topology.grid_shape),
        "rollout_steps": args.steps,
        "loss_before": float(value),
        "loss_after_one_update": float(updated_value),
        "gradient_norm": float(gradient_norm),
        "finite_trajectory": finite,
        "scope": "synthetic integration smoke test; not forecast validation",
    }, indent=2))


if __name__ == "__main__":
    main()
