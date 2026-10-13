#!/usr/bin/env python3
"""Run A/B/C/D on synthetic dynamics; this is NOT a NeuralGCM experiment.

The known step shifts fields periodically in longitude. The target adds weak
diffusion plus broad nonlocal coupling. A small learned tendency must model
that residual. No weather data, pretrained checkpoints, or skill claims enter
this smoke test. D adds a zonal-mean diagnostic proxy, not a circulation law.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Also support `python examples/train_toy.py` from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from neuralgcm_graph.geometry import build_graph
from neuralgcm_graph.model import GraphConditionedPhysics, ModelConfig
from neuralgcm_graph.training import (
    autoregressive_rollout, forecast_loss, init_train_state, make_train_step,
)


EXPERIMENTS = {
    "A": {"context_mode": "none", "circulation_weight": 0.0},
    "B": {"context_mode": "local", "circulation_weight": 0.0},
    "C": {"context_mode": "mesh", "circulation_weight": 0.0},
    "D": {"context_mode": "mesh", "circulation_weight": 0.1},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional A/B/C/D JSON configuration")
    parser.add_argument("--experiment", choices=EXPERIMENTS, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--nlon", type=int, default=16)
    parser.add_argument("--nlat", type=int, default=8)
    parser.add_argument("--latent-size", type=int, default=None)
    parser.add_argument("--message-passing-steps", type=int, default=None)
    parser.add_argument("--mesh-level", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/toy_metrics.json"))
    args = parser.parse_args()
    if args.steps < 1 or args.rollout_steps < 1 or args.nlon < 4 or args.nlat < 3:
        parser.error("steps/rollout-steps must be positive; nlon >= 4, nlat >= 3")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if args.output.suffix.lower() != ".json":
        parser.error("output must end in .json (parameter snapshot uses the matching .npz)")
    return args


def resolve_config(args):
    supplied = json.loads(args.config.read_text()) if args.config else {}
    experiment = args.experiment or supplied.get("experiment", "C")
    if experiment not in EXPERIMENTS:
        raise ValueError("config experiment must be A, B, C, or D")
    defaults = {"latent_size": 32, "message_passing_steps": 4,
                "film_scale": 0.1, "film_init_scale": 0.0, "mesh_level": 1}
    config = {**defaults, **EXPERIMENTS[experiment], **supplied, "experiment": experiment}
    # An explicit experiment selects its context/loss together even with a config.
    if args.experiment is not None:
        config.update(EXPERIMENTS[experiment])
    for name in ("latent_size", "message_passing_steps", "mesh_level"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
    if config["circulation_weight"] < 0:
        raise ValueError("circulation_weight must be nonnegative")
    return config


def synthetic_problem(seed, longitude_deg, latitude_deg, area_weights, num_steps):
    """Six independently sampled trajectories, with scales fitted to four train cases."""
    rng = np.random.default_rng(seed)
    lon = np.deg2rad(longitude_deg)[:, None]
    lat = np.deg2rad(latitude_deg)[None, :]
    bases = np.stack([
        np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat),
        np.cos(2 * lon) * np.cos(lat) ** 2,
        np.sin(3 * lon) * np.cos(lat) ** 3,
        np.ones_like(lon) * np.sin(lat),
        np.ones_like(lon) * (1.5 * np.sin(lat) ** 2 - 0.5),
    ]).astype(np.float32)
    coefficients = rng.normal(size=(6, 2, len(bases))).astype(np.float32)
    initial = jnp.asarray(np.einsum("bcm,mij->bcij", coefficients, bases) / 2)
    area = jnp.asarray(area_weights)
    projection_basis = jnp.asarray(bases[0])
    forcing_basis = jnp.asarray(np.ones_like(lon) * np.sin(2 * lat))

    def reference_step(state):
        shifted = jnp.roll(state, 1, axis=-2)
        laplacian_lon = (jnp.roll(shifted, 1, axis=-2)
                         + jnp.roll(shifted, -1, axis=-2) - 2 * shifted)
        amplitude = jnp.sum(state * projection_basis * area, axis=(-2, -1))
        # Cross-field global coefficient generates a smooth zonal-mean response.
        broad_coupling = jnp.roll(amplitude, 1)[:, None, None] * forcing_basis
        return shifted + 0.035 * laplacian_lon + 0.06 * broad_coupling

    def reference_rollout(state):
        def step(current, _):
            next_state = reference_step(current)
            return next_state, next_state
        return jax.lax.scan(step, state, None, length=num_steps)[1]

    targets = jax.vmap(reference_rollout)(initial)
    # Training partition only. All fields have arbitrary synthetic units.
    train_targets = targets[:4]
    mean = jnp.mean(jnp.sum(train_targets * area, axis=(-2, -1)), axis=(0, 1))
    variance = jnp.mean(jnp.sum(
        jnp.square(train_targets - mean[None, None, :, None, None]) * area,
        axis=(-2, -1)), axis=(0, 1))
    scales = jnp.sqrt(jnp.maximum(variance, 1e-6))
    return (initial[:4], targets[:4]), (initial[4:], targets[4:]), scales


def main():
    args = parse_args()
    config = resolve_config(args)
    lon = np.linspace(0, 360, args.nlon, endpoint=False, dtype=np.float32)
    lat = np.linspace(-90 + 90 / args.nlat, 90 - 90 / args.nlat,
                      args.nlat, dtype=np.float32)
    topology = build_graph(lon, lat, mesh_level=config["mesh_level"])
    model_config = ModelConfig(**{name: config[name] for name in (
        "context_mode", "latent_size", "message_passing_steps", "film_scale", "film_init_scale")})

    def forward(state):
        return GraphConditionedPhysics(output_size=2, topology=topology,
                                       config=model_config)(state)

    transformed = hk.without_apply_rng(hk.transform(forward))
    train_batch, eval_batch, scales = synthetic_problem(
        args.seed, lon, lat, topology.grid_area_weights, args.rollout_steps)
    params = transformed.init(jax.random.PRNGKey(args.seed), train_batch[0][0])
    # A fixed seed alone does not align local weights: context initialization
    # consumes additional RNG splits before the decoder. Copy an independent
    # baseline initialization into the matching shared local parameter leaves.
    baseline = hk.without_apply_rng(hk.transform(lambda state:
        GraphConditionedPhysics(output_size=2, topology=topology,
                                config=replace(model_config, context_mode="none"))(state)))
    shared_params = baseline.init(jax.random.PRNGKey(args.seed), train_batch[0][0])
    params = hk.data_structures.to_mutable_dict(params)
    for module, leaves in shared_params.items():
        if module not in params:
            raise RuntimeError(f"Shared baseline module missing in chosen model: {module}")
        for name, value in leaves.items():
            if name not in params[module] or params[module][name].shape != value.shape:
                raise RuntimeError(f"Shared baseline parameter mismatch: {module}/{name}")
            params[module][name] = value

    def physical_step(state, tendency, dt):
        # Known toy transport operator. This is NOT the NeuralGCM dynamical core.
        return jnp.roll(state, 1, axis=-2) + dt * tendency

    def loss_fn(current_params, batch):
        initial, targets = batch
        predictions = jax.vmap(lambda state: autoregressive_rollout(
            current_params, state, transformed.apply, physical_step,
            args.rollout_steps))(initial)
        return forecast_loss(predictions, targets, topology.grid_area_weights, scales,
                             time_weights=jnp.arange(1, args.rollout_steps + 1),
                             zonal_weight=config["circulation_weight"])

    evaluate = jax.jit(loss_fn)
    initial_train_loss = float(evaluate(params, train_batch)[0])
    initial_eval_loss = float(evaluate(params, eval_batch)[0])
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate))
    state = init_train_state(params, optimizer)
    train_step = make_train_step(loss_fn, optimizer)
    history = []
    print(f"TOY ONLY | experiment {config['experiment']} | {config['context_mode']} context", flush=True)
    for step in range(args.steps):
        state, metrics = train_step(state, train_batch)
        numeric = {key: float(value) for key, value in metrics.items()}
        history.append({"step": step + 1, **numeric})
        if step == 0 or step == args.steps - 1 or (step + 1) % 10 == 0:
            print(f"step={step + 1} loss={numeric['loss']:.6g} "
                  f"accepted={bool(numeric['update_accepted'])}", flush=True)
    train_loss, train_components = evaluate(state.params, train_batch)
    eval_loss, eval_components = evaluate(state.params, eval_batch)
    report = {
        "scope": "synthetic smoke test; not NeuralGCM training or weather skill",
        "config": config, "seed": args.seed, "steps": args.steps,
        "rollout_steps": args.rollout_steps, "grid_shape": [args.nlon, args.nlat],
        "field_units": "arbitrary synthetic units", "field_scales_train_only": np.asarray(scales).tolist(),
        "parameter_count": int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(state.params))),
        "parameter_matching": "A/B/C counts differ; B is not automatically parameter-matched to C",
        "local_weight_initialization": "identical independent column-only initialization copied into each mode",
        "diagnostic_loss": "area-weighted normalized zonal-mean MSE proxy",
        "initial_train_loss": initial_train_loss, "initial_eval_loss": initial_eval_loss,
        "final_train_loss": float(train_loss), "final_eval_loss": float(eval_loss),
        "final_train_components": {key: float(value) for key, value in train_components.items()},
        "final_eval_components": {key: float(value) for key, value in eval_components.items()},
        "accepted_steps": int(state.accepted_steps), "rejected_steps": int(state.rejected_steps),
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_suffix(".npz")
    # Array-only Haiku parameter snapshot: never serialize Python objects/pickle.
    arrays = {f"{module}/{name}": np.asarray(value)
              for module, module_params in state.params.items()
              for name, value in module_params.items()}
    np.savez_compressed(checkpoint_path, **arrays)
    report["parameter_snapshot"] = checkpoint_path.name
    report["snapshot_scope"] = "model parameters only; optimizer state is not a resumable checkpoint"
    def json_safe(value):
        # Preserve rejection reports even if a rejected attempt produced NaN.
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value
    args.output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    print(json.dumps({"metrics": str(args.output), "parameters": str(checkpoint_path),
                      "final_train_loss": float(train_loss), "final_eval_loss": float(eval_loss)}), flush=True)
    if int(state.rejected_steps):
        raise SystemExit("Nonfinite updates were rejected; inspect metrics before using this run.")


if __name__ == "__main__":
    main()
