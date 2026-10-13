"""Scale-aware forecast losses and circulation *diagnostics*.

Array convention is ``[..., field, longitude, latitude]``. A forecast time
dimension, when present, is immediately before ``field``. These losses do not
impose closed streamlines, mass conservation, or an atmospheric balance law.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array


def _validate_fields(prediction, target, area_weights, field_scales):
    prediction, target = jnp.asarray(prediction), jnp.asarray(target)
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError("prediction and target must share [..., field, lon, lat] shape")
    weights = jnp.asarray(area_weights)
    if weights.shape != prediction.shape[-2:]:
        raise ValueError("area_weights must have [lon, lat] shape")
    scales = jnp.asarray(field_scales)
    if scales.ndim == 0:
        scales = jnp.broadcast_to(scales, (prediction.shape[-3],))
    if scales.shape != (prediction.shape[-3],):
        raise ValueError("field_scales must be scalar or one positive scale per field")
    # Invalid dynamic inputs yield NaN so the training finite guard rejects them.
    valid = (jnp.all(weights >= 0) & (jnp.sum(weights) > 0)
             & jnp.all(scales > 0) & jnp.all(jnp.isfinite(scales)))
    weights = jnp.where(valid, weights / jnp.sum(weights), jnp.nan)
    return prediction, target, weights, scales[:, None, None]


def _reduce_forecast_errors(errors, time_weights):
    """Reduce [..., time] errors; uniform averaging if time weights are omitted."""
    if time_weights is None:
        return jnp.mean(errors)
    weights = jnp.asarray(time_weights)
    if errors.ndim == 0 or weights.ndim != 1 or errors.shape[-1] != weights.size:
        raise ValueError("time_weights must match the axis immediately before field")
    valid = jnp.all(weights >= 0) & (jnp.sum(weights) > 0)
    weights = jnp.where(valid, weights / jnp.sum(weights), jnp.nan)
    return jnp.mean(jnp.sum(errors * weights, axis=-1))


def area_weighted_mse(prediction, target, area_weights, field_scales=1.0,
                      time_weights=None) -> Array:
    """Area-integrated mean square error after per-field scale normalization.

    ``field_scales`` should be fixed standard deviations or reference magnitudes
    from the **training** partition, in the same units as each field. Each field
    receives equal weight after normalization. ``time_weights`` can emphasize
    later rollout leads and is normalized to sum to one. Remaining batch axes
    are averaged uniformly. Non-positive scales or invalid weights return NaN.
    """
    prediction, target, weights, scales = _validate_fields(
        prediction, target, area_weights, field_scales)
    squared = jnp.square((prediction - target) / scales)
    errors = jnp.mean(jnp.sum(squared * weights, axis=(-2, -1)), axis=-1)
    return _reduce_forecast_errors(errors, time_weights)


def zonal_mean_mse(prediction, target, area_weights, field_scales=1.0,
                   time_weights=None) -> Array:
    """Normalized zonal-mean diagnostic loss, not closed-circulation enforcement.

    Longitude averages use the grid cell areas at each latitude. This is a useful
    broad-scale diagnostic for arbitrary fields. For meridional overturning,
    use ``mass_streamfunction`` on pressure-level meridional wind instead.
    """
    prediction, target, weights, scales = _validate_fields(
        prediction, target, area_weights, field_scales)
    latitude_weights = jnp.sum(weights, axis=0)
    zonal_weights = weights / jnp.maximum(latitude_weights[None, :], 1e-30)
    error = jnp.sum((prediction - target) / scales * zonal_weights, axis=-2)
    errors = jnp.mean(jnp.sum(jnp.square(error) * latitude_weights, axis=-1), axis=-1)
    return _reduce_forecast_errors(errors, time_weights)


def mass_streamfunction(meridional_wind, pressure_pa, latitude_deg, *,
                        longitude_weights=None, radius_m=6_371_000.0,
                        gravity_m_s2=9.80665) -> Array:
    r"""Pressure-coordinate zonal-mean mass streamfunction in kg/s.

    Input wind is ``[..., pressure, longitude, latitude]`` in m/s; pressures
    are strictly increasing **Pa** (top to bottom); latitude is in degrees.
    Trapezoidal integration computes

    ``psi(p, lat) = 2*pi*a*cos(lat)/g * integral[p_top:p] mean_lon(v) dp``.

    Output is ``[..., pressure, latitude]``. The top supplied pressure surface
    has psi=0: if ``pressure_pa[0] > 0``, the unobserved layer above it is not
    extrapolated. The sign convention is positive northward wind in the upper
    integral. This is a pressure-level diagnostic, not a sigma-coordinate
    primitive-equation solver. Pressure levels below local terrain require an
    explicitly chosen mask/interpolation policy before calling this function.
    """
    wind = jnp.asarray(meridional_wind)
    if wind.ndim < 3:
        raise ValueError("wind must be [..., pressure, lon, lat]")
    pressure, latitude = jnp.asarray(pressure_pa), jnp.asarray(latitude_deg)
    if pressure.ndim != 1 or pressure.size != wind.shape[-3] or pressure.size < 2:
        raise ValueError("at least two pressure levels matching wind are required")
    if latitude.ndim != 1 or latitude.size != wind.shape[-1]:
        raise ValueError("latitude_deg must match the last wind axis")
    # Raise useful errors outside tracing; retain dynamic checks under jit.
    try:
        concrete_pressure = np.asarray(pressure_pa)
        if (not np.all(np.isfinite(concrete_pressure))
                or np.any(concrete_pressure < 0)
                or np.any(np.diff(concrete_pressure) <= 0)):
            raise ValueError("pressure_pa must increase strictly from top to bottom")
    except jax.errors.TracerArrayConversionError:
        pass
    if longitude_weights is None:
        lon_weights = jnp.ones((wind.shape[-2],), dtype=wind.dtype)
    else:
        lon_weights = jnp.asarray(longitude_weights)
        if lon_weights.shape != (wind.shape[-2],):
            raise ValueError("longitude_weights must match the longitude dimension")
    valid = (jnp.all(jnp.diff(pressure) > 0) & jnp.all(pressure >= 0)
             & jnp.all(jnp.isfinite(pressure)) & jnp.all(jnp.abs(latitude) <= 90)
             & jnp.all(lon_weights >= 0) & (jnp.sum(lon_weights) > 0)
             & (radius_m > 0) & (gravity_m_s2 > 0))
    mean_wind = jnp.sum(wind * (lon_weights / jnp.sum(lon_weights))[:, None], axis=-2)
    increments = 0.5 * (mean_wind[..., 1:, :] + mean_wind[..., :-1, :])
    increments = increments * jnp.diff(pressure)[:, None]
    integral = jnp.concatenate((jnp.zeros_like(mean_wind[..., :1, :]),
                                jnp.cumsum(increments, axis=-2)), axis=-2)
    factor = 2 * jnp.pi * radius_m * jnp.cos(jnp.deg2rad(latitude)) / gravity_m_s2
    return jnp.where(valid, integral * factor, jnp.nan)


def mass_streamfunction_mse(predicted_v, target_v, pressure_pa, latitude_deg, *,
                            scale_kg_s, longitude_weights=None,
                            latitude_weights=None, time_weights=None,
                            radius_m=6_371_000.0, gravity_m_s2=9.80665) -> Array:
    """Dimensionless meridional-overturning diagnostic supervision.

    Wind arrays use ``[..., time, pressure, longitude, latitude]`` (the time
    dimension may be omitted if ``time_weights`` is None). Wind/pressure units,
    integration order, and top boundary are exactly those of
    :func:`mass_streamfunction`. ``scale_kg_s`` is a required positive scalar
    reference magnitude, selected from training data or specified physically.
    Supplied pressure levels receive equal loss weight. Latitude defaults to
    cos(latitude); pass actual latitude-band areas for a nonuniform grid.

    This matches an observed streamfunction. It does not impose zero bottom
    streamfunction, mass closure, or closed trajectories. Missing/below-ground
    pressure-level values must be handled consistently before this loss.
    """
    if jnp.shape(predicted_v) != jnp.shape(target_v):
        raise ValueError("predicted_v and target_v must have identical shapes")
    predicted = mass_streamfunction(
        predicted_v, pressure_pa, latitude_deg, longitude_weights=longitude_weights,
        radius_m=radius_m, gravity_m_s2=gravity_m_s2)
    target = mass_streamfunction(
        target_v, pressure_pa, latitude_deg, longitude_weights=longitude_weights,
        radius_m=radius_m, gravity_m_s2=gravity_m_s2)
    scale = jnp.asarray(scale_kg_s)
    if scale.ndim != 0:
        raise ValueError("scale_kg_s must be a positive scalar")
    if latitude_weights is None:
        weights = jnp.maximum(jnp.cos(jnp.deg2rad(jnp.asarray(latitude_deg))), 0)
    else:
        weights = jnp.asarray(latitude_weights)
    if weights.shape != (predicted.shape[-1],):
        raise ValueError("latitude_weights must match latitude_deg")
    valid = (jnp.all(weights >= 0) & (jnp.sum(weights) > 0)
             & (scale > 0) & jnp.isfinite(scale))
    normalized = jnp.square((predicted - target) / scale)
    errors = jnp.mean(jnp.sum(normalized * (weights / jnp.sum(weights)), axis=-1), axis=-1)
    return jnp.where(valid, _reduce_forecast_errors(errors, time_weights), jnp.nan)
