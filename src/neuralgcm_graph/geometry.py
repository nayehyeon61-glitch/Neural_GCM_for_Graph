"""Sparse, coordinate-aware spherical graphs prepared outside JAX tracing.

The grid is longitude-major: ``(nlon, nlat)`` flattens with latitude varying
fastest. Longitudes and latitudes are supplied in degrees, including for
Gaussian grids. This is a small independent implementation of the multiscale
icosahedral idea used by GraphCast, not GraphCast's exact mesh connectivity.

Grid/mesh links use spherical k-nearest neighbours, not GraphCast's radius
encoder and containing-triangle decoder. A coverage pass gives every mesh
vertex an incoming grid edge, even for coarse input grids. No dense grid by
mesh distance matrix is formed. Decoder edges are learned message-passing
links; they are not prescribed barycentric interpolation weights.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class EdgeSet:
    """Directed sparse edges, with geometry expressed at the receiver.

    ``features[:, :]`` = (east displacement, north displacement, radial
    displacement, great-circle distance), all on a unit sphere. Displacement
    is sender minus receiver in the receiver's orthonormal frame. At either
    exact pole, east is chosen as +y to make the frame deterministic.
    """

    senders: np.ndarray
    receivers: np.ndarray
    features: np.ndarray


@dataclass(frozen=True)
class GraphGeometry:
    grid_shape: tuple[int, int]
    grid_xyz: np.ndarray
    mesh_xyz: np.ndarray
    grid_to_mesh: EdgeSet
    mesh_to_mesh: EdgeSet
    mesh_to_grid: EdgeSet
    grid_area_weights: np.ndarray
    mesh_faces: np.ndarray
    grid_to_grid: EdgeSet
    longitudes: np.ndarray
    latitudes: np.ndarray


def _coordinates(longitudes, latitudes):
    lon = np.asarray(longitudes, dtype=np.float64)
    lat = np.asarray(latitudes, dtype=np.float64)
    for name, axis in (("longitudes", lon), ("latitudes", lat)):
        if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
            raise ValueError(f"{name} must be a nonempty finite one-dimensional degree vector")
    if np.any(np.abs(lat) > 90):
        raise ValueError("latitudes must lie in [-90, 90] degrees")
    if lat.size > 1 and not (np.all(np.diff(lat) > 1e-9) or np.all(np.diff(lat) < -1e-9)):
        raise ValueError("latitudes must be strictly monotone without duplicates")
    lon = np.mod(lon, 360.0)
    if lon.size > 1:
        gaps = np.diff(np.r_[np.sort(lon), np.min(lon) + 360])
        if np.any(gaps < 1e-9):
            raise ValueError("longitudes contain duplicates, including a repeated periodic endpoint")
    return lon, lat


def _grid_xyz(lon, lat):
    lam, phi = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat), indexing="ij")
    return np.stack((np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)), axis=-1).reshape(-1, 3)


def _longitude_widths(lon):
    if len(lon) == 1:
        return np.array([2 * np.pi])
    order = np.argsort(lon)
    sorted_lon = np.deg2rad(lon[order])
    forward = np.diff(np.r_[sorted_lon, sorted_lon[0] + 2 * np.pi])
    widths = np.empty_like(sorted_lon)
    widths[order] = (forward + np.roll(forward, 1)) / 2
    return widths


def spherical_area_weights(longitudes, latitudes, area_weights=None):
    """Return positive global-cell weights of shape ``(nlon,nlat)``, sum 1.

    By default, longitude boundaries are periodic midpoints and latitude
    boundaries are midpoints in latitude, capped at the poles. Explicit
    Gaussian latitudes work with this approximation; pass their quadrature
    weights for exact Gaussian integration. ``area_weights`` can be latitude
    quadrature weights ``[nlat]`` or complete cell weights ``[nlon,nlat]``.
    Coordinates describe a global grid; regional-domain areas are not implied.
    """
    lon, lat = _coordinates(longitudes, latitudes)
    lon_width = _longitude_widths(lon)
    if area_weights is None:
        order = np.argsort(lat)
        phi = np.deg2rad(lat[order])
        boundaries = np.r_[-np.pi / 2, (phi[:-1] + phi[1:]) / 2, np.pi / 2]
        bands = np.empty_like(phi)
        bands[order] = np.diff(np.sin(boundaries))
        weights = lon_width[:, None] * bands[None, :]
    else:
        supplied = np.asarray(area_weights, dtype=np.float64)
        if supplied.shape == (len(lat),):
            weights = lon_width[:, None] * supplied[None, :]
        elif supplied.shape == (len(lon), len(lat)):
            weights = supplied.copy()
        else:
            raise ValueError("area_weights must have shape [nlat] or [nlon,nlat]")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("area_weights must be finite and strictly positive")
    return (weights / weights.sum()).astype(np.float32)


def _icosahedron():
    t = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)
    return vertices, faces


def _subdivide(vertices, faces):
    points = list(vertices)
    midpoints = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key not in midpoints:
            point = vertices[a] + vertices[b]
            point /= np.linalg.norm(point)
            midpoints[key] = len(points)
            points.append(point)
        return midpoints[key]

    refined = []
    for a, b, c in faces:
        ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
        refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
    return np.asarray(points), np.asarray(refined, dtype=np.int32)


def _face_edges(faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    return np.concatenate((edges, edges[:, ::-1]))


def _edge_set(sender_xyz, receiver_xyz, pairs):
    pairs = np.unique(np.asarray(pairs, dtype=np.int32).reshape(-1, 2), axis=0)
    senders, receivers = pairs[:, 0], pairs[:, 1]
    src, dst = sender_xyz[senders], receiver_xyz[receivers]
    horizontal = np.hypot(dst[:, 0], dst[:, 1])
    east = np.zeros_like(dst)
    away_from_pole = horizontal > 1e-12
    east[away_from_pole, 0] = -dst[away_from_pole, 1] / horizontal[away_from_pole]
    east[away_from_pole, 1] = dst[away_from_pole, 0] / horizontal[away_from_pole]
    east[~away_from_pole, 1] = 1
    north = np.cross(dst, east)
    delta = src - dst
    distance = np.arctan2(np.linalg.norm(np.cross(src, dst), axis=1), np.sum(src * dst, axis=1))
    features = np.stack((np.sum(delta * east, axis=1), np.sum(delta * north, axis=1), np.sum(delta * dst, axis=1), distance), axis=1)
    return EdgeSet(senders, receivers, features.astype(np.float32))


def build_grid_edges(longitudes, latitudes):
    """Eight-neighbour grid graph: wrap longitude, never wrap latitude.

    Longitude neighbours follow geographic cyclic order even if the supplied
    longitude axis is permuted. Exact pole nodes stay separate grid nodes.
    """
    lon, lat = _coordinates(longitudes, latitudes)
    xyz = _grid_xyz(lon, lat)
    nlon, nlat = len(lon), len(lat)
    # Ranks recover longitude neighbours in the original flattened indexing.
    order = np.argsort(lon)
    ranks = np.argsort(order)
    ids = np.arange(nlon * nlat).reshape(nlon, nlat)
    pairs = []
    for di in (-1, 0, 1):
        target_lon = order[(ranks + di) % nlon]
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            start, stop = max(0, -dj), min(nlat, nlat - dj)
            src = ids[:, start:stop].ravel()
            dst = ids[target_lon, start + dj:stop + dj].ravel()
            keep = src != dst
            pairs.append(np.stack((src[keep], dst[keep]), axis=1))
    return _edge_set(xyz, xyz, np.concatenate(pairs))


def build_graph(longitudes, latitudes, mesh_level=1, grid_to_mesh_k=3,
                mesh_to_grid_k=3, multiscale=True, area_weights=None):
    """Build all sparse topology once on the host, before JIT/training.

    A level ``L`` mesh has ``10 * 4**L + 2`` nodes and ``20 * 4**L`` faces.
    Levels 0--5 are supported to prevent accidental oversized construction.
    Multiscale edges retain all coarser mesh edges on their nested vertices.
    Euclidean kNN on unit vectors is equivalent to great-circle kNN.
    """
    if isinstance(mesh_level, bool) or not isinstance(mesh_level, Integral) or not 0 <= mesh_level <= 5:
        raise ValueError("mesh_level must be an integer between 0 and 5")
    lon, lat = _coordinates(longitudes, latitudes)
    grid_xyz = _grid_xyz(lon, lat)
    vertices, faces = _icosahedron()
    all_mesh_edges = [_face_edges(faces)]
    for _ in range(mesh_level):
        vertices, faces = _subdivide(vertices, faces)
        all_mesh_edges.append(_face_edges(faces))
    for name, k in (("grid_to_mesh_k", grid_to_mesh_k), ("mesh_to_grid_k", mesh_to_grid_k)):
        if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= len(vertices):
            raise ValueError(f"{name} must be an integer between 1 and mesh node count ({len(vertices)})")
    tree = cKDTree(vertices)
    nearest = tree.query(grid_xyz, k=grid_to_mesh_k)[1].reshape(len(grid_xyz), grid_to_mesh_k)
    grid_to_mesh_pairs = np.stack((np.repeat(np.arange(len(grid_xyz)), grid_to_mesh_k), nearest.ravel()), axis=1)
    uncovered = np.setdiff1d(np.arange(len(vertices)), nearest.ravel())
    if len(uncovered):
        covering_grid = cKDTree(grid_xyz).query(vertices[uncovered], k=1)[1]
        grid_to_mesh_pairs = np.concatenate((grid_to_mesh_pairs, np.stack((covering_grid, uncovered), axis=1)))
    nearest = tree.query(grid_xyz, k=mesh_to_grid_k)[1].reshape(len(grid_xyz), mesh_to_grid_k)
    mesh_to_grid_pairs = np.stack((nearest.ravel(), np.repeat(np.arange(len(grid_xyz)), mesh_to_grid_k)), axis=1)
    mesh_pairs = np.concatenate(all_mesh_edges if multiscale else all_mesh_edges[-1:])
    return GraphGeometry(
        grid_shape=(len(lon), len(lat)),
        grid_xyz=grid_xyz.astype(np.float32),
        mesh_xyz=vertices.astype(np.float32),
        grid_to_mesh=_edge_set(grid_xyz, vertices, grid_to_mesh_pairs),
        mesh_to_mesh=_edge_set(vertices, vertices, mesh_pairs),
        mesh_to_grid=_edge_set(vertices, grid_xyz, mesh_to_grid_pairs),
        grid_area_weights=spherical_area_weights(lon, lat, area_weights),
        mesh_faces=faces,
        grid_to_grid=build_grid_edges(lon, lat),
        longitudes=lon.copy(),
        latitudes=lat.copy(),
    )
