import numpy as np
import pytest

from neuralgcm_graph.geometry import build_graph, build_grid_edges, spherical_area_weights


def coordinates():
    return np.arange(0.0, 360.0, 45.0), np.array([-90.0, -45.0, 0.0, 45.0, 90.0])


@pytest.mark.parametrize("level", [0, 1, 2])
def test_icosphere_counts_and_connectivity(level):
    graph = build_graph(*coordinates(), mesh_level=level)
    assert graph.mesh_xyz.shape == (10 * 4**level + 2, 3)
    assert graph.mesh_faces.shape == (20 * 4**level, 3)
    np.testing.assert_allclose(np.linalg.norm(graph.mesh_xyz, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(graph.grid_xyz, axis=1), 1, atol=1e-7)
    for edge, nsrc, ndst in ((graph.grid_to_mesh, 40, len(graph.mesh_xyz)),
                             (graph.mesh_to_mesh, len(graph.mesh_xyz), len(graph.mesh_xyz)),
                             (graph.mesh_to_grid, len(graph.mesh_xyz), 40),
                             (graph.grid_to_grid, 40, 40)):
        assert edge.senders.min() >= 0 and edge.senders.max() < nsrc
        assert edge.receivers.min() >= 0 and edge.receivers.max() < ndst
        assert edge.features.shape == (len(edge.senders), 4)
        assert np.all(np.isfinite(edge.features))
    assert len(np.unique(graph.grid_to_mesh.receivers)) == len(graph.mesh_xyz)
    assert len(np.unique(graph.grid_to_mesh.senders)) == 40
    assert len(np.unique(graph.mesh_to_grid.receivers)) == 40


def test_multiscale_retains_long_edges():
    single = build_graph(*coordinates(), mesh_level=2, multiscale=False)
    multi = build_graph(*coordinates(), mesh_level=2, multiscale=True)
    assert len(single.mesh_to_mesh.senders) == 60 * 4**2
    assert len(multi.mesh_to_mesh.senders) == 60 * (1 + 4 + 16)
    assert multi.mesh_to_mesh.features[:, 3].max() > 2 * single.mesh_to_mesh.features[:, 3].max()
    assert set(zip(single.mesh_to_mesh.senders, single.mesh_to_mesh.receivers)) <= set(zip(multi.mesh_to_mesh.senders, multi.mesh_to_mesh.receivers))


def test_area_weights_poles_dateline_and_gaussian_quadrature():
    lon, lat = coordinates()
    weights = spherical_area_weights(lon, lat)
    assert weights.shape == (8, 5)
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-7)
    np.testing.assert_allclose(weights[:, 0], weights[:, -1])
    np.testing.assert_allclose(spherical_area_weights(lon - 180, lat), weights)
    np.testing.assert_allclose(spherical_area_weights(lon, lat[::-1]), weights[:, ::-1])
    sinlat, quadrature = np.polynomial.legendre.leggauss(8)
    gaussian_lat = np.rad2deg(np.arcsin(sinlat))
    graph = build_graph(lon, gaussian_lat, area_weights=quadrature)
    np.testing.assert_allclose(graph.grid_area_weights.sum(axis=0), quadrature / 2, atol=1e-7)
    # The exact integral average of sin(latitude)^2 on a sphere is 1/3.
    np.testing.assert_allclose(np.sum(graph.grid_area_weights * sinlat[None, :]**2), 1 / 3, atol=1e-7)
    assert not np.allclose(graph.grid_xyz, build_graph(lon, np.linspace(-80, 80, 8)).grid_xyz)


def test_longitude_order_and_grid_boundary():
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    lat = np.array([-60.0, 0.0, 60.0])
    edges = build_grid_edges(lon, lat)
    pairs = set(zip(edges.senders.tolist(), edges.receivers.tolist()))
    assert (0, 9) in pairs  # Longitude wraps.
    assert (0, 2) not in pairs  # Latitude never wraps.
    assert (4, 8) in pairs  # Diagonal neighbour.
    assert not any(a == b for a, b in pairs)
    shuffled = np.array([180.0, 0.0, 270.0, 90.0])
    edges = build_grid_edges(shuffled, lat)
    pairs = set(zip(edges.senders.tolist(), edges.receivers.tolist()))
    assert (3, 6) in pairs  # 0 degrees is adjacent to 270 degrees.
    assert (3, 0) not in pairs  # 0 degrees is not adjacent to 180 degrees.


def test_tangent_features_and_sparse_coverage():
    graph = build_graph([0.0, 90.0], [-30.0, 30.0], mesh_level=2, grid_to_mesh_k=1, mesh_to_grid_k=1)
    assert len(np.unique(graph.grid_to_mesh.receivers)) == len(graph.mesh_xyz)
    assert len(graph.grid_to_mesh.senders) <= len(graph.grid_xyz) + len(graph.mesh_xyz)
    edge = graph.grid_to_mesh
    delta = graph.grid_xyz[edge.senders] - graph.mesh_xyz[edge.receivers]
    np.testing.assert_allclose(np.sum(edge.features[:, :3]**2, axis=1), np.sum(delta**2, axis=1), atol=2e-7)
    assert np.all(edge.features[:, 3] >= 0)
    assert np.all(edge.features[:, 3] <= np.pi)


@pytest.mark.parametrize("lon,lat", [([0, 360], [-45, 45]), ([0, 90], [0, 0]),
                                     ([0, 90], [0, 30, -30]), ([0, 90], [91]),
                                     ([np.nan], [0]), ([], [0])])
def test_invalid_coordinates_rejected(lon, lat):
    with pytest.raises(ValueError):
        build_graph(lon, lat)


@pytest.mark.parametrize("kwargs", [{"mesh_level": 6}, {"mesh_level": 1.5},
                                    {"grid_to_mesh_k": 0}, {"mesh_to_grid_k": 100},
                                    {"area_weights": [-1, 1]}])
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        build_graph([0, 180], [-45, 45], **kwargs)
