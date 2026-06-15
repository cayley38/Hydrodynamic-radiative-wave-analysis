"""Geodesic angular grids on the sphere for radiation transport discretization.

The grid directions are taken from recursively refined icosahedra (an
``icosphere``). Their spherical Voronoi dual cells form the familiar
pentagon-hexagon tiling:

- 12 pentagons at every generation
- all remaining cells are hexagons for generations >= 1

With recursive generation ``g``, the number of directions is

    N_dir = 10 * 4**g + 2

so generation 2 gives 162 directions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from angular_grid import AngularGrid

try:
    from scipy.spatial import ConvexHull
except Exception:  # pragma: no cover - keep import optional until used
    ConvexHull = None

try:
    from scipy.spatial import SphericalVoronoi
except Exception:  # pragma: no cover - keep import optional until used
    SphericalVoronoi = None


@dataclass(frozen=True)
class GeodesicAngularGrid(AngularGrid):
    """Angular grid based on an icosahedral geodesic sphere."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    generation: int
    cell_order: np.ndarray
    voronoi_vertices: np.ndarray

    @property
    def n_directions(self) -> int:
        return int(self.mu.size)

    @property
    def n_pentagons(self) -> int:
        return int(np.count_nonzero(self.cell_order == 5))

    @property
    def n_hexagons(self) -> int:
        return int(np.count_nonzero(self.cell_order == 6))


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("Encountered a zero-length vector during normalization.")
    return vectors / norms


def _base_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    sqrt5 = math.sqrt(5.0)
    vertices = np.array(
        [
            (0.0, 0.0, 1.0),
            (0.0, 2.0 / sqrt5, 1.0 / sqrt5),
            (math.sqrt((5.0 + sqrt5) / 10.0), (5.0 - sqrt5) / 10.0, 1.0 / sqrt5),
            (math.sqrt((5.0 - sqrt5) / 10.0), (-5.0 - sqrt5) / 10.0, 1.0 / sqrt5),
            (-math.sqrt((5.0 - sqrt5) / 10.0), (-5.0 - sqrt5) / 10.0, 1.0 / sqrt5),
            (-math.sqrt((5.0 + sqrt5) / 10.0), (5.0 - sqrt5) / 10.0, 1.0 / sqrt5),
            (0.0, -2.0 / sqrt5, -1.0 / sqrt5),
            (-math.sqrt((5.0 + sqrt5) / 10.0), (-5.0 + sqrt5) / 10.0, -1.0 / sqrt5),
            (-math.sqrt((5.0 - sqrt5) / 10.0), (5.0 + sqrt5) / 10.0, -1.0 / sqrt5),
            (math.sqrt((5.0 - sqrt5) / 10.0), (5.0 + sqrt5) / 10.0, -1.0 / sqrt5),
            (math.sqrt((5.0 + sqrt5) / 10.0), (-5.0 + sqrt5) / 10.0, -1.0 / sqrt5),
            (0.0, 0.0, -1.0),
        ],
        dtype=float,
    )
    vertices = _normalize_rows(vertices)
    if ConvexHull is None:
        raise ImportError(
            "SciPy with `scipy.spatial.ConvexHull` is required to build the "
            "geodesic angular grid."
        )
    hull = ConvexHull(vertices)
    faces = np.asarray(hull.simplices, dtype=int)
    return vertices, faces


def _midpoint_index(
    vertices: list[np.ndarray],
    midpoint_cache: dict[tuple[int, int], int],
    left: int,
    right: int,
) -> int:
    edge = (left, right) if left < right else (right, left)
    if edge in midpoint_cache:
        return midpoint_cache[edge]

    midpoint = 0.5 * (vertices[left] + vertices[right])
    midpoint /= np.linalg.norm(midpoint)
    index = len(vertices)
    vertices.append(midpoint)
    midpoint_cache[edge] = index
    return index


def _refine_icosahedron(
    vertices: np.ndarray,
    faces: np.ndarray,
    generation: int,
) -> tuple[np.ndarray, np.ndarray]:
    refined_vertices = [vertex.copy() for vertex in vertices]
    refined_faces = faces.copy()

    for _ in range(generation):
        midpoint_cache: dict[tuple[int, int], int] = {}
        next_faces: list[tuple[int, int, int]] = []

        for tri0, tri1, tri2 in refined_faces:
            mid01 = _midpoint_index(refined_vertices, midpoint_cache, tri0, tri1)
            mid12 = _midpoint_index(refined_vertices, midpoint_cache, tri1, tri2)
            mid20 = _midpoint_index(refined_vertices, midpoint_cache, tri2, tri0)

            next_faces.extend(
                [
                    (tri0, mid01, mid20),
                    (tri1, mid12, mid01),
                    (tri2, mid20, mid12),
                    (mid01, mid12, mid20),
                ]
            )

        refined_faces = np.asarray(next_faces, dtype=int)

    return np.asarray(refined_vertices, dtype=float), refined_faces


def _cartesian_to_x_polar_angles(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    theta = np.arccos(np.clip(x, -1.0, 1.0))
    phi = np.mod(np.arctan2(z, y), 2.0 * math.pi)
    return theta, phi, x


def make_geodesic_angular_grid(generation: int = 2) -> GeodesicAngularGrid:
    """Return a geodesic angular grid based on a refined icosahedron.

    Parameters
    ----------
    generation:
        Number of recursive triangle refinements applied to the base
        icosahedron. The resulting number of directions is ``10 * 4**g + 2``.
        In particular, ``generation=2`` yields 162 directions.
    """

    if generation < 0:
        raise ValueError("`generation` must be non-negative.")
    if SphericalVoronoi is None or ConvexHull is None:
        raise ImportError(
            "SciPy with `scipy.spatial.ConvexHull` and "
            "`scipy.spatial.SphericalVoronoi` is required to build the "
            "geodesic angular grid."
        )

    vertices, faces = _base_icosahedron()
    if generation > 0:
        vertices, faces = _refine_icosahedron(vertices, faces, generation=generation)
    del faces  # the Voronoi dual is built directly from the final vertex set

    sphere = SphericalVoronoi(vertices, radius=1.0, center=np.zeros(3))
    sphere.sort_vertices_of_regions()
    delta_omega = sphere.calculate_areas()
    cell_order = np.asarray([len(region) for region in sphere.regions], dtype=int)

    theta, phi, mu = _cartesian_to_x_polar_angles(vertices)

    return GeodesicAngularGrid(
        mu=mu.copy(),
        delta_omega=np.asarray(delta_omega, dtype=float),
        theta=theta,
        phi=phi,
        x=vertices[:, 0].copy(),
        y=vertices[:, 1].copy(),
        z=vertices[:, 2].copy(),
        generation=int(generation),
        cell_order=cell_order,
        voronoi_vertices=np.asarray(sphere.vertices, dtype=float),
    )


def geodesic_grid_summary(grid: GeodesicAngularGrid) -> dict[str, float | int]:
    """Return a compact summary of a geodesic angular grid."""

    return {
        "generation": grid.generation,
        "n_directions": grid.n_directions,
        "n_pentagons": grid.n_pentagons,
        "n_hexagons": grid.n_hexagons,
        "total_solid_angle": float(np.sum(grid.delta_omega)),
        "min_solid_angle": float(np.min(grid.delta_omega)),
        "max_solid_angle": float(np.max(grid.delta_omega)),
    }
