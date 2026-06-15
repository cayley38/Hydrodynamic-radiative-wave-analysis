"""Angular quadratures used by the wave-dispersion tools.

The dispersion solver only depends on the directional cosine

    mu = n^x,

but the helper grids keep enough geometric metadata to inspect directional
eigenvectors on the sphere and to compare different angular discretizations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AngularGrid:
    """Discrete angular grid with one quadrature weight per direction."""

    mu: np.ndarray
    delta_omega: np.ndarray
    theta: np.ndarray
    phi: np.ndarray


def make_longitude_latitude_grid(
    n_latitude: int = 5,
    n_longitude: int = 16,
) -> AngularGrid:
    """Return a uniform longitude-latitude tiling of the sphere.

    The grid is uniform in ``mu = cos(theta)`` and azimuthal angle ``phi``.
    Each cell therefore carries the exact solid angle

        DeltaOmega = DeltaPhi * (mu_lo - mu_hi).

    This is convenient for visualization and for comparing the dispersion
    solver against a simple full-sphere discretization.
    """

    if n_latitude <= 0 or n_longitude <= 0:
        raise ValueError("`n_latitude` and `n_longitude` must both be positive.")

    mu_edges = np.linspace(1.0, -1.0, n_latitude + 1, dtype=float)
    phi_edges = np.linspace(0.0, 2.0 * math.pi, n_longitude + 1, dtype=float)

    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    theta_centers = np.arccos(np.clip(mu_centers, -1.0, 1.0))
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    delta_phi = phi_edges[1:] - phi_edges[:-1]
    band_weights = mu_edges[:-1] - mu_edges[1:]

    theta_2d, phi_2d = np.meshgrid(theta_centers, phi_centers, indexing="ij")
    mu_2d, _ = np.meshgrid(mu_centers, phi_centers, indexing="ij")
    delta_omega_2d = band_weights[:, None] * delta_phi[None, :]

    return AngularGrid(
        mu=mu_2d.reshape(-1),
        delta_omega=delta_omega_2d.reshape(-1),
        theta=theta_2d.reshape(-1),
        phi=phi_2d.reshape(-1),
    )


def make_gauss_legendre_angular_grid(n_angles: int = 16) -> AngularGrid:
    """Return a Gauss-Legendre quadrature over the full sphere.

    For the 1D wave problem, the angular dependence enters through ``mu`` only,
    so a Gauss-Legendre rule in ``mu`` gives a compact and accurate quadrature
    with full-sphere weights

        DeltaOmega_i = 2 pi w_i.
    """

    if n_angles <= 0:
        raise ValueError("`n_angles` must be positive.")

    mu, weights = np.polynomial.legendre.leggauss(n_angles)
    delta_omega = 2.0 * math.pi * weights
    theta = np.arccos(np.clip(mu, -1.0, 1.0))
    phi = np.zeros_like(mu)
    return AngularGrid(mu=mu, delta_omega=delta_omega, theta=theta, phi=phi)
