"""Comoving-frame radiation initialization and isotropy diagnostics.

The helpers here are intended for angular radiation fields stored on a discrete
solid-angle quadrature. They support two common tasks:

1. Build a lab-frame intensity field whose radiation is isotropic in the
   comoving frame, which guarantees zero comoving-frame flux initially.
2. Diagnose how far a given lab-frame intensity departs from isotropy in the
   comoving frame.
"""

from __future__ import annotations

import math

import numpy as np


def lorentz_factor_from_u1(u1: float) -> float:
    """Return ``u^0 = sqrt(1 + (u^1)^2)`` for the 1D flow."""

    return float(math.sqrt(1.0 + u1 * u1))


def _grid_arrays(grid: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        mu = np.asarray(grid.mu, dtype=float)
        delta_omega = np.asarray(grid.delta_omega, dtype=float)
        theta = np.asarray(grid.theta, dtype=float)
        phi = np.asarray(grid.phi, dtype=float)
    except AttributeError as exc:  # pragma: no cover - defensive interface guard
        raise TypeError(
            "`grid` must provide `mu`, `delta_omega`, `theta`, and `phi` arrays."
        ) from exc

    if mu.shape != delta_omega.shape or mu.shape != theta.shape or mu.shape != phi.shape:
        raise ValueError("Angular-grid arrays must all have the same shape.")
    return mu, delta_omega, theta, phi


def _doppler_factor(mu: np.ndarray, u1: float, u0: float | None = None) -> np.ndarray:
    if u0 is None:
        u0 = lorentz_factor_from_u1(u1)
    return float(u0) - float(u1) * mu


def comoving_isotropic_intensity(
    urad_comoving0: float,
    grid: object,
    u1: float,
    u0: float | None = None,
) -> np.ndarray:
    """Return a lab-frame intensity field isotropic in the comoving frame.

    Parameters
    ----------
    urad_comoving0:
        Radiation energy density in the comoving frame.
    grid:
        Angular grid with ``mu`` and ``delta_omega``.
    u1, u0:
        Fluid four-velocity components. If ``u0`` is omitted it is recovered
        from ``u1``.
    """

    if urad_comoving0 < 0.0:
        raise ValueError("`urad_comoving0` must be non-negative.")

    mu, _, _, _ = _grid_arrays(grid)
    factor = _doppler_factor(mu, u1=u1, u0=u0)
    ihat0 = urad_comoving0 / (4.0 * math.pi)
    return ihat0 * factor ** (-4)


def comoving_zero_flux_intensity(
    urad_comoving0: float,
    grid: object,
    u1: float,
    u0: float | None = None,
) -> np.ndarray:
    """Return a lab-frame field with exact discrete zero comoving-frame flux.

    On a finite angular quadrature, the analytically transformed isotropic field
    may not give exactly zero comoving flux because the transformed angular
    integrals are only approximate. This helper instead constructs the
    comoving-frame field

        I_hat(mu_hat) = a0 + a1 mu_hat

    whose discrete comoving energy matches ``urad_comoving0`` and whose
    discrete comoving x-flux is exactly zero on the supplied quadrature. The
    resulting lab-frame intensity is then obtained by inverse transformation.
    """

    if urad_comoving0 < 0.0:
        raise ValueError("`urad_comoving0` must be non-negative.")

    transformed = comoving_frame_intensity(
        np.ones_like(np.asarray(grid.mu, dtype=float)),
        grid,
        u1=u1,
        u0=u0,
    )
    mu_hat = transformed["mu_comoving"]
    delta_omega_hat = transformed["delta_omega_comoving"]

    s0 = float(np.sum(delta_omega_hat))
    s1 = float(np.sum(mu_hat * delta_omega_hat))
    s2 = float(np.sum(mu_hat * mu_hat * delta_omega_hat))

    matrix = np.array([[s0, s1], [s1, s2]], dtype=float)
    rhs = np.array([urad_comoving0, 0.0], dtype=float)
    a0, a1 = np.linalg.solve(matrix, rhs)

    intensity_hat = a0 + a1 * mu_hat
    factor = transformed["doppler_factor"]
    return intensity_hat * factor ** (-4)


def comoving_frame_intensity(
    intensity_lab: np.ndarray,
    grid: object,
    u1: float,
    u0: float | None = None,
) -> dict[str, np.ndarray]:
    """Transform a lab-frame angular intensity field to the comoving frame."""

    mu_lab, delta_omega_lab, theta_lab, phi_lab = _grid_arrays(grid)
    intensity_lab = np.asarray(intensity_lab, dtype=float)
    if intensity_lab.shape != mu_lab.shape:
        raise ValueError("`intensity_lab` must have the same shape as `grid.mu`.")

    if u0 is None:
        u0 = lorentz_factor_from_u1(u1)
    u0 = float(u0)
    beta = float(u1) / u0
    factor = _doppler_factor(mu_lab, u1=u1, u0=u0)

    mu_comoving = (mu_lab - beta) / (1.0 - beta * mu_lab)
    mu_comoving = np.clip(mu_comoving, -1.0, 1.0)
    theta_comoving = np.arccos(mu_comoving)
    delta_omega_comoving = delta_omega_lab / factor**2
    intensity_comoving = intensity_lab * factor**4

    return {
        "mu_lab": mu_lab,
        "theta_lab": theta_lab,
        "phi_lab": phi_lab,
        "delta_omega_lab": delta_omega_lab,
        "mu_comoving": mu_comoving,
        "theta_comoving": theta_comoving,
        "phi_comoving": phi_lab.copy(),
        "delta_omega_comoving": delta_omega_comoving,
        "intensity_lab": intensity_lab,
        "intensity_comoving": intensity_comoving,
        "doppler_factor": factor,
    }


def comoving_isotropy_diagnostics(
    intensity_lab: np.ndarray,
    grid: object,
    u1: float,
    u0: float | None = None,
) -> dict[str, float | np.ndarray]:
    """Return comoving-frame isotropy diagnostics for a discrete intensity field."""

    transformed = comoving_frame_intensity(intensity_lab, grid, u1=u1, u0=u0)
    intensity_comoving = transformed["intensity_comoving"]
    mu_comoving = transformed["mu_comoving"]
    delta_omega_comoving = transformed["delta_omega_comoving"]

    erad_comoving = float(np.sum(intensity_comoving * delta_omega_comoving))
    flux_comoving = float(
        np.sum(intensity_comoving * mu_comoving * delta_omega_comoving)
    )
    pressure_xx_comoving = float(
        np.sum(intensity_comoving * mu_comoving * mu_comoving * delta_omega_comoving)
    )

    mean_intensity = erad_comoving / (4.0 * math.pi)
    fluctuation = intensity_comoving - mean_intensity
    weighted_variance = float(
        np.sum(fluctuation * fluctuation * delta_omega_comoving) / (4.0 * math.pi)
    )
    weighted_std = math.sqrt(max(weighted_variance, 0.0))

    if mean_intensity != 0.0:
        rms_relative = weighted_std / abs(mean_intensity)
        max_relative = float(np.max(np.abs(fluctuation))) / abs(mean_intensity)
    else:
        rms_relative = math.nan
        max_relative = math.nan

    if erad_comoving != 0.0:
        flux_ratio = abs(flux_comoving) / abs(erad_comoving)
        eddington_xx = pressure_xx_comoving / erad_comoving
    else:
        flux_ratio = math.nan
        eddington_xx = math.nan

    return {
        **transformed,
        "u0": float(lorentz_factor_from_u1(u1) if u0 is None else u0),
        "u1": float(u1),
        "erad_comoving": erad_comoving,
        "flux_comoving": flux_comoving,
        "pressure_xx_comoving": pressure_xx_comoving,
        "mean_intensity_comoving": mean_intensity,
        "rms_relative_departure": float(rms_relative),
        "max_relative_departure": float(max_relative),
        "flux_to_energy_ratio": float(flux_ratio),
        "eddington_xx": float(eddington_xx),
    }


def print_comoving_isotropy_report(
    intensity_lab: np.ndarray,
    grid: object,
    u1: float,
    u0: float | None = None,
    *,
    label: str = "Radiation field",
) -> dict[str, float | np.ndarray]:
    """Print a compact comoving-frame isotropy report and return the diagnostics."""

    diagnostics = comoving_isotropy_diagnostics(intensity_lab, grid, u1=u1, u0=u0)

    print(f"{label} comoving-frame isotropy diagnostics")
    print(f"  u0 = {diagnostics['u0']:.16e}")
    print(f"  u1 = {diagnostics['u1']:.16e}")
    print(f"  Erad_hat = {diagnostics['erad_comoving']:.16e}")
    print(f"  Frad_hat_x = {diagnostics['flux_comoving']:.16e}")
    print(f"  |Frad_hat_x| / Erad_hat = {diagnostics['flux_to_energy_ratio']:.16e}")
    print(
        "  RMS relative departure from isotropy = "
        f"{diagnostics['rms_relative_departure']:.16e}"
    )
    print(
        "  Max relative departure from isotropy = "
        f"{diagnostics['max_relative_departure']:.16e}"
    )
    print(f"  Pxx_hat / Erad_hat = {diagnostics['eddington_xx']:.16e}")

    return diagnostics
