"""Comoving-frame isotropy diagnostics for dispersion-relation studies."""

from __future__ import annotations

import math

import numpy as np

from comoving_isotropy import (
    comoving_isotropic_intensity,
    comoving_zero_flux_intensity,
    lorentz_factor_from_u1,
)
from radiative_dispersion import DispersionBackground, DispersionMode


def background_directional_intensity(
    background: DispersionBackground,
    *,
    isotropy_frame: str = "coordinate",
    background_u1: float = 0.0,
    background_u0: float | None = None,
) -> np.ndarray:
    """Return the directional background intensity field on the dispersion grid."""

    if isotropy_frame == "coordinate":
        return np.full(background.n_angles, background.i0, dtype=float)
    if isotropy_frame == "comoving":
        return comoving_isotropic_intensity(
            background.erad0,
            background,
            u1=background_u1,
            u0=background_u0,
        )
    if isotropy_frame == "comoving_zero_flux":
        return comoving_zero_flux_intensity(
            background.erad0,
            background,
            u1=background_u1,
            u0=background_u0,
        )
    raise ValueError(
        "`isotropy_frame` must be one of `coordinate`, `comoving`, or "
        "`comoving_zero_flux`."
    )


def complex_comoving_frame_intensity(
    intensity_lab: np.ndarray,
    background: DispersionBackground,
    *,
    background_u1: float = 0.0,
    background_u0: float | None = None,
) -> dict[str, np.ndarray | float]:
    """Transform a possibly complex directional intensity field to the comoving frame."""

    intensity_lab = np.asarray(intensity_lab, dtype=complex)
    if intensity_lab.shape != background.mu.shape:
        raise ValueError(
            "`intensity_lab` must have the same shape as the dispersion angular grid."
        )

    if background_u0 is None:
        background_u0 = lorentz_factor_from_u1(background_u1)
    background_u0 = float(background_u0)
    beta = float(background_u1) / background_u0
    doppler_factor = background_u0 - float(background_u1) * background.mu

    mu_comoving = (background.mu - beta) / (1.0 - beta * background.mu)
    mu_comoving = np.clip(mu_comoving, -1.0, 1.0)
    theta_comoving = np.arccos(mu_comoving)
    delta_omega_comoving = background.delta_omega / doppler_factor**2
    intensity_comoving = intensity_lab * doppler_factor**4

    return {
        "u0": background_u0,
        "u1": float(background_u1),
        "mu_lab": background.mu.copy(),
        "theta_lab": background.theta.copy(),
        "phi_lab": background.phi.copy(),
        "delta_omega_lab": background.delta_omega.copy(),
        "mu_comoving": mu_comoving,
        "theta_comoving": theta_comoving,
        "phi_comoving": background.phi.copy(),
        "delta_omega_comoving": delta_omega_comoving,
        "intensity_lab": intensity_lab,
        "intensity_comoving": intensity_comoving,
        "doppler_factor": doppler_factor,
    }


def _complex_isotropy_diagnostics(
    intensity_lab: np.ndarray,
    background: DispersionBackground,
    *,
    background_u1: float = 0.0,
    background_u0: float | None = None,
) -> dict[str, np.ndarray | float | complex]:
    transformed = complex_comoving_frame_intensity(
        intensity_lab,
        background,
        background_u1=background_u1,
        background_u0=background_u0,
    )

    intensity_comoving = transformed["intensity_comoving"]
    mu_comoving = transformed["mu_comoving"]
    delta_omega_comoving = transformed["delta_omega_comoving"]

    energy_comoving = np.sum(intensity_comoving * delta_omega_comoving)
    flux_comoving = np.sum(intensity_comoving * mu_comoving * delta_omega_comoving)
    pressure_xx_comoving = np.sum(
        intensity_comoving * mu_comoving * mu_comoving * delta_omega_comoving
    )

    mean_intensity = energy_comoving / (4.0 * math.pi)
    fluctuation = intensity_comoving - mean_intensity
    weighted_rms = math.sqrt(
        max(
            float(np.sum(np.abs(fluctuation) ** 2 * delta_omega_comoving) / (4.0 * math.pi)),
            0.0,
        )
    )

    mean_scale = abs(mean_intensity)
    if mean_scale > 0.0:
        rms_relative = weighted_rms / mean_scale
        max_relative = float(np.max(np.abs(fluctuation))) / mean_scale
    else:
        rms_relative = math.nan
        max_relative = math.nan

    energy_scale = abs(energy_comoving)
    if energy_scale > 0.0:
        flux_ratio = abs(flux_comoving) / energy_scale
        eddington_xx = pressure_xx_comoving / energy_comoving
    else:
        flux_ratio = math.nan
        eddington_xx = complex(np.nan, np.nan)

    return {
        **transformed,
        "energy_comoving": energy_comoving,
        "flux_comoving": flux_comoving,
        "pressure_xx_comoving": pressure_xx_comoving,
        "mean_intensity_comoving": mean_intensity,
        "rms_relative_departure": float(rms_relative),
        "max_relative_departure": float(max_relative),
        "flux_to_energy_ratio": float(flux_ratio),
        "eddington_xx": eddington_xx,
    }


def background_comoving_isotropy_diagnostics(
    background: DispersionBackground,
    *,
    isotropy_frame: str = "coordinate",
    background_u1: float = 0.0,
    background_u0: float | None = None,
) -> dict[str, np.ndarray | float | complex]:
    """Diagnose the angular isotropy of the background radiation field."""

    intensity_lab = background_directional_intensity(
        background,
        isotropy_frame=isotropy_frame,
        background_u1=background_u1,
        background_u0=background_u0,
    )
    diagnostics = _complex_isotropy_diagnostics(
        intensity_lab,
        background,
        background_u1=background_u1,
        background_u0=background_u0,
    )
    diagnostics["isotropy_frame"] = isotropy_frame
    return diagnostics


def mode_comoving_isotropy_diagnostics(
    mode: DispersionMode,
    background: DispersionBackground,
    *,
    background_u1: float = 0.0,
    background_u0: float | None = None,
) -> dict[str, np.ndarray | float | complex]:
    """Diagnose comoving-frame anisotropy of a dispersion-mode intensity perturbation."""

    diagnostics = _complex_isotropy_diagnostics(
        mode.intensity1,
        background,
        background_u1=background_u1,
        background_u0=background_u0,
    )
    diagnostics["mode_index"] = mode.index
    diagnostics["omega"] = mode.omega
    diagnostics["energy1_comoving"] = diagnostics["energy_comoving"]
    diagnostics["flux1_comoving"] = diagnostics["flux_comoving"]
    diagnostics["pressure1_xx_comoving"] = diagnostics["pressure_xx_comoving"]
    diagnostics["mean_intensity1_comoving"] = diagnostics["mean_intensity_comoving"]
    return diagnostics


def print_background_comoving_isotropy_report(
    background: DispersionBackground,
    *,
    isotropy_frame: str = "coordinate",
    background_u1: float = 0.0,
    background_u0: float | None = None,
    label: str = "Dispersion background",
) -> dict[str, np.ndarray | float | complex]:
    """Print a compact isotropy report for the dispersion background field."""

    diagnostics = background_comoving_isotropy_diagnostics(
        background,
        isotropy_frame=isotropy_frame,
        background_u1=background_u1,
        background_u0=background_u0,
    )

    print(f"{label} comoving-frame isotropy diagnostics")
    print(f"  isotropy_frame = {isotropy_frame}")
    print(f"  u0 = {diagnostics['u0']:.16e}")
    print(f"  u1 = {diagnostics['u1']:.16e}")
    print(f"  Erad_hat = {diagnostics['energy_comoving']:.16e}")
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


def print_mode_comoving_isotropy_report(
    mode: DispersionMode,
    background: DispersionBackground,
    *,
    background_u1: float = 0.0,
    background_u0: float | None = None,
    label: str | None = None,
) -> dict[str, np.ndarray | float | complex]:
    """Print a compact isotropy report for one dispersion-mode intensity eigenvector."""

    diagnostics = mode_comoving_isotropy_diagnostics(
        mode,
        background,
        background_u1=background_u1,
        background_u0=background_u0,
    )

    if label is None:
        label = f"Dispersion mode {mode.index}"

    print(f"{label} comoving-frame isotropy diagnostics")
    print(f"  omega = {mode.omega.real:.16e} + {mode.omega.imag:.16e}j")
    print(f"  u0 = {diagnostics['u0']:.16e}")
    print(f"  u1 = {diagnostics['u1']:.16e}")
    print(f"  delta Erad_hat = {diagnostics['energy1_comoving']:.16e}")
    print(f"  delta Frad_hat_x = {diagnostics['flux1_comoving']:.16e}")
    print(
        "  |delta Frad_hat_x| / |delta Erad_hat| = "
        f"{diagnostics['flux_to_energy_ratio']:.16e}"
    )
    print(
        "  RMS relative departure from isotropy = "
        f"{diagnostics['rms_relative_departure']:.16e}"
    )
    print(
        "  Max relative departure from isotropy = "
        f"{diagnostics['max_relative_departure']:.16e}"
    )
    print(
        "  delta Pxx_hat / delta Erad_hat = "
        f"{diagnostics['eddington_xx'].real:.16e} + {diagnostics['eddington_xx'].imag:.16e}j"
    )

    return diagnostics
