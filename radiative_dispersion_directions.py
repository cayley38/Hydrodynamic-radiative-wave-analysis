"""Helpers to inspect directional specific-intensity eigenvectors.

This module complements :mod:`radiative_dispersion` by exposing each angular
component of an eigenmode individually, together with its direction metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radiative_dispersion import DispersionBackground, DispersionMode


@dataclass(frozen=True)
class DirectionalIntensityView:
    """Notebook-friendly directional decomposition of one eigenmode."""

    mode_index: int
    omega: complex
    angle_index: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    delta_omega: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    nz: np.ndarray
    intensity1: np.ndarray

    @property
    def mu(self) -> np.ndarray:
        return self.nx

    @property
    def intensity1_real(self) -> np.ndarray:
        return self.intensity1.real

    @property
    def intensity1_imag(self) -> np.ndarray:
        return self.intensity1.imag

    @property
    def intensity1_abs(self) -> np.ndarray:
        return np.abs(self.intensity1)

    @property
    def intensity1_phase(self) -> np.ndarray:
        return np.angle(self.intensity1)


def directional_mode_view(
    mode: DispersionMode,
    background: DispersionBackground,
) -> DirectionalIntensityView:
    """Return the directional intensity components of one eigenmode."""

    n_angles = background.n_angles
    if mode.intensity1.size != n_angles:
        raise ValueError(
            "Mode/background mismatch: the number of intensity components does not "
            "match the angular grid."
        )

    return DirectionalIntensityView(
        mode_index=mode.index,
        omega=mode.omega,
        angle_index=np.arange(n_angles, dtype=int),
        theta=background.theta.copy(),
        phi=background.phi.copy(),
        delta_omega=background.delta_omega.copy(),
        nx=background.nx.copy(),
        ny=background.ny.copy(),
        nz=background.nz.copy(),
        intensity1=mode.intensity1.copy(),
    )


def directional_mode_table(
    mode: DispersionMode,
    background: DispersionBackground,
) -> np.ndarray:
    """Return a structured array with one row per angular intensity component."""

    view = directional_mode_view(mode, background)
    dtype = [
        ("angle", int),
        ("theta", float),
        ("phi", float),
        ("delta_omega", float),
        ("nx", float),
        ("ny", float),
        ("nz", float),
        ("I1_real", float),
        ("I1_imag", float),
        ("I1_abs", float),
        ("I1_phase", float),
    ]
    table = np.zeros(view.angle_index.size, dtype=dtype)
    table["angle"] = view.angle_index
    table["theta"] = view.theta
    table["phi"] = view.phi
    table["delta_omega"] = view.delta_omega
    table["nx"] = view.nx
    table["ny"] = view.ny
    table["nz"] = view.nz
    table["I1_real"] = view.intensity1_real
    table["I1_imag"] = view.intensity1_imag
    table["I1_abs"] = view.intensity1_abs
    table["I1_phase"] = view.intensity1_phase
    return table


def directional_mode_dict(
    mode: DispersionMode,
    background: DispersionBackground,
) -> dict[str, np.ndarray | complex | int]:
    """Return a plain dictionary for convenient notebook use."""

    view = directional_mode_view(mode, background)
    return {
        "mode_index": view.mode_index,
        "omega": view.omega,
        "angle_index": view.angle_index,
        "theta": view.theta,
        "phi": view.phi,
        "delta_omega": view.delta_omega,
        "nx": view.nx,
        "ny": view.ny,
        "nz": view.nz,
        "mu": view.mu,
        "I1": view.intensity1,
        "I1_real": view.intensity1_real,
        "I1_imag": view.intensity1_imag,
        "I1_abs": view.intensity1_abs,
        "I1_phase": view.intensity1_phase,
    }
