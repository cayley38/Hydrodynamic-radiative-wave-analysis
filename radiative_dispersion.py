"""Linear dispersion solver for the 1D radiation-hydrodynamic wave system.

The perturbations are assumed to vary as

    exp(i k x - i omega t),

so damped modes have ``omega.imag < 0``.

The linear system implemented here matches the equations supplied in the
notebook discussion:

    -i omega rho_1 + i k rho_0 v_1 = 0
    -i omega (rho_1 + u_g,1) + i k w_0 v_1
        = alpha_a sum_n I_1,n DeltaOmega_n - 4 a_rad alpha_a T_0^3 T_1
    -i omega w_0 v_1 + i k (Gamma - 1) u_g,1
        = (alpha_a + alpha_s) sum_n I_1,n mu_n DeltaOmega_n
          - (16 pi / 3) I_0 (alpha_a + alpha_s) v_1
    -i omega I_1,a + i k mu_a I_1,a
        = 3 v_1 mu_a (a_rad alpha_a T_0^4 + 4 pi alpha_s I_0) / (4 pi)
          + (4 a_rad alpha_a T_0^3 T_1 + alpha_s sum_b I_1,b DeltaOmega_b) / (4 pi)
          - (alpha_a + alpha_s) I_1,a
          + (alpha_a + alpha_s) v_1 mu_a I_0

with

    T_1 = T_0 (p_1 / p_0 - rho_1 / rho_0),
    p_1 = (Gamma - 1) u_g,1.

Because the equations are linear in ``lambda = -i omega``, the dispersion
relation is solved as a generalized eigenvalue problem rather than by directly
root-finding the determinant. This returns all modes and their eigenvectors at
once for a given background and wavenumber.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

try:
    from scipy import linalg as scipy_linalg
except Exception:  # pragma: no cover - only used when SciPy is unavailable
    scipy_linalg = None

from angular_grid import AngularGrid, make_gauss_legendre_angular_grid


RHO_INDEX = 0
UG_INDEX = 1
V_INDEX = 2


def _as_1d_array(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"`{name}` must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"`{name}` must not be empty.")
    return array


@dataclass(frozen=True)
class DispersionBackground:
    """Uniform background state for the linear wave calculation."""

    rho0: float
    ug0: float
    p0: float
    temperature0: float
    w0: float
    gamma_ad: float
    absorption_opacity: float
    scattering_opacity: float
    arad: float
    i0: float
    mu: np.ndarray
    delta_omega: np.ndarray
    theta: np.ndarray
    phi: np.ndarray

    @classmethod
    def from_parameters(
        cls,
        *,
        rho0: float,
        gamma_ad: float,
        absorption_opacity: float,
        scattering_opacity: float,
        arad: float = 1.0,
        require_thermal_equilibrium: bool = False,
        equilibrium_rtol: float = 1.0e-10,
        equilibrium_atol: float = 1.0e-12,
        ug0: float | None = None,
        p0: float | None = None,
        temperature0: float | None = None,
        w0: float | None = None,
        i0: float | None = None,
        erad0: float | None = None,
        mu: Iterable[float] | None = None,
        delta_omega: Iterable[float] | None = None,
        theta: Iterable[float] | None = None,
        phi: Iterable[float] | None = None,
        grid: AngularGrid | None = None,
        n_angles: int = 16,
    ) -> "DispersionBackground":
        """Build a validated background state.

        Any one of ``ug0``, ``p0``, or ``temperature0`` is enough to recover
        the others through

            p0 = (Gamma - 1) ug0 = rho0 T0.

        Likewise, either ``i0`` or ``erad0`` may be supplied for the isotropic
        radiation background.
        """

        if rho0 <= 0.0:
            raise ValueError("`rho0` must be positive.")
        if gamma_ad <= 1.0:
            raise ValueError("`gamma_ad` must be greater than 1.")
        if absorption_opacity < 0.0 or scattering_opacity < 0.0:
            raise ValueError("Opacities must be non-negative.")
        if arad <= 0.0:
            raise ValueError("`arad` must be positive.")

        if ug0 is not None:
            ug0 = float(ug0)
            if ug0 <= 0.0:
                raise ValueError("`ug0` must be positive.")
            p0 = (gamma_ad - 1.0) * ug0
            temperature0 = p0 / rho0
        elif p0 is not None:
            p0 = float(p0)
            if p0 <= 0.0:
                raise ValueError("`p0` must be positive.")
            ug0 = p0 / (gamma_ad - 1.0)
            temperature0 = p0 / rho0
        elif temperature0 is not None:
            temperature0 = float(temperature0)
            if temperature0 <= 0.0:
                raise ValueError("`temperature0` must be positive.")
            p0 = rho0 * temperature0
            ug0 = p0 / (gamma_ad - 1.0)
        else:
            raise ValueError(
                "Provide one of `ug0`, `p0`, or `temperature0` to define the gas background."
            )

        if w0 is None:
            w0 = rho0 + gamma_ad * ug0
        w0 = float(w0)
        if w0 <= 0.0:
            raise ValueError("`w0` must be positive.")

        if i0 is None:
            if erad0 is None:
                raise ValueError("Provide one of `i0` or `erad0` for the radiation background.")
            i0 = float(erad0) / (4.0 * math.pi)
        i0 = float(i0)
        if i0 < 0.0:
            raise ValueError("`i0` must be non-negative.")

        erad0_value = 4.0 * math.pi * i0
        equilibrium_erad0 = arad * temperature0**4
        if require_thermal_equilibrium and not np.isclose(
            erad0_value,
            equilibrium_erad0,
            rtol=equilibrium_rtol,
            atol=equilibrium_atol,
        ):
            raise ValueError(
                "Background is out of thermal equilibrium: "
                f"`erad0 = {erad0_value:.16e}` while "
                f"`arad * temperature0**4 = {equilibrium_erad0:.16e}`. "
                "For thermal equilibrium, use `erad0 = arad * temperature0**4` "
                "or equivalently `i0 = arad * temperature0**4 / (4*pi)`."
            )

        if grid is not None:
            if (
                mu is not None
                or delta_omega is not None
                or theta is not None
                or phi is not None
            ):
                raise ValueError(
                    "Provide either `grid` or explicit angular arrays, not both."
                )
            mu = grid.mu
            delta_omega = grid.delta_omega
            theta = grid.theta
            phi = grid.phi
        elif mu is None or delta_omega is None:
            default_grid = make_gauss_legendre_angular_grid(n_angles=n_angles)
            mu = default_grid.mu
            delta_omega = default_grid.delta_omega
            theta = default_grid.theta
            phi = default_grid.phi

        mu_array = _as_1d_array(mu, "mu")
        delta_omega_array = _as_1d_array(delta_omega, "delta_omega")
        if mu_array.size != delta_omega_array.size:
            raise ValueError("`mu` and `delta_omega` must have the same length.")

        if theta is None:
            theta_array = np.arccos(np.clip(mu_array, -1.0, 1.0))
        else:
            theta_array = _as_1d_array(theta, "theta")

        if phi is None:
            phi_array = np.zeros_like(mu_array)
        else:
            phi_array = _as_1d_array(phi, "phi")

        if theta_array.size != mu_array.size or phi_array.size != mu_array.size:
            raise ValueError("`theta` and `phi` must match the length of `mu`.")

        return cls(
            rho0=float(rho0),
            ug0=float(ug0),
            p0=float(p0),
            temperature0=float(temperature0),
            w0=w0,
            gamma_ad=float(gamma_ad),
            absorption_opacity=float(absorption_opacity),
            scattering_opacity=float(scattering_opacity),
            arad=float(arad),
            i0=i0,
            mu=mu_array,
            delta_omega=delta_omega_array,
            theta=theta_array,
            phi=phi_array,
        )

    @property
    def n_angles(self) -> int:
        return int(self.mu.size)

    @property
    def erad0(self) -> float:
        return float(4.0 * math.pi * self.i0)

    @property
    def equilibrium_erad0(self) -> float:
        return float(self.arad * self.temperature0**4)

    @property
    def thermal_equilibrium_mismatch(self) -> float:
        return float(self.erad0 - self.equilibrium_erad0)

    @property
    def nx(self) -> np.ndarray:
        return self.mu

    @property
    def ny(self) -> np.ndarray:
        sin_theta = np.sin(self.theta)
        return sin_theta * np.cos(self.phi)

    @property
    def nz(self) -> np.ndarray:
        sin_theta = np.sin(self.theta)
        return sin_theta * np.sin(self.phi)

    @property
    def temperature_rho_coeff(self) -> float:
        return -self.temperature0 / self.rho0

    @property
    def temperature_ug_coeff(self) -> float:
        return self.temperature0 / self.ug0


@dataclass(frozen=True)
class DispersionMode:
    """A single eigenmode of the linearized system."""

    index: int
    k: float
    omega: complex
    lambda_value: complex
    eigenvector: np.ndarray
    rho1: complex
    ug1: complex
    v1: complex
    intensity1: np.ndarray
    p1: complex
    temperature1: complex
    erad1: complex
    frad1: complex
    four_force_energy1: complex
    four_force_momentum_x1: complex
    residual_norm: float

    @property
    def phase_speed(self) -> float:
        return float(np.real(self.omega) / self.k) if self.k != 0.0 else np.nan

    @property
    def damping_rate(self) -> float:
        return float(-np.imag(self.omega))


@dataclass(frozen=True)
class DispersionSolution:
    """Full solution bundle for one wavenumber."""

    k: float
    background: DispersionBackground
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    modes: tuple[DispersionMode, ...]

    @property
    def omega(self) -> np.ndarray:
        return np.asarray([mode.omega for mode in self.modes], dtype=complex)


def variable_names(background: DispersionBackground) -> list[str]:
    """Return the unknown ordering used in the matrix system."""

    return ["rho1", "ug1", "v1"] + [f"I1_{angle}" for angle in range(background.n_angles)]


def build_dispersion_matrices(
    background: DispersionBackground,
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``A`` and ``B`` for ``(A + lambda B) x = 0`` with ``lambda = -i omega``."""

    size = 3 + background.n_angles
    a_matrix = np.zeros((size, size), dtype=complex)
    b_matrix = np.zeros((size, size), dtype=complex)

    rho0 = background.rho0
    ug0 = background.ug0
    w0 = background.w0
    gamma_ad = background.gamma_ad
    temp0 = background.temperature0
    alpha_a = background.absorption_opacity
    alpha_s = background.scattering_opacity
    chi = alpha_a + alpha_s
    arad = background.arad
    i0 = background.i0
    mu = background.mu
    delta_omega = background.delta_omega

    temp_rho = background.temperature_rho_coeff
    temp_ug = background.temperature_ug_coeff
    intensity_slice = slice(3, size)

    # Mass conservation:
    #   lambda rho1 + i k rho0 v1 = 0
    b_matrix[0, RHO_INDEX] = 1.0
    a_matrix[0, V_INDEX] = 1j * k * rho0

    # Gas internal-energy equation:
    #   lambda (rho1 + ug1) + i k w0 v1
    #       + 4 a_rad alpha_a T0^3 T1 - alpha_a * delta Erad = 0
    b_matrix[1, RHO_INDEX] = 1.0
    b_matrix[1, UG_INDEX] = 1.0
    a_matrix[1, RHO_INDEX] = 4.0 * arad * alpha_a * temp0**3 * temp_rho
    a_matrix[1, UG_INDEX] = 4.0 * arad * alpha_a * temp0**3 * temp_ug
    a_matrix[1, V_INDEX] = 1j * k * w0
    a_matrix[1, intensity_slice] = -alpha_a * delta_omega

    # Gas momentum equation:
    #   lambda w0 v1 + i k p1
    #       + (16 pi / 3) I0 chi v1 - chi * delta Frad = 0
    b_matrix[2, V_INDEX] = w0
    a_matrix[2, UG_INDEX] = 1j * k * (gamma_ad - 1.0)
    a_matrix[2, V_INDEX] = (16.0 * math.pi / 3.0) * i0 * chi
    a_matrix[2, intensity_slice] = -chi * mu * delta_omega

    velocity_source_prefactor = (
        3.0 * (arad * alpha_a * temp0**4 + 4.0 * math.pi * alpha_s * i0)
        / (4.0 * math.pi)
        + chi * i0
    )
    temperature_source = arad * alpha_a * temp0**3 / math.pi
    isotropic_scattering = alpha_s * delta_omega / (4.0 * math.pi)

    for angle_index, mu_angle in enumerate(mu):
        row = 3 + angle_index
        column = 3 + angle_index

        # Direction-by-direction transfer equation for I1(angle_index).
        b_matrix[row, column] = 1.0
        a_matrix[row, RHO_INDEX] = -temperature_source * temp_rho
        a_matrix[row, UG_INDEX] = -temperature_source * temp_ug
        a_matrix[row, V_INDEX] = -mu_angle * velocity_source_prefactor
        a_matrix[row, intensity_slice] = -isotropic_scattering
        a_matrix[row, column] += 1j * k * mu_angle + chi

    return a_matrix, b_matrix


def dispersion_matrix(
    background: DispersionBackground,
    k: float,
    omega: complex,
) -> np.ndarray:
    """Return the complex matrix ``M(omega, k)`` whose determinant is the dispersion relation."""

    a_matrix, b_matrix = build_dispersion_matrices(background, k)
    return a_matrix - 1j * omega * b_matrix


def dispersion_determinant(
    background: DispersionBackground,
    k: float,
    omega: complex,
) -> complex:
    """Return ``det(M(omega, k))`` for optional external root-finding."""

    return complex(np.linalg.det(dispersion_matrix(background, k, omega)))


def _reference_index(background: DispersionBackground, reference: str | int) -> int:
    if isinstance(reference, int):
        index = reference
    else:
        names = variable_names(background)
        if reference not in names:
            raise ValueError(
                f"Unknown normalization reference `{reference}`. Choose from {names}."
            )
        index = names.index(reference)
    size = 3 + background.n_angles
    if index < 0 or index >= size:
        raise ValueError("Normalization index is out of range.")
    return index


def normalize_eigenvector(
    eigenvector: np.ndarray,
    background: DispersionBackground,
    reference: str | int = "v1",
    tol: float = 1.0e-14,
) -> np.ndarray:
    """Normalize an eigenvector using one component, or the largest one if needed."""

    vector = np.asarray(eigenvector, dtype=complex).copy()
    index = _reference_index(background, reference)
    scale = vector[index]
    if abs(scale) < tol:
        largest = int(np.argmax(np.abs(vector)))
        scale = vector[largest]
    if abs(scale) < tol:
        raise ValueError("Cannot normalize a near-zero eigenvector.")
    vector /= scale
    return vector


def _mode_from_vector(
    *,
    background: DispersionBackground,
    k: float,
    mode_index: int,
    omega: complex,
    lambda_value: complex,
    eigenvector: np.ndarray,
    residual_norm: float,
) -> DispersionMode:
    rho1 = eigenvector[RHO_INDEX]
    ug1 = eigenvector[UG_INDEX]
    v1 = eigenvector[V_INDEX]
    intensity1 = eigenvector[3:].copy()
    p1 = (background.gamma_ad - 1.0) * ug1
    temperature1 = (
        background.temperature_rho_coeff * rho1
        + background.temperature_ug_coeff * ug1
    )
    erad1 = np.sum(intensity1 * background.delta_omega)
    frad1 = np.sum(intensity1 * background.mu * background.delta_omega)
    chi = background.absorption_opacity + background.scattering_opacity
    four_force_energy1 = (
        background.absorption_opacity * erad1
        - 4.0 * background.arad * background.absorption_opacity
        * background.temperature0**3 * temperature1
    )
    four_force_momentum_x1 = (
        chi * frad1
        - (16.0 * math.pi / 3.0) * background.i0 * chi * v1
    )

    return DispersionMode(
        index=mode_index,
        k=float(k),
        omega=complex(omega),
        lambda_value=complex(lambda_value),
        eigenvector=eigenvector,
        rho1=complex(rho1),
        ug1=complex(ug1),
        v1=complex(v1),
        intensity1=intensity1,
        p1=complex(p1),
        temperature1=complex(temperature1),
        erad1=complex(erad1),
        frad1=complex(frad1),
        four_force_energy1=complex(four_force_energy1),
        four_force_momentum_x1=complex(four_force_momentum_x1),
        residual_norm=float(residual_norm),
    )


def solve_dispersion_relation(
    background: DispersionBackground,
    k: float,
    *,
    normalize_to: str | int = "v1",
    sort_by: str = "omega_real",
) -> DispersionSolution:
    """Solve the linear dispersion relation for one wavenumber.

    Parameters
    ----------
    background:
        Uniform background state and angular quadrature.
    k:
        Wave number.
    normalize_to:
        Eigenvector normalization component. Examples: ``"rho1"``, ``"ug1"``,
        ``"v1"``, or an integer index in the matrix unknown ordering.
    sort_by:
        One of ``"omega_real"``, ``"omega_imag"``, or ``"magnitude"``.
    """

    a_matrix, b_matrix = build_dispersion_matrices(background, k)

    if scipy_linalg is not None:
        lambda_values, eigenvectors = scipy_linalg.eig(-a_matrix, b_matrix)
    else:  # pragma: no cover - intended only as a lightweight fallback
        operator = -np.linalg.solve(b_matrix, a_matrix)
        lambda_values, eigenvectors = np.linalg.eig(operator)

    omega_values = 1j * lambda_values

    if sort_by == "omega_real":
        order = np.lexsort((omega_values.imag, omega_values.real))
    elif sort_by == "omega_imag":
        order = np.lexsort((omega_values.real, omega_values.imag))
    elif sort_by == "magnitude":
        order = np.argsort(np.abs(omega_values))
    else:
        raise ValueError("`sort_by` must be `omega_real`, `omega_imag`, or `magnitude`.")

    modes: list[DispersionMode] = []
    for output_index, raw_index in enumerate(order):
        raw_vector = eigenvectors[:, raw_index]
        normalized_vector = normalize_eigenvector(
            raw_vector,
            background=background,
            reference=normalize_to,
        )
        residual = np.linalg.norm((a_matrix + lambda_values[raw_index] * b_matrix) @ normalized_vector)
        modes.append(
            _mode_from_vector(
                background=background,
                k=k,
                mode_index=output_index,
                omega=omega_values[raw_index],
                lambda_value=lambda_values[raw_index],
                eigenvector=normalized_vector,
                residual_norm=residual,
            )
        )

    return DispersionSolution(
        k=float(k),
        background=background,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        modes=tuple(modes),
    )


def find_closest_mode(
    solution: DispersionSolution,
    omega_guess: complex,
) -> DispersionMode:
    """Return the mode whose frequency is closest to a supplied guess."""

    distances = np.abs(solution.omega - omega_guess)
    index = int(np.argmin(distances))
    return solution.modes[index]


def summarize_modes(solution: DispersionSolution) -> np.ndarray:
    """Return a compact notebook-friendly summary table."""

    dtype = [
        ("mode", int),
        ("omega_real", float),
        ("omega_imag", float),
        ("phase_speed", float),
        ("damping_rate", float),
        ("residual_norm", float),
    ]
    summary = np.zeros(len(solution.modes), dtype=dtype)
    for row, mode in enumerate(solution.modes):
        summary[row] = (
            mode.index,
            mode.omega.real,
            mode.omega.imag,
            mode.phase_speed,
            mode.damping_rate,
            mode.residual_norm,
        )
    return summary
