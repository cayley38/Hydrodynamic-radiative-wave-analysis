# Manual

## Purpose

This repository computes the linear wave spectrum of a coupled gas-radiation
system using a finite-solid-angle discretization of the radiation field. The
main workflow is the generalized eigenvalue solve in
`radiative_dispersion.py`, with examples and mode inspection in
`Dispersion_relation.ipynb`.

It was created in support of the wave-analysis test presented in the revised
manuscript by Wallace, Begue, and Pe'er (2026),
["Direct Solution of the Time-Dependent Covariant Radiative Transfer Equation and its Coupling to General Relativistic Magnetohydrodynamics with cuHARM"](https://arxiv.org/html/2508.15532v1).
If this repository is useful in your work, please consider citing that paper.

The code is designed to answer questions of the form:

- What are the complex mode frequencies `omega(k)` for a chosen uniform background?
- How strongly damped is each acoustic or radiation mode?
- How anisotropic is the radiation perturbation carried by a given eigenmode?
- How do the results change when the angular quadrature is refined?

## Equations Solved

The perturbations are assumed to vary as

```text
exp(i k x - i omega t),
```

so `Re(omega)` is the oscillation frequency and `-Im(omega)` is the damping
rate.

For a uniform background with zero bulk velocity, the code solves the
linearized 1D gas-radiation system

```text
-i omega rho_1 + i k rho_0 v_1 = 0

-i omega (rho_1 + u_g,1) + i k w_0 v_1
    = alpha_a sum_a I_1,a DeltaOmega_a - 4 a_rad alpha_a T_0^3 T_1

-i omega w_0 v_1 + i k (Gamma - 1) u_g,1
    = (alpha_a + alpha_s) sum_a mu_a I_1,a DeltaOmega_a
      - (16 pi / 3) I_0 (alpha_a + alpha_s) v_1

-i omega I_1,a + i k mu_a I_1,a
    = 3 v_1 mu_a (a_rad alpha_a T_0^4 + 4 pi alpha_s I_0) / (4 pi)
      + (4 a_rad alpha_a T_0^3 T_1
         + alpha_s sum_b I_1,b DeltaOmega_b) / (4 pi)
      - (alpha_a + alpha_s) I_1,a
      + (alpha_a + alpha_s) v_1 mu_a I_0
```

with

```text
p_1 = (Gamma - 1) u_g,1
T_1 = T_0 (p_1 / p_0 - rho_1 / rho_0)
w_0 = rho_0 + Gamma u_g,0
E_r,1 = sum_a I_1,a DeltaOmega_a
F_r,1 = sum_a mu_a I_1,a DeltaOmega_a
```

Here `I_1,a` is the radiation-intensity perturbation in angular bin `a`, and
`DeltaOmega_a` is that bin's solid-angle weight.

## Numerical Formulation

The unknown vector is

```text
x = [rho_1, u_g,1, v_1, I_1,0, I_1,1, ..., I_1,N-1]^T.
```

The code rewrites the system as

```text
(A + lambda B) x = 0,    lambda = -i omega,
```

and solves the generalized eigenvalue problem for all modes at once. This is
implemented in `solve_dispersion_relation()`.

Two angular-grid families are provided:

- `make_gauss_legendre_angular_grid()`: compact quadrature in `mu`, efficient for the strictly 1D dispersion problem.
- `make_geodesic_angular_grid()`: full-sphere geodesic discretization, useful when you want directional mode content on a nearly uniform spherical mesh.

The notebook-oriented helpers expose:

- mode summaries (`summarize_modes`)
- directional intensity tables (`radiative_dispersion_directions.py`)
- comoving-frame isotropy diagnostics (`radiative_dispersion_isotropy.py`)

## Difference From Moment Formalism

This repository does not evolve only low-order radiation moments such as
`E_r`, `F_r`, and `P_r`. Instead, it keeps the directional intensity
perturbations `I_1,a` themselves.

That changes the method in several important ways:

1. No closure relation is needed.
   In a moment formalism, the hierarchy must be truncated and closed, for
   example by an Eddington approximation or an `M1` closure. Here the pressure
   and higher moments come directly from the angular quadrature over `I_1,a`.

2. Angular anisotropy is resolved explicitly.
   Moment methods compress the radiation field into a few moments, which is
   efficient but hides the directional structure. This code can inspect the
   full eigenvector over angle and diagnose how isotropic or beamed each mode
   is.

3. The linear algebra is larger.
   A moment model keeps only a handful of radiation variables. This solver adds
   one unknown per angular bin, so the matrix size grows with angular
   resolution.

4. The results depend on quadrature rather than closure.
   In a moment scheme, the approximation error is tied to the closure. In this
   finite-solid-angle approach, the main controllable approximation is the
   angular discretization itself.

In short, moment formalisms are usually cheaper and better suited to large
nonlinear simulations, while this repository is aimed at detailed linear wave
analysis where directional information matters.

## Capabilities

- Solve the full linear spectrum for a chosen wavenumber `k`
- Include absorption and isotropic scattering
- Enforce or test thermal-equilibrium backgrounds
- Use Gauss-Legendre, longitude-latitude, or geodesic angular grids
- Inspect directional eigenvector content mode by mode
- Diagnose comoving-frame isotropy of backgrounds and perturbations

## Current Scope And Limitations

- The implemented wave problem is 1D in space.
- The background is uniform and stationary.
- The code studies linear perturbations, not nonlinear time evolution.
- Radiation is represented with discrete ordinates / finite solid angles.
- The repository is a research analysis tool, not a production radiation-hydrodynamics solver.

## Typical Workflow

```python
import numpy as np

from geodesic_angular_grid import make_geodesic_angular_grid
from radiative_dispersion import DispersionBackground, solve_dispersion_relation, summarize_modes

grid = make_geodesic_angular_grid(generation=2)

background = DispersionBackground.from_parameters(
    rho0=1.0,
    temperature0=0.1,
    gamma_ad=5.0 / 3.0,
    absorption_opacity=10.0,
    scattering_opacity=10.0,
    erad0=0.1,
    arad=0.1 / (0.1**4.0),
    grid=grid,
    require_thermal_equilibrium=True,
)

solution = solve_dispersion_relation(
    background,
    k=2.0 * np.pi,
    normalize_to="rho1",
)

table = summarize_modes(solution)
```

## Repository Layout

- `radiative_dispersion.py`: core matrix assembly and generalized eigenvalue solve
- `angular_grid.py`: generic angular quadratures
- `geodesic_angular_grid.py`: refined icosahedral full-sphere quadrature
- `radiative_dispersion_directions.py`: directional eigenvector inspection
- `radiative_dispersion_isotropy.py`: comoving-frame isotropy diagnostics
- `comoving_isotropy.py`: frame transforms and isotropy utilities
- `Dispersion_relation.ipynb`: worked examples and plotting
