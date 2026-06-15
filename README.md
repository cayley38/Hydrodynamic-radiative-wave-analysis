# Radiation-Hydrodynamic Wave Dispersion

This repository computes the linear wave spectrum of a coupled gas-radiation
system using a finite-solid-angle discretization of the radiation field. The
main deliverable is a generalized eigenvalue solver for the complex mode
frequencies `omega(k)`, together with utilities to inspect directional
eigenvectors and comoving-frame isotropy.

The repository is centered on `Dispersion_relation.ipynb` and the solver in
`radiative_dispersion.py`. It is not a nonlinear radiation-hydrodynamics code;
its purpose is linear wave analysis.

This code was created in support of the wave-analysis test presented in the
revised manuscript by Wallace, Begue, and Pe'er (2026),
["Direct Solution of the Time-Dependent Covariant Radiative Transfer Equation and its Coupling to General Relativistic Magnetohydrodynamics with cuHARM"](https://arxiv.org/html/2508.15532v1).
If this repository is useful in your work, please consider citing that paper.

## Main Features

- generalized-eigenvalue solution of the linear dispersion relation
- absorption and isotropic scattering
- multiple angular quadratures, including Gauss-Legendre and geodesic grids
- directional inspection of radiation eigenvectors
- comoving-frame isotropy diagnostics for backgrounds and modes

## Quick Start

Install the scientific Python stack used by the notebook and solver:

- `numpy`
- `scipy`
- `matplotlib`
- `jupyter`

Then open [`Dispersion_relation.ipynb`](Dispersion_relation.ipynb) or run the core workflow below:

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

solution = solve_dispersion_relation(background, k=2.0 * np.pi, normalize_to="rho1")
summary = summarize_modes(solution)
```

## Documentation

The detailed write-up is in [`MANUAL.md`](MANUAL.md). It covers:

- the main equations solved by the code
- the matrix/eigenvalue formulation
- the difference between this finite-solid-angle approach and moment formalisms
- the current capabilities and limitations of the repository
