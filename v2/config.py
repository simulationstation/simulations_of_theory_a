#!/usr/bin/env python3
"""
config.py

Default physical constants and cosmological parameters for the v2 pipeline.

All values are in SI units unless otherwise noted.

References:
- Planck 2018 results (arXiv:1807.06209)
- PDG 2022 for physical constants
"""

import numpy as np

# =============================================================================
# Physical Constants (SI)
# =============================================================================

C_LIGHT = 2.99792458e8      # Speed of light [m/s]
G_NEWTON = 6.67430e-11      # Gravitational constant [m^3 kg^-1 s^-2]
HBAR = 1.054571817e-34      # Reduced Planck constant [J s]
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant [J/K]

# =============================================================================
# Astrophysical Constants
# =============================================================================

M_SUN = 1.98892e30          # Solar mass [kg]
MPC_TO_M = 3.085677581e22   # Megaparsec to meters
GYR_TO_S = 3.15576e16       # Gigayear to seconds (365.25 days/yr)
PC_TO_M = 3.085677581e16    # Parsec to meters

# =============================================================================
# Cosmological Parameters (Planck 2018 baseline)
# =============================================================================

# Hubble constant
H0_KM_S_MPC = 67.4          # Hubble constant [km/s/Mpc]
H0_SI = H0_KM_S_MPC * 1000.0 / MPC_TO_M  # [1/s]
H0_INV_GYR = H0_SI * GYR_TO_S  # H0 in Gyr^{-1}

# Hubble parameter h = H0 / (100 km/s/Mpc)
LITTLE_H = H0_KM_S_MPC / 100.0

# Density parameters (flat LCDM)
OMEGA_M0 = 0.315            # Total matter density
OMEGA_B0 = 0.0493           # Baryon density
OMEGA_CDM0 = OMEGA_M0 - OMEGA_B0  # Cold dark matter density
OMEGA_LAMBDA0 = 0.685       # Dark energy density (our universe)
OMEGA_R0 = 9.0e-5           # Radiation density (approximate, includes neutrinos)
OMEGA_K0 = 0.0              # Curvature (flat universe)

# Power spectrum parameters
SIGMA_8 = 0.811             # RMS fluctuation in 8 Mpc/h spheres
N_S = 0.965                 # Scalar spectral index

# =============================================================================
# Derived Quantities
# =============================================================================

# Critical density today
RHO_CRIT_0 = 3.0 * H0_SI**2 / (8.0 * np.pi * G_NEWTON)  # [kg/m^3]
RHO_CRIT_0_MSUN_MPC3 = RHO_CRIT_0 * MPC_TO_M**3 / M_SUN  # [M_sun/Mpc^3]

# Hubble time
T_HUBBLE = 1.0 / H0_SI  # [s]
T_HUBBLE_GYR = T_HUBBLE / GYR_TO_S  # [Gyr] ~ 14.5 Gyr

# Hubble radius
R_HUBBLE = C_LIGHT / H0_SI  # [m]
R_HUBBLE_MPC = R_HUBBLE / MPC_TO_M  # [Mpc] ~ 4400 Mpc

# =============================================================================
# Black Hole Parameters (defaults for calculations)
# =============================================================================

# Reference black hole mass (10^8 solar masses, typical SMBH)
M_BH_REF = 1e8 * M_SUN  # [kg]

# Minimum halo mass to host a BH (for N_BH calculations)
M_HALO_MIN_DEFAULT = 1e10  # [M_sun]

# Characteristic BH formation redshift
Z_FORM_DEFAULT = 2.0

# =============================================================================
# Mass Inflation Parameters
# =============================================================================

# Inner horizon surface gravity parameter
KAPPA_DEFAULT = 0.5

# Curvature blowup threshold
K_CRIT_DEFAULT = 1e6

# =============================================================================
# Numerical Parameters
# =============================================================================

# Default grid sizes
N_OMEGA_LAMBDA = 30         # Points in Omega_Lambda scan
N_MASS = 50                 # Points in mass integration
N_REDSHIFT = 20             # Points in redshift integration
N_TIME = 50                 # Points in time integration

# Integration limits
A_MIN = 1e-8                # Minimum scale factor
A_MAX = 100.0               # Maximum scale factor
Z_MAX = 20.0                # Maximum redshift
M_MAX = 1e16                # Maximum halo mass [M_sun]

# =============================================================================
# Utility Functions
# =============================================================================

def H0_from_h(h):
    """Convert little h to H0 in SI units."""
    return h * 100.0 * 1000.0 / MPC_TO_M


def Omega_m_from_flat(Omega_Lambda, Omega_r=OMEGA_R0):
    """Compute Omega_m for flat universe given Omega_Lambda."""
    return 1.0 - Omega_Lambda - Omega_r


def print_parameters():
    """Print current parameter values."""
    print("=" * 60)
    print("v2 Configuration Parameters")
    print("=" * 60)
    print("\nPhysical Constants:")
    print(f"  c = {C_LIGHT:.6e} m/s")
    print(f"  G = {G_NEWTON:.6e} m^3 kg^-1 s^-2")
    print(f"  M_sun = {M_SUN:.6e} kg")
    print()
    print("Cosmological Parameters (Planck 2018):")
    print(f"  H0 = {H0_KM_S_MPC:.1f} km/s/Mpc")
    print(f"  h = {LITTLE_H:.3f}")
    print(f"  Omega_m = {OMEGA_M0:.4f}")
    print(f"  Omega_b = {OMEGA_B0:.4f}")
    print(f"  Omega_Lambda = {OMEGA_LAMBDA0:.4f}")
    print(f"  Omega_r = {OMEGA_R0:.2e}")
    print(f"  sigma_8 = {SIGMA_8:.3f}")
    print(f"  n_s = {N_S:.3f}")
    print()
    print("Derived Quantities:")
    print(f"  t_Hubble = {T_HUBBLE_GYR:.2f} Gyr")
    print(f"  R_Hubble = {R_HUBBLE_MPC:.0f} Mpc")
    print(f"  rho_crit = {RHO_CRIT_0:.3e} kg/m^3")
    print()
    print("Default Model Parameters:")
    print(f"  M_BH = {M_BH_REF/M_SUN:.0e} M_sun")
    print(f"  M_halo_min = {M_HALO_MIN_DEFAULT:.0e} M_sun")
    print(f"  z_form = {Z_FORM_DEFAULT:.1f}")
    print(f"  kappa = {KAPPA_DEFAULT:.2f}")
    print()


if __name__ == "__main__":
    print_parameters()
