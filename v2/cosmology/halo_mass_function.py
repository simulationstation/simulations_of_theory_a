#!/usr/bin/env python3
"""
halo_mass_function.py

Press-Schechter and extended Press-Schechter halo mass functions.

Provides:
- dn/dM: comoving number density of halos per unit mass
- N(>M): integrated number density above a threshold mass
- Variants: original PS, Sheth-Tormen, Tinker

These functions compute the abundance of dark matter halos as a function
of mass and redshift, which is the foundation for estimating black hole
populations in different cosmologies.

References:
- Press & Schechter (1974), ApJ 187, 425
- Sheth & Tormen (1999), MNRAS 308, 119
- Tinker et al. (2008), ApJ 688, 709
"""

import numpy as np
from scipy import integrate

from .structure_formation import sigma_M, critical_overdensity, growth_factor_normalized

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OMEGA_M0, OMEGA_B0, SIGMA_8, N_S


# =============================================================================
# Press-Schechter Mass Function
# =============================================================================

def f_PS(nu):
    """
    Press-Schechter multiplicity function f(nu).

    f_PS(nu) = sqrt(2/pi) * nu * exp(-nu^2/2)

    where nu = delta_c / sigma(M).

    Parameters
    ----------
    nu : float or ndarray
        Peak height nu = delta_c / sigma

    Returns
    -------
    f : float or ndarray
        Multiplicity function
    """
    return np.sqrt(2.0 / np.pi) * nu * np.exp(-nu**2 / 2.0)


def f_ST(nu, A=0.3222, a=0.707, p=0.3):
    """
    Sheth-Tormen multiplicity function f(nu).

    f_ST(nu) = A * sqrt(2a/pi) * nu * [1 + (a*nu^2)^(-p)] * exp(-a*nu^2/2)

    This is a modification of PS that better fits N-body simulations.

    Parameters
    ----------
    nu : float or ndarray
        Peak height nu = delta_c / sigma
    A : float
        Normalization (default 0.3222 for integral = 1)
    a : float
        Shape parameter (default 0.707)
    p : float
        Low-mass slope parameter (default 0.3)

    Returns
    -------
    f : float or ndarray
        Multiplicity function
    """
    nu_prime = np.sqrt(a) * nu
    return (A * np.sqrt(2.0 / np.pi) * nu_prime *
            (1.0 + nu_prime**(-2*p)) *
            np.exp(-nu_prime**2 / 2.0))


def dln_sigma_dln_M(M, z=0, sigma8=SIGMA_8, n_s=N_S, Omega_m0=OMEGA_M0,
                    Omega_b0=OMEGA_B0, Omega_Lambda=None, h=0.7,
                    dM_frac=0.01):
    """
    Compute d(ln sigma) / d(ln M) numerically.

    Parameters
    ----------
    M : float
        Mass scale [M_sun]
    z : float
        Redshift
    dM_frac : float
        Fractional step for numerical derivative

    Returns
    -------
    dlns_dlnM : float
        Logarithmic derivative
    """
    dM = M * dM_frac

    sigma_plus = sigma_M(M + dM, z, sigma8, n_s, Omega_m0, Omega_b0,
                         Omega_Lambda, h)
    sigma_minus = sigma_M(M - dM, z, sigma8, n_s, Omega_m0, Omega_b0,
                          Omega_Lambda, h)

    dlns = np.log(sigma_plus) - np.log(sigma_minus)
    dlnM = np.log(M + dM) - np.log(M - dM)

    return dlns / dlnM


def press_schechter_mass_function(M, z=0, sigma8=SIGMA_8, n_s=N_S,
                                   Omega_m0=OMEGA_M0, Omega_b0=OMEGA_B0,
                                   Omega_Lambda=None, h=0.7,
                                   multiplicity='PS'):
    """
    Press-Schechter halo mass function dn/dM.

    dn/dM = (rho_m / M) * |d ln sigma / d ln M| * f(nu) * (nu / sigma)

    where nu = delta_c / sigma(M).

    Parameters
    ----------
    M : float or ndarray
        Halo mass [M_sun]
    z : float
        Redshift
    sigma8 : float
        sigma_8 normalization
    n_s : float
        Spectral index
    Omega_m0 : float
        Matter density parameter
    Omega_b0 : float
        Baryon density parameter
    Omega_Lambda : float, optional
        Dark energy parameter
    h : float
        Hubble parameter
    multiplicity : str
        'PS' for Press-Schechter, 'ST' for Sheth-Tormen

    Returns
    -------
    dndM : float or ndarray
        Mass function dn/dM [M_sun^{-1} (Mpc/h)^{-3}]
    """
    if Omega_Lambda is None:
        Omega_Lambda = 1.0 - Omega_m0

    # Mean matter density [M_sun / (Mpc/h)^3]
    rho_crit_0 = 2.775e11  # h^2 M_sun / Mpc^3
    rho_m0 = Omega_m0 * rho_crit_0 * h**2

    # Handle scalar/array
    scalar_input = np.isscalar(M)
    M_array = np.atleast_1d(M)

    dndM_values = np.zeros_like(M_array, dtype=float)

    # Critical overdensity
    delta_c = critical_overdensity(z, Omega_m0, Omega_Lambda)

    for i, M_val in enumerate(M_array):
        # sigma(M)
        sig = sigma_M(M_val, z, sigma8, n_s, Omega_m0, Omega_b0,
                      Omega_Lambda, h)

        # Peak height
        nu = delta_c / sig

        # Multiplicity function
        if multiplicity == 'PS':
            f_nu = f_PS(nu)
        elif multiplicity == 'ST':
            f_nu = f_ST(nu)
        else:
            raise ValueError(f"Unknown multiplicity: {multiplicity}")

        # Logarithmic derivative
        dlns_dlnM = dln_sigma_dln_M(M_val, z, sigma8, n_s, Omega_m0,
                                     Omega_b0, Omega_Lambda, h)

        # Mass function
        dndM_values[i] = (rho_m0 / M_val**2) * np.abs(dlns_dlnM) * f_nu

    if scalar_input:
        return dndM_values[0]
    return dndM_values


def integrated_halo_number(M_min, z=0, M_max=1e16, sigma8=SIGMA_8, n_s=N_S,
                            Omega_m0=OMEGA_M0, Omega_b0=OMEGA_B0,
                            Omega_Lambda=None, h=0.7,
                            multiplicity='PS', n_points=50):
    """
    Integrated halo number density n(>M_min).

    n(>M) = integral_{M_min}^{M_max} (dn/dM) dM

    Parameters
    ----------
    M_min : float
        Minimum mass threshold [M_sun]
    z : float
        Redshift
    M_max : float
        Maximum mass for integration [M_sun]
    sigma8, n_s : float
        Power spectrum parameters
    Omega_m0, Omega_b0, Omega_Lambda, h : float
        Cosmological parameters
    multiplicity : str
        'PS' or 'ST'
    n_points : int
        Number of points for integration

    Returns
    -------
    n_gt_M : float
        Comoving number density of halos with M > M_min [(Mpc/h)^{-3}]
    """
    if Omega_Lambda is None:
        Omega_Lambda = 1.0 - Omega_m0

    # Log-spaced mass array
    log_M = np.linspace(np.log10(M_min), np.log10(M_max), n_points)
    M_array = 10**log_M

    # Compute mass function at each point
    dndM = press_schechter_mass_function(
        M_array, z, sigma8, n_s, Omega_m0, Omega_b0, Omega_Lambda, h,
        multiplicity
    )

    # Integrate using trapezoidal rule in log space
    # integral dndM dM = integral dndM * M * d(ln M)
    integrand = dndM * M_array
    d_lnM = np.diff(log_M) * np.log(10)

    # Trapezoidal integration
    n_gt_M = np.sum(0.5 * (integrand[:-1] + integrand[1:]) * d_lnM)

    return n_gt_M


def total_collapsed_fraction(z=0, M_min=0, sigma8=SIGMA_8, n_s=N_S,
                              Omega_m0=OMEGA_M0, Omega_Lambda=None):
    """
    Total fraction of mass in collapsed halos above M_min.

    For Press-Schechter, this is erfc(nu_min / sqrt(2)) where nu_min = delta_c / sigma(M_min).

    Parameters
    ----------
    z : float
        Redshift
    M_min : float
        Minimum halo mass [M_sun]. If 0, returns total collapsed fraction.
    sigma8, n_s : float
        Power spectrum parameters
    Omega_m0, Omega_Lambda : float
        Cosmological parameters

    Returns
    -------
    f_coll : float
        Collapsed mass fraction
    """
    from scipy.special import erfc

    if Omega_Lambda is None:
        Omega_Lambda = 1.0 - Omega_m0

    delta_c = critical_overdensity(z, Omega_m0, Omega_Lambda)

    if M_min <= 0:
        # Total collapsed fraction (formally sigma -> infinity at M -> 0)
        # In practice, use a very small mass
        M_min = 1e4  # 10^4 M_sun as minimum

    sig = sigma_M(M_min, z, sigma8, n_s, Omega_m0, OMEGA_B0, Omega_Lambda)
    nu_min = delta_c / sig

    # Press-Schechter collapsed fraction
    f_coll = erfc(nu_min / np.sqrt(2.0))

    return f_coll


# =============================================================================
# Main: Test the functions
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Halo Mass Function Module Test")
    print("=" * 70)

    # Test mass function
    print("\n1. Press-Schechter Mass Function at z=0:")
    print("-" * 50)
    M_values = [1e8, 1e10, 1e12, 1e14]
    print(f"{'M [M_sun]':>12} | {'dn/dM (PS)':>15} | {'dn/dM (ST)':>15}")
    print("-" * 50)
    for M in M_values:
        dndM_PS = press_schechter_mass_function(M, multiplicity='PS')
        dndM_ST = press_schechter_mass_function(M, multiplicity='ST')
        print(f"{M:>12.0e} | {dndM_PS:>15.4e} | {dndM_ST:>15.4e}")

    # Test integrated number
    print("\n2. Integrated Halo Number n(>M) at z=0:")
    print("-" * 50)
    M_thresholds = [1e6, 1e8, 1e10, 1e12]
    print(f"{'M_min [M_sun]':>14} | {'n(>M) PS':>15} | {'n(>M) ST':>15}")
    print("-" * 50)
    for M_min in M_thresholds:
        n_PS = integrated_halo_number(M_min, multiplicity='PS')
        n_ST = integrated_halo_number(M_min, multiplicity='ST')
        print(f"{M_min:>14.0e} | {n_PS:>15.4e} | {n_ST:>15.4e}")

    # Test collapsed fraction
    print("\n3. Collapsed Mass Fraction at different z:")
    print("-" * 40)
    z_values = [0, 1, 2, 5]
    for z in z_values:
        f_coll = total_collapsed_fraction(z, M_min=1e8)
        print(f"  z = {z}: f_coll(M > 10^8 M_sun) = {f_coll:.4f}")

    print("\nDone!")
