#!/usr/bin/env python3
"""
N_BH_v3_press_schechter_sfr.py

Compute black hole abundance N_BH(Omega_Lambda) using:
1. Press-Schechter collapsed fraction F_coll(z, Omega_Lambda)
2. Cosmic star formation history (CSFH) baseline
3. Integration over redshift and comoving volume

Physical picture:
- Black holes form from massive stars in galaxies
- Star formation rate depends on available gas and cosmic epoch
- The collapsed fraction modulates how much structure exists
- Higher Omega_Lambda suppresses structure, reducing BH formation
- We integrate over cosmic history to get total BH count

Key equations:
- SFR0(z) ~ (1+z)^a / (1 + ((1+z)/b)^c)  [Madau & Dickinson 2014 form]
- n_dot_BH(z) ~ SFR0(z) * F_coll(z, OL) / F_coll(z, OL_ref)
- N_BH ~ integral dz * n_dot_BH * dVc/dz * |dt/dz|

References:
- Madau & Dickinson (2014), ARA&A 52, 415
- Press & Schechter (1974), ApJ 187, 425
"""

import numpy as np
from cosmology_utils import (
    H_of_z, E_of_z, comoving_distance, dVc_dz, dt_dz,
    collapsed_fraction
)


# =============================================================================
# Cosmic Star Formation Rate Density
# =============================================================================

def sfr_baseline(z, a=2.5, b=2.0, c=4.0):
    """
    Baseline cosmic star formation rate density (arbitrary normalization).

    This follows the functional form from Madau & Dickinson (2014):
    SFR0(z) ~ (1+z)^a / (1 + ((1+z)/b)^c)

    Physical interpretation:
    - Rising phase at low z: SFR increases as (1+z)^a due to higher gas fractions
    - Peak around z ~ 1-2: balance between gas availability and structure formation
    - Declining phase at high z: fewer massive halos, less star formation

    Parameters
    ----------
    z : float or ndarray
        Redshift
    a : float
        Low-z power law index (default 2.5)
        Controls how fast SFR rises from z=0
    b : float
        Turnover redshift parameter (default 2.0)
        Peak SFR occurs around z ~ b - 1
    c : float
        High-z suppression power (default 4.0)
        Controls how fast SFR drops at high z

    Returns
    -------
    sfr : float or ndarray
        Star formation rate density (arbitrary units)

    Notes
    -----
    The default parameters give a peak around z ~ 1.5-2, consistent
    with observations. The normalization is arbitrary since we only
    need relative values for comparing different Omega_Lambda.
    """
    one_plus_z = 1.0 + z

    # Numerator: rising phase
    numerator = one_plus_z**a

    # Denominator: turnover and high-z suppression
    denominator = 1.0 + (one_plus_z / b)**c

    return numerator / denominator


# =============================================================================
# BH Formation Rate Density
# =============================================================================

def bh_formation_rate_density(z, omega_lambda, omega_lambda_ref=0.7,
                               M_min=1e12, tiny=1e-10):
    """
    Black hole formation rate density as function of z and Omega_Lambda.

    n_dot_BH(z; OL) ~ SFR0(z) * F_coll(z, OL)

    Physical interpretation:
    - SFR0(z) gives the baseline star formation history
    - F_coll(z, OL) gives the collapsed fraction for this cosmology
    - Higher OL means less structure, lower F_coll, fewer BHs

    Parameters
    ----------
    z : float
        Redshift
    omega_lambda : float
        Dark energy density parameter
    omega_lambda_ref : float
        Reference Omega_Lambda (not used in this simpler version)
    M_min : float
        Minimum halo mass for collapsed fraction [M_sun]
    tiny : float
        Floor to prevent issues with zero collapsed fraction

    Returns
    -------
    n_dot_BH : float
        BH formation rate density (arbitrary units)

    Notes
    -----
    The absolute normalization doesn't matter since we normalize
    the final N_BH. We use F_coll directly rather than a ratio
    to avoid numerical instabilities when F_coll_ref is tiny at high z.
    """
    # Baseline star formation rate
    sfr = sfr_baseline(z)

    # Collapsed fraction for this cosmology
    F_coll_OL = collapsed_fraction(z, omega_lambda, M_min=M_min)

    # Floor to avoid numerical issues
    F_coll_safe = max(F_coll_OL, tiny)

    # BH formation rate density: SFR weighted by collapsed fraction
    n_dot_BH = sfr * F_coll_safe

    return n_dot_BH


# =============================================================================
# Total BH Count
# =============================================================================

def N_BH_v3(omega_lambda, z_max=10.0, nz=400, omega_lambda_ref=0.7,
            M_min=1e12):
    """
    Compute v3 estimate of total black hole count as function of Omega_Lambda.

    Integrates the BH formation rate over redshift and comoving volume:

    N_BH(OL) ~ integral_0^{z_max} dz * n_dot_BH(z; OL) * dVc/dz * |dt/dz|

    Parameters
    ----------
    omega_lambda : float
        Dark energy density parameter
    z_max : float
        Maximum redshift for integration (default 10)
    nz : int
        Number of redshift grid points (default 400)
    omega_lambda_ref : float
        Reference Omega_Lambda for BH formation rate
    M_min : float
        Minimum halo mass for collapsed fraction [M_sun]

    Returns
    -------
    N_BH : float
        Total BH count (arbitrary normalization)

    Notes
    -----
    The integral combines:
    - n_dot_BH: BH formation rate density [#/Mpc^3/Gyr]
    - dVc/dz: comoving volume element [Mpc^3/sr/dz]
    - |dt/dz|: time per unit redshift [Gyr/dz]

    We use trapezoidal integration over a uniform z grid.
    """
    # Check for valid cosmology
    omega_m = 1.0 - omega_lambda
    if omega_m <= 0.01 or omega_m >= 0.99:
        return 0.0

    # Build redshift grid
    z_array = np.linspace(0, z_max, nz)
    dz = z_array[1] - z_array[0]

    # Compute integrand at each z
    integrand = np.zeros(nz)

    for i, z in enumerate(z_array):
        # Skip z=0 to avoid potential issues
        if z < 1e-6:
            integrand[i] = 0.0
            continue

        # BH formation rate density
        n_dot = bh_formation_rate_density(z, omega_lambda, omega_lambda_ref,
                                           M_min=M_min)

        # Comoving volume element
        dV = dVc_dz(z, omega_lambda)

        # Time per unit redshift
        dt = dt_dz(z, omega_lambda)

        # Integrand: n_dot * dV/dz * |dt/dz|
        integrand[i] = n_dot * dV * dt

    # Trapezoidal integration
    N_BH = np.trapz(integrand, z_array)

    return N_BH


def N_BH_v3_detailed(omega_lambda, z_max=10.0, nz=400, omega_lambda_ref=0.7,
                      M_min=1e12):
    """
    Compute N_BH_v3 and return detailed breakdown by redshift.

    Parameters
    ----------
    omega_lambda : float
        Dark energy density parameter
    z_max : float
        Maximum redshift for integration
    nz : int
        Number of redshift grid points
    omega_lambda_ref : float
        Reference Omega_Lambda
    M_min : float
        Minimum halo mass [M_sun]

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'z': redshift array
        - 'n_dot_BH': BH formation rate density
        - 'dVc_dz': comoving volume element
        - 'dt_dz': time per unit redshift
        - 'integrand': full integrand
        - 'N_BH': total integral
    """
    omega_m = 1.0 - omega_lambda
    if omega_m <= 0.01 or omega_m >= 0.99:
        return {'N_BH': 0.0}

    z_array = np.linspace(0, z_max, nz)

    n_dot_array = np.zeros(nz)
    dV_array = np.zeros(nz)
    dt_array = np.zeros(nz)
    integrand = np.zeros(nz)

    for i, z in enumerate(z_array):
        if z < 1e-6:
            continue

        n_dot_array[i] = bh_formation_rate_density(z, omega_lambda,
                                                    omega_lambda_ref, M_min)
        dV_array[i] = dVc_dz(z, omega_lambda)
        dt_array[i] = dt_dz(z, omega_lambda)
        integrand[i] = n_dot_array[i] * dV_array[i] * dt_array[i]

    N_BH = np.trapz(integrand, z_array)

    return {
        'z': z_array,
        'n_dot_BH': n_dot_array,
        'dVc_dz': dV_array,
        'dt_dz': dt_array,
        'integrand': integrand,
        'N_BH': N_BH
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("N_BH v3 Module Test")
    print("=" * 60)

    # Test SFR baseline
    print("\n1. Baseline SFR(z):")
    print("-" * 40)
    test_z = [0, 0.5, 1, 2, 3, 5, 8]
    for z in test_z:
        sfr = sfr_baseline(z)
        print(f"  z = {z}: SFR0 = {sfr:.4f}")

    # Test N_BH for a few Omega_Lambda values
    print("\n2. N_BH_v3(Omega_Lambda):")
    print("-" * 40)
    test_ol = [0.1, 0.3, 0.5, 0.7, 0.9]
    N_values = []
    for ol in test_ol:
        N = N_BH_v3(ol, nz=200)
        N_values.append(N)
        print(f"  Omega_Lambda = {ol}: N_BH = {N:.4e}")

    # Normalize
    N_max = max(N_values)
    print("\n3. Normalized N_BH:")
    print("-" * 40)
    for ol, N in zip(test_ol, N_values):
        N_norm = N / N_max if N_max > 0 else 0
        print(f"  Omega_Lambda = {ol}: N_BH_norm = {N_norm:.4f}")

    print("\nDone!")
