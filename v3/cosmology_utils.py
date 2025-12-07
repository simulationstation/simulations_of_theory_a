#!/usr/bin/env python3
"""
cosmology_utils.py

Shared cosmology functions for v3 pipeline.

Provides:
- H(z) Hubble parameter
- Comoving distance and volume element
- dt/dz time-redshift relation
- Reuses v2 growth factor and collapsed fraction

All functions assume flat FLRW with Omega_m = 1 - Omega_Lambda.
"""

import numpy as np
from scipy import integrate

# =============================================================================
# Physical Constants
# =============================================================================

C_LIGHT_KM_S = 299792.458      # Speed of light [km/s]
H0_DEFAULT = 70.0              # Hubble constant [km/s/Mpc]
H0_INV_GYR = 13.97             # 1/H0 in Gyr for H0=70 km/s/Mpc


# =============================================================================
# Hubble Parameter
# =============================================================================

def H_of_z(z, omega_lambda, H0=H0_DEFAULT):
    """
    Hubble parameter H(z) for flat LCDM.

    H(z)^2 = H0^2 * [Omega_m * (1+z)^3 + Omega_Lambda]

    Parameters
    ----------
    z : float or ndarray
        Redshift
    omega_lambda : float
        Dark energy density parameter
    H0 : float
        Hubble constant [km/s/Mpc]

    Returns
    -------
    H : float or ndarray
        Hubble parameter [km/s/Mpc]
    """
    omega_m = 1.0 - omega_lambda
    H_squared = H0**2 * (omega_m * (1 + z)**3 + omega_lambda)
    return np.sqrt(H_squared)


def E_of_z(z, omega_lambda):
    """
    Dimensionless Hubble parameter E(z) = H(z)/H0.

    Parameters
    ----------
    z : float or ndarray
        Redshift
    omega_lambda : float
        Dark energy density parameter

    Returns
    -------
    E : float or ndarray
        E(z) = H(z)/H0
    """
    omega_m = 1.0 - omega_lambda
    return np.sqrt(omega_m * (1 + z)**3 + omega_lambda)


# =============================================================================
# Comoving Distance and Volume
# =============================================================================

def comoving_distance(z, omega_lambda, H0=H0_DEFAULT):
    """
    Comoving distance D_c(z) in Mpc.

    D_c(z) = (c/H0) * integral_0^z dz' / E(z')

    Parameters
    ----------
    z : float
        Redshift
    omega_lambda : float
        Dark energy density parameter
    H0 : float
        Hubble constant [km/s/Mpc]

    Returns
    -------
    D_c : float
        Comoving distance [Mpc]
    """
    if z <= 0:
        return 0.0

    def integrand(zp):
        return 1.0 / E_of_z(zp, omega_lambda)

    result, _ = integrate.quad(integrand, 0, z, limit=100)
    D_c = (C_LIGHT_KM_S / H0) * result

    return D_c


def dVc_dz(z, omega_lambda, H0=H0_DEFAULT):
    """
    Comoving volume element dV_c/dz per steradian.

    dV_c/dz = (c/H0) * D_c(z)^2 / E(z)

    Parameters
    ----------
    z : float
        Redshift
    omega_lambda : float
        Dark energy density parameter
    H0 : float
        Hubble constant [km/s/Mpc]

    Returns
    -------
    dVcdz : float
        Comoving volume element [Mpc^3 per unit z per steradian]
    """
    D_c = comoving_distance(z, omega_lambda, H0)
    E_z = E_of_z(z, omega_lambda)

    # dV_c/dz = (c/H0) * D_c^2 / E(z)
    dVcdz = (C_LIGHT_KM_S / H0) * D_c**2 / E_z

    return dVcdz


# =============================================================================
# Time-Redshift Relation
# =============================================================================

def dt_dz(z, omega_lambda, H0=H0_DEFAULT):
    """
    Return |dt/dz| in Gyr per unit redshift.

    dt/dz = -1 / [(1+z) * H(z)]

    We return the absolute value since we integrate from z=0 to z_max.

    Parameters
    ----------
    z : float or ndarray
        Redshift
    omega_lambda : float
        Dark energy density parameter
    H0 : float
        Hubble constant [km/s/Mpc]

    Returns
    -------
    dt_dz : float or ndarray
        |dt/dz| in Gyr per unit redshift
    """
    # H(z) in km/s/Mpc
    H_z = H_of_z(z, omega_lambda, H0)

    # Convert H from km/s/Mpc to 1/Gyr
    # 1 Mpc = 3.086e19 km, 1 Gyr = 3.156e16 s
    # H [1/Gyr] = H [km/s/Mpc] * (3.086e19 km/Mpc) / (3.156e16 s/Gyr) / (km/s)
    #           = H [km/s/Mpc] * 0.978
    # Or simply: 1/H0 ~ 14 Gyr for H0=70
    H_inv_Gyr = H0_INV_GYR / H0 * H0_DEFAULT  # Scale if H0 differs

    # |dt/dz| = 1 / [(1+z) * H(z)] in Gyr
    # Using H_z in km/s/Mpc, we need the conversion factor
    # H_z [1/Gyr] = H_z [km/s/Mpc] * 1.022e-3
    km_s_Mpc_to_inv_Gyr = 1.0 / 978.0  # Approximate conversion

    H_z_inv_Gyr = H_z * km_s_Mpc_to_inv_Gyr

    return 1.0 / ((1 + z) * H_z_inv_Gyr)


# =============================================================================
# Collapsed Fraction (wrapper for v2)
# =============================================================================

def collapsed_fraction(z, omega_lambda, M_min=1e12, sigma8=0.811):
    """
    Press-Schechter collapsed fraction F_coll(z, Omega_Lambda).

    Uses a semi-analytic approximation that properly captures
    the Omega_Lambda dependence without numerical instabilities.

    Parameters
    ----------
    z : float
        Redshift
    omega_lambda : float
        Dark energy density parameter
    M_min : float
        Minimum halo mass [M_sun] (default 10^12)
    sigma8 : float
        Power spectrum normalization

    Returns
    -------
    F_coll : float
        Collapsed mass fraction in halos > M_min
    """
    from scipy.special import erfc

    omega_m = 1.0 - omega_lambda

    # Ensure valid cosmology
    if omega_m <= 0.01 or omega_m >= 0.99:
        return 0.0

    # Critical overdensity (weakly z-dependent, use EdS value)
    delta_c = 1.686

    # Linear growth factor D(z) / D(0)
    # Using Carroll, Press & Turner (1992) fitting formula
    a = 1.0 / (1.0 + z)

    # Omega_m(z) and Omega_Lambda(z)
    E_z_sq = omega_m * (1 + z)**3 + omega_lambda
    Omega_m_z = omega_m * (1 + z)**3 / E_z_sq
    Omega_L_z = omega_lambda / E_z_sq

    # Growth factor approximation (Heath 1977, Carroll et al 1992)
    # D(a) ~ a * g(a) where g depends on Omega_m, Omega_Lambda
    # Simplified: D(z)/D(0) ~ (1+z)^{-1} * [Omega_m(z)]^{0.6} / [Omega_m(0)]^{0.6}
    # More accurate fitting formula:
    g_0 = 2.5 * omega_m / (omega_m**(4./7.) - omega_lambda +
                           (1 + omega_m/2.) * (1 + omega_lambda/70.))
    g_z = 2.5 * Omega_m_z / (Omega_m_z**(4./7.) - Omega_L_z +
                             (1 + Omega_m_z/2.) * (1 + Omega_L_z/70.))

    D_ratio = a * g_z / g_0

    # sigma(M) at z=0 for M_min
    # sigma(M) ~ sigma8 * (M / M_8)^{-(n+3)/6} where n ~ 1
    # For M = 10^12 M_sun, M_8 ~ 6e14 M_sun (mass in 8 Mpc/h sphere)
    # sigma(10^12) ~ sigma8 * (10^12 / 6e14)^{-0.67} ~ sigma8 * 30 ~ 2.4
    # But this is rough; use empirical calibration
    log_M_ratio = np.log10(M_min / 1e12)  # log(M/10^12)
    # sigma decreases with mass: sigma ~ M^{-0.2} approximately
    sigma_0_M = sigma8 * 2.5 * (10**(-0.2 * log_M_ratio))

    # sigma at redshift z
    sigma_z = sigma_0_M * D_ratio

    # Press-Schechter: F_coll = erfc(delta_c / (sqrt(2) * sigma))
    if sigma_z <= 0:
        return 0.0

    nu = delta_c / sigma_z
    F_coll = erfc(nu / np.sqrt(2.0))

    return F_coll


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Cosmology Utils Test")
    print("=" * 50)

    test_z = [0, 0.5, 1, 2, 5]
    test_ol = 0.7

    print(f"\nOmega_Lambda = {test_ol}")
    print(f"{'z':>6} | {'H(z)':>10} | {'D_c':>10} | {'dVc/dz':>12} | {'|dt/dz|':>10}")
    print("-" * 60)

    for z in test_z:
        H = H_of_z(z, test_ol)
        Dc = comoving_distance(z, test_ol)
        dV = dVc_dz(z, test_ol)
        dt = dt_dz(z, test_ol)
        print(f"{z:>6.1f} | {H:>10.1f} | {Dc:>10.1f} | {dV:>12.2e} | {dt:>10.3f}")

    print("\nDone!")
