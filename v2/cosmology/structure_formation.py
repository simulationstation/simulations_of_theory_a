#!/usr/bin/env python3
"""
structure_formation.py

Linear perturbation theory for structure formation in LCDM cosmology.

Provides:
- Linear growth factor D(a) and its normalized form D(a)/D(a=1)
- Critical overdensity delta_c(z) for spherical collapse
- Matter variance sigma(M) from the power spectrum

These are the building blocks for Press-Schechter theory.

References:
- Peebles (1980), "The Large-Scale Structure of the Universe"
- Carroll, Press & Turner (1992), ARA&A 30, 499
- Mo, van den Bosch & White (2010), "Galaxy Formation and Evolution"
"""

import numpy as np
from scipy import integrate, interpolate

# Import constants from config (will be created)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    H0_SI, OMEGA_M0, OMEGA_R0, OMEGA_B0,
    SIGMA_8, N_S, C_LIGHT,
    M_SUN, MPC_TO_M, GYR_TO_S
)


# =============================================================================
# Hubble Parameter and Scale Factor
# =============================================================================

def E_of_a(a, Omega_m0, Omega_r0, Omega_Lambda):
    """
    Dimensionless Hubble parameter E(a) = H(a)/H0.

    For flat LCDM: E^2(a) = Omega_r0 * a^{-4} + Omega_m0 * a^{-3} + Omega_Lambda

    Parameters
    ----------
    a : float or ndarray
        Scale factor (a=1 today)
    Omega_m0 : float
        Matter density parameter today
    Omega_r0 : float
        Radiation density parameter today
    Omega_Lambda : float
        Dark energy density parameter

    Returns
    -------
    E : float or ndarray
        E(a) = H(a)/H0
    """
    E_squared = Omega_r0 * a**(-4) + Omega_m0 * a**(-3) + Omega_Lambda
    return np.sqrt(E_squared)


def H_of_a(a, H0, Omega_m0, Omega_r0, Omega_Lambda):
    """
    Hubble parameter H(a) in SI units [1/s].

    Parameters
    ----------
    a : float or ndarray
        Scale factor
    H0 : float
        Hubble constant today [1/s]
    Omega_m0, Omega_r0, Omega_Lambda : float
        Density parameters

    Returns
    -------
    H : float or ndarray
        Hubble parameter [1/s]
    """
    return H0 * E_of_a(a, Omega_m0, Omega_r0, Omega_Lambda)


# =============================================================================
# Linear Growth Factor D(a)
# =============================================================================

def growth_factor_integrand(a, Omega_m0, Omega_Lambda):
    """
    Integrand for the linear growth factor integral.

    D(a) propto H(a) * integral_0^a [ da' / (a' H(a'))^3 ]

    For matter + Lambda (ignoring radiation at late times):
    E^2(a) = Omega_m0 * a^{-3} + Omega_Lambda

    Parameters
    ----------
    a : float
        Scale factor
    Omega_m0 : float
        Matter density parameter
    Omega_Lambda : float
        Dark energy density parameter

    Returns
    -------
    integrand : float
        1 / (a * E(a))^3
    """
    # Ignore radiation for growth factor (valid for a > 0.001)
    E = np.sqrt(Omega_m0 * a**(-3) + Omega_Lambda)
    return 1.0 / (a * E)**3


def growth_factor(a, Omega_m0=OMEGA_M0, Omega_Lambda=None):
    """
    Compute the linear growth factor D(a) for flat LCDM.

    Uses the integral formula:
    D(a) = (5/2) * Omega_m0 * E(a) * integral_0^a [ da' / (a' E(a'))^3 ]

    Normalized such that D(a) -> a in the matter-dominated era.

    Parameters
    ----------
    a : float or ndarray
        Scale factor(s) at which to evaluate D
    Omega_m0 : float
        Matter density parameter today (default from config)
    Omega_Lambda : float, optional
        Dark energy parameter. If None, computed from flatness: 1 - Omega_m0

    Returns
    -------
    D : float or ndarray
        Linear growth factor D(a)
    """
    if Omega_Lambda is None:
        Omega_Lambda = 1.0 - Omega_m0

    # Handle scalar vs array input
    scalar_input = np.isscalar(a)
    a_array = np.atleast_1d(a)

    D_values = np.zeros_like(a_array, dtype=float)

    for i, a_val in enumerate(a_array):
        if a_val <= 0:
            D_values[i] = 0.0
            continue

        # Integrate from small a to a_val
        a_min = 1e-6
        if a_val < a_min:
            # In matter domination, D ~ a
            D_values[i] = a_val
            continue

        result, _ = integrate.quad(
            growth_factor_integrand, a_min, a_val,
            args=(Omega_m0, Omega_Lambda),
            limit=100
        )

        # Prefactor
        E_a = np.sqrt(Omega_m0 * a_val**(-3) + Omega_Lambda)
        D_values[i] = (5.0 / 2.0) * Omega_m0 * E_a * result

    if scalar_input:
        return D_values[0]
    return D_values


def growth_factor_normalized(a, Omega_m0=OMEGA_M0, Omega_Lambda=None):
    """
    Normalized linear growth factor D(a)/D(a=1).

    Parameters
    ----------
    a : float or ndarray
        Scale factor(s)
    Omega_m0 : float
        Matter density parameter
    Omega_Lambda : float, optional
        Dark energy parameter

    Returns
    -------
    D_norm : float or ndarray
        D(a) / D(1)
    """
    D_a = growth_factor(a, Omega_m0, Omega_Lambda)
    D_1 = growth_factor(1.0, Omega_m0, Omega_Lambda)
    return D_a / D_1


def growth_rate_f(a, Omega_m0=OMEGA_M0, Omega_Lambda=None):
    """
    Logarithmic growth rate f = d ln D / d ln a.

    Approximation: f ~ Omega_m(a)^gamma where gamma ~ 0.55 for LCDM.

    Parameters
    ----------
    a : float
        Scale factor
    Omega_m0 : float
        Matter density parameter today
    Omega_Lambda : float, optional
        Dark energy parameter

    Returns
    -------
    f : float
        Growth rate f(a)
    """
    if Omega_Lambda is None:
        Omega_Lambda = 1.0 - Omega_m0

    # Omega_m(a) = Omega_m0 * a^{-3} / E^2(a)
    E_sq = Omega_m0 * a**(-3) + Omega_Lambda
    Omega_m_a = Omega_m0 * a**(-3) / E_sq

    # Approximation for LCDM
    gamma = 0.55
    return Omega_m_a**gamma


# =============================================================================
# Critical Overdensity for Collapse
# =============================================================================

def critical_overdensity(z=0, Omega_m0=OMEGA_M0, Omega_Lambda=None):
    """
    Critical linear overdensity delta_c for spherical collapse.

    For Einstein-de Sitter (Omega_m = 1): delta_c = 1.686
    For LCDM, there is a weak dependence on Omega_m(z).

    Approximation from Kitayama & Suto (1996):
    delta_c(z) ~ 1.686 * [1 + 0.0123 * log10(Omega_m(z))]

    Parameters
    ----------
    z : float
        Redshift
    Omega_m0 : float
        Matter density parameter today
    Omega_Lambda : float, optional
        Dark energy parameter

    Returns
    -------
    delta_c : float
        Critical overdensity for collapse at redshift z
    """
    if Omega_Lambda is None:
        Omega_Lambda = 1.0 - Omega_m0

    # EdS value
    delta_c_EdS = 1.686

    if Omega_Lambda == 0 and Omega_m0 == 1:
        return delta_c_EdS

    # Compute Omega_m(z)
    a = 1.0 / (1.0 + z)
    E_sq = Omega_m0 * a**(-3) + Omega_Lambda
    Omega_m_z = Omega_m0 * a**(-3) / E_sq

    # Kitayama & Suto approximation
    delta_c = delta_c_EdS * (1.0 + 0.0123 * np.log10(Omega_m_z))

    return delta_c


# =============================================================================
# Matter Power Spectrum and Variance
# =============================================================================

def transfer_function_BBKS(k, Omega_m0=OMEGA_M0, Omega_b0=OMEGA_B0, h=0.7):
    """
    BBKS transfer function T(k) for CDM.

    From Bardeen, Bond, Kaiser & Szalay (1986), Eq. 7.70 in Mo et al.

    Parameters
    ----------
    k : float or ndarray
        Wavenumber [h/Mpc]
    Omega_m0 : float
        Total matter density parameter
    Omega_b0 : float
        Baryon density parameter
    h : float
        Hubble parameter h = H0 / (100 km/s/Mpc)

    Returns
    -------
    T : float or ndarray
        Transfer function T(k)
    """
    # Shape parameter
    Gamma = Omega_m0 * h * np.exp(-Omega_b0 * (1 + np.sqrt(2*h) / Omega_m0))

    q = k / Gamma  # Dimensionless wavenumber

    # BBKS fitting formula
    T = (np.log(1 + 2.34*q) / (2.34*q)) * (
        1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
    )**(-0.25)

    return T


def power_spectrum(k, sigma8=SIGMA_8, n_s=N_S, Omega_m0=OMEGA_M0,
                   Omega_b0=OMEGA_B0, h=0.7):
    """
    Linear matter power spectrum P(k).

    P(k) = A * k^n_s * T(k)^2

    Normalized to sigma_8 at z=0.

    Parameters
    ----------
    k : float or ndarray
        Wavenumber [h/Mpc]
    sigma8 : float
        RMS fluctuation in 8 Mpc/h spheres at z=0
    n_s : float
        Scalar spectral index
    Omega_m0 : float
        Matter density parameter
    Omega_b0 : float
        Baryon density parameter
    h : float
        Hubble parameter

    Returns
    -------
    P : float or ndarray
        Power spectrum P(k) [(Mpc/h)^3]
    """
    T_k = transfer_function_BBKS(k, Omega_m0, Omega_b0, h)

    # Primordial power spectrum times transfer function squared
    P_unnorm = k**n_s * T_k**2

    # Normalization: we need to match sigma_8
    # This requires computing sigma(R=8 Mpc/h) and normalizing
    # For now, use approximate normalization
    # A proper implementation would integrate to find A such that sigma(8) = sigma8

    # Approximate amplitude (will be renormalized)
    A = 1.0

    return A * P_unnorm


def window_function_tophat(k, R):
    """
    Fourier-space top-hat window function.

    W(kR) = 3 * [sin(kR) - kR*cos(kR)] / (kR)^3

    Parameters
    ----------
    k : float or ndarray
        Wavenumber
    R : float
        Smoothing radius

    Returns
    -------
    W : float or ndarray
        Window function W(kR)
    """
    x = k * R
    # Handle x -> 0 limit
    x = np.atleast_1d(x)
    W = np.zeros_like(x)

    small = np.abs(x) < 1e-4
    W[small] = 1.0 - x[small]**2 / 10.0  # Taylor expansion

    large = ~small
    W[large] = 3.0 * (np.sin(x[large]) - x[large]*np.cos(x[large])) / x[large]**3

    if np.isscalar(k * R):
        return W[0]
    return W


def sigma_M(M, z=0, sigma8=SIGMA_8, n_s=N_S, Omega_m0=OMEGA_M0,
            Omega_b0=OMEGA_B0, Omega_Lambda=None, h=0.7):
    """
    RMS mass fluctuation sigma(M) at redshift z.

    sigma^2(M) = (1/2pi^2) * integral k^2 P(k) W^2(kR) dk

    where R = (3M / 4pi rho_m)^{1/3} is the Lagrangian radius.

    Parameters
    ----------
    M : float or ndarray
        Mass scale [M_sun]
    z : float
        Redshift
    sigma8 : float
        RMS fluctuation at R=8 Mpc/h at z=0
    n_s : float
        Scalar spectral index
    Omega_m0 : float
        Matter density parameter
    Omega_b0 : float
        Baryon density parameter
    Omega_Lambda : float, optional
        Dark energy parameter
    h : float
        Hubble parameter

    Returns
    -------
    sigma : float or ndarray
        RMS fluctuation sigma(M, z)
    """
    if Omega_Lambda is None:
        Omega_Lambda = 1.0 - Omega_m0

    # Mean matter density today [M_sun / (Mpc/h)^3]
    rho_crit_0 = 2.775e11  # h^2 M_sun / Mpc^3
    rho_m0 = Omega_m0 * rho_crit_0 * h**2  # M_sun / (Mpc/h)^3

    # Handle scalar/array input
    scalar_input = np.isscalar(M)
    M_array = np.atleast_1d(M)

    sigma_values = np.zeros_like(M_array, dtype=float)

    for i, M_val in enumerate(M_array):
        # Lagrangian radius [Mpc/h]
        R = (3.0 * M_val / (4.0 * np.pi * rho_m0))**(1.0/3.0)

        # Integrate sigma^2
        def integrand(k):
            if k < 1e-6:
                return 0.0
            T_k = transfer_function_BBKS(k, Omega_m0, Omega_b0, h)
            W_k = window_function_tophat(k, R)
            P_k = k**n_s * T_k**2
            return k**2 * P_k * W_k**2

        # Integration limits
        k_min = 1e-4
        k_max = 1e3

        result, _ = integrate.quad(integrand, k_min, k_max, limit=200)
        sigma_sq_unnorm = result / (2.0 * np.pi**2)

        sigma_values[i] = np.sqrt(sigma_sq_unnorm)

    # Normalize to sigma_8
    # Compute sigma at R = 8 Mpc/h
    R_8 = 8.0  # Mpc/h
    M_8 = (4.0/3.0) * np.pi * R_8**3 * rho_m0  # Mass in 8 Mpc/h sphere

    def integrand_8(k):
        if k < 1e-6:
            return 0.0
        T_k = transfer_function_BBKS(k, Omega_m0, Omega_b0, h)
        W_k = window_function_tophat(k, R_8)
        P_k = k**n_s * T_k**2
        return k**2 * P_k * W_k**2

    result_8, _ = integrate.quad(integrand_8, 1e-4, 1e3, limit=200)
    sigma_8_unnorm = np.sqrt(result_8 / (2.0 * np.pi**2))

    # Normalization factor
    norm = sigma8 / sigma_8_unnorm
    sigma_values *= norm

    # Apply growth factor for z > 0
    if z > 0:
        a = 1.0 / (1.0 + z)
        D_ratio = growth_factor_normalized(a, Omega_m0, Omega_Lambda)
        sigma_values *= D_ratio

    if scalar_input:
        return sigma_values[0]
    return sigma_values


# =============================================================================
# Main: Test the functions
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Structure Formation Module Test")
    print("=" * 70)

    # Test growth factor
    print("\n1. Linear Growth Factor D(a):")
    print("-" * 40)
    a_values = [0.1, 0.2, 0.5, 1.0]
    for a in a_values:
        D = growth_factor(a)
        D_norm = growth_factor_normalized(a)
        print(f"  a = {a:.2f}: D(a) = {D:.4f}, D(a)/D(1) = {D_norm:.4f}")

    # Test critical overdensity
    print("\n2. Critical Overdensity delta_c(z):")
    print("-" * 40)
    z_values = [0, 0.5, 1, 2, 5]
    for z in z_values:
        dc = critical_overdensity(z)
        print(f"  z = {z}: delta_c = {dc:.4f}")

    # Test sigma(M)
    print("\n3. Mass Variance sigma(M) at z=0:")
    print("-" * 40)
    M_values = [1e8, 1e10, 1e12, 1e14]  # M_sun
    for M in M_values:
        sig = sigma_M(M)
        print(f"  M = {M:.0e} M_sun: sigma(M) = {sig:.4f}")

    print("\nDone!")
