#!/usr/bin/env python3
"""
interior_lifetime.py

Compute the black hole interior lifetime T_int(Omega_Lambda) using a
mass-inflation-inspired flux integral model.

Physical picture:
- Black hole interiors undergo mass inflation near the inner horizon
- The Weyl curvature grows exponentially: C ~ exp(kappa * v)
- The growth rate depends on the ingoing perturbation flux
- Higher flux -> faster blowup -> shorter interior lifetime
- The cosmological constant affects the ambient radiation flux

Model:
------
We use a simplified mass-inflation model where the interior lifetime
is determined by when the curvature blowup reaches a critical threshold.

T_int ~ integral_0^{v_max} dv / [1 + L_in * exp(kappa * v)]

For small L_in (isolated BH): T_int is large
For large L_in (early universe): T_int is small

The effective flux L_in depends on Omega_Lambda through the cosmic
radiation history.

References:
- Poisson & Israel (1990), Phys. Rev. D 41, 1796 (mass inflation)
- Ori (1991), Phys. Rev. Lett. 67, 789
- Dafermos (2005), Comm. Pure Appl. Math. 58, 445
"""

import numpy as np
from scipy import integrate, optimize

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    H0_SI, OMEGA_M0, OMEGA_R0, C_LIGHT, G_NEWTON,
    M_SUN, MPC_TO_M, GYR_TO_S
)


# =============================================================================
# Cosmological Flux Model
# =============================================================================

def E_squared(a, Omega_m0, Omega_r0, Omega_Lambda):
    """
    Squared dimensionless Hubble parameter E^2(a) = (H/H0)^2.
    """
    return Omega_r0 * a**(-4) + Omega_m0 * a**(-3) + Omega_Lambda


def cosmic_time_to_scale_factor(t_Gyr, Omega_m0, Omega_r0, Omega_Lambda,
                                  H0=H0_SI, a_min=1e-8, a_max=100.0):
    """
    Invert t(a) to get a(t) numerically.

    Parameters
    ----------
    t_Gyr : float
        Cosmic time in Gyr
    Omega_m0, Omega_r0, Omega_Lambda : float
        Density parameters
    H0 : float
        Hubble constant [1/s]

    Returns
    -------
    a : float
        Scale factor at time t
    """
    t_s = t_Gyr * GYR_TO_S

    def t_of_a(a):
        """Cosmic time as function of scale factor."""
        def integrand(a_prime):
            E = np.sqrt(E_squared(a_prime, Omega_m0, Omega_r0, Omega_Lambda))
            return 1.0 / (a_prime * E * H0)
        result, _ = integrate.quad(integrand, a_min, a, limit=200)
        return result

    # Find a such that t_of_a(a) = t_s
    def objective(a):
        return t_of_a(a) - t_s

    try:
        a_solution = optimize.brentq(objective, a_min * 10, a_max)
    except ValueError:
        # If t is outside range, return boundary
        if t_of_a(a_max) < t_s:
            return a_max
        return a_min * 10

    return a_solution


def radiation_energy_density(a, Omega_r0, H0=H0_SI):
    """
    Radiation energy density at scale factor a.

    rho_r(a) = rho_r0 * a^{-4}
    rho_r0 = Omega_r0 * rho_crit * c^2

    Parameters
    ----------
    a : float
        Scale factor
    Omega_r0 : float
        Radiation density parameter today
    H0 : float
        Hubble constant [1/s]

    Returns
    -------
    rho_r : float
        Radiation energy density [J/m^3]
    """
    # Critical density
    rho_crit = 3.0 * H0**2 / (8.0 * np.pi * G_NEWTON)  # kg/m^3

    # Radiation energy density today
    rho_r0 = Omega_r0 * rho_crit * C_LIGHT**2  # J/m^3

    return rho_r0 * a**(-4)


def flux_onto_horizon(t_Gyr, M_bh, Omega_m0, Omega_r0, Omega_Lambda,
                       H0=H0_SI, C_geom=1.0):
    """
    Compute the radiation flux onto a black hole horizon at cosmic time t.

    L_in(t) = C_geom * rho_r(t) * c * A_horizon

    Parameters
    ----------
    t_Gyr : float
        Cosmic time [Gyr]
    M_bh : float
        Black hole mass [kg]
    Omega_m0, Omega_r0, Omega_Lambda : float
        Density parameters
    H0 : float
        Hubble constant [1/s]
    C_geom : float
        Geometric factor (order unity)

    Returns
    -------
    L_in : float
        Power incident on horizon [W]
    """
    # Get scale factor at time t
    a = cosmic_time_to_scale_factor(t_Gyr, Omega_m0, Omega_r0, Omega_Lambda, H0)

    # Radiation energy density
    rho_r = radiation_energy_density(a, Omega_r0, H0)

    # Horizon area
    r_s = 2.0 * G_NEWTON * M_bh / C_LIGHT**2
    A_horizon = 4.0 * np.pi * r_s**2

    # Flux
    L_in = C_geom * rho_r * C_LIGHT * A_horizon

    return L_in


def effective_flux(Omega_Lambda, M_bh=1e8 * M_SUN, t_min_Gyr=0.5,
                   t_max_Gyr=50.0, n_points=50, Omega_m0=None,
                   Omega_r0=OMEGA_R0, H0=H0_SI, method='geometric_mean'):
    """
    Compute effective perturbation flux for a given Omega_Lambda.

    Averages the flux over cosmic time to get a characteristic value.

    Parameters
    ----------
    Omega_Lambda : float
        Dark energy parameter
    M_bh : float
        Black hole mass [kg]
    t_min_Gyr : float
        Start of averaging window [Gyr]
    t_max_Gyr : float
        End of averaging window [Gyr]
    n_points : int
        Number of time points
    Omega_m0 : float, optional
        Matter density (computed from flatness if None)
    Omega_r0 : float
        Radiation density parameter
    H0 : float
        Hubble constant [1/s]
    method : str
        'geometric_mean', 'arithmetic_mean', or 'time_weighted'

    Returns
    -------
    L_eff : float
        Effective flux [W]
    """
    if Omega_m0 is None:
        Omega_m0 = 1.0 - Omega_Lambda - Omega_r0

    if Omega_m0 <= 0:
        return np.inf  # Invalid cosmology

    # Time array
    t_array = np.linspace(t_min_Gyr, t_max_Gyr, n_points)

    # Compute flux at each time
    L_array = np.array([
        flux_onto_horizon(t, M_bh, Omega_m0, Omega_r0, Omega_Lambda, H0)
        for t in t_array
    ])

    # Filter out invalid values
    valid = (L_array > 0) & np.isfinite(L_array)
    L_valid = L_array[valid]

    if len(L_valid) == 0:
        return np.inf

    if method == 'geometric_mean':
        L_eff = np.exp(np.mean(np.log(L_valid)))
    elif method == 'arithmetic_mean':
        L_eff = np.mean(L_valid)
    elif method == 'time_weighted':
        # Weight by 1/L to emphasize low-flux epochs
        weights = 1.0 / L_valid
        L_eff = np.sum(L_valid * weights) / np.sum(weights)
    else:
        L_eff = np.mean(L_valid)

    return L_eff


# =============================================================================
# Interior Lifetime Model
# =============================================================================

def T_int_from_flux(L_in, kappa=0.5, L_scale=1e20, T_scale=10.0,
                    blowup_threshold=1e6):
    """
    Compute interior lifetime from perturbation flux using scaling law.

    Uses a power-law scaling inspired by mass inflation:
    T_int ~ T_scale * (L_scale / L_in)^p

    where p is determined by the mass inflation exponent.

    Parameters
    ----------
    L_in : float
        Perturbation flux [W]
    kappa : float
        Inner horizon surface gravity parameter
    L_scale : float
        Reference flux scale [W]
    T_scale : float
        Reference time scale [dimensionless]
    blowup_threshold : float
        Curvature threshold for blowup

    Returns
    -------
    T_int : float
        Interior lifetime [dimensionless units]
    """
    # Power law exponent from mass inflation theory
    # Higher kappa -> faster growth -> smaller p
    p = 1.0 / (1.0 + kappa)

    # Scaling law
    T_int = T_scale * (L_scale / L_in)**p

    return T_int


def T_int_integral_model(L_in, kappa=0.5, v_max=20.0, n_v=200,
                          blowup_threshold=1e6):
    """
    Compute interior lifetime by integrating the mass inflation equation.

    The mass function evolves as:
    dm/dv = L_in * exp(kappa * v) * (1 + b * m)

    Interior lifetime is the v value when m exceeds threshold.

    Parameters
    ----------
    L_in : float
        Perturbation flux parameter (dimensionless)
    kappa : float
        Inner horizon surface gravity
    v_max : float
        Maximum v coordinate to integrate
    n_v : int
        Number of integration steps
    blowup_threshold : float
        Mass threshold for blowup

    Returns
    -------
    T_int : float
        Interior lifetime (v value at blowup)
    """
    # Initial conditions
    m0 = 1.0
    b = 0.1

    # v grid
    v_array = np.linspace(0, v_max, n_v)
    dv = v_array[1] - v_array[0]

    # Integrate
    m = m0
    for v in v_array[1:]:
        dm_dv = L_in * np.exp(kappa * v) * (1.0 + b * m)
        m += dv * dm_dv

        if m**2 > blowup_threshold:
            return v

    # No blowup in range
    return v_max


def compute_T_int(Omega_Lambda, M_bh=1e8 * M_SUN, kappa=0.5,
                   Omega_m0=None, Omega_r0=OMEGA_R0, H0=H0_SI,
                   t_avg_min=0.5, t_avg_max=50.0,
                   L_ref=None, normalize=True, Omega_Lambda_ref=0.7):
    """
    Compute interior lifetime T_int(Omega_Lambda).

    Pipeline:
    1. Compute effective flux L_eff(Omega_Lambda) from cosmology
    2. Map L_eff to interior lifetime using mass inflation scaling

    Parameters
    ----------
    Omega_Lambda : float or ndarray
        Dark energy parameter
    M_bh : float
        Black hole mass [kg]
    kappa : float
        Inner horizon surface gravity parameter
    Omega_m0 : float, optional
        Matter density (from flatness if None)
    Omega_r0 : float
        Radiation density parameter
    H0 : float
        Hubble constant [1/s]
    t_avg_min, t_avg_max : float
        Time range for flux averaging [Gyr]
    L_ref : float, optional
        Reference flux for normalization
    normalize : bool
        If True, normalize to T_int(Omega_Lambda_ref) = 1
    Omega_Lambda_ref : float
        Reference Omega_Lambda for normalization

    Returns
    -------
    T_int : float or ndarray
        Interior lifetime (normalized if requested)
    """
    # Handle scalar/array
    scalar_input = np.isscalar(Omega_Lambda)
    OL_array = np.atleast_1d(Omega_Lambda)

    T_int_values = np.zeros_like(OL_array, dtype=float)

    # Compute reference flux if needed
    if L_ref is None:
        if Omega_m0 is None:
            Om_ref = 1.0 - Omega_Lambda_ref - Omega_r0
        else:
            Om_ref = Omega_m0
        L_ref = effective_flux(Omega_Lambda_ref, M_bh, t_avg_min, t_avg_max,
                                Omega_m0=Om_ref, Omega_r0=Omega_r0, H0=H0)

    for i, OL in enumerate(OL_array):
        if Omega_m0 is None:
            Om = 1.0 - OL - Omega_r0
        else:
            Om = Omega_m0

        if Om <= 0:
            T_int_values[i] = 0.0
            continue

        # Effective flux
        L_eff = effective_flux(OL, M_bh, t_avg_min, t_avg_max,
                                Omega_m0=Om, Omega_r0=Omega_r0, H0=H0)

        if not np.isfinite(L_eff) or L_eff <= 0:
            T_int_values[i] = 0.0
            continue

        # Interior lifetime from flux scaling
        # Normalize L_eff to dimensionless parameter
        L_dimless = L_eff / L_ref * 0.1  # Map to toy model scale

        # Use integral model
        T_int_values[i] = T_int_integral_model(L_dimless, kappa)

    # Normalize if requested
    if normalize:
        T_ref = compute_T_int(Omega_Lambda_ref, M_bh, kappa,
                               Omega_m0=Omega_m0, Omega_r0=Omega_r0, H0=H0,
                               t_avg_min=t_avg_min, t_avg_max=t_avg_max,
                               L_ref=L_ref, normalize=False)
        if T_ref > 0:
            T_int_values /= T_ref

    if scalar_input:
        return T_int_values[0]
    return T_int_values


# =============================================================================
# Main: Test and Demonstrate
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Interior Lifetime T_int(Omega_Lambda) - v2 Model")
    print("=" * 70)
    print()

    # Parameters
    M_bh = 1e8 * M_SUN
    print(f"Black hole mass: {M_bh/M_SUN:.0e} M_sun")
    print()

    # Omega_Lambda scan
    OL_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    print("Computing effective flux L_eff(Omega_Lambda)...")
    L_eff_values = []
    for OL in OL_values:
        Om = 1.0 - OL - OMEGA_R0
        if Om > 0:
            L_eff = effective_flux(OL, M_bh, Omega_m0=Om)
            L_eff_values.append(L_eff)
        else:
            L_eff_values.append(np.nan)
    L_eff_values = np.array(L_eff_values)

    print("\nComputing T_int(Omega_Lambda)...")
    T_int_values = compute_T_int(OL_values, M_bh, normalize=True)

    print()
    print("-" * 60)
    print(f"{'Omega_Lambda':>12} | {'L_eff [W]':>12} | {'T_int (norm)':>12}")
    print("-" * 60)
    for OL, L, T in zip(OL_values, L_eff_values, T_int_values):
        print(f"{OL:>12.2f} | {L:>12.2e} | {T:>12.4f}")
    print("-" * 60)
    print()

    print("Physical interpretation:")
    print("-" * 60)
    print("""
- Higher Omega_Lambda -> faster cosmic acceleration
- Faster acceleration -> radiation dilutes more quickly
- Lower effective flux L_eff onto the black hole
- Lower flux -> slower mass inflation -> longer interior lifetime

The relationship T_int ~ L_eff^(-p) captures this inverse scaling.
""")

    # Optional plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Effective flux
        ax1 = axes[0]
        ax1.semilogy(OL_values, L_eff_values, 'ro-', markersize=10, linewidth=2)
        ax1.axvline(0.7, color='green', linestyle='--', alpha=0.7)
        ax1.set_xlabel('$\\Omega_\\Lambda$', fontsize=14)
        ax1.set_ylabel('Effective Flux $L_{eff}$ [W]', fontsize=14)
        ax1.set_title('Effective Perturbation Flux vs $\\Omega_\\Lambda$', fontsize=13)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Interior lifetime
        ax2 = axes[1]
        ax2.plot(OL_values, T_int_values, 'bo-', markersize=10, linewidth=2)
        ax2.axvline(0.7, color='green', linestyle='--', alpha=0.7,
                    label='Our Universe')
        ax2.set_xlabel('$\\Omega_\\Lambda$', fontsize=14)
        ax2.set_ylabel('$T_{int}$ (normalized)', fontsize=14)
        ax2.set_title('Interior Lifetime vs $\\Omega_\\Lambda$', fontsize=13)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(__file__), 'T_int_vs_OmegaLambda_v2.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.show()

    except ImportError:
        print("matplotlib not available for plotting")

    print("\nDone!")
