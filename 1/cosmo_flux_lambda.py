#!/usr/bin/env python3
"""
cosmo_flux_lambda.py

A toy flat ΛCDM cosmology model that computes:
1. Scale factor evolution a(t)
2. Radiation energy density ρ_r(t)
3. Flux onto a black hole horizon L_in(t)
4. Isolation time when flux drops below a threshold

This is a simplified pedagogical model, not intended for precision cosmology.

Units: SI throughout (meters, seconds, kilograms, Watts)
"""

import numpy as np
from scipy import integrate, optimize
import matplotlib.pyplot as plt


# =============================================================================
# Physical Constants (SI units)
# =============================================================================
c = 2.998e8           # Speed of light [m/s]
G = 6.674e-11         # Gravitational constant [m^3 kg^-1 s^-2]
M_sun = 1.989e30      # Solar mass [kg]
Mpc_to_m = 3.086e22   # Megaparsec to meters
Gyr_to_s = 3.156e16   # Gigayear to seconds


# =============================================================================
# Cosmological Functions
# =============================================================================

def H_of_a(a, H0, Omega_m0, Omega_r0, Omega_Lambda):
    """
    Hubble parameter H(a) for flat ΛCDM cosmology.

    H²(a) = H0² [ Omega_r0 * a^{-4} + Omega_m0 * a^{-3} + Omega_Lambda ]

    Parameters
    ----------
    a : float
        Scale factor (a=1 today)
    H0 : float
        Hubble constant [1/s]
    Omega_m0 : float
        Matter density parameter today
    Omega_r0 : float
        Radiation density parameter today
    Omega_Lambda : float
        Dark energy (cosmological constant) density parameter

    Returns
    -------
    H : float
        Hubble parameter at scale factor a [1/s]
    """
    H_squared = H0**2 * (Omega_r0 * a**(-4) + Omega_m0 * a**(-3) + Omega_Lambda)
    return np.sqrt(H_squared)


def t_of_a(a, H0, Omega_m0, Omega_r0, Omega_Lambda, a_min=1e-10):
    """
    Compute cosmic time t as a function of scale factor a.

    Integrates dt/da = 1 / (a * H(a)) from a_min to a.

    Parameters
    ----------
    a : float
        Scale factor
    H0, Omega_m0, Omega_r0, Omega_Lambda : float
        Cosmological parameters
    a_min : float
        Lower integration limit (near Big Bang)

    Returns
    -------
    t : float
        Cosmic time [s]
    """
    def integrand(a_prime):
        H = H_of_a(a_prime, H0, Omega_m0, Omega_r0, Omega_Lambda)
        return 1.0 / (a_prime * H)

    result, _ = integrate.quad(integrand, a_min, a, limit=200)
    return result


def a_of_t(t, H0, Omega_m0, Omega_r0, Omega_Lambda, a_min=1e-10, a_max_init=100.0):
    """
    Invert t(a) to get a(t) using root finding.

    Finds a such that t_of_a(a) = t.

    Parameters
    ----------
    t : float
        Cosmic time [s]
    H0, Omega_m0, Omega_r0, Omega_Lambda : float
        Cosmological parameters
    a_min : float
        Lower bound for scale factor search
    a_max_init : float
        Initial upper bound (will be expanded if needed)

    Returns
    -------
    a : float
        Scale factor at time t
    """
    # We need to find a such that t_of_a(a) - t = 0
    def objective(a):
        return t_of_a(a, H0, Omega_m0, Omega_r0, Omega_Lambda, a_min=a_min) - t

    # Find appropriate bounds - expand a_max if needed
    a_max = a_max_init
    t_at_amin = t_of_a(a_min * 10, H0, Omega_m0, Omega_r0, Omega_Lambda, a_min=a_min)

    if t < t_at_amin:
        # Very early time - use a small a
        return a_min * 10

    # Expand a_max until t_of_a(a_max) > t
    for _ in range(20):  # Safety limit
        t_at_amax = t_of_a(a_max, H0, Omega_m0, Omega_r0, Omega_Lambda, a_min=a_min)
        if t_at_amax > t:
            break
        a_max *= 10.0
    else:
        # If we couldn't find a large enough a_max, return the largest
        return a_max

    # Use Brent's method for root finding
    a_solution = optimize.brentq(objective, a_min * 10, a_max)
    return a_solution


def redshift_of_t(t, H0, Omega_m0, Omega_r0, Omega_Lambda):
    """
    Compute redshift z(t) = 1/a(t) - 1.
    """
    a = a_of_t(t, H0, Omega_m0, Omega_r0, Omega_Lambda)
    return 1.0 / a - 1.0


# =============================================================================
# Radiation Density
# =============================================================================

def rho_critical(H0):
    """
    Critical density of the universe.

    ρ_crit = 3 H0² / (8 π G)

    Returns
    -------
    rho_crit : float
        Critical density [kg/m^3]
    """
    return 3.0 * H0**2 / (8.0 * np.pi * G)


def rho_radiation(t, H0, Omega_m0, Omega_r0, Omega_Lambda):
    """
    Radiation energy density at cosmic time t.

    ρ_r(t) = ρ_r0 * (1 + z(t))^4
           = Omega_r0 * ρ_crit * (1 + z)^4

    Parameters
    ----------
    t : float
        Cosmic time [s]

    Returns
    -------
    rho_r : float
        Radiation energy density [J/m^3 = kg/(m·s²)]
    """
    rho_crit = rho_critical(H0)
    rho_r0 = Omega_r0 * rho_crit * c**2  # Convert to energy density [J/m^3]

    a = a_of_t(t, H0, Omega_m0, Omega_r0, Omega_Lambda)
    z = 1.0 / a - 1.0

    return rho_r0 * (1.0 + z)**4


# =============================================================================
# Black Hole Flux Model
# =============================================================================

def schwarzschild_radius(M_bh):
    """
    Schwarzschild radius r_s = 2GM/c².

    Parameters
    ----------
    M_bh : float
        Black hole mass [kg]

    Returns
    -------
    r_s : float
        Schwarzschild radius [m]
    """
    return 2.0 * G * M_bh / c**2


def horizon_area(M_bh):
    """
    Horizon area A = 4π r_s².

    Parameters
    ----------
    M_bh : float
        Black hole mass [kg]

    Returns
    -------
    A : float
        Horizon area [m^2]
    """
    r_s = schwarzschild_radius(M_bh)
    return 4.0 * np.pi * r_s**2


def L_in(t, M_bh, H0, Omega_m0, Omega_r0, Omega_Lambda, C_flux=1.0):
    """
    Incoming radiation flux (power) onto a black hole horizon.

    L_in(t) = C_flux * ρ_r(t) * c * A

    This is a simplified model assuming:
    - Isotropic radiation field
    - Geometric cross-section capture
    - No relativistic corrections

    Parameters
    ----------
    t : float
        Cosmic time [s]
    M_bh : float
        Black hole mass [kg]
    C_flux : float
        Dimensionless flux coefficient (order unity)

    Returns
    -------
    L : float
        Power incident on horizon [W]
    """
    A = horizon_area(M_bh)
    rho_r = rho_radiation(t, H0, Omega_m0, Omega_r0, Omega_Lambda)

    return C_flux * rho_r * c * A


# =============================================================================
# Isolation Time Calculation
# =============================================================================

def isolation_time(M_bh, L_crit, H0, Omega_m0, Omega_r0, Omega_Lambda,
                   t_min_Gyr=0.5, t_max_Gyr=350.0, n_samples=700, t_array_s=None):
    """
    Find the isolation time when L_in drops below L_crit.

    Samples L_in(t) over a time range and finds the earliest time
    where L_in(t) < L_crit.

    Parameters
    ----------
    M_bh : float
        Black hole mass [kg]
    L_crit : float
        Critical flux threshold [W]
    H0, Omega_m0, Omega_r0, Omega_Lambda : float
        Cosmological parameters
    t_min_Gyr, t_max_Gyr : float
        Time range to search [Gyr]
    n_samples : int
        Number of time samples

    Returns
    -------
    t_iso : float or None
        Isolation time [s], or None if not found in range
    """
    t_min = t_min_Gyr * Gyr_to_s
    t_max = t_max_Gyr * Gyr_to_s

    # Use linear spacing for more uniform coverage at late times
    # This is important since isolation can happen at very late cosmic times
    if t_array_s is None:
        t_array = np.linspace(t_min, t_max, n_samples)
    else:
        t_array = t_array_s

    for t in t_array:
        try:
            L = L_in(t, M_bh, H0, Omega_m0, Omega_r0, Omega_Lambda)
            if L < L_crit:
                return t
        except Exception:
            continue

    return None


# =============================================================================
# Wrapper Function for Module Use
# =============================================================================

def compute_flux_history_and_isolation(Omega_Lambda, M_bh, L_crit,
                                        t_min_Gyr=0.5, t_max_Gyr=200.0, n_t=500,
                                        H0_km_s_Mpc=70.0, Omega_m0=0.3, Omega_r0=9e-5):
    """
    Compute flux history L_in(t) and isolation time for a given Omega_Lambda.

    This is the main wrapper function for importing this module.

    Parameters
    ----------
    Omega_Lambda : float
        Cosmological constant density parameter
    M_bh : float
        Black hole mass [kg]
    L_crit : float
        Critical flux threshold [W]
    t_min_Gyr : float
        Minimum time [Gyr]
    t_max_Gyr : float
        Maximum time [Gyr]
    n_t : int
        Number of time points
    H0_km_s_Mpc : float
        Hubble constant [km/s/Mpc]
    Omega_m0 : float
        Matter density parameter
    Omega_r0 : float
        Radiation density parameter

    Returns
    -------
    t_array_Gyr : ndarray
        Time array [Gyr]
    L_in_array_W : ndarray
        Incoming flux array [W]
    t_iso_Gyr : float or None
        Isolation time [Gyr], or None if not found in range
    """
    # Convert H0 to SI units
    H0 = H0_km_s_Mpc * 1000.0 / Mpc_to_m  # [1/s]

    # Build time grid
    t_Gyr_array = np.linspace(t_min_Gyr, t_max_Gyr, n_t)
    t_s_array = t_Gyr_array * Gyr_to_s

    # Compute L_in(t)
    L_in_array = []
    for t in t_s_array:
        try:
            L = L_in(t, M_bh, H0, Omega_m0, Omega_r0, Omega_Lambda)
            L_in_array.append(L)
        except Exception:
            L_in_array.append(np.nan)

    L_in_array = np.array(L_in_array)

    # Find isolation time
    t_iso_s = isolation_time(M_bh, L_crit, H0, Omega_m0, Omega_r0, Omega_Lambda,
                              t_min_Gyr=t_min_Gyr, t_max_Gyr=t_max_Gyr,
                              n_samples=n_t * 2, t_array_s=None)

    if t_iso_s is not None:
        t_iso_Gyr = t_iso_s / Gyr_to_s
    else:
        t_iso_Gyr = None

    return t_Gyr_array, L_in_array, t_iso_Gyr


def get_default_L_crit(M_bh, Omega_Lambda_ref=0.7, t_ref_Gyr=0.5,
                       H0_km_s_Mpc=70.0, Omega_m0=0.3, Omega_r0=9e-5,
                       factor=1e-15):
    """
    Compute default L_crit as a fraction of the flux at a reference time.

    Parameters
    ----------
    M_bh : float
        Black hole mass [kg]
    Omega_Lambda_ref : float
        Reference Omega_Lambda for computing L_ref
    t_ref_Gyr : float
        Reference time [Gyr]
    H0_km_s_Mpc : float
        Hubble constant [km/s/Mpc]
    Omega_m0 : float
        Matter density parameter
    Omega_r0 : float
        Radiation density parameter
    factor : float
        Factor to multiply L_ref by (default 1e-15)

    Returns
    -------
    L_crit : float
        Critical flux threshold [W]
    L_ref : float
        Reference flux at t_ref [W]
    """
    H0 = H0_km_s_Mpc * 1000.0 / Mpc_to_m
    t_ref_s = t_ref_Gyr * Gyr_to_s
    L_ref = L_in(t_ref_s, M_bh, H0, Omega_m0, Omega_r0, Omega_Lambda_ref)
    L_crit = factor * L_ref
    return L_crit, L_ref


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Toy ΛCDM Cosmology + Black Hole Flux Model")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Cosmological Parameters
    # -------------------------------------------------------------------------
    # Hubble constant: H0 = 70 km/s/Mpc converted to 1/s
    H0_km_s_Mpc = 70.0
    H0 = H0_km_s_Mpc * 1000.0 / Mpc_to_m  # [1/s]

    # Density parameters (for flat universe with variable Omega_Lambda)
    Omega_m0 = 0.3      # Matter density
    Omega_r0 = 9e-5     # Radiation density (approximate)

    # Values of Omega_Lambda to scan
    Omega_Lambda_values = [0.1, 0.3, 0.7, 0.9]

    # -------------------------------------------------------------------------
    # Black Hole Parameters
    # -------------------------------------------------------------------------
    M_bh = 1e8 * M_sun  # 10^8 solar masses (supermassive BH)

    r_s = schwarzschild_radius(M_bh)
    A_horizon = horizon_area(M_bh)

    print(f"\nBlack Hole Parameters:")
    print(f"  Mass: {M_bh:.2e} kg ({M_bh/M_sun:.2e} M_sun)")
    print(f"  Schwarzschild radius: {r_s:.2e} m ({r_s/1e9:.2f} million km)")
    print(f"  Horizon area: {A_horizon:.2e} m²")

    # -------------------------------------------------------------------------
    # Critical Flux Threshold
    # -------------------------------------------------------------------------
    # We define L_crit as a very small fraction of the flux at t = 0.5 Gyr.
    # This represents the point where the ambient radiation becomes
    # negligible compared to, e.g., Hawking radiation or instrumental noise.
    #
    # Choice: L_crit = 1e-15 * L_in(t=0.5 Gyr)
    # This represents when flux drops to a tiny fraction of early universe values.
    # We use 1e-15 to ensure we find isolation times within ~100-200 Gyr.
    # A factor of 10^-30 would require waiting essentially forever.

    t_ref = 0.5 * Gyr_to_s
    # Use a reference Omega_Lambda to set L_crit
    Omega_Lambda_ref = 0.7
    L_ref = L_in(t_ref, M_bh, H0, Omega_m0, Omega_r0, Omega_Lambda_ref)
    L_crit = 1e-15 * L_ref

    print(f"\nFlux Parameters:")
    print(f"  Reference flux at t=0.5 Gyr (Ω_Λ=0.7): {L_ref:.2e} W")
    print(f"  Critical threshold L_crit: {L_crit:.2e} W")
    print(f"  (L_crit = 10^-15 × L_ref)")

    # -------------------------------------------------------------------------
    # Time Grid for Plotting
    # -------------------------------------------------------------------------
    t_min_Gyr = 0.5
    t_max_Gyr = 150.0
    n_points = 100

    t_Gyr_array = np.logspace(np.log10(t_min_Gyr), np.log10(t_max_Gyr), n_points)
    t_s_array = t_Gyr_array * Gyr_to_s

    # -------------------------------------------------------------------------
    # Compute L_in(t) and Isolation Times
    # -------------------------------------------------------------------------
    print("\nComputing fluxes and isolation times...")

    results = {}
    isolation_times = {}

    for Omega_Lambda in Omega_Lambda_values:
        print(f"  Ω_Λ = {Omega_Lambda}...", end=" ", flush=True)

        # Compute L_in(t) for this Omega_Lambda
        L_array = []
        for t in t_s_array:
            try:
                L = L_in(t, M_bh, H0, Omega_m0, Omega_r0, Omega_Lambda)
                L_array.append(L)
            except Exception as e:
                L_array.append(np.nan)

        results[Omega_Lambda] = np.array(L_array)

        # Find isolation time
        t_iso = isolation_time(M_bh, L_crit, H0, Omega_m0, Omega_r0, Omega_Lambda)

        if t_iso is not None:
            isolation_times[Omega_Lambda] = t_iso / Gyr_to_s  # Convert to Gyr
            print(f"t_iso = {isolation_times[Omega_Lambda]:.2f} Gyr")
        else:
            isolation_times[Omega_Lambda] = None
            print("t_iso = Not found in range")

    # -------------------------------------------------------------------------
    # Plot 1: L_in(t) vs t for multiple Omega_Lambda
    # -------------------------------------------------------------------------
    print("\nGenerating plots...")

    fig1, ax1 = plt.subplots(figsize=(10, 7))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, Omega_Lambda in enumerate(Omega_Lambda_values):
        label = f'$\\Omega_\\Lambda = {Omega_Lambda}$'
        ax1.semilogy(t_Gyr_array, results[Omega_Lambda],
                     color=colors[i], linewidth=2, label=label)

    # Mark isolation times
    for i, Omega_Lambda in enumerate(Omega_Lambda_values):
        t_iso = isolation_times[Omega_Lambda]
        if t_iso is not None and t_iso < t_max_Gyr:
            ax1.axvline(t_iso, color=colors[i], linestyle='--', alpha=0.5)

    # Mark L_crit threshold
    ax1.axhline(L_crit, color='gray', linestyle=':', linewidth=2,
                label=f'$L_{{crit}} = 10^{{-15}} L_{{ref}}$')

    ax1.set_xlabel('Cosmic Time $t$ [Gyr]', fontsize=12)
    ax1.set_ylabel('Incoming Flux $L_{in}$ [W]', fontsize=12)
    ax1.set_title('Toy Model: Radiation Flux onto Black Hole Horizon\n'
                  f'($M_{{BH}} = 10^8 M_\\odot$, flat ΛCDM)', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(t_min_Gyr, t_max_Gyr)

    plt.tight_layout()
    plt.savefig('flux_vs_time.png', dpi=150, bbox_inches='tight')
    print("  Saved: flux_vs_time.png")

    # -------------------------------------------------------------------------
    # Plot 2: Isolation Time vs Omega_Lambda
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Prepare data for plotting
    OL_plot = []
    t_iso_plot = []

    for Omega_Lambda in Omega_Lambda_values:
        t_iso = isolation_times[Omega_Lambda]
        if t_iso is not None:
            OL_plot.append(Omega_Lambda)
            t_iso_plot.append(t_iso)

    if len(OL_plot) > 0:
        ax2.scatter(OL_plot, t_iso_plot, s=100, c='blue', marker='o',
                   edgecolors='black', linewidths=1.5, zorder=5)
        ax2.plot(OL_plot, t_iso_plot, 'b-', linewidth=2, alpha=0.7)

    ax2.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=12)
    ax2.set_ylabel('Isolation Time $t_{iso}$ [Gyr]', fontsize=12)
    ax2.set_title('Toy Model: Black Hole Isolation Time vs $\\Omega_\\Lambda$\n'
                  f'($L_{{crit}} = 10^{{-15}} L_{{ref}}$)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    # Add annotation
    ax2.text(0.05, 0.95,
             'Higher $\\Omega_\\Lambda$ → faster cosmic expansion\n'
             '→ faster radiation dilution → earlier isolation',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('isolation_time_vs_omega_lambda.png', dpi=150, bbox_inches='tight')
    print("  Saved: isolation_time_vs_omega_lambda.png")

    # -------------------------------------------------------------------------
    # Summary Output
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary of Results")
    print("=" * 70)
    print(f"\n{'Ω_Λ':<10} {'t_iso [Gyr]':<15} {'L_in(t=0.5 Gyr) [W]':<20}")
    print("-" * 45)

    for Omega_Lambda in Omega_Lambda_values:
        t_iso = isolation_times[Omega_Lambda]
        L_early = results[Omega_Lambda][0]  # At t = 0.5 Gyr

        t_iso_str = f"{t_iso:.2f}" if t_iso is not None else "N/A"
        print(f"{Omega_Lambda:<10} {t_iso_str:<15} {L_early:.2e}")

    print("\n" + "=" * 70)
    print("Physical Interpretation (Toy Model)")
    print("=" * 70)
    print("""
This simplified model shows how the cosmological constant affects
the "isolation" of a black hole from the ambient radiation bath:

1. Higher Ω_Λ leads to faster cosmic acceleration
2. Faster expansion means the radiation density ρ_r ∝ (1+z)^4 drops faster
3. The flux L_in onto the black hole decreases more rapidly
4. The black hole becomes "isolated" (L_in < L_crit) earlier

Caveats:
- This is a toy model ignoring many physical effects
- Real black holes have complex accretion physics
- The flux model assumes geometric cross-section only
- Hawking radiation is not included
- The critical threshold L_crit is arbitrary
""")

    # Show plots
    plt.show()

    print("\nDone!")
