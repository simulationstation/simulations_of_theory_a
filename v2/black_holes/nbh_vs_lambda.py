#!/usr/bin/env python3
"""
nbh_vs_lambda.py

Compute the relative black hole abundance N_BH(Omega_Lambda) using
Press-Schechter structure formation theory.

Physical picture:
- Black holes form in galaxies which form in dark matter halos
- The number of halos above a threshold mass depends on cosmology
- Higher Omega_Lambda leads to earlier acceleration and suppressed structure
- Lower Omega_Lambda allows more structure but may have other effects

This module provides N_BH(Omega_Lambda) by integrating the halo mass function
over halos massive enough to host black holes (M > M_BH_threshold).

We assume a simple proportionality: N_BH ~ N_halo(M > M_threshold)
A more refined model could include BH occupation fractions, merger histories, etc.
"""

import numpy as np
from scipy import integrate

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmology.structure_formation import (
    growth_factor_normalized,
    critical_overdensity,
    sigma_M,
)
from cosmology.halo_mass_function import (
    integrated_halo_number,
    total_collapsed_fraction,
    press_schechter_mass_function,
)
from config import OMEGA_M0, OMEGA_B0, SIGMA_8, N_S, H0_SI, MPC_TO_M


# =============================================================================
# Black Hole Abundance from Structure Formation
# =============================================================================

def compute_N_BH(Omega_Lambda, M_halo_min=1e10, z_form=2.0,
                 sigma8=SIGMA_8, n_s=N_S, Omega_m0=None,
                 Omega_b0=OMEGA_B0, h=0.7, multiplicity='ST',
                 normalize=True, N_BH_ref=None, Omega_Lambda_ref=0.7):
    """
    Compute relative black hole abundance N_BH(Omega_Lambda).

    Uses Press-Schechter theory to estimate the number of halos above
    a threshold mass at a characteristic formation redshift.

    N_BH ~ n(>M_halo_min, z=z_form)

    Parameters
    ----------
    Omega_Lambda : float or ndarray
        Dark energy density parameter
    M_halo_min : float
        Minimum halo mass to host a significant BH [M_sun]
        Default 10^10 M_sun (roughly MW-sized halo)
    z_form : float
        Characteristic redshift for BH formation
        Default z=2 (peak of cosmic star formation)
    sigma8 : float
        Power spectrum normalization at z=0
    n_s : float
        Spectral index
    Omega_m0 : float, optional
        Matter density. If None, computed from flatness: 1 - Omega_Lambda
    Omega_b0 : float
        Baryon density parameter
    h : float
        Hubble parameter h = H0/(100 km/s/Mpc)
    multiplicity : str
        'PS' or 'ST' for mass function
    normalize : bool
        If True, normalize to N_BH(Omega_Lambda_ref) = 1
    N_BH_ref : float, optional
        Reference value for normalization (computed if None)
    Omega_Lambda_ref : float
        Reference Omega_Lambda for normalization

    Returns
    -------
    N_BH : float or ndarray
        Black hole abundance (relative if normalized)
    """
    # Handle scalar/array
    scalar_input = np.isscalar(Omega_Lambda)
    OL_array = np.atleast_1d(Omega_Lambda)

    N_BH_values = np.zeros_like(OL_array, dtype=float)

    for i, OL in enumerate(OL_array):
        # For flat universe: Omega_m = 1 - Omega_Lambda
        if Omega_m0 is None:
            Om = 1.0 - OL
        else:
            Om = Omega_m0

        # Check for valid cosmology
        if Om <= 0 or Om > 1:
            N_BH_values[i] = 0.0
            continue

        # Integrated halo number density at z_form
        try:
            n_halo = integrated_halo_number(
                M_halo_min, z=z_form, M_max=1e16,
                sigma8=sigma8, n_s=n_s,
                Omega_m0=Om, Omega_b0=Omega_b0,
                Omega_Lambda=OL, h=h,
                multiplicity=multiplicity
            )
            N_BH_values[i] = n_halo
        except Exception:
            N_BH_values[i] = 0.0

    # Normalize if requested
    if normalize:
        if N_BH_ref is None:
            # Compute reference value
            if Omega_m0 is None:
                Om_ref = 1.0 - Omega_Lambda_ref
            else:
                Om_ref = Omega_m0

            N_BH_ref = integrated_halo_number(
                M_halo_min, z=z_form, M_max=1e16,
                sigma8=sigma8, n_s=n_s,
                Omega_m0=Om_ref, Omega_b0=Omega_b0,
                Omega_Lambda=Omega_Lambda_ref, h=h,
                multiplicity=multiplicity
            )

        if N_BH_ref > 0:
            N_BH_values /= N_BH_ref

    if scalar_input:
        return N_BH_values[0]
    return N_BH_values


def compute_N_BH_integrated(Omega_Lambda, M_halo_min=1e10, z_max=10.0,
                             sigma8=SIGMA_8, n_s=N_S,
                             Omega_b0=OMEGA_B0, h=0.7,
                             multiplicity='ST', n_z_points=20):
    """
    Compute N_BH by integrating over cosmic history.

    N_BH ~ integral_0^{z_max} n(>M, z) * |dt/dz| dz

    This accounts for BH formation over cosmic time rather than
    at a single snapshot.

    Parameters
    ----------
    Omega_Lambda : float
        Dark energy parameter
    M_halo_min : float
        Minimum halo mass [M_sun]
    z_max : float
        Maximum redshift for integration
    n_z_points : int
        Number of redshift points

    Returns
    -------
    N_BH_int : float
        Time-integrated BH abundance proxy
    """
    Omega_m0 = 1.0 - Omega_Lambda

    if Omega_m0 <= 0:
        return 0.0

    # Redshift array
    z_array = np.linspace(0, z_max, n_z_points)

    # |dt/dz| in units where we can compare
    # dt/dz = -1 / [(1+z) * H(z)]
    # H(z) = H0 * E(z) where E(z) = sqrt(Omega_m*(1+z)^3 + Omega_Lambda)

    def E_of_z(z):
        return np.sqrt(Omega_m0 * (1+z)**3 + Omega_Lambda)

    # Compute integrand at each z
    integrand = np.zeros_like(z_array)
    for j, z in enumerate(z_array):
        try:
            n_halo = integrated_halo_number(
                M_halo_min, z=z, M_max=1e16,
                sigma8=sigma8, n_s=n_s,
                Omega_m0=Omega_m0, Omega_b0=Omega_b0,
                Omega_Lambda=Omega_Lambda, h=h,
                multiplicity=multiplicity, n_points=30
            )
            dt_dz = 1.0 / ((1 + z) * E_of_z(z))  # In H0^{-1} units
            integrand[j] = n_halo * dt_dz
        except Exception:
            integrand[j] = 0.0

    # Trapezoidal integration
    N_BH_int = np.trapz(integrand, z_array)

    return N_BH_int


def N_BH_suppression_factor(Omega_Lambda, Omega_Lambda_crit=0.9, steepness=10.0):
    """
    Additional suppression factor for very high Omega_Lambda.

    At very high Omega_Lambda, structure formation is severely suppressed
    and the universe may not even form galaxies. This factor provides
    additional suppression beyond what Press-Schechter captures.

    Parameters
    ----------
    Omega_Lambda : float or ndarray
        Dark energy parameter
    Omega_Lambda_crit : float
        Critical value above which suppression kicks in
    steepness : float
        How sharply the suppression occurs

    Returns
    -------
    factor : float or ndarray
        Suppression factor (0 to 1)
    """
    x = (Omega_Lambda - Omega_Lambda_crit) * steepness
    return 1.0 / (1.0 + np.exp(x))


# =============================================================================
# Main: Test and Demonstrate
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Black Hole Abundance N_BH(Omega_Lambda) - v2 Model")
    print("=" * 70)
    print()

    # Omega_Lambda values to scan
    OL_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    print("Computing N_BH(Omega_Lambda) from Press-Schechter...")
    print(f"  M_halo_min = 10^10 M_sun")
    print(f"  z_form = 2.0")
    print(f"  Using Sheth-Tormen multiplicity function")
    print()

    # Compute N_BH for each Omega_Lambda
    N_BH_values = compute_N_BH(OL_values, M_halo_min=1e10, z_form=2.0,
                                normalize=True, Omega_Lambda_ref=0.7)

    print("-" * 50)
    print(f"{'Omega_Lambda':>12} | {'Omega_m':>10} | {'N_BH (norm)':>12}")
    print("-" * 50)
    for OL, N in zip(OL_values, N_BH_values):
        Om = 1.0 - OL
        print(f"{OL:>12.2f} | {Om:>10.2f} | {N:>12.4f}")

    # Find peak
    idx_max = np.argmax(N_BH_values)
    print("-" * 50)
    print(f"Peak at Omega_Lambda = {OL_values[idx_max]:.2f}, "
          f"N_BH = {N_BH_values[idx_max]:.4f}")
    print()

    # Compare with our universe
    print(f"Our universe (Omega_Lambda ~ 0.7): N_BH = {N_BH_values[OL_values == 0.7][0]:.4f}")
    print()

    print("Physical interpretation:")
    print("-" * 50)
    print("""
- Low Omega_Lambda (high Omega_m): More matter -> more structure
  But if Omega_Lambda is too low, late-time structure can recollapse
  or the universe evolves differently.

- High Omega_Lambda: Accelerated expansion freezes out structure
  formation early, suppressing halo (and thus BH) formation.

- Intermediate Omega_Lambda: "Goldilocks zone" for structure.

The peak depends on M_halo_min and z_form assumptions.
""")

    # Optional: plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 7))

        ax.plot(OL_values, N_BH_values, 'bo-', markersize=10, linewidth=2,
                label='$N_{BH}(\\Omega_\\Lambda)$ from Press-Schechter')

        ax.axvline(0.7, color='green', linestyle='--', linewidth=2,
                   label='Our Universe ($\\Omega_\\Lambda \\approx 0.7$)')
        ax.axvline(OL_values[idx_max], color='red', linestyle=':', linewidth=2,
                   label=f'Peak ($\\Omega_\\Lambda = {OL_values[idx_max]:.2f}$)')

        ax.set_xlabel('$\\Omega_\\Lambda$', fontsize=14)
        ax.set_ylabel('$N_{BH}$ (normalized)', fontsize=14)
        ax.set_title('v2 Model: Black Hole Abundance from Structure Formation',
                     fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

        plt.tight_layout()
        save_path = os.path.join(os.path.dirname(__file__), 'N_BH_vs_OmegaLambda_v2.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.show()

    except ImportError:
        print("matplotlib not available for plotting")

    print("\nDone!")
