#!/usr/bin/env python3
"""
fertility.py

Compute the fertility functional F(Omega_Lambda) = N_BH(Omega_Lambda) * T_int(Omega_Lambda).

This is the central quantity in cosmological natural selection (CNS) theory,
representing the "reproductive fitness" of a universe with a given cosmological
constant.

Physical interpretation:
- N_BH: Number of black holes formed (offspring count)
- T_int: Interior lifetime available for baby universe formation
- F = N_BH * T_int: Total "fertility" or reproductive potential

The CNS hypothesis predicts that the observed cosmological constant should
be near the peak of F(Omega_Lambda).

This module combines:
- N_BH from Press-Schechter structure formation (black_holes/nbh_vs_lambda.py)
- T_int from mass-inflation flux integral (black_holes/interior_lifetime.py)
"""

import numpy as np

from black_holes.nbh_vs_lambda import compute_N_BH
from black_holes.interior_lifetime import compute_T_int
from config import OMEGA_M0, OMEGA_R0, SIGMA_8, N_S, M_SUN


# =============================================================================
# Fertility Functional
# =============================================================================

def compute_fertility(Omega_Lambda, M_bh=1e8 * M_SUN, M_halo_min=1e10,
                       z_form=2.0, kappa=0.5,
                       sigma8=SIGMA_8, n_s=N_S,
                       normalize=True, Omega_Lambda_ref=0.7,
                       return_components=False):
    """
    Compute the fertility functional F(Omega_Lambda) = N_BH * T_int.

    Parameters
    ----------
    Omega_Lambda : float or ndarray
        Dark energy density parameter
    M_bh : float
        Black hole mass for interior lifetime calculation [kg]
    M_halo_min : float
        Minimum halo mass for BH formation [M_sun]
    z_form : float
        Characteristic redshift for BH formation
    kappa : float
        Inner horizon surface gravity for mass inflation
    sigma8 : float
        Power spectrum normalization
    n_s : float
        Spectral index
    normalize : bool
        If True, normalize F to max = 1
    Omega_Lambda_ref : float
        Reference Omega_Lambda for component normalization
    return_components : bool
        If True, also return N_BH and T_int arrays

    Returns
    -------
    F : float or ndarray
        Fertility functional (normalized if requested)
    N_BH : float or ndarray (optional)
        Black hole abundance (if return_components=True)
    T_int : float or ndarray (optional)
        Interior lifetime (if return_components=True)
    """
    # Compute N_BH(Omega_Lambda)
    N_BH = compute_N_BH(
        Omega_Lambda,
        M_halo_min=M_halo_min,
        z_form=z_form,
        sigma8=sigma8,
        n_s=n_s,
        normalize=True,
        Omega_Lambda_ref=Omega_Lambda_ref
    )

    # Compute T_int(Omega_Lambda)
    T_int = compute_T_int(
        Omega_Lambda,
        M_bh=M_bh,
        kappa=kappa,
        normalize=True,
        Omega_Lambda_ref=Omega_Lambda_ref
    )

    # Fertility functional
    F = N_BH * T_int

    # Normalize to max = 1 if requested
    if normalize:
        F_max = np.max(F) if hasattr(F, '__len__') else F
        if F_max > 0:
            F = F / F_max

    if return_components:
        return F, N_BH, T_int
    return F


def find_fertility_peak(Omega_Lambda_min=0.05, Omega_Lambda_max=0.95,
                         n_points=50, **kwargs):
    """
    Find the Omega_Lambda that maximizes the fertility functional.

    Parameters
    ----------
    Omega_Lambda_min, Omega_Lambda_max : float
        Search range for Omega_Lambda
    n_points : int
        Number of points for grid search
    **kwargs : dict
        Additional arguments passed to compute_fertility

    Returns
    -------
    Omega_peak : float
        Omega_Lambda at the peak
    F_peak : float
        Maximum fertility value
    omega_array : ndarray
        Omega_Lambda values scanned
    F_array : ndarray
        Fertility values
    """
    omega_array = np.linspace(Omega_Lambda_min, Omega_Lambda_max, n_points)

    F_array = compute_fertility(omega_array, normalize=False, **kwargs)

    # Find peak
    idx_peak = np.argmax(F_array)
    Omega_peak = omega_array[idx_peak]
    F_peak = F_array[idx_peak]

    # Normalize
    if F_peak > 0:
        F_array = F_array / F_peak

    return Omega_peak, F_peak, omega_array, F_array


def fertility_ratio(Omega_Lambda, Omega_Lambda_ref=0.7, **kwargs):
    """
    Compute F(Omega_Lambda) / F(Omega_Lambda_ref).

    Useful for comparing the fertility of different cosmologies.

    Parameters
    ----------
    Omega_Lambda : float or ndarray
        Dark energy parameter to evaluate
    Omega_Lambda_ref : float
        Reference cosmology (default: our universe)
    **kwargs : dict
        Additional arguments passed to compute_fertility

    Returns
    -------
    ratio : float or ndarray
        F(Omega_Lambda) / F(Omega_Lambda_ref)
    """
    F = compute_fertility(Omega_Lambda, normalize=False, **kwargs)
    F_ref = compute_fertility(Omega_Lambda_ref, normalize=False, **kwargs)

    if np.isscalar(F_ref) and F_ref > 0:
        return F / F_ref
    return F


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_fertility(omega_array=None, verbose=True, **kwargs):
    """
    Perform a comprehensive analysis of the fertility functional.

    Parameters
    ----------
    omega_array : ndarray, optional
        Omega_Lambda values to scan. Default: linspace(0.05, 0.95, 30)
    verbose : bool
        If True, print results
    **kwargs : dict
        Additional arguments for compute_fertility

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'omega': Omega_Lambda array
        - 'F': Fertility array (normalized)
        - 'N_BH': Black hole abundance array
        - 'T_int': Interior lifetime array
        - 'Omega_peak': Peak location
        - 'F_at_0.7': Fertility at our universe's value
        - 'ratio_to_peak': F(0.7) / F_peak
    """
    if omega_array is None:
        omega_array = np.linspace(0.05, 0.95, 30)

    # Compute components
    F, N_BH, T_int = compute_fertility(
        omega_array, normalize=False, return_components=True, **kwargs
    )

    # Normalize each component
    N_BH_norm = N_BH / np.max(N_BH) if np.max(N_BH) > 0 else N_BH
    T_int_norm = T_int / np.max(T_int) if np.max(T_int) > 0 else T_int
    F_norm = F / np.max(F) if np.max(F) > 0 else F

    # Find peak
    idx_peak = np.argmax(F)
    Omega_peak = omega_array[idx_peak]

    # Value at our universe
    idx_07 = np.argmin(np.abs(omega_array - 0.7))
    F_at_07 = F_norm[idx_07]

    results = {
        'omega': omega_array,
        'F': F_norm,
        'N_BH': N_BH_norm,
        'T_int': T_int_norm,
        'Omega_peak': Omega_peak,
        'F_at_0.7': F_at_07,
        'ratio_to_peak': F_at_07 / F_norm[idx_peak] if F_norm[idx_peak] > 0 else 0
    }

    if verbose:
        print("=" * 60)
        print("Fertility Functional Analysis")
        print("=" * 60)
        print(f"\nPeak location: Omega_Lambda = {Omega_peak:.3f}")
        print(f"Our universe (Omega_Lambda = 0.7): F = {F_at_07:.3f}")
        print(f"Ratio F(0.7) / F_peak = {results['ratio_to_peak']:.1%}")
        print()
        print("Sample values:")
        print("-" * 50)
        print(f"{'Omega_Lambda':>12} | {'N_BH':>10} | {'T_int':>10} | {'F':>10}")
        print("-" * 50)
        for i in range(0, len(omega_array), max(1, len(omega_array)//10)):
            print(f"{omega_array[i]:>12.2f} | {N_BH_norm[i]:>10.3f} | "
                  f"{T_int_norm[i]:>10.3f} | {F_norm[i]:>10.3f}")
        print()

    return results


# =============================================================================
# Main: Run Analysis
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("v2 Fertility Functional: F(Omega_Lambda) = N_BH * T_int")
    print("=" * 70)
    print()

    # Run analysis
    results = analyze_fertility(verbose=True)

    # Create plot
    try:
        import matplotlib.pyplot as plt
        import os

        omega = results['omega']
        F = results['F']
        N_BH = results['N_BH']
        T_int = results['T_int']
        Omega_peak = results['Omega_peak']

        # Create figure
        fig = plt.figure(figsize=(16, 10))

        # Color scheme
        color_F = '#E63946'
        color_N = '#457B9D'
        color_T = '#2A9D8F'
        color_obs = '#F4A261'

        # Main plot: Fertility
        ax1 = fig.add_subplot(2, 2, (1, 2))

        ax1.plot(omega, F, 'o-', color=color_F, markersize=8, linewidth=2.5,
                 label='$F(\\Omega_\\Lambda) = N_{BH} \\times T_{int}$')
        ax1.fill_between(omega, 0, F, alpha=0.2, color=color_F)

        # Mark peak
        idx_peak = np.argmax(F)
        ax1.axvline(Omega_peak, color=color_F, linestyle='--', alpha=0.7)
        ax1.plot(Omega_peak, F[idx_peak], 'o', markersize=15, color=color_F,
                 markeredgecolor='darkred', markeredgewidth=2)
        ax1.annotate(f'Peak\n$\\Omega_\\Lambda = {Omega_peak:.2f}$',
                     xy=(Omega_peak, F[idx_peak]),
                     xytext=(Omega_peak - 0.15, F[idx_peak] + 0.1),
                     fontsize=11, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='darkred'))

        # Mark our universe
        ax1.axvline(0.7, color=color_obs, linestyle=':', linewidth=2.5)
        idx_07 = np.argmin(np.abs(omega - 0.7))
        ax1.plot(0.7, F[idx_07], 's', markersize=12, color=color_obs,
                 markeredgecolor='darkorange', markeredgewidth=2)
        ax1.annotate(f'Our Universe\n$\\Omega_\\Lambda = 0.7$\n$F = {F[idx_07]:.2f}$',
                     xy=(0.7, F[idx_07]),
                     xytext=(0.75, F[idx_07] + 0.15),
                     fontsize=10, color='darkorange',
                     arrowprops=dict(arrowstyle='->', color='darkorange'))

        ax1.set_xlabel('$\\Omega_\\Lambda$', fontsize=14)
        ax1.set_ylabel('Fertility $F(\\Omega_\\Lambda)$ (normalized)', fontsize=14)
        ax1.set_title('v2 Model: Fertility Functional from Press-Schechter + Mass Inflation',
                      fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1.15)

        # Subplot: N_BH
        ax2 = fig.add_subplot(2, 2, 3)
        ax2.plot(omega, N_BH, 'o-', color=color_N, markersize=6, linewidth=2)
        ax2.fill_between(omega, 0, N_BH, alpha=0.15, color=color_N)
        ax2.axvline(0.7, color=color_obs, linestyle=':', linewidth=2)
        ax2.set_xlabel('$\\Omega_\\Lambda$', fontsize=12)
        ax2.set_ylabel('$N_{BH}$ (normalized)', fontsize=12)
        ax2.set_title('Black Hole Abundance\n(Press-Schechter)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)

        # Subplot: T_int
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.plot(omega, T_int, 'o-', color=color_T, markersize=6, linewidth=2)
        ax3.fill_between(omega, 0, T_int, alpha=0.15, color=color_T)
        ax3.axvline(0.7, color=color_obs, linestyle=':', linewidth=2)
        ax3.set_xlabel('$\\Omega_\\Lambda$', fontsize=12)
        ax3.set_ylabel('$T_{int}$ (normalized)', fontsize=12)
        ax3.set_title('Interior Lifetime\n(Mass Inflation)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)

        plt.tight_layout()

        save_path = os.path.join(os.path.dirname(__file__), 'F_vs_OmegaLambda_v2.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {save_path}")

        plt.show()

    except ImportError:
        print("matplotlib not available for plotting")

    # Summary
    print()
    print("=" * 70)
    print("Physical Interpretation")
    print("=" * 70)
    print(f"""
The v2 fertility functional combines:

1. N_BH(Omega_Lambda) from Press-Schechter theory:
   - Uses linear growth factor D(a) and halo mass function
   - Counts halos above M > 10^10 M_sun at z ~ 2
   - Peaks at low Omega_Lambda (more structure formation)

2. T_int(Omega_Lambda) from mass-inflation model:
   - Uses cosmological radiation flux history
   - Higher Omega_Lambda -> lower flux -> longer interior lifetime
   - Increases monotonically with Omega_Lambda

3. F = N_BH * T_int balances these competing effects:
   - Peak at Omega_Lambda ~ {Omega_peak:.2f}
   - Our universe at Omega_Lambda = 0.7 has F = {results['F_at_0.7']:.2f}
   - Ratio to peak: {results['ratio_to_peak']:.1%}

INTERPRETATION:
Our universe sits {'to the right of' if Omega_peak < 0.7 else 'to the left of' if Omega_peak > 0.7 else 'at'} the fertility peak.
This is {'consistent with' if results['ratio_to_peak'] > 0.5 else 'somewhat inconsistent with'} CNS if we expect
observed values to be within ~50% of the peak.

CAVEATS:
- Press-Schechter is approximate; Sheth-Tormen improves fit to N-body
- Mass inflation model is simplified; real GR is more complex
- BH occupation fraction, mergers, etc. not included
- The "baby universe" mechanism is speculative
""")

    print("Done!")
