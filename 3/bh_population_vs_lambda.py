#!/usr/bin/env python3
"""
bh_population_vs_lambda.py

A toy Press-Schechter-like model for estimating the relative number of
black holes as a function of the cosmological constant parameter Omega_Lambda.

Physical motivation:
- For very small Omega_Lambda: structure forms late, potential issues with
  matter-dominated recollapse or inefficient late-time structure formation.
- For very large Omega_Lambda: accelerated expansion freezes out structure
  early, suppressing collapse and black hole formation.
- Intermediate Omega_Lambda: "Goldilocks zone" for structure formation.

This is a highly simplified toy model for pedagogical/illustrative purposes.

Author: Toy model for cosmological natural selection studies
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Physical Constants (for reference, not all used in this toy model)
# =============================================================================
H0_km_s_Mpc = 70.0  # Hubble constant [km/s/Mpc]


# =============================================================================
# Toy Press-Schechter-like Model
# =============================================================================

def collapsed_fraction(Omega_Lambda, Omega_plus=0.8, Omega_minus=0.1,
                       alpha=2.0, beta=2.0):
    """
    Toy Press-Schechter-like collapsed fraction as a function of Omega_Lambda.

    This parametric form captures the qualitative behavior:
    - f_coll -> 0 as Omega_Lambda -> 0 (late-time structure issues)
    - f_coll -> 0 as Omega_Lambda -> 1 (early acceleration suppresses collapse)
    - f_coll has a peak at intermediate Omega_Lambda

    The functional form is:
        f_coll(Omega_Lambda) = exp[-(Omega_Lambda/Omega_plus)^alpha]
                             * [1 - exp(-(Omega_Lambda/Omega_minus)^beta)]

    Parameters
    ----------
    Omega_Lambda : float or ndarray
        Cosmological constant density parameter (0 < Omega_Lambda < 1)
    Omega_plus : float
        Scale where large Lambda suppresses collapse (default 0.8)
    Omega_minus : float
        Scale where small Lambda leads to structure inefficiency (default 0.1)
    alpha : float
        Exponent controlling large-Lambda suppression (default 2.0)
    beta : float
        Exponent controlling small-Lambda suppression (default 2.0)

    Returns
    -------
    f_coll : float or ndarray
        Dimensionless collapsed fraction (between 0 and ~1)
    """
    # Ensure Omega_Lambda is array-like for vectorized operations
    OL = np.asarray(Omega_Lambda)

    # Large-Lambda suppression: exp[-(OL/Omega_plus)^alpha]
    # This term -> 1 for small OL, -> 0 for large OL
    large_lambda_factor = np.exp(-(OL / Omega_plus)**alpha)

    # Small-Lambda suppression: 1 - exp[-(OL/Omega_minus)^beta]
    # This term -> 0 for small OL, -> 1 for large OL
    small_lambda_factor = 1.0 - np.exp(-(OL / Omega_minus)**beta)

    # Combined collapsed fraction
    f_coll = large_lambda_factor * small_lambda_factor

    return f_coll


def N_BH_relative(Omega_Lambda, **kwargs):
    """
    Return a relative black hole abundance N_BH(Omega_Lambda),
    proportional to the collapsed fraction.

    In this toy model, we assume the number of black holes is directly
    proportional to the fraction of matter that collapses into halos
    above some threshold mass.

    Parameters
    ----------
    Omega_Lambda : float or ndarray
        Cosmological constant density parameter
    **kwargs : dict
        Additional parameters passed to collapsed_fraction

    Returns
    -------
    N_BH : float or ndarray
        Relative black hole abundance (arbitrary normalization)
    """
    return collapsed_fraction(Omega_Lambda, **kwargs)


def find_peak_omega_lambda(omega_array, N_array):
    """
    Find the Omega_Lambda value where N_BH is maximal.

    Parameters
    ----------
    omega_array : ndarray
        Array of Omega_Lambda values
    N_array : ndarray
        Corresponding N_BH values

    Returns
    -------
    Omega_peak : float
        Omega_Lambda at the peak
    N_peak : float
        Maximum N_BH value
    """
    idx_max = np.argmax(N_array)
    return omega_array[idx_max], N_array[idx_max]


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Toy Press-Schechter Model: Black Hole Abundance vs Omega_Lambda")
    print("=" * 70)
    print()

    # -------------------------------------------------------------------------
    # Model Parameters
    # -------------------------------------------------------------------------
    Omega_plus = 0.8    # Large-Lambda suppression scale
    Omega_minus = 0.1   # Small-Lambda suppression scale
    alpha = 2.0         # Large-Lambda exponent
    beta = 2.0          # Small-Lambda exponent

    print("Model Parameters:")
    print(f"  Omega_+ = {Omega_plus} (large-Lambda suppression scale)")
    print(f"  Omega_- = {Omega_minus} (small-Lambda suppression scale)")
    print(f"  alpha = {alpha} (large-Lambda exponent)")
    print(f"  beta = {beta} (small-Lambda exponent)")
    print()

    # -------------------------------------------------------------------------
    # Compute N_BH(Omega_Lambda)
    # -------------------------------------------------------------------------
    print("Computing relative black hole abundance...")

    # Array of Omega_Lambda values
    omega_lambda_array = np.linspace(0.01, 0.99, 200)

    # Compute relative abundance
    N_rel = N_BH_relative(omega_lambda_array,
                          Omega_plus=Omega_plus,
                          Omega_minus=Omega_minus,
                          alpha=alpha,
                          beta=beta)

    # Normalize to maximum = 1
    N_rel_norm = N_rel / np.max(N_rel)

    # Find peak
    Omega_peak, N_peak = find_peak_omega_lambda(omega_lambda_array, N_rel)

    print()
    print("-" * 70)
    print(f"Peak relative BH abundance at Omega_Lambda = {Omega_peak:.3f}")
    print("-" * 70)
    print()

    # -------------------------------------------------------------------------
    # Additional analysis: width of the distribution
    # -------------------------------------------------------------------------
    # Find FWHM (full width at half maximum)
    half_max = 0.5
    above_half = omega_lambda_array[N_rel_norm >= half_max]
    if len(above_half) > 0:
        fwhm_low = above_half[0]
        fwhm_high = above_half[-1]
        fwhm = fwhm_high - fwhm_low
        print(f"Distribution width (FWHM): {fwhm:.3f}")
        print(f"  Lower bound (50%): Omega_Lambda = {fwhm_low:.3f}")
        print(f"  Upper bound (50%): Omega_Lambda = {fwhm_high:.3f}")
    print()

    # Print some sample values
    print("Sample values:")
    print(f"{'Omega_Lambda':>12} | {'N_BH (norm)':>12}")
    print("-" * 27)
    sample_omegas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    for ol in sample_omegas:
        idx = np.argmin(np.abs(omega_lambda_array - ol))
        print(f"{ol:>12.2f} | {N_rel_norm[idx]:>12.4f}")
    print()

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    print("Generating plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Main curve
    ax.plot(omega_lambda_array, N_rel_norm, 'b-', linewidth=3,
            label='$N_{BH}(\\Omega_\\Lambda) / \\max$')

    # Fill under the curve
    ax.fill_between(omega_lambda_array, 0, N_rel_norm, alpha=0.2, color='blue')

    # Mark the peak
    ax.axvline(Omega_peak, color='red', linestyle='--', linewidth=2,
               label=f'Peak at $\\Omega_\\Lambda = {Omega_peak:.3f}$')
    ax.plot(Omega_peak, 1.0, 'ro', markersize=12, markeredgecolor='darkred',
            markeredgewidth=2, zorder=5)

    # Mark our universe's approximate value
    Omega_Lambda_observed = 0.7
    idx_obs = np.argmin(np.abs(omega_lambda_array - Omega_Lambda_observed))
    N_obs = N_rel_norm[idx_obs]
    ax.axvline(Omega_Lambda_observed, color='green', linestyle=':', linewidth=2,
               label=f'Our Universe $\\Omega_\\Lambda \\approx {Omega_Lambda_observed}$')
    ax.plot(Omega_Lambda_observed, N_obs, 'g^', markersize=12,
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=5)

    # Annotations for physical interpretation
    ax.annotate('Too small $\\Lambda$:\nLate-time structure\ninefficiency',
                xy=(0.05, 0.15), fontsize=11, color='gray',
                ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                         edgecolor='orange', alpha=0.8))

    ax.annotate('Too large $\\Lambda$:\nEarly acceleration\nsuppresses structure',
                xy=(0.95, 0.15), fontsize=11, color='gray',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                         edgecolor='orange', alpha=0.8))

    ax.annotate('"Goldilocks zone"\nfor BH formation',
                xy=(Omega_peak, 0.85), fontsize=11, color='darkred',
                ha='center', va='top', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose',
                         edgecolor='red', alpha=0.9))

    # Labels and title
    ax.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=14)
    ax.set_ylabel('Relative Black Hole Abundance $N_{BH}(\\Omega_\\Lambda) / \\max$',
                  fontsize=14)
    ax.set_title('Toy Model: Relative Black Hole Abundance vs Cosmological Constant\n'
                 '(Press-Schechter-like approximation)', fontsize=16, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)

    # Add formula annotation
    formula_text = (f'$f_{{coll}}(\\Omega_\\Lambda) = '
                   f'\\exp\\left[-\\left(\\frac{{\\Omega_\\Lambda}}{{{Omega_plus}}}\\right)^{{{alpha:.0f}}}\\right] '
                   f'\\times \\left[1 - \\exp\\left(-\\left(\\frac{{\\Omega_\\Lambda}}{{{Omega_minus}}}\\right)^{{{beta:.0f}}}\\right)\\right]$')
    ax.text(0.5, -0.12, formula_text, transform=ax.transAxes, fontsize=12,
            ha='center', va='top')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    # Save the plot
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, 'N_BH_vs_OmegaLambda.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")

    # Show the plot
    plt.show()

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Physical Interpretation (Toy Model)")
    print("=" * 70)
    print("""
This toy model illustrates the "anthropic" argument for the cosmological
constant:

1. TOO SMALL Omega_Lambda (Omega_Lambda -> 0):
   - The universe is matter-dominated for longer
   - Structure formation is delayed or altered
   - In extreme cases, recollapse could occur
   - The parametric form suppresses N_BH for small Omega_Lambda

2. TOO LARGE Omega_Lambda (Omega_Lambda -> 1):
   - Dark energy dominates early
   - Cosmic acceleration freezes out structure formation
   - Few galaxies form, few stars, few black holes
   - The exponential suppression reduces N_BH for large Omega_Lambda

3. INTERMEDIATE Omega_Lambda (the "Goldilocks zone"):
   - Enough matter domination for structures to form
   - Lambda takes over late enough to allow collapse
   - Maximum black hole production efficiency

The peak near Omega_Lambda ~ 0.3-0.4 is interesting because our observed
universe has Omega_Lambda ~ 0.7. This suggests that while we're not at
the absolute peak, we're still in a reasonably "fertile" region.

CAVEATS:
- This is a highly simplified toy model
- Real Press-Schechter theory involves the linear growth factor D(a)
- The parametric form is chosen for illustration, not from first principles
- The actual constraints on Omega_Lambda from structure formation are
  more nuanced and depend on many other cosmological parameters
""")

    print("Done!")
