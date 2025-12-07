#!/usr/bin/env python3
"""
fertility_vs_lambda_toy.py

Computes and plots the toy fertility functional:

    F(Omega_Lambda) = N_BH(Omega_Lambda) * T_int(Omega_Lambda)

where:
- N_BH(Omega_Lambda) is the relative black hole abundance from Program 3
  (Press-Schechter-like toy model)
- T_int(Omega_Lambda) is the interior lifetime from Programs 1+2
  (cosmology + mass inflation toy model)

This functional represents a toy measure of "reproductive fitness" for
universes with different cosmological constants in a cosmological natural
selection scenario.

Author: Combined from Programs 1, 2, and 3
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory and module directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, '1'))
sys.path.insert(0, os.path.join(parent_dir, '2'))
sys.path.insert(0, os.path.join(parent_dir, '3'))

# Import from our modules
from T_int_vs_lambda_toy import get_T_int_vs_OmegaLambda
from bh_population_vs_lambda import N_BH_relative


def compute_fertility_functional(omega_lambda_values, verbose=True):
    """
    Compute the fertility functional F(Omega_Lambda) = N_BH * T_int.

    Parameters
    ----------
    omega_lambda_values : array-like
        Omega_Lambda values to compute
    verbose : bool
        If True, print progress messages

    Returns
    -------
    omega_array : ndarray
        Omega_Lambda values
    F_array : ndarray
        Fertility functional values (normalized)
    T_int_norm : ndarray
        Normalized interior lifetime
    N_BH_norm : ndarray
        Normalized BH abundance
    """
    if verbose:
        print("Computing T_int(Omega_Lambda)...")

    # Get interior lifetime data
    omega_T, T_int = get_T_int_vs_OmegaLambda(
        omega_lambda_values=omega_lambda_values,
        verbose=verbose
    )

    if verbose:
        print("\nComputing N_BH(Omega_Lambda)...")

    # Compute relative BH abundance on the same grid
    N_rel = N_BH_relative(omega_T)

    if verbose:
        print("  Done.")

    # Normalize each component to max = 1
    T_norm = T_int / np.max(T_int)
    N_norm = N_rel / np.max(N_rel)

    # Compute fertility functional
    F = N_norm * T_norm

    # Normalize F to max = 1
    F_norm = F / np.max(F)

    return omega_T, F_norm, T_norm, N_norm


if __name__ == "__main__":
    print("=" * 70)
    print("Toy Fertility Functional: F(Omega_Lambda) = N_BH * T_int")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 1: Compute fertility functional
    # =========================================================================
    print("-" * 70)
    print("Step 1: Computing fertility functional components")
    print("-" * 70)
    print()

    # Use a finer grid of Omega_Lambda values
    omega_lambda_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                          0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    omega_T, F_norm, T_norm, N_norm = compute_fertility_functional(
        omega_lambda_values, verbose=True
    )

    # =========================================================================
    # Step 2: Find peak and analyze
    # =========================================================================
    print()
    print("-" * 70)
    print("Step 2: Analysis of fertility functional")
    print("-" * 70)
    print()

    # Find peak of F(Omega_Lambda)
    idx_max = np.argmax(F_norm)
    Omega_peak = omega_T[idx_max]
    F_peak = F_norm[idx_max]

    print("Fertility functional peak:")
    print(f"  Omega_Lambda_peak = {Omega_peak:.3f}")
    print(f"  F_peak (normalized) = {F_peak:.3f}")
    print()

    # Our universe's value
    Omega_obs = 0.7
    idx_obs = np.argmin(np.abs(omega_T - Omega_obs))
    F_obs = F_norm[idx_obs]
    T_obs = T_norm[idx_obs]
    N_obs = N_norm[idx_obs]

    print(f"Our universe (Omega_Lambda = {Omega_obs}):")
    print(f"  F(0.7) = {F_obs:.3f}")
    print(f"  T_int(0.7) normalized = {T_obs:.3f}")
    print(f"  N_BH(0.7) normalized = {N_obs:.3f}")
    print()

    # Ratio: how close are we to the peak?
    print(f"Ratio F(our universe) / F(peak) = {F_obs/F_peak:.3f}")
    print()

    # =========================================================================
    # Step 3: Summary table
    # =========================================================================
    print("-" * 70)
    print("Step 3: Summary table")
    print("-" * 70)
    print()
    print(f"{'Omega_Lambda':>12} | {'N_BH (norm)':>12} | {'T_int (norm)':>12} | {'F (norm)':>12}")
    print("-" * 55)
    for i, ol in enumerate(omega_T):
        marker = " <-- PEAK" if i == idx_max else (" <-- Our Universe" if abs(ol - Omega_obs) < 0.01 else "")
        print(f"{ol:>12.2f} | {N_norm[i]:>12.4f} | {T_norm[i]:>12.4f} | {F_norm[i]:>12.4f}{marker}")
    print()

    # =========================================================================
    # Step 4: Generate plots
    # =========================================================================
    print("-" * 70)
    print("Step 4: Generating plots")
    print("-" * 70)

    # Create main figure with 3 subplots
    fig = plt.figure(figsize=(16, 12))

    # Color scheme
    color_F = '#E63946'      # Red for fertility
    color_T = '#2A9D8F'      # Teal for T_int
    color_N = '#457B9D'      # Blue for N_BH
    color_obs = '#F4A261'    # Orange for our universe

    # -------------------------------------------------------------------------
    # Main plot: Fertility functional F(Omega_Lambda)
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(2, 2, (1, 2))

    # Plot F(Omega_Lambda)
    ax1.plot(omega_T, F_norm, 'o-', color=color_F, markersize=10, linewidth=3,
             markeredgecolor='white', markeredgewidth=1.5,
             label='$F(\\Omega_\\Lambda) = N_{BH} \\times T_{int}$')
    ax1.fill_between(omega_T, 0, F_norm, alpha=0.2, color=color_F)

    # Mark the peak
    ax1.axvline(Omega_peak, color=color_F, linestyle='--', linewidth=2, alpha=0.7)
    ax1.plot(Omega_peak, F_peak, 'o', markersize=18, color=color_F,
             markeredgecolor='darkred', markeredgewidth=3, zorder=5)
    ax1.annotate(f'Peak\n$\\Omega_\\Lambda = {Omega_peak:.2f}$',
                 xy=(Omega_peak, F_peak), xytext=(Omega_peak - 0.15, F_peak + 0.08),
                 fontsize=12, fontweight='bold', color='darkred',
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose',
                          edgecolor='darkred', alpha=0.9))

    # Mark our universe
    ax1.axvline(Omega_obs, color=color_obs, linestyle=':', linewidth=3)
    ax1.plot(Omega_obs, F_obs, 's', markersize=15, color=color_obs,
             markeredgecolor='darkorange', markeredgewidth=2, zorder=5)
    ax1.annotate(f'Our Universe\n$\\Omega_\\Lambda \\approx {Omega_obs}$\n$F = {F_obs:.2f}$',
                 xy=(Omega_obs, F_obs), xytext=(Omega_obs + 0.1, F_obs + 0.15),
                 fontsize=11, fontweight='bold', color='darkorange',
                 arrowprops=dict(arrowstyle='->', color='darkorange', lw=2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='papayawhip',
                          edgecolor='darkorange', alpha=0.9))

    ax1.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=14)
    ax1.set_ylabel('Normalized Fertility $F(\\Omega_\\Lambda)$', fontsize=14)
    ax1.set_title('Toy Fertility Functional: $F(\\Omega_\\Lambda) = N_{BH}(\\Omega_\\Lambda) \\times T_{int}(\\Omega_\\Lambda)$',
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.15)

    # -------------------------------------------------------------------------
    # Subplot: N_BH(Omega_Lambda)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(2, 2, 3)

    ax2.plot(omega_T, N_norm, 'o-', color=color_N, markersize=8, linewidth=2.5,
             markeredgecolor='white', markeredgewidth=1,
             label='$N_{BH}(\\Omega_\\Lambda)$ (normalized)')
    ax2.fill_between(omega_T, 0, N_norm, alpha=0.15, color=color_N)

    # Mark our universe
    ax2.axvline(Omega_obs, color=color_obs, linestyle=':', linewidth=2)
    ax2.plot(Omega_obs, N_obs, 's', markersize=10, color=color_obs,
             markeredgecolor='darkorange', markeredgewidth=1.5)

    ax2.set_xlabel('$\\Omega_\\Lambda$', fontsize=12)
    ax2.set_ylabel('$N_{BH}$ (normalized)', fontsize=12)
    ax2.set_title('Black Hole Abundance\n(Press-Schechter-like)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.1)

    # Add annotation
    ax2.text(0.05, 0.95, 'Peaks at low $\\Omega_\\Lambda$\n(more structure formation)',
             transform=ax2.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

    # -------------------------------------------------------------------------
    # Subplot: T_int(Omega_Lambda)
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(2, 2, 4)

    ax3.plot(omega_T, T_norm, 'o-', color=color_T, markersize=8, linewidth=2.5,
             markeredgecolor='white', markeredgewidth=1,
             label='$T_{int}(\\Omega_\\Lambda)$ (normalized)')
    ax3.fill_between(omega_T, 0, T_norm, alpha=0.15, color=color_T)

    # Mark our universe
    ax3.axvline(Omega_obs, color=color_obs, linestyle=':', linewidth=2)
    ax3.plot(Omega_obs, T_obs, 's', markersize=10, color=color_obs,
             markeredgecolor='darkorange', markeredgewidth=1.5)

    ax3.set_xlabel('$\\Omega_\\Lambda$', fontsize=12)
    ax3.set_ylabel('$T_{int}$ (normalized)', fontsize=12)
    ax3.set_title('Interior Lifetime\n(Mass Inflation Model)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.1)

    # Add annotation
    ax3.text(0.95, 0.05, 'Higher $\\Omega_\\Lambda$ →\nlonger interior lifetime',
             transform=ax3.transAxes, fontsize=9, va='bottom', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(script_dir, 'F_vs_OmegaLambda.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")

    # -------------------------------------------------------------------------
    # Additional: Combined comparison plot
    # -------------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(12, 8))

    # Plot all three on same axes
    ax.plot(omega_T, N_norm, 'o-', color=color_N, markersize=8, linewidth=2,
            label='$N_{BH}(\\Omega_\\Lambda)$ - BH abundance')
    ax.plot(omega_T, T_norm, 's-', color=color_T, markersize=8, linewidth=2,
            label='$T_{int}(\\Omega_\\Lambda)$ - Interior lifetime')
    ax.plot(omega_T, F_norm, 'D-', color=color_F, markersize=10, linewidth=3,
            label='$F(\\Omega_\\Lambda) = N_{BH} \\times T_{int}$ - Fertility')

    # Mark peak and our universe
    ax.axvline(Omega_peak, color=color_F, linestyle='--', linewidth=2, alpha=0.5,
               label=f'Fertility peak ($\\Omega_\\Lambda = {Omega_peak:.2f}$)')
    ax.axvline(Omega_obs, color=color_obs, linestyle=':', linewidth=2.5,
               label=f'Our universe ($\\Omega_\\Lambda = {Omega_obs}$)')

    ax.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=14)
    ax.set_ylabel('Normalized Value', fontsize=14)
    ax.set_title('Toy Model Components: How $N_{BH}$ and $T_{int}$ Combine into Fertility $F$',
                 fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)

    # Add physics explanation
    textstr = ('Physical Picture:\n'
               '• $N_{BH}$: peaks at low $\\Omega_\\Lambda$ (more structure)\n'
               '• $T_{int}$: increases with $\\Omega_\\Lambda$ (longer lifetime)\n'
               '• $F = N_{BH} \\times T_{int}$: balance between the two\n'
               f'• Peak at $\\Omega_\\Lambda \\approx {Omega_peak:.2f}$')
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                 edgecolor='goldenrod', alpha=0.9)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()

    save_path2 = os.path.join(script_dir, 'F_vs_OmegaLambda_comparison.png')
    plt.savefig(save_path2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path2}")

    # Show plots
    plt.show()

    # =========================================================================
    # Final Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Physical Interpretation (Toy Model)")
    print("=" * 70)
    print(f"""
The fertility functional F(Omega_Lambda) = N_BH * T_int represents a
toy measure of "reproductive fitness" in cosmological natural selection:

1. N_BH(Omega_Lambda) - Black Hole Abundance:
   - Peaks at low Omega_Lambda (~0.2)
   - High Omega_Lambda suppresses structure formation
   - Represents "number of offspring" in CNS

2. T_int(Omega_Lambda) - Interior Lifetime:
   - Increases with Omega_Lambda
   - Higher Lambda -> lower flux -> longer interior lifetime
   - Represents "time available for baby universe formation"

3. F(Omega_Lambda) = N_BH * T_int - Fertility:
   - Balances abundance vs lifetime
   - Peak at Omega_Lambda = {Omega_peak:.3f}
   - Our universe (Omega_Lambda = 0.7) has F = {F_obs:.3f}
   - Ratio to peak: {F_obs/F_peak:.1%}

INTERPRETATION:
Our universe sits somewhat to the RIGHT of the fertility peak.
This could suggest:
- We're in a reasonably "fertile" region (F ~ {F_obs:.0%} of max)
- The actual peak depends on model parameters
- A real CNS model would need proper GR interior dynamics

CAVEATS:
- This is a highly simplified toy model
- The scaling laws are qualitative approximations
- Real physics would require full numerical relativity
- The Press-Schechter model is also simplified
""")

    print("Done!")
