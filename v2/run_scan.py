#!/usr/bin/env python3
"""
run_scan.py

Main driver script for v2 cosmological natural selection model.

Performs a 1D scan over Omega_Lambda and computes:
1. N_BH(Omega_Lambda) from Press-Schechter structure formation
2. T_int(Omega_Lambda) from mass-inflation flux integral
3. F(Omega_Lambda) = N_BH * T_int (fertility functional)

Outputs:
- Summary table to stdout
- Plots of all quantities vs Omega_Lambda
- Optional CSV/JSON export

Usage:
    python run_scan.py [--n-points 50] [--save-data] [--no-plot]

Author: v2 pipeline
"""

import argparse
import numpy as np
import os
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    OMEGA_LAMBDA0, OMEGA_M0, OMEGA_R0, SIGMA_8, N_S,
    M_SUN, M_BH_REF, M_HALO_MIN_DEFAULT, Z_FORM_DEFAULT,
    KAPPA_DEFAULT, print_parameters
)
from fertility import compute_fertility, analyze_fertility, find_fertility_peak
from black_holes.nbh_vs_lambda import compute_N_BH
from black_holes.interior_lifetime import compute_T_int


def run_omega_lambda_scan(n_points=30, omega_min=0.05, omega_max=0.95,
                           M_bh=M_BH_REF, M_halo_min=M_HALO_MIN_DEFAULT,
                           z_form=Z_FORM_DEFAULT, kappa=KAPPA_DEFAULT,
                           verbose=True):
    """
    Run a complete scan over Omega_Lambda.

    Parameters
    ----------
    n_points : int
        Number of Omega_Lambda points
    omega_min, omega_max : float
        Range for Omega_Lambda scan
    M_bh : float
        Black hole mass [kg]
    M_halo_min : float
        Minimum halo mass [M_sun]
    z_form : float
        Characteristic BH formation redshift
    kappa : float
        Mass inflation surface gravity parameter
    verbose : bool
        Print progress messages

    Returns
    -------
    results : dict
        Dictionary with all computed quantities
    """
    # Omega_Lambda array
    omega_array = np.linspace(omega_min, omega_max, n_points)

    if verbose:
        print(f"\nScanning Omega_Lambda from {omega_min:.2f} to {omega_max:.2f} "
              f"({n_points} points)...")
        print()

    # Initialize arrays
    N_BH_array = np.zeros(n_points)
    T_int_array = np.zeros(n_points)
    F_array = np.zeros(n_points)

    start_time = time.time()

    # Compute each quantity
    for i, OL in enumerate(omega_array):
        if verbose and (i % 5 == 0 or i == n_points - 1):
            print(f"  Processing Omega_Lambda = {OL:.3f} ({i+1}/{n_points})...",
                  end='\r')

        # N_BH
        N_BH_array[i] = compute_N_BH(
            OL, M_halo_min=M_halo_min, z_form=z_form,
            normalize=False
        )

        # T_int
        T_int_array[i] = compute_T_int(
            OL, M_bh=M_bh, kappa=kappa,
            normalize=False
        )

        # F = N_BH * T_int
        F_array[i] = N_BH_array[i] * T_int_array[i]

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n  Completed in {elapsed:.1f} seconds.")
        print()

    # Normalize
    N_BH_norm = N_BH_array / np.max(N_BH_array) if np.max(N_BH_array) > 0 else N_BH_array
    T_int_norm = T_int_array / np.max(T_int_array) if np.max(T_int_array) > 0 else T_int_array
    F_norm = F_array / np.max(F_array) if np.max(F_array) > 0 else F_array

    # Find peaks
    idx_N_peak = np.argmax(N_BH_array)
    idx_T_peak = np.argmax(T_int_array)
    idx_F_peak = np.argmax(F_array)

    # Find value at our universe (Omega_Lambda = 0.7)
    idx_07 = np.argmin(np.abs(omega_array - 0.7))

    results = {
        'omega': omega_array,
        'N_BH': N_BH_array,
        'N_BH_norm': N_BH_norm,
        'T_int': T_int_array,
        'T_int_norm': T_int_norm,
        'F': F_array,
        'F_norm': F_norm,
        'Omega_N_peak': omega_array[idx_N_peak],
        'Omega_T_peak': omega_array[idx_T_peak],
        'Omega_F_peak': omega_array[idx_F_peak],
        'idx_07': idx_07,
        'F_at_07': F_norm[idx_07],
        'ratio_to_peak': F_norm[idx_07] / F_norm[idx_F_peak] if F_norm[idx_F_peak] > 0 else 0,
        'elapsed_time': elapsed,
        'parameters': {
            'M_bh': M_bh,
            'M_halo_min': M_halo_min,
            'z_form': z_form,
            'kappa': kappa,
        }
    }

    return results


def print_results(results):
    """Print formatted results table."""
    print("=" * 70)
    print("RESULTS: Fertility Functional F(Omega_Lambda) = N_BH * T_int")
    print("=" * 70)
    print()

    # Peak information
    print("Peak Locations:")
    print("-" * 40)
    print(f"  N_BH peaks at Omega_Lambda = {results['Omega_N_peak']:.3f}")
    print(f"  T_int peaks at Omega_Lambda = {results['Omega_T_peak']:.3f}")
    print(f"  F peaks at Omega_Lambda = {results['Omega_F_peak']:.3f}")
    print()

    # Our universe
    print("Our Universe (Omega_Lambda = 0.7):")
    print("-" * 40)
    print(f"  F(0.7) / F_max = {results['F_at_07']:.3f}")
    print(f"  Ratio to peak = {results['ratio_to_peak']:.1%}")
    print()

    # Data table
    omega = results['omega']
    N_BH = results['N_BH_norm']
    T_int = results['T_int_norm']
    F = results['F_norm']

    print("Data Table (normalized to max = 1):")
    print("-" * 60)
    print(f"{'Omega_Lambda':>12} | {'N_BH':>10} | {'T_int':>10} | {'F':>10} | {'Notes':>10}")
    print("-" * 60)

    for i in range(len(omega)):
        notes = ""
        if i == np.argmax(F):
            notes = "F peak"
        elif abs(omega[i] - 0.7) < 0.02:
            notes = "Our Universe"

        print(f"{omega[i]:>12.3f} | {N_BH[i]:>10.4f} | {T_int[i]:>10.4f} | "
              f"{F[i]:>10.4f} | {notes:>10}")

    print("-" * 60)
    print()


def create_plots(results, save_dir=None, show=True):
    """Create and optionally save plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    omega = results['omega']
    N_BH = results['N_BH_norm']
    T_int = results['T_int_norm']
    F = results['F_norm']
    Omega_peak = results['Omega_F_peak']

    # Color scheme
    color_F = '#E63946'
    color_N = '#457B9D'
    color_T = '#2A9D8F'
    color_obs = '#F4A261'

    # Figure 1: Main fertility plot
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    ax1.plot(omega, F, 'o-', color=color_F, markersize=8, linewidth=2.5,
             label='$F(\\Omega_\\Lambda) = N_{BH} \\times T_{int}$')
    ax1.fill_between(omega, 0, F, alpha=0.2, color=color_F)

    # Peak
    idx_peak = np.argmax(F)
    ax1.axvline(Omega_peak, color=color_F, linestyle='--', alpha=0.7)
    ax1.plot(Omega_peak, F[idx_peak], 'o', markersize=15, color=color_F,
             markeredgecolor='darkred', markeredgewidth=2)

    # Our universe
    ax1.axvline(0.7, color=color_obs, linestyle=':', linewidth=2.5,
                label='Our Universe ($\\Omega_\\Lambda = 0.7$)')
    idx_07 = results['idx_07']
    ax1.plot(0.7, F[idx_07], 's', markersize=12, color=color_obs,
             markeredgecolor='darkorange', markeredgewidth=2)

    ax1.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant)', fontsize=14)
    ax1.set_ylabel('Fertility $F(\\Omega_\\Lambda)$ (normalized)', fontsize=14)
    ax1.set_title('v2 Model: Cosmological Natural Selection Fertility Functional',
                  fontsize=15, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.15)

    # Add annotation box
    textstr = (f'Peak: $\\Omega_\\Lambda = {Omega_peak:.2f}$\n'
               f'Our Universe: $F = {F[idx_07]:.2f}$\n'
               f'Ratio to peak: {results["ratio_to_peak"]:.0%}')
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                 edgecolor='goldenrod', alpha=0.9)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)

    plt.tight_layout()

    if save_dir:
        path1 = os.path.join(save_dir, 'F_vs_OmegaLambda_v2.png')
        plt.savefig(path1, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path1}")

    # Figure 2: All components
    fig2, axes = plt.subplots(1, 3, figsize=(16, 5))

    # N_BH
    ax = axes[0]
    ax.plot(omega, N_BH, 'o-', color=color_N, markersize=6, linewidth=2)
    ax.fill_between(omega, 0, N_BH, alpha=0.2, color=color_N)
    ax.axvline(0.7, color=color_obs, linestyle=':', linewidth=2)
    ax.set_xlabel('$\\Omega_\\Lambda$', fontsize=12)
    ax.set_ylabel('$N_{BH}$ (normalized)', fontsize=12)
    ax.set_title('Black Hole Abundance\n(Press-Schechter)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # T_int
    ax = axes[1]
    ax.plot(omega, T_int, 'o-', color=color_T, markersize=6, linewidth=2)
    ax.fill_between(omega, 0, T_int, alpha=0.2, color=color_T)
    ax.axvline(0.7, color=color_obs, linestyle=':', linewidth=2)
    ax.set_xlabel('$\\Omega_\\Lambda$', fontsize=12)
    ax.set_ylabel('$T_{int}$ (normalized)', fontsize=12)
    ax.set_title('Interior Lifetime\n(Mass Inflation)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # F
    ax = axes[2]
    ax.plot(omega, F, 'o-', color=color_F, markersize=6, linewidth=2)
    ax.fill_between(omega, 0, F, alpha=0.2, color=color_F)
    ax.axvline(0.7, color=color_obs, linestyle=':', linewidth=2)
    ax.axvline(Omega_peak, color=color_F, linestyle='--', alpha=0.7)
    ax.set_xlabel('$\\Omega_\\Lambda$', fontsize=12)
    ax.set_ylabel('$F$ (normalized)', fontsize=12)
    ax.set_title('Fertility Functional\n$F = N_{BH} \\times T_{int}$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    plt.tight_layout()

    if save_dir:
        path2 = os.path.join(save_dir, 'components_vs_OmegaLambda_v2.png')
        plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path2}")

    if show:
        plt.show()


def save_data(results, save_dir):
    """Save results to CSV file."""
    import csv

    filepath = os.path.join(save_dir, 'fertility_scan_results.csv')

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['Omega_Lambda', 'N_BH', 'N_BH_norm', 'T_int', 'T_int_norm', 'F', 'F_norm'])

        # Data
        for i in range(len(results['omega'])):
            writer.writerow([
                f"{results['omega'][i]:.6f}",
                f"{results['N_BH'][i]:.6e}",
                f"{results['N_BH_norm'][i]:.6f}",
                f"{results['T_int'][i]:.6e}",
                f"{results['T_int_norm'][i]:.6f}",
                f"{results['F'][i]:.6e}",
                f"{results['F_norm'][i]:.6f}",
            ])

    print(f"Saved data to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description='v2 Cosmological Natural Selection Model: Omega_Lambda Scan'
    )
    parser.add_argument('--n-points', type=int, default=30,
                        help='Number of Omega_Lambda points (default: 30)')
    parser.add_argument('--omega-min', type=float, default=0.05,
                        help='Minimum Omega_Lambda (default: 0.05)')
    parser.add_argument('--omega-max', type=float, default=0.95,
                        help='Maximum Omega_Lambda (default: 0.95)')
    parser.add_argument('--save-data', action='store_true',
                        help='Save results to CSV file')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')

    args = parser.parse_args()

    verbose = not args.quiet

    # Print header
    if verbose:
        print()
        print("=" * 70)
        print("v2 COSMOLOGICAL NATURAL SELECTION MODEL")
        print("Fertility Functional: F(Omega_Lambda) = N_BH * T_int")
        print("=" * 70)
        print()
        print_parameters()

    # Run scan
    results = run_omega_lambda_scan(
        n_points=args.n_points,
        omega_min=args.omega_min,
        omega_max=args.omega_max,
        verbose=verbose
    )

    # Print results
    if verbose:
        print_results(results)

    # Save data
    save_dir = os.path.dirname(os.path.abspath(__file__))
    if args.save_data:
        save_data(results, save_dir)

    # Create plots
    if not args.no_plot:
        create_plots(results, save_dir=save_dir, show=True)

    if verbose:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"""
The v2 model computes F(Omega_Lambda) using:
- Press-Schechter theory with linear growth factor D(a)
- Mass-inflation-inspired interior lifetime model

KEY RESULTS:
- Fertility peak at Omega_Lambda = {results['Omega_F_peak']:.3f}
- Our universe (Omega_Lambda = 0.7) has F/F_max = {results['ratio_to_peak']:.1%}
- Computation time: {results['elapsed_time']:.1f} seconds

This {'supports' if results['ratio_to_peak'] > 0.5 else 'challenges'} CNS if we expect
observed parameters to be within ~50% of the fertility peak.
""")

    print("Done!")


if __name__ == "__main__":
    main()
