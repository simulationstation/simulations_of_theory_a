#!/usr/bin/env python3
"""
scan_N_BH_vs_omega_lambda.py

Driver script to scan N_BH(Omega_Lambda) using Press-Schechter theory.

Outputs:
- CSV file: N_BH_vs_omega_lambda_v2.csv
- PNG plot: N_BH_vs_omega_lambda_v2.png
- Sample table to stdout

Usage:
    python scan_N_BH_vs_omega_lambda.py
"""

import numpy as np
import os
import sys
import csv
import time

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from black_holes.nbh_vs_lambda import compute_N_BH
from config import M_HALO_MIN_DEFAULT, Z_FORM_DEFAULT, SIGMA_8, N_S


def scan_N_BH(omega_min=0.01, omega_max=0.95, n_points=40,
              M_halo_min=1e10, z_form=2.0):
    """
    Scan N_BH over a range of Omega_Lambda values.

    Parameters
    ----------
    omega_min : float
        Minimum Omega_Lambda (default 0.01)
    omega_max : float
        Maximum Omega_Lambda (default 0.95)
    n_points : int
        Number of scan points (default 40)
    M_halo_min : float
        Minimum halo mass for BH hosting [M_sun]
    z_form : float
        Characteristic formation redshift

    Returns
    -------
    omega_array : ndarray
        Omega_Lambda values
    N_BH_raw : ndarray
        Unnormalized N_BH values
    N_BH_norm : ndarray
        Normalized N_BH values (max = 1)
    """
    omega_array = np.linspace(omega_min, omega_max, n_points)

    print(f"Scanning Omega_Lambda from {omega_min} to {omega_max} ({n_points} points)...")
    print(f"  M_halo_min = {M_halo_min:.0e} M_sun")
    print(f"  z_form = {z_form}")
    print()

    start = time.time()

    # Compute N_BH (unnormalized)
    N_BH_raw = compute_N_BH(
        omega_array,
        M_halo_min=M_halo_min,
        z_form=z_form,
        normalize=False  # Get raw values
    )

    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f} seconds.")

    # Normalize to max = 1
    N_max = np.max(N_BH_raw)
    N_BH_norm = N_BH_raw / N_max if N_max > 0 else N_BH_raw

    return omega_array, N_BH_raw, N_BH_norm


def save_csv(omega_array, N_BH_raw, N_BH_norm, filename="N_BH_vs_omega_lambda_v2.csv"):
    """
    Save results to CSV file.

    Parameters
    ----------
    omega_array : ndarray
        Omega_Lambda values
    N_BH_raw : ndarray
        Raw (unnormalized) N_BH values
    N_BH_norm : ndarray
        Normalized N_BH values
    filename : str
        Output filename
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Omega_Lambda', 'N_BH_raw', 'N_BH_normalized'])
        for i in range(len(omega_array)):
            writer.writerow([
                f"{omega_array[i]:.6f}",
                f"{N_BH_raw[i]:.6e}",
                f"{N_BH_norm[i]:.6f}"
            ])

    print(f"\nSaved CSV to: {filepath}")
    return filepath


def create_plot(omega_array, N_BH_norm, filename="N_BH_vs_omega_lambda_v2.png"):
    """
    Create and save plot of N_BH_norm vs Omega_Lambda.

    Parameters
    ----------
    omega_array : ndarray
        Omega_Lambda values
    N_BH_norm : ndarray
        Normalized N_BH values
    filename : str
        Output filename
    """
    import matplotlib.pyplot as plt

    filepath = os.path.join(os.path.dirname(__file__), filename)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Main curve
    ax.plot(omega_array, N_BH_norm, 'b-', linewidth=2.5, label='$\\tilde{N}_{BH}(\\Omega_\\Lambda)$')
    ax.scatter(omega_array, N_BH_norm, c='blue', s=40, zorder=5, edgecolors='white', linewidths=0.5)

    # Fill under curve
    ax.fill_between(omega_array, 0, N_BH_norm, alpha=0.15, color='blue')

    # Find and mark peak
    idx_peak = np.argmax(N_BH_norm)
    omega_peak = omega_array[idx_peak]
    ax.axvline(omega_peak, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Peak at $\\Omega_\\Lambda = {omega_peak:.2f}$')
    ax.plot(omega_peak, N_BH_norm[idx_peak], 'ro', markersize=12,
            markeredgecolor='darkred', markeredgewidth=2, zorder=6)

    # Mark our universe
    ax.axvline(0.7, color='green', linestyle=':', linewidth=2,
               label='Our Universe ($\\Omega_\\Lambda \\approx 0.7$)')
    idx_07 = np.argmin(np.abs(omega_array - 0.7))
    ax.plot(0.7, N_BH_norm[idx_07], 'g^', markersize=10,
            markeredgecolor='darkgreen', markeredgewidth=1.5, zorder=6)

    # Labels
    ax.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=14)
    ax.set_ylabel('$\\tilde{N}_{BH} = N_{BH} / \\max(N_{BH})$', fontsize=14)
    ax.set_title('v2 Model: Black Hole Abundance vs Cosmological Constant\n'
                 '(Press-Schechter + Sheth-Tormen)', fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)

    # Add annotation
    ax.text(0.02, 0.02,
            f'$M_{{halo,min}} = 10^{{10}} M_\\odot$\n$z_{{form}} = 2.0$',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {filepath}")

    plt.show()

    return filepath


def print_sample_table(omega_array, N_BH_raw, N_BH_norm):
    """
    Print a sample table for selected Omega_Lambda values.

    Parameters
    ----------
    omega_array : ndarray
        All Omega_Lambda values
    N_BH_raw : ndarray
        Raw N_BH values
    N_BH_norm : ndarray
        Normalized N_BH values
    """
    sample_omegas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    print("\n" + "=" * 60)
    print("SAMPLE TABLE: N_BH vs Omega_Lambda")
    print("=" * 60)
    print(f"{'Omega_Lambda':>12} | {'N_BH (raw)':>14} | {'N_BH (norm)':>12}")
    print("-" * 44)

    for ol in sample_omegas:
        idx = np.argmin(np.abs(omega_array - ol))
        actual_ol = omega_array[idx]
        print(f"{actual_ol:>12.3f} | {N_BH_raw[idx]:>14.4e} | {N_BH_norm[idx]:>12.4f}")

    print("-" * 44)

    # Peak info
    idx_peak = np.argmax(N_BH_norm)
    print(f"\nPeak: Omega_Lambda = {omega_array[idx_peak]:.3f}, N_BH_norm = {N_BH_norm[idx_peak]:.4f}")

    # Our universe
    idx_07 = np.argmin(np.abs(omega_array - 0.7))
    print(f"Our Universe (Omega_Lambda ~ 0.7): N_BH_norm = {N_BH_norm[idx_07]:.4f}")
    print()


def main():
    """Main entry point."""
    print()
    print("=" * 60)
    print("v2 MODEL: N_BH(Omega_Lambda) from Press-Schechter")
    print("=" * 60)
    print()

    # Run scan
    omega_array, N_BH_raw, N_BH_norm = scan_N_BH(
        omega_min=0.01,
        omega_max=0.95,
        n_points=40,
        M_halo_min=1e10,
        z_form=2.0
    )

    # Save CSV
    save_csv(omega_array, N_BH_raw, N_BH_norm)

    # Print sample table
    print_sample_table(omega_array, N_BH_raw, N_BH_norm)

    # Create plot
    try:
        create_plot(omega_array, N_BH_norm)
    except ImportError:
        print("matplotlib not available, skipping plot")

    print("Done!")


if __name__ == "__main__":
    main()
