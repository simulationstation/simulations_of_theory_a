#!/usr/bin/env python3
"""
scan_T_int_vs_omega_lambda_v2.py

Driver script to scan T_int(Omega_Lambda) using the mass-inflation-motivated
logarithmic formula with cosmological flux history.

Pipeline:
1. For each Omega_Lambda, compute Phi_eff = integral L_in(t) dt
2. Map to T_int = (1/kappa) * log(C / Phi_eff)
3. Normalize and output results

Outputs:
- CSV file: T_int_vs_omega_lambda_v2.csv
- PNG plot: T_int_vs_omega_lambda_v2.png
- Sample table to stdout

Usage:
    python3 scan_T_int_vs_omega_lambda_v2.py
"""

import numpy as np
import os
import sys
import csv
import time

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from T_int_from_flux_v2 import scan_T_int_vs_omega_lambda
from cosmo_flux_lambda import M_sun


def load_omega_grid_from_csv(csv_path):
    """
    Load Omega_Lambda grid from the N_BH scan CSV for consistency.

    Parameters
    ----------
    csv_path : str
        Path to N_BH_vs_omega_lambda_v2.csv

    Returns
    -------
    omega_array : ndarray
        Omega_Lambda values from the CSV
    """
    omega_list = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            omega_list.append(float(row['Omega_Lambda']))
    return np.array(omega_list)


def save_csv(omega_array, Phi_eff_array, T_int_array, T_int_norm, t_iso_array,
             filename="T_int_vs_omega_lambda_v2.csv"):
    """
    Save results to CSV file.

    Parameters
    ----------
    omega_array : ndarray
        Omega_Lambda values
    Phi_eff_array : ndarray
        Effective perturbation amplitudes
    T_int_array : ndarray
        Raw interior lifetimes
    T_int_norm : ndarray
        Normalized interior lifetimes
    t_iso_array : ndarray
        Isolation times [Gyr]
    filename : str
        Output filename
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Omega_Lambda', 'Phi_eff_W_Gyr', 't_iso_Gyr',
                         'T_int_raw', 'T_int_normalized'])
        for i in range(len(omega_array)):
            t_iso_str = f"{t_iso_array[i]:.2f}" if np.isfinite(t_iso_array[i]) else "NaN"
            writer.writerow([
                f"{omega_array[i]:.6f}",
                f"{Phi_eff_array[i]:.6e}",
                t_iso_str,
                f"{T_int_array[i]:.6f}",
                f"{T_int_norm[i]:.6f}"
            ])

    print(f"\nSaved CSV to: {filepath}")
    return filepath


def create_plot(omega_array, T_int_norm, Phi_eff_array, t_iso_array,
                filename="T_int_vs_omega_lambda_v2.png"):
    """
    Create and save plot of T_int_norm vs Omega_Lambda.

    Parameters
    ----------
    omega_array : ndarray
        Omega_Lambda values
    T_int_norm : ndarray
        Normalized T_int values
    Phi_eff_array : ndarray
        Effective perturbation amplitudes (for subplot)
    t_iso_array : ndarray
        Isolation times (for subplot)
    filename : str
        Output filename
    """
    import matplotlib.pyplot as plt

    filepath = os.path.join(os.path.dirname(__file__), filename)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ----- Left plot: T_int_norm vs Omega_Lambda -----
    ax1 = axes[0]

    ax1.plot(omega_array, T_int_norm, 'b-', linewidth=2.5,
             label='$\\tilde{T}_{int}(\\Omega_\\Lambda)$')
    ax1.scatter(omega_array, T_int_norm, c='blue', s=40, zorder=5,
                edgecolors='white', linewidths=0.5)
    ax1.fill_between(omega_array, 0, T_int_norm, alpha=0.15, color='blue')

    # Find and mark peak
    idx_peak = np.argmax(T_int_norm)
    omega_peak = omega_array[idx_peak]
    ax1.axvline(omega_peak, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Peak at $\\Omega_\\Lambda = {omega_peak:.2f}$')
    ax1.plot(omega_peak, T_int_norm[idx_peak], 'ro', markersize=12,
             markeredgecolor='darkred', markeredgewidth=2, zorder=6)

    # Mark our universe
    ax1.axvline(0.7, color='green', linestyle=':', linewidth=2,
                label='Our Universe ($\\Omega_\\Lambda \\approx 0.7$)')
    idx_07 = np.argmin(np.abs(omega_array - 0.7))
    ax1.plot(0.7, T_int_norm[idx_07], 'g^', markersize=10,
             markeredgecolor='darkgreen', markeredgewidth=1.5, zorder=6)

    ax1.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant)', fontsize=14)
    ax1.set_ylabel('$\\tilde{T}_{int} = T_{int} / \\max(T_{int})$', fontsize=14)
    ax1.set_title('v2 Model: Interior Lifetime vs $\\Omega_\\Lambda$\n'
                  '(Mass Inflation + Flux Integral)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.1)

    # Add physics annotation
    ax1.text(0.02, 0.98,
             'Higher $\\Omega_\\Lambda$ → lower flux\n→ smaller $\\Phi_{eff}$\n→ longer $T_{int}$',
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # ----- Right plot: Phi_eff and t_iso vs Omega_Lambda -----
    ax2 = axes[1]

    # Phi_eff on left y-axis
    color_phi = 'tab:red'
    ax2.set_xlabel('$\\Omega_\\Lambda$', fontsize=14)
    ax2.set_ylabel('$\\Phi_{eff}$ [W $\\cdot$ Gyr]', color=color_phi, fontsize=12)
    ax2.semilogy(omega_array, Phi_eff_array, 'o-', color=color_phi,
                 markersize=5, linewidth=2, label='$\\Phi_{eff}$')
    ax2.tick_params(axis='y', labelcolor=color_phi)

    # t_iso on right y-axis
    ax2_twin = ax2.twinx()
    color_tiso = 'tab:blue'
    ax2_twin.set_ylabel('$t_{iso}$ [Gyr]', color=color_tiso, fontsize=12)
    valid_tiso = np.isfinite(t_iso_array)
    ax2_twin.plot(omega_array[valid_tiso], t_iso_array[valid_tiso], 's--',
                  color=color_tiso, markersize=5, linewidth=1.5, label='$t_{iso}$')
    ax2_twin.tick_params(axis='y', labelcolor=color_tiso)

    ax2.axvline(0.7, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax2.set_title('Intermediate Quantities', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 1)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {filepath}")

    plt.show()

    return filepath


def print_sample_table(omega_array, Phi_eff_array, T_int_array, T_int_norm, t_iso_array):
    """
    Print a sample table for selected Omega_Lambda values.

    Parameters
    ----------
    omega_array : ndarray
        All Omega_Lambda values
    Phi_eff_array : ndarray
        Effective perturbation amplitudes
    T_int_array : ndarray
        Raw T_int values
    T_int_norm : ndarray
        Normalized T_int values
    t_iso_array : ndarray
        Isolation times [Gyr]
    """
    sample_omegas = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("\n" + "=" * 75)
    print("SAMPLE TABLE: T_int vs Omega_Lambda")
    print("=" * 75)
    print(f"{'Omega_Lambda':>12} | {'Phi_eff [W*Gyr]':>16} | {'t_iso [Gyr]':>12} | "
          f"{'T_int':>10} | {'T_int_norm':>10}")
    print("-" * 75)

    for ol in sample_omegas:
        idx = np.argmin(np.abs(omega_array - ol))
        actual_ol = omega_array[idx]
        t_iso_str = f"{t_iso_array[idx]:.1f}" if np.isfinite(t_iso_array[idx]) else "N/A"
        print(f"{actual_ol:>12.3f} | {Phi_eff_array[idx]:>16.4e} | {t_iso_str:>12} | "
              f"{T_int_array[idx]:>10.4f} | {T_int_norm[idx]:>10.4f}")

    print("-" * 75)

    # Peak and our universe info
    idx_peak = np.argmax(T_int_norm)
    print(f"\nPeak: Omega_Lambda = {omega_array[idx_peak]:.3f}, T_int_norm = {T_int_norm[idx_peak]:.4f}")

    idx_07 = np.argmin(np.abs(omega_array - 0.7))
    print(f"Our Universe (Omega_Lambda ~ 0.7): T_int_norm = {T_int_norm[idx_07]:.4f}")
    print()


def main():
    """Main entry point."""
    print()
    print("=" * 70)
    print("v2 MODEL: T_int(Omega_Lambda) from Flux Integral + Mass Inflation")
    print("=" * 70)
    print()

    # Check if we can load the N_BH grid for consistency
    nbh_csv = os.path.join(os.path.dirname(__file__), "N_BH_vs_omega_lambda_v2.csv")
    if os.path.exists(nbh_csv):
        print(f"Loading Omega_Lambda grid from {nbh_csv}...")
        omega_array = load_omega_grid_from_csv(nbh_csv)
        print(f"  Loaded {len(omega_array)} points from N_BH scan.")
    else:
        print("N_BH CSV not found, using default grid...")
        omega_array = np.linspace(0.01, 0.95, 40)

    print()

    # Run the scan
    start_time = time.time()

    omega, Phi_eff, T_int, T_int_norm, t_iso, C = scan_T_int_vs_omega_lambda(
        omega_array,
        kappa_minus=1.0,
        M_bh=1e8 * M_sun,
        use_isolation_cutoff=True,
        C_factor=10.0,
        verbose=True
    )

    elapsed = time.time() - start_time
    print(f"Total computation time: {elapsed:.1f} seconds")

    # Save CSV
    save_csv(omega, Phi_eff, T_int, T_int_norm, t_iso)

    # Print sample table
    print_sample_table(omega, Phi_eff, T_int, T_int_norm, t_iso)

    # Create plot
    try:
        create_plot(omega, T_int_norm, Phi_eff, t_iso)
    except ImportError:
        print("matplotlib not available, skipping plot")

    # Summary
    print("=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print(f"""
The T_int(Omega_Lambda) curve shows:

1. Higher Omega_Lambda -> faster cosmic acceleration
2. Faster expansion -> radiation dilutes more quickly
3. Lower cumulative flux Phi_eff = integral L_in dt
4. Smaller Phi_eff -> slower mass inflation -> longer T_int

Formula: T_int = (1/kappa) * log(C / Phi_eff)

With C = {C:.4e} W*Gyr (set to {10.0}x max Phi_eff),
the interior lifetime increases monotonically with Omega_Lambda.

Peak at Omega_Lambda = {omega[np.argmax(T_int_norm)]:.2f}
Our Universe (Omega_Lambda ~ 0.7): T_int_norm = {T_int_norm[np.argmin(np.abs(omega - 0.7))]:.3f}
""")

    print("Done!")


if __name__ == "__main__":
    main()
