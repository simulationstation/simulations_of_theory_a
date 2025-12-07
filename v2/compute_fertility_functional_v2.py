#!/usr/bin/env python3
"""
compute_fertility_functional_v2.py

Combine v2 structure formation (N_BH) and interior lifetime (T_int) results
into the fertility functional F(Omega_Lambda) = N_BH * T_int.

Loads data from:
- N_BH_vs_omega_lambda_v2.csv
- T_int_vs_omega_lambda_v2.csv

Outputs:
- F_vs_omega_lambda_v2.csv
- F_vs_omega_lambda_v2.png (main plot)
- F_components_vs_omega_lambda_v2.png (comparison plot)

Usage:
    python3 compute_fertility_functional_v2.py
"""

import numpy as np
import os
import sys
import csv


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_N_BH_data(csv_path):
    """
    Load N_BH data from CSV file.

    Parameters
    ----------
    csv_path : str
        Path to N_BH_vs_omega_lambda_v2.csv

    Returns
    -------
    data : dict
        Dictionary with 'omega', 'N_BH_raw', 'N_BH_norm' arrays
    """
    omega = []
    N_BH_raw = []
    N_BH_norm = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            omega.append(float(row['Omega_Lambda']))
            N_BH_raw.append(float(row['N_BH_raw']))
            N_BH_norm.append(float(row['N_BH_normalized']))

    return {
        'omega': np.array(omega),
        'N_BH_raw': np.array(N_BH_raw),
        'N_BH_norm': np.array(N_BH_norm)
    }


def load_T_int_data(csv_path):
    """
    Load T_int data from CSV file.

    Parameters
    ----------
    csv_path : str
        Path to T_int_vs_omega_lambda_v2.csv

    Returns
    -------
    data : dict
        Dictionary with 'omega', 'Phi_eff', 'T_int_raw', 'T_int_norm' arrays
    """
    omega = []
    Phi_eff = []
    t_iso = []
    T_int_raw = []
    T_int_norm = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            omega.append(float(row['Omega_Lambda']))
            Phi_eff.append(float(row['Phi_eff_W_Gyr']))
            t_iso_val = row['t_iso_Gyr']
            t_iso.append(float(t_iso_val) if t_iso_val != 'NaN' else np.nan)
            T_int_raw.append(float(row['T_int_raw']))
            T_int_norm.append(float(row['T_int_normalized']))

    return {
        'omega': np.array(omega),
        'Phi_eff': np.array(Phi_eff),
        't_iso': np.array(t_iso),
        'T_int_raw': np.array(T_int_raw),
        'T_int_norm': np.array(T_int_norm)
    }


def verify_omega_grids(omega1, omega2, tol=1e-6):
    """
    Verify that two omega_lambda grids match.

    Parameters
    ----------
    omega1 : ndarray
        First omega grid
    omega2 : ndarray
        Second omega grid
    tol : float
        Tolerance for floating point comparison

    Raises
    ------
    ValueError
        If grids don't match
    """
    if len(omega1) != len(omega2):
        raise ValueError(
            f"Omega grids have different lengths: {len(omega1)} vs {len(omega2)}"
        )

    max_diff = np.max(np.abs(omega1 - omega2))
    if max_diff > tol:
        raise ValueError(
            f"Omega grids don't match. Max difference: {max_diff:.2e}"
        )

    print(f"  Omega grids match ({len(omega1)} points, max diff = {max_diff:.2e})")


# =============================================================================
# Fertility Computation
# =============================================================================

def compute_fertility(N_BH_raw, T_int_raw):
    """
    Compute unnormalized and normalized fertility functional.

    F(Omega_Lambda) = N_BH(Omega_Lambda) * T_int(Omega_Lambda)

    Parameters
    ----------
    N_BH_raw : ndarray
        Unnormalized black hole abundance
    T_int_raw : ndarray
        Unnormalized interior lifetime

    Returns
    -------
    F_raw : ndarray
        Unnormalized fertility
    F_norm : ndarray
        Normalized fertility (max = 1)
    """
    F_raw = N_BH_raw * T_int_raw

    F_max = np.max(F_raw)
    F_norm = F_raw / F_max if F_max > 0 else F_raw

    return F_raw, F_norm


# =============================================================================
# Output Functions
# =============================================================================

def save_csv(omega, N_BH_raw, N_BH_norm, T_int_raw, T_int_norm, F_raw, F_norm,
             filename="F_vs_omega_lambda_v2.csv"):
    """
    Save combined fertility results to CSV.

    Parameters
    ----------
    omega : ndarray
        Omega_Lambda values
    N_BH_raw, N_BH_norm : ndarray
        Black hole abundance (raw and normalized)
    T_int_raw, T_int_norm : ndarray
        Interior lifetime (raw and normalized)
    F_raw, F_norm : ndarray
        Fertility (raw and normalized)
    filename : str
        Output filename
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Omega_Lambda', 'N_BH', 'N_BH_norm',
            'T_int', 'T_int_norm', 'F', 'F_norm'
        ])
        for i in range(len(omega)):
            writer.writerow([
                f"{omega[i]:.6f}",
                f"{N_BH_raw[i]:.6e}",
                f"{N_BH_norm[i]:.6f}",
                f"{T_int_raw[i]:.6f}",
                f"{T_int_norm[i]:.6f}",
                f"{F_raw[i]:.6e}",
                f"{F_norm[i]:.6f}"
            ])

    print(f"Saved CSV to: {filepath}")
    return filepath


def create_main_plot(omega, F_norm, filename="F_vs_omega_lambda_v2.png"):
    """
    Create main fertility plot: F_norm vs Omega_Lambda.

    Parameters
    ----------
    omega : ndarray
        Omega_Lambda values
    F_norm : ndarray
        Normalized fertility values
    filename : str
        Output filename
    """
    import matplotlib.pyplot as plt

    filepath = os.path.join(os.path.dirname(__file__), filename)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Main curve
    ax.plot(omega, F_norm, 'b-', linewidth=2.5,
            label='$\\tilde{F}(\\Omega_\\Lambda) = N_{BH} \\times T_{int}$ (normalized)')
    ax.scatter(omega, F_norm, c='blue', s=50, zorder=5,
               edgecolors='white', linewidths=0.5)
    ax.fill_between(omega, 0, F_norm, alpha=0.15, color='blue')

    # Find and mark peak
    idx_peak = np.argmax(F_norm)
    omega_peak = omega[idx_peak]
    F_peak = F_norm[idx_peak]

    ax.axvline(omega_peak, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Peak at $\\Omega_\\Lambda = {omega_peak:.3f}$')
    ax.plot(omega_peak, F_peak, 'ro', markersize=15,
            markeredgecolor='darkred', markeredgewidth=2, zorder=6)

    # Annotate peak
    ax.annotate(f'Peak\n$\\Omega_\\Lambda = {omega_peak:.3f}$\n$\\tilde{{F}} = {F_peak:.3f}$',
                xy=(omega_peak, F_peak),
                xytext=(omega_peak + 0.12, F_peak - 0.15),
                fontsize=11, fontweight='bold', color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                bbox=dict(boxstyle='round,pad=0.4', facecolor='mistyrose',
                          edgecolor='darkred', alpha=0.9))

    # Mark our universe
    ax.axvline(0.7, color='green', linestyle=':', linewidth=2.5,
               label='Our Universe ($\\Omega_\\Lambda \\approx 0.7$)')
    idx_07 = np.argmin(np.abs(omega - 0.7))
    F_07 = F_norm[idx_07]
    ax.plot(0.7, F_07, 'g^', markersize=12,
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=6)

    # Annotate our universe
    ax.annotate(f'Our Universe\n$\\Omega_\\Lambda = 0.7$\n$\\tilde{{F}} = {F_07:.3f}$',
                xy=(0.7, F_07),
                xytext=(0.7 + 0.1, F_07 + 0.12),
                fontsize=10, color='darkgreen',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew',
                          edgecolor='darkgreen', alpha=0.9))

    # Labels and formatting
    ax.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=14)
    ax.set_ylabel('$\\tilde{F} = F / \\max(F)$ (Normalized Fertility)', fontsize=14)
    ax.set_title('v2 Model: Fertility Functional $F(\\Omega_\\Lambda) = N_{BH} \\times T_{int}$\n'
                 '(Press-Schechter Structure Formation + Mass Inflation Interior)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)

    # Add ratio annotation
    ratio = F_07 / F_peak
    ratio_pct = ratio * 100
    ax.text(0.02, 0.02,
            f'F(0.7) / F(peak) = {ratio_pct:.1f}%',
            transform=ax.transAxes, fontsize=12, va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='goldenrod', alpha=0.9))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved main plot to: {filepath}")

    return filepath


def create_components_plot(omega, N_BH_norm, T_int_norm, F_norm,
                            filename="F_components_vs_omega_lambda_v2.png"):
    """
    Create comparison plot showing N_BH, T_int, and F together.

    Parameters
    ----------
    omega : ndarray
        Omega_Lambda values
    N_BH_norm : ndarray
        Normalized N_BH
    T_int_norm : ndarray
        Normalized T_int
    F_norm : ndarray
        Normalized fertility
    filename : str
        Output filename
    """
    import matplotlib.pyplot as plt

    filepath = os.path.join(os.path.dirname(__file__), filename)

    # Color scheme
    color_F = '#E63946'    # Red for fertility
    color_N = '#457B9D'    # Blue for N_BH
    color_T = '#2A9D8F'    # Teal for T_int
    color_obs = '#F4A261'  # Orange for our universe

    fig = plt.figure(figsize=(16, 10))

    # ----- Top panel: All three on same axes -----
    ax1 = fig.add_subplot(2, 1, 1)

    ax1.plot(omega, N_BH_norm, 'o-', color=color_N, markersize=5, linewidth=2,
             label='$\\tilde{N}_{BH}(\\Omega_\\Lambda)$ — BH Abundance')
    ax1.plot(omega, T_int_norm, 's-', color=color_T, markersize=5, linewidth=2,
             label='$\\tilde{T}_{int}(\\Omega_\\Lambda)$ — Interior Lifetime')
    ax1.plot(omega, F_norm, 'D-', color=color_F, markersize=6, linewidth=2.5,
             label='$\\tilde{F}(\\Omega_\\Lambda) = \\tilde{N}_{BH} \\times \\tilde{T}_{int}$ — Fertility')

    # Mark peak of F
    idx_peak = np.argmax(F_norm)
    ax1.axvline(omega[idx_peak], color=color_F, linestyle='--', linewidth=1.5, alpha=0.7)

    # Mark our universe
    ax1.axvline(0.7, color=color_obs, linestyle=':', linewidth=2.5)

    ax1.set_xlabel('$\\Omega_\\Lambda$', fontsize=13)
    ax1.set_ylabel('Normalized Value', fontsize=13)
    ax1.set_title('v2 Model: Components of the Fertility Functional',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.15)

    # Physics annotation
    textstr = ('Physical Picture:\n'
               '• $N_{BH}$: peaks at low $\\Omega_\\Lambda$ (more structure)\n'
               '• $T_{int}$: nearly flat (weak $\\Omega_\\Lambda$ dependence)\n'
               '• $F = N_{BH} \\times T_{int}$: dominated by $N_{BH}$')
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                 edgecolor='goldenrod', alpha=0.9)
    ax1.text(0.02, 0.35, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # ----- Bottom panels: Individual components -----
    ax2 = fig.add_subplot(2, 3, 4)
    ax2.plot(omega, N_BH_norm, 'o-', color=color_N, markersize=4, linewidth=2)
    ax2.fill_between(omega, 0, N_BH_norm, alpha=0.2, color=color_N)
    ax2.axvline(0.7, color=color_obs, linestyle=':', linewidth=2)
    ax2.set_xlabel('$\\Omega_\\Lambda$', fontsize=11)
    ax2.set_ylabel('$\\tilde{N}_{BH}$', fontsize=11)
    ax2.set_title('Black Hole Abundance\n(Press-Schechter)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.1)

    ax3 = fig.add_subplot(2, 3, 5)
    ax3.plot(omega, T_int_norm, 's-', color=color_T, markersize=4, linewidth=2)
    ax3.fill_between(omega, 0, T_int_norm, alpha=0.2, color=color_T)
    ax3.axvline(0.7, color=color_obs, linestyle=':', linewidth=2)
    ax3.set_xlabel('$\\Omega_\\Lambda$', fontsize=11)
    ax3.set_ylabel('$\\tilde{T}_{int}$', fontsize=11)
    ax3.set_title('Interior Lifetime\n(Mass Inflation)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.1)

    ax4 = fig.add_subplot(2, 3, 6)
    ax4.plot(omega, F_norm, 'D-', color=color_F, markersize=4, linewidth=2)
    ax4.fill_between(omega, 0, F_norm, alpha=0.2, color=color_F)
    ax4.axvline(omega[idx_peak], color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.axvline(0.7, color=color_obs, linestyle=':', linewidth=2)
    ax4.set_xlabel('$\\Omega_\\Lambda$', fontsize=11)
    ax4.set_ylabel('$\\tilde{F}$', fontsize=11)
    ax4.set_title('Fertility Functional\n$F = N_{BH} \\times T_{int}$', fontsize=11)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved components plot to: {filepath}")

    return filepath


def print_sample_table(omega, N_BH_norm, T_int_norm, F_norm):
    """
    Print sample table for selected Omega_Lambda values.

    Parameters
    ----------
    omega : ndarray
        Omega_Lambda values
    N_BH_norm, T_int_norm, F_norm : ndarray
        Normalized quantities
    """
    sample_omegas = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    print("\n" + "=" * 65)
    print("SAMPLE TABLE: Fertility Functional Components")
    print("=" * 65)
    print(f"{'Omega_Lambda':>12} | {'N_BH_norm':>12} | {'T_int_norm':>12} | {'F_norm':>12}")
    print("-" * 65)

    for ol in sample_omegas:
        idx = np.argmin(np.abs(omega - ol))
        actual_ol = omega[idx]
        print(f"{actual_ol:>12.3f} | {N_BH_norm[idx]:>12.4f} | "
              f"{T_int_norm[idx]:>12.4f} | {F_norm[idx]:>12.4f}")

    print("-" * 65)


def print_summary(omega, F_norm):
    """
    Print summary statistics.

    Parameters
    ----------
    omega : ndarray
        Omega_Lambda values
    F_norm : ndarray
        Normalized fertility
    """
    # Find peak
    idx_peak = np.argmax(F_norm)
    omega_peak = omega[idx_peak]
    F_peak = F_norm[idx_peak]

    # Find value at our universe
    idx_07 = np.argmin(np.abs(omega - 0.7))
    omega_07 = omega[idx_07]
    F_07 = F_norm[idx_07]

    ratio = F_07 / F_peak

    print("\n" + "=" * 65)
    print("SUMMARY: Fertility Functional F(Omega_Lambda)")
    print("=" * 65)
    print(f"\nPEAK:")
    print(f"  Omega_Lambda_peak = {omega_peak:.4f}")
    print(f"  F_norm(peak) = {F_peak:.4f}")

    print(f"\nOUR UNIVERSE (Omega_Lambda ≈ 0.7):")
    print(f"  Omega_Lambda = {omega_07:.4f}")
    print(f"  F_norm(0.7) = {F_07:.4f}")

    print(f"\nRATIO:")
    print(f"  F(0.7) / F(peak) = {ratio:.4f} = {ratio:.1%}")
    print()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    print()
    print("=" * 70)
    print("v2 MODEL: Fertility Functional F(Omega_Lambda) = N_BH × T_int")
    print("=" * 70)
    print()

    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    n_bh_csv = os.path.join(script_dir, "N_BH_vs_omega_lambda_v2.csv")
    t_int_csv = os.path.join(script_dir, "T_int_vs_omega_lambda_v2.csv")

    # Check files exist
    if not os.path.exists(n_bh_csv):
        print(f"ERROR: {n_bh_csv} not found.")
        print("Please run scan_N_BH_vs_omega_lambda.py first.")
        sys.exit(1)

    if not os.path.exists(t_int_csv):
        print(f"ERROR: {t_int_csv} not found.")
        print("Please run scan_T_int_vs_omega_lambda_v2.py first.")
        sys.exit(1)

    # Load data
    print("Loading data...")
    print(f"  N_BH data: {n_bh_csv}")
    n_bh_data = load_N_BH_data(n_bh_csv)

    print(f"  T_int data: {t_int_csv}")
    t_int_data = load_T_int_data(t_int_csv)

    # Verify grids match
    print("\nVerifying Omega_Lambda grids...")
    verify_omega_grids(n_bh_data['omega'], t_int_data['omega'])

    # Extract arrays
    omega = n_bh_data['omega']
    N_BH_raw = n_bh_data['N_BH_raw']
    N_BH_norm = n_bh_data['N_BH_norm']
    T_int_raw = t_int_data['T_int_raw']
    T_int_norm = t_int_data['T_int_norm']

    # Compute fertility
    print("\nComputing fertility functional...")
    F_raw, F_norm = compute_fertility(N_BH_raw, T_int_raw)
    print(f"  F_raw range: [{np.min(F_raw):.4e}, {np.max(F_raw):.4e}]")
    print(f"  F_norm range: [{np.min(F_norm):.4f}, {np.max(F_norm):.4f}]")

    # Save CSV
    print("\nSaving results...")
    save_csv(omega, N_BH_raw, N_BH_norm, T_int_raw, T_int_norm, F_raw, F_norm)

    # Print sample table
    print_sample_table(omega, N_BH_norm, T_int_norm, F_norm)

    # Print summary
    print_summary(omega, F_norm)

    # Create plots
    print("Generating plots...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        create_main_plot(omega, F_norm)
        create_components_plot(omega, N_BH_norm, T_int_norm, F_norm)

    except ImportError:
        print("matplotlib not available, skipping plots")

    # Final interpretation
    print("=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)

    idx_peak = np.argmax(F_norm)
    omega_peak = omega[idx_peak]
    idx_07 = np.argmin(np.abs(omega - 0.7))
    ratio = F_norm[idx_07] / F_norm[idx_peak]

    print(f"""
The v2 fertility functional F(Omega_Lambda) = N_BH × T_int shows:

SHAPE: Monotonically decreasing from low to high Omega_Lambda
  - Dominated by N_BH (structure formation)
  - T_int is nearly flat (~0.5% variation)
  - Net effect: F ≈ N_BH × const

PEAK LOCATION: Omega_Lambda = {omega_peak:.3f}
  - At the lowest Omega_Lambda in the scan
  - This is where structure formation is maximized

OUR UNIVERSE: Omega_Lambda ≈ 0.7
  - F(0.7) / F(peak) = {ratio:.1%}
  - We are NOT near the fertility peak in this model

IMPLICATIONS FOR CNS:
  - If fertility = "reproductive fitness", CNS would predict
    universes should cluster near the peak (low Omega_Lambda)
  - Our universe's Omega_Lambda = 0.7 is far from optimal
  - This CHALLENGES simple CNS predictions

CAVEATS:
  - T_int model may underestimate Omega_Lambda dependence
  - The v1 toy model showed more T_int variation
  - Additional factors (star formation, metallicity) not included
  - The "baby universe" mechanism remains speculative
""")

    print("Done!")


if __name__ == "__main__":
    main()
