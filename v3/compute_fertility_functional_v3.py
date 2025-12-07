#!/usr/bin/env python3
"""
compute_fertility_functional_v3.py

Combine N_BH_v3_density and T_int_v3 into the v3 fertility functional:

    F_v3(Omega_Lambda) = N_BH_norm(Omega_Lambda) * T_int_norm(Omega_Lambda)

This represents the "reproductive fitness" of a universe with a given
cosmological constant, combining:
- N_BH: How many black holes form (potential "offspring")
- T_int: How long their interiors survive (time for physics to occur)

Outputs:
- CSV: F_v3_vs_omega_lambda.csv
- PNG: F_v3_vs_omega_lambda.png
- Sample table to stdout

Usage:
    python3 compute_fertility_functional_v3.py
"""

import numpy as np
import csv
import os
import sys

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_csv(filepath):
    """Load CSV file and return header and data as dict of arrays."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    data = {}
    for key in rows[0].keys():
        data[key] = np.array([float(row[key]) for row in rows])

    return data


def load_data():
    """
    Load N_BH and T_int CSVs and merge on omega_lambda.

    Returns
    -------
    data : dict
        Dictionary with arrays for omega_lambda, N_BH_norm, T_int_norm, etc.
    """
    # Load N_BH data
    nbh_path = os.path.join(SCRIPT_DIR, "N_BH_v3_density_vs_omega_lambda.csv")
    nbh_data = load_csv(nbh_path)

    # Load T_int data
    tint_path = os.path.join(SCRIPT_DIR, "T_int_v3_vs_omega_lambda.csv")
    tint_data = load_csv(tint_path)

    print(f"Loaded N_BH data: {len(nbh_data['Omega_Lambda'])} rows")
    print(f"Loaded T_int data: {len(tint_data['Omega_Lambda'])} rows")

    # Verify grids match (round for comparison)
    nbh_omega = np.round(nbh_data['Omega_Lambda'], 6)
    tint_omega = np.round(tint_data['Omega_Lambda'], 6)

    if not np.allclose(nbh_omega, tint_omega):
        raise ValueError("Omega_Lambda grids do not match between CSVs!")

    # Build merged data dict
    data = {
        'omega_lambda': nbh_omega,
        'N_BH_norm': nbh_data['N_BH_v3_density_norm'],
        'T_int_norm': tint_data['T_int_norm'],
        't_form_Gyr': tint_data['t_form_Gyr'],
        'L_eff': tint_data['L_eff'],
    }

    print(f"Merged data: {len(data['omega_lambda'])} rows")

    return data


def compute_fertility(data):
    """
    Compute fertility functional F = N_BH_norm * T_int_norm.

    Parameters
    ----------
    data : dict
        Data dictionary with N_BH_norm and T_int_norm arrays

    Returns
    -------
    data : dict
        Input data with added F_raw and F_norm arrays
    """
    # F_raw = N_BH_norm * T_int_norm
    data['F_raw'] = data['N_BH_norm'] * data['T_int_norm']

    # F_norm = F_raw / max(F_raw)
    F_max = np.max(data['F_raw'])
    data['F_norm'] = data['F_raw'] / F_max

    print(f"\nFertility functional computed:")
    print(f"  F_raw range: [{np.min(data['F_raw']):.6f}, {np.max(data['F_raw']):.6f}]")
    print(f"  F_norm range: [{np.min(data['F_norm']):.6f}, {np.max(data['F_norm']):.6f}]")

    return data


def save_csv(data, filename="F_v3_vs_omega_lambda.csv"):
    """
    Save fertility results to CSV.

    Parameters
    ----------
    data : dict
        Data dictionary with fertility results
    filename : str
        Output filename
    """
    filepath = os.path.join(SCRIPT_DIR, filename)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Omega_Lambda', 'N_BH_norm', 'T_int_norm', 'F_raw', 'F_norm'])
        for i in range(len(data['omega_lambda'])):
            writer.writerow([
                f"{data['omega_lambda'][i]:.6f}",
                f"{data['N_BH_norm'][i]:.6f}",
                f"{data['T_int_norm'][i]:.6f}",
                f"{data['F_raw'][i]:.6f}",
                f"{data['F_norm'][i]:.6f}"
            ])

    print(f"\nSaved CSV to: {filepath}")
    return filepath


def create_plot(data, filename="F_v3_vs_omega_lambda.png"):
    """
    Create plot of F_norm vs Omega_Lambda.

    Parameters
    ----------
    data : dict
        Data dictionary with fertility results
    filename : str
        Output filename
    """
    import matplotlib.pyplot as plt

    filepath = os.path.join(SCRIPT_DIR, filename)

    omega = data['omega_lambda']
    F_norm = data['F_norm']
    N_BH_norm = data['N_BH_norm']
    T_int_norm = data['T_int_norm']

    fig, ax = plt.subplots(figsize=(12, 8))

    # Main F_norm curve
    ax.plot(omega, F_norm, 'b-', linewidth=3,
            label='$\\tilde{F}_{v3}(\\Omega_\\Lambda) = \\tilde{N}_{BH} \\times \\tilde{T}_{int}$')
    ax.scatter(omega, F_norm, c='blue', s=60, zorder=5,
               edgecolors='white', linewidths=0.5)
    ax.fill_between(omega, 0, F_norm, alpha=0.15, color='blue')

    # Component curves (lighter)
    ax.plot(omega, N_BH_norm, 'g--', linewidth=1.5, alpha=0.6,
            label='$\\tilde{N}_{BH}$ (density)')
    ax.plot(omega, T_int_norm, 'r--', linewidth=1.5, alpha=0.6,
            label='$\\tilde{T}_{int}$')

    # Find and mark peak
    idx_peak = np.argmax(F_norm)
    omega_peak = omega[idx_peak]
    F_peak = F_norm[idx_peak]

    ax.axvline(omega_peak, color='darkblue', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Peak at $\\Omega_\\Lambda = {omega_peak:.3f}$')
    ax.plot(omega_peak, F_peak, 'o', markersize=16, color='darkblue',
            markeredgecolor='navy', markeredgewidth=2, zorder=6)

    # Annotate peak
    ax.annotate(f'Peak\n$\\Omega_\\Lambda = {omega_peak:.3f}$',
                xy=(omega_peak, F_peak),
                xytext=(omega_peak - 0.15, F_peak - 0.18),
                fontsize=11, fontweight='bold', color='darkblue',
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightsteelblue',
                          edgecolor='darkblue', alpha=0.9))

    # Mark our universe (Omega_Lambda = 0.7)
    ax.axvline(0.7, color='green', linestyle=':', linewidth=2.5,
               label='Our Universe ($\\Omega_\\Lambda \\approx 0.7$)')
    idx_07 = np.argmin(np.abs(omega - 0.7))
    F_07 = F_norm[idx_07]
    ax.plot(0.7, F_07, '^', markersize=14, color='green',
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=6)

    # Annotate our universe
    ratio_pct = 100 * F_07 / F_peak
    ax.annotate(f'Our Universe\n$\\Omega_\\Lambda = 0.7$\n$\\tilde{{F}} = {F_07:.3f}$\n({ratio_pct:.1f}% of peak)',
                xy=(0.7, F_07),
                xytext=(0.78, F_07 + 0.22),
                fontsize=10, color='darkgreen',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew',
                          edgecolor='darkgreen', alpha=0.9))

    # Labels
    ax.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=14)
    ax.set_ylabel('Normalized Fertility $\\tilde{F}_{v3}$', fontsize=14)
    ax.set_title('v3 Fertility Functional: $F(\\Omega_\\Lambda) = N_{BH} \\times T_{int}$\n'
                 '(BH density per comoving volume $\\times$ interior lifetime)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)

    # Physics note
    ax.text(0.02, 0.02,
            '$N_{BH}$: BH number density (time integral of formation rate)\n'
            '$T_{int}$: Interior lifetime ($\\propto L_{eff}^{-0.95}$, mass inflation)',
            transform=ax.transAxes, fontsize=9, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {filepath}")

    plt.show()

    return filepath


def print_sample_table(data):
    """
    Print sample table for selected Omega_Lambda values.

    Parameters
    ----------
    data : dict
        Data dictionary with fertility results
    """
    sample_omegas = [0.1, 0.3, 0.5, 0.7, 0.9]

    omega = data['omega_lambda']
    N_BH_norm = data['N_BH_norm']
    T_int_norm = data['T_int_norm']
    F_norm = data['F_norm']

    print("\n" + "=" * 65)
    print("SAMPLE TABLE: v3 Fertility Functional")
    print("=" * 65)
    print(f"{'Omega_Lambda':>12} | {'N_BH_norm':>10} | {'T_int_norm':>10} | {'F_norm':>10}")
    print("-" * 65)

    for ol in sample_omegas:
        idx = np.argmin(np.abs(omega - ol))
        actual_ol = omega[idx]
        print(f"{actual_ol:>12.3f} | {N_BH_norm[idx]:>10.4f} | {T_int_norm[idx]:>10.4f} | {F_norm[idx]:>10.4f}")

    print("-" * 65)

    # Peak info
    idx_peak = np.argmax(F_norm)
    print(f"\nPeak: Omega_Lambda = {omega[idx_peak]:.4f}, F_norm = {F_norm[idx_peak]:.4f}")

    # Our universe
    idx_07 = np.argmin(np.abs(omega - 0.7))
    print(f"Our Universe (Omega_Lambda ~ 0.7): F_norm = {F_norm[idx_07]:.4f}")

    # Ratio
    ratio = F_norm[idx_07] / F_norm[idx_peak]
    print(f"Ratio F(0.7) / F(peak) = {ratio:.1%}")
    print()


def main():
    """Main entry point."""
    print()
    print("=" * 70)
    print("v3 FERTILITY FUNCTIONAL: F(Omega_Lambda) = N_BH * T_int")
    print("=" * 70)
    print()

    print("Loading pre-computed data...")

    # Load and merge data
    data = load_data()

    # Compute fertility
    data = compute_fertility(data)

    # Save CSV
    save_csv(data)

    # Print sample table
    print_sample_table(data)

    # Create plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        create_plot(data)
    except ImportError:
        print("matplotlib not available, skipping plot")

    # Physical interpretation
    omega = data['omega_lambda']
    F_norm = data['F_norm']

    idx_peak = np.argmax(F_norm)
    idx_07 = np.argmin(np.abs(omega - 0.7))
    ratio = F_norm[idx_07] / F_norm[idx_peak]

    print("=" * 70)
    print("PHYSICAL INTERPRETATION")
    print("=" * 70)
    print(f"""
The v3 fertility functional combines two factors:

1. N_BH (BH number density):
   - Integrates BH formation rate over cosmic time
   - Increases with Omega_Lambda (more time for formation)

2. T_int (interior lifetime):
   - Based on post-formation flux and mass inflation scaling
   - Increases with Omega_Lambda (later formation = lower flux)

RESULT: F = N_BH * T_int

Both factors increase with Omega_Lambda in this model, so F is
monotonically increasing, peaking at the highest Omega_Lambda value.

KEY RESULTS:
- Peak at Omega_Lambda = {omega[idx_peak]:.3f}
- F_norm(0.7) = {F_norm[idx_07]:.3f}
- Ratio F(0.7)/F(peak) = {ratio:.1%}

INTERPRETATION:
Our universe (Omega_Lambda ~ 0.7) achieves {ratio:.1%} of the maximum
possible fertility in this v3 model.
""")

    print("Done!")


if __name__ == "__main__":
    main()
