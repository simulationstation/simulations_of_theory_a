#!/usr/bin/env python3
"""
scan_N_BH_v3_vs_omega_lambda.py

Driver script to scan N_BH_v3(Omega_Lambda) using Press-Schechter
collapsed fraction combined with cosmic star formation history.

Outputs:
- CSV file: N_BH_v3_vs_omega_lambda.csv
- PNG plot: N_BH_v3_vs_omega_lambda.png
- Sample table to stdout

Usage:
    python3 scan_N_BH_v3_vs_omega_lambda.py
"""

import numpy as np
import os
import sys
import csv
import time

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from N_BH_v3_press_schechter_sfr import N_BH_v3, sfr_baseline


def scan_N_BH_v3(omega_min=0.01, omega_max=0.95, n_points=35,
                  z_max=10.0, nz=300):
    """
    Scan N_BH_v3 over a range of Omega_Lambda values.

    Parameters
    ----------
    omega_min : float
        Minimum Omega_Lambda
    omega_max : float
        Maximum Omega_Lambda
    n_points : int
        Number of scan points
    z_max : float
        Maximum redshift for integration
    nz : int
        Number of redshift points

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
    print(f"  z_max = {z_max}, nz = {nz}")
    print()

    start = time.time()

    N_BH_raw = np.zeros(n_points)

    for i, ol in enumerate(omega_array):
        if i % 5 == 0 or i == n_points - 1:
            print(f"  Processing Omega_Lambda = {ol:.3f} ({i+1}/{n_points})...",
                  end='\r')

        N_BH_raw[i] = N_BH_v3(ol, z_max=z_max, nz=nz)

    print()  # Newline after progress
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f} seconds.")

    # Normalize to max = 1
    N_max = np.max(N_BH_raw)
    N_BH_norm = N_BH_raw / N_max if N_max > 0 else N_BH_raw

    return omega_array, N_BH_raw, N_BH_norm


def save_csv(omega_array, N_BH_raw, N_BH_norm,
             filename="N_BH_v3_vs_omega_lambda.csv"):
    """
    Save results to CSV file.

    Parameters
    ----------
    omega_array : ndarray
        Omega_Lambda values
    N_BH_raw : ndarray
        Raw N_BH values
    N_BH_norm : ndarray
        Normalized N_BH values
    filename : str
        Output filename
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Omega_Lambda', 'N_BH_v3_raw', 'N_BH_v3_norm'])
        for i in range(len(omega_array)):
            writer.writerow([
                f"{omega_array[i]:.6f}",
                f"{N_BH_raw[i]:.6e}",
                f"{N_BH_norm[i]:.6f}"
            ])

    print(f"\nSaved CSV to: {filepath}")
    return filepath


def create_plot(omega_array, N_BH_norm, filename="N_BH_v3_vs_omega_lambda.png"):
    """
    Create and save plot of N_BH_v3_norm vs Omega_Lambda.

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

    fig, ax = plt.subplots(figsize=(11, 8))

    # Main curve
    ax.plot(omega_array, N_BH_norm, 'b-', linewidth=2.5,
            label='$\\tilde{N}_{BH}^{v3}(\\Omega_\\Lambda)$')
    ax.scatter(omega_array, N_BH_norm, c='blue', s=50, zorder=5,
               edgecolors='white', linewidths=0.5)
    ax.fill_between(omega_array, 0, N_BH_norm, alpha=0.15, color='blue')

    # Find and mark peak
    idx_peak = np.argmax(N_BH_norm)
    omega_peak = omega_array[idx_peak]
    N_peak = N_BH_norm[idx_peak]

    ax.axvline(omega_peak, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Peak at $\\Omega_\\Lambda = {omega_peak:.3f}$')
    ax.plot(omega_peak, N_peak, 'ro', markersize=14,
            markeredgecolor='darkred', markeredgewidth=2, zorder=6)

    # Annotate peak
    ax.annotate(f'Peak\n$\\Omega_\\Lambda = {omega_peak:.3f}$',
                xy=(omega_peak, N_peak),
                xytext=(omega_peak + 0.12, N_peak - 0.12),
                fontsize=11, fontweight='bold', color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose',
                          edgecolor='darkred', alpha=0.9))

    # Mark our universe (Omega_Lambda = 0.7)
    ax.axvline(0.7, color='green', linestyle=':', linewidth=2.5,
               label='Our Universe ($\\Omega_\\Lambda \\approx 0.7$)')
    idx_07 = np.argmin(np.abs(omega_array - 0.7))
    N_07 = N_BH_norm[idx_07]
    ax.plot(0.7, N_07, 'g^', markersize=12,
            markeredgecolor='darkgreen', markeredgewidth=2, zorder=6)

    # Annotate our universe
    ax.annotate(f'Our Universe\n$\\Omega_\\Lambda = 0.7$\n$\\tilde{{N}}_{{BH}} = {N_07:.3f}$',
                xy=(0.7, N_07),
                xytext=(0.78, N_07 + 0.15),
                fontsize=10, color='darkgreen',
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew',
                          edgecolor='darkgreen', alpha=0.9))

    # Labels
    ax.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=14)
    ax.set_ylabel('$\\tilde{N}_{BH}^{v3} = N_{BH} / \\max(N_{BH})$', fontsize=14)
    ax.set_title('v3 Model: Black Hole Abundance vs $\\Omega_\\Lambda$\n'
                 '(Press-Schechter + Cosmic Star Formation History)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)

    # Add physics note
    ax.text(0.02, 0.02,
            'Integration: $N_{BH} \\propto \\int dz\\, \\dot{n}_{BH}(z)\\, '
            '\\frac{dV_c}{dz}\\, \\left|\\frac{dt}{dz}\\right|$',
            transform=ax.transAxes, fontsize=10, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {filepath}")

    plt.show()

    return filepath


def print_sample_table(omega_array, N_BH_raw, N_BH_norm):
    """
    Print sample table for selected Omega_Lambda values.

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

    print("\n" + "=" * 55)
    print("SAMPLE TABLE: N_BH_v3 vs Omega_Lambda")
    print("=" * 55)
    print(f"{'Omega_Lambda':>12} | {'N_BH_v3 (raw)':>14} | {'N_BH_v3 (norm)':>14}")
    print("-" * 55)

    for ol in sample_omegas:
        idx = np.argmin(np.abs(omega_array - ol))
        actual_ol = omega_array[idx]
        print(f"{actual_ol:>12.3f} | {N_BH_raw[idx]:>14.4e} | {N_BH_norm[idx]:>14.4f}")

    print("-" * 55)

    # Peak info
    idx_peak = np.argmax(N_BH_norm)
    print(f"\nPeak: Omega_Lambda = {omega_array[idx_peak]:.4f}, "
          f"N_BH_v3_norm = {N_BH_norm[idx_peak]:.4f}")

    # Our universe
    idx_07 = np.argmin(np.abs(omega_array - 0.7))
    print(f"Our Universe (Omega_Lambda ~ 0.7): N_BH_v3_norm = {N_BH_norm[idx_07]:.4f}")

    # Ratio
    ratio = N_BH_norm[idx_07] / N_BH_norm[idx_peak]
    print(f"Ratio N_BH(0.7) / N_BH(peak) = {ratio:.1%}")
    print()


def main():
    """Main entry point."""
    print()
    print("=" * 65)
    print("v3 MODEL: N_BH(Omega_Lambda) with Cosmic Star Formation History")
    print("=" * 65)
    print()

    print("Physics:")
    print("  - Press-Schechter collapsed fraction F_coll(z, Omega_Lambda)")
    print("  - Cosmic SFR: SFR0(z) ~ (1+z)^a / (1 + ((1+z)/b)^c)")
    print("  - BH rate: n_dot_BH ~ SFR0 * F_coll / F_coll_ref")
    print("  - Integral over redshift and comoving volume")
    print()

    # Run scan
    omega_array, N_BH_raw, N_BH_norm = scan_N_BH_v3(
        omega_min=0.01,
        omega_max=0.95,
        n_points=35,
        z_max=10.0,
        nz=300
    )

    # Save CSV
    save_csv(omega_array, N_BH_raw, N_BH_norm)

    # Print sample table
    print_sample_table(omega_array, N_BH_raw, N_BH_norm)

    # Create plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        create_plot(omega_array, N_BH_norm)
    except ImportError:
        print("matplotlib not available, skipping plot")

    # Physical interpretation
    idx_peak = np.argmax(N_BH_norm)
    idx_07 = np.argmin(np.abs(omega_array - 0.7))
    ratio = N_BH_norm[idx_07] / N_BH_norm[idx_peak]

    print("=" * 65)
    print("PHYSICAL INTERPRETATION")
    print("=" * 65)
    print(f"""
The v3 model integrates BH formation over cosmic history:

N_BH(OL) ~ integral dz * SFR0(z) * [F_coll(z,OL)/F_coll(z,0.7)] * dVc/dz * |dt/dz|

KEY FEATURES:
1. SFR0(z) peaks at z ~ 1.5-2 (cosmic noon)
2. F_coll(z, OL) encodes structure formation suppression
3. dVc/dz weights by comoving volume
4. |dt/dz| converts to cosmic time

RESULTS:
- Peak at Omega_Lambda = {omega_array[idx_peak]:.3f}
- Our universe (OL = 0.7): N_BH_norm = {N_BH_norm[idx_07]:.3f}
- Ratio to peak: {ratio:.1%}

COMPARISON TO v2:
- v2: Single-redshift halo count, monotonic decrease
- v3: Integrated over z with SFR weighting, may show non-monotonic behavior
""")

    print("Done!")


if __name__ == "__main__":
    main()
