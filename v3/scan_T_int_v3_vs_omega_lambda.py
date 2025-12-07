#!/usr/bin/env python3
"""
scan_T_int_v3_vs_omega_lambda.py

Driver script to scan T_int_v3(Omega_Lambda) using:
1. BH formation time from v3 formation rate distribution
2. Post-formation flux from v1 cosmological flux model
3. Power-law scaling T_int ~ L_eff^{-p}

Outputs:
- CSV file: T_int_v3_vs_omega_lambda.csv
- PNG plot: T_int_v3_vs_omega_lambda.png
- Sample table to stdout

Usage:
    python3 scan_T_int_v3_vs_omega_lambda.py
"""

import numpy as np
import os
import sys
import csv
import time

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from T_int_v3_from_formation_history import scan_T_int_v3_vs_omega_lambda


def save_csv(omega_array, t_form_array, L_eff_array, T_int_raw, T_int_norm,
             filename="T_int_v3_vs_omega_lambda.csv"):
    """
    Save results to CSV file.

    Parameters
    ----------
    omega_array : ndarray
        Omega_Lambda values
    t_form_array : ndarray
        Formation times [Gyr]
    L_eff_array : ndarray
        Effective post-formation flux [W]
    T_int_raw : ndarray
        Raw T_int values
    T_int_norm : ndarray
        Normalized T_int values
    filename : str
        Output filename
    """
    filepath = os.path.join(os.path.dirname(__file__), filename)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Omega_Lambda', 't_form_Gyr', 'L_eff', 'T_int_raw', 'T_int_norm'])
        for i in range(len(omega_array)):
            writer.writerow([
                f"{omega_array[i]:.6f}",
                f"{t_form_array[i]:.4f}",
                f"{L_eff_array[i]:.6e}",
                f"{T_int_raw[i]:.6e}",
                f"{T_int_norm[i]:.6f}"
            ])

    print(f"\nSaved CSV to: {filepath}")
    return filepath


def create_plot(omega_array, t_form_array, L_eff_array, T_int_norm,
                filename="T_int_v3_vs_omega_lambda.png"):
    """
    Create and save plot of T_int_v3_norm and L_eff vs Omega_Lambda.

    Parameters
    ----------
    omega_array : ndarray
        Omega_Lambda values
    t_form_array : ndarray
        Formation times [Gyr]
    L_eff_array : ndarray
        Effective flux [W]
    T_int_norm : ndarray
        Normalized T_int values
    filename : str
        Output filename
    """
    import matplotlib.pyplot as plt

    filepath = os.path.join(os.path.dirname(__file__), filename)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)

    # -------------------------------------------------------------------------
    # Top panel: T_int_norm vs Omega_Lambda
    # -------------------------------------------------------------------------
    ax1 = axes[0]

    ax1.plot(omega_array, T_int_norm, 'b-', linewidth=2.5,
             label='$\\tilde{T}_{int}^{v3}(\\Omega_\\Lambda)$')
    ax1.scatter(omega_array, T_int_norm, c='blue', s=50, zorder=5,
                edgecolors='white', linewidths=0.5)
    ax1.fill_between(omega_array, 0, T_int_norm, alpha=0.15, color='blue')

    # Find and mark peak
    idx_peak = np.argmax(T_int_norm)
    omega_peak = omega_array[idx_peak]
    T_peak = T_int_norm[idx_peak]

    ax1.axvline(omega_peak, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'Peak at $\\Omega_\\Lambda = {omega_peak:.3f}$')
    ax1.plot(omega_peak, T_peak, 'ro', markersize=12,
             markeredgecolor='darkred', markeredgewidth=2, zorder=6)

    # Mark our universe
    ax1.axvline(0.7, color='green', linestyle=':', linewidth=2.5,
                label='Our Universe ($\\Omega_\\Lambda \\approx 0.7$)')
    idx_07 = np.argmin(np.abs(omega_array - 0.7))
    T_07 = T_int_norm[idx_07]
    ax1.plot(0.7, T_07, 'g^', markersize=10,
             markeredgecolor='darkgreen', markeredgewidth=2, zorder=6)

    ax1.annotate(f'Our Universe\n$\\tilde{{T}}_{{int}} = {T_07:.3f}$',
                 xy=(0.7, T_07),
                 xytext=(0.78, T_07 + 0.15),
                 fontsize=10, color='darkgreen',
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew',
                           edgecolor='darkgreen', alpha=0.9))

    ax1.set_ylabel('$\\tilde{T}_{int}^{v3} = T_{int} / \\max(T_{int})$', fontsize=13)
    ax1.set_title('v3 Model: Interior Lifetime vs $\\Omega_\\Lambda$\n'
                  '(Formation history + post-formation flux)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 1.15)

    # Physics note
    ax1.text(0.02, 0.02,
             '$T_{int} \\propto L_{eff}^{-0.95}$ (mass inflation scaling)',
             transform=ax1.transAxes, fontsize=10, va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # -------------------------------------------------------------------------
    # Bottom panel: L_eff and t_form vs Omega_Lambda
    # -------------------------------------------------------------------------
    ax2 = axes[1]

    # Normalize L_eff for plotting
    L_eff_norm = L_eff_array / np.max(L_eff_array)

    ax2.plot(omega_array, L_eff_norm, 'r-', linewidth=2,
             label='$L_{eff}$ (normalized)', marker='o', markersize=4)

    # Normalize t_form for overlay
    t_form_norm = t_form_array / np.max(t_form_array)
    ax2.plot(omega_array, t_form_norm, 'purple', linewidth=2, linestyle='--',
             label='$t_{form}$ (normalized)', marker='s', markersize=4)

    ax2.axvline(0.7, color='green', linestyle=':', linewidth=2.5, alpha=0.7)

    ax2.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=13)
    ax2.set_ylabel('Normalized values', fontsize=13)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.15)

    # Add actual value annotations
    ax2.text(0.98, 0.95,
             f'$L_{{eff}}$ range: [{np.min(L_eff_array):.2e}, {np.max(L_eff_array):.2e}] W\n'
             f'$t_{{form}}$ range: [{np.min(t_form_array):.1f}, {np.max(t_form_array):.1f}] Gyr',
             transform=ax2.transAxes, fontsize=9, va='top', ha='right',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.8))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved plot to: {filepath}")

    plt.show()

    return filepath


def print_sample_table(omega_array, t_form_array, L_eff_array, T_int_raw, T_int_norm):
    """
    Print sample table for selected Omega_Lambda values.

    Parameters
    ----------
    omega_array : ndarray
        All Omega_Lambda values
    t_form_array : ndarray
        Formation times [Gyr]
    L_eff_array : ndarray
        Effective flux [W]
    T_int_raw : ndarray
        Raw T_int values
    T_int_norm : ndarray
        Normalized T_int values
    """
    sample_omegas = [0.1, 0.3, 0.5, 0.7, 0.9]

    print("\n" + "=" * 80)
    print("SAMPLE TABLE: T_int_v3 vs Omega_Lambda")
    print("=" * 80)
    print(f"{'OL':>8} | {'t_form [Gyr]':>12} | {'L_eff [W]':>14} | {'T_int_raw':>12} | {'T_int_norm':>10}")
    print("-" * 80)

    for ol in sample_omegas:
        idx = np.argmin(np.abs(omega_array - ol))
        actual_ol = omega_array[idx]
        print(f"{actual_ol:>8.3f} | {t_form_array[idx]:>12.2f} | {L_eff_array[idx]:>14.4e} | "
              f"{T_int_raw[idx]:>12.4e} | {T_int_norm[idx]:>10.4f}")

    print("-" * 80)

    # Peak info
    idx_peak = np.argmax(T_int_norm)
    print(f"\nPeak: Omega_Lambda = {omega_array[idx_peak]:.4f}, "
          f"T_int_v3_norm = {T_int_norm[idx_peak]:.4f}")

    # Our universe
    idx_07 = np.argmin(np.abs(omega_array - 0.7))
    print(f"Our Universe (Omega_Lambda ~ 0.7): T_int_v3_norm = {T_int_norm[idx_07]:.4f}")

    # Ratio
    ratio = T_int_norm[idx_07] / T_int_norm[idx_peak]
    print(f"Ratio T_int(0.7) / T_int(peak) = {ratio:.1%}")
    print()


def main():
    """Main entry point."""
    print()
    print("=" * 75)
    print("v3 MODEL: T_int(Omega_Lambda) from Formation History + Post-Formation Flux")
    print("=" * 75)
    print()

    print("Physics:")
    print("  - BH formation PDF: p_BH(z) ~ n_dot_BH(z) * |dt/dz|")
    print("  - Mean formation redshift: <z_form> = integral z * p_BH(z) dz")
    print("  - Formation time: t_form = cosmic_time(<z_form>)")
    print("  - Effective flux: L_eff = time-averaged L_in from t_form to t_iso")
    print("  - Interior lifetime: T_int ~ L_eff^{-0.95} (mass inflation scaling)")
    print()

    # Use same grid as N_BH_v3 scan
    omega_min = 0.01
    omega_max = 0.95
    n_points = 35

    omega_array = np.linspace(omega_min, omega_max, n_points)

    start = time.time()

    # Run scan
    omega_array, t_form_array, L_eff_array, T_int_raw, T_int_norm = \
        scan_T_int_v3_vs_omega_lambda(
            omega_array,
            p=0.95,
            t_max_Gyr=150.0,
            verbose=True
        )

    elapsed = time.time() - start
    print(f"Scan completed in {elapsed:.1f} seconds.")

    # Save CSV
    save_csv(omega_array, t_form_array, L_eff_array, T_int_raw, T_int_norm)

    # Print sample table
    print_sample_table(omega_array, t_form_array, L_eff_array, T_int_raw, T_int_norm)

    # Create plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        create_plot(omega_array, t_form_array, L_eff_array, T_int_norm)
    except ImportError:
        print("matplotlib not available, skipping plot")

    # Physical interpretation
    idx_peak = np.argmax(T_int_norm)
    idx_07 = np.argmin(np.abs(omega_array - 0.7))
    ratio = T_int_norm[idx_07] / T_int_norm[idx_peak]

    print("=" * 75)
    print("PHYSICAL INTERPRETATION")
    print("=" * 75)
    print(f"""
The v3 interior lifetime model combines formation history with flux physics:

1. BH FORMATION TIME:
   - Higher Omega_Lambda → delayed structure formation → later BH formation
   - t_form range: [{np.min(t_form_array):.1f}, {np.max(t_form_array):.1f}] Gyr

2. POST-FORMATION FLUX:
   - BHs that form later see lower ambient radiation (cosmic expansion dilutes flux)
   - L_eff range: [{np.min(L_eff_array):.2e}, {np.max(L_eff_array):.2e}] W

3. INTERIOR LIFETIME:
   - T_int ~ L_eff^{{-0.95}} (mass inflation scaling)
   - Lower flux → longer interior lifetime

RESULTS:
- Peak T_int at Omega_Lambda = {omega_array[idx_peak]:.3f}
- Our universe (OL = 0.7): T_int_norm = {T_int_norm[idx_07]:.3f}
- Ratio to peak: {ratio:.1%}

KEY PHYSICS:
- Higher Omega_Lambda generally means LATER BH formation and LOWER flux
- This leads to LONGER interior lifetimes at higher Omega_Lambda
- The competition between these effects determines the T_int curve shape
""")

    print("Done!")


if __name__ == "__main__":
    main()
