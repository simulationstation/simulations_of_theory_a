#!/usr/bin/env python3
"""
T_int_vs_lambda_toy.py

Combined script that links cosmology (Program 1) with black hole interior
dynamics (Program 2) to compute effective interior lifetime as a function
of the cosmological constant Omega_Lambda.

Pipeline:
  Omega_Lambda -> flux history L_in(t; Omega_Lambda) [Program 1]
               -> effective L_in
               -> T_int(L_in) scaling [Program 2]
               -> T_int(Omega_Lambda) curve

Author: Combined from Programs 1 and 2
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add module directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '2'))

# Import from Program 1 (cosmology)
from cosmo_flux_lambda import (
    compute_flux_history_and_isolation,
    get_default_L_crit,
    M_sun
)

# Import from Program 2 (black hole interior)
from bh_interior_toy import fit_scaling_law


def main():
    print("=" * 70)
    print("Toy Model: Interior Lifetime vs Cosmological Constant")
    print("=" * 70)
    print()
    print("Pipeline: Omega_Lambda -> L_in(t) -> L_eff -> T_int(L_eff)")
    print()

    # =========================================================================
    # Step 1: Fit the scaling law T_int(L_in) from Program 2
    # =========================================================================
    print("-" * 70)
    print("Step 1: Fitting T_int(L_in) scaling law from toy interior model")
    print("-" * 70)

    # Use the fit_scaling_law function from Program 2
    # This runs the mass inflation simulations and fits T_int = A * L_in^(-p)
    A, p, L_fit, T_fit = fit_scaling_law(
        L_in_values=[0.01, 0.02, 0.05, 0.1, 0.2],
        N_u=200, N_v=200, U_max=10.0, V_max=10.0
    )

    print(f"\nScaling law fit results:")
    print(f"  T_int = A * L_in^(-p)")
    print(f"  A = {A:.6f}")
    print(f"  p = {p:.6f}")
    print()
    print(f"  Data points used in fit:")
    for L, T in zip(L_fit, T_fit):
        print(f"    L_in = {L:.3f}  ->  T_int = {T:.4f}")
    print()

    # =========================================================================
    # Step 2: Set up cosmological parameters
    # =========================================================================
    print("-" * 70)
    print("Step 2: Setting up cosmology parameters")
    print("-" * 70)

    # Black hole parameters
    M_bh = 1e8 * M_sun  # 10^8 solar masses

    # Get default L_crit
    L_crit, L_ref = get_default_L_crit(M_bh)

    print(f"\nBlack hole mass: {M_bh:.2e} kg ({M_bh/M_sun:.0e} M_sun)")
    print(f"Reference flux at t=0.5 Gyr: {L_ref:.2e} W")
    print(f"Critical threshold L_crit: {L_crit:.2e} W")
    print()

    # Omega_Lambda values to scan
    omega_lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # =========================================================================
    # Step 3: Compute T_int(Omega_Lambda)
    # =========================================================================
    print("-" * 70)
    print("Step 3: Computing T_int for each Omega_Lambda")
    print("-" * 70)
    print()

    results = []

    for Omega_Lambda in omega_lambda_values:
        print(f"  Processing Omega_Lambda = {Omega_Lambda}...", end=" ", flush=True)

        # Get flux history and isolation time from Program 1
        t_array_Gyr, L_in_array_W, t_iso_Gyr = compute_flux_history_and_isolation(
            Omega_Lambda, M_bh, L_crit,
            t_min_Gyr=0.5, t_max_Gyr=200.0, n_t=500
        )

        # Compute effective flux: average from t_min to t_iso
        if t_iso_Gyr is not None:
            mask = t_array_Gyr <= t_iso_Gyr
            L_eff = np.nanmean(L_in_array_W[mask])
        else:
            # If no isolation, use average over first 50 Gyr
            mask = t_array_Gyr <= 50.0
            L_eff = np.nanmean(L_in_array_W[mask])
            t_iso_Gyr = np.nan

        # Also compute flux at t=5 Gyr for comparison
        idx_5Gyr = np.argmin(np.abs(t_array_Gyr - 5.0))
        L_at_5Gyr = L_in_array_W[idx_5Gyr]

        # Normalize L_eff to the toy model scale
        # The toy model uses L_in ~ O(0.01 - 0.2)
        # We need to map the physical flux to this scale
        # Use L_ref as the reference scale: L_toy = L_physical / L_ref * scale_factor
        scale_factor = 0.1  # Map L_ref -> 0.1 in toy units
        L_eff_toy = L_eff / L_ref * scale_factor
        L_5Gyr_toy = L_at_5Gyr / L_ref * scale_factor

        # Compute T_int using the scaling law
        T_int_eff = A * (L_eff_toy ** (-p))
        T_int_5Gyr = A * (L_5Gyr_toy ** (-p))

        results.append({
            'Omega_Lambda': Omega_Lambda,
            't_iso_Gyr': t_iso_Gyr,
            'L_eff_W': L_eff,
            'L_eff_toy': L_eff_toy,
            'L_5Gyr_toy': L_5Gyr_toy,
            'T_int_eff': T_int_eff,
            'T_int_5Gyr': T_int_5Gyr,
            't_array': t_array_Gyr,
            'L_array': L_in_array_W
        })

        print(f"t_iso = {t_iso_Gyr:.1f} Gyr, T_int = {T_int_eff:.4f}")

    print()

    # =========================================================================
    # Step 4: Print summary table
    # =========================================================================
    print("-" * 70)
    print("Step 4: Summary of Results")
    print("-" * 70)
    print()
    print(f"{'Omega_Lambda':>12} | {'t_iso [Gyr]':>12} | {'L_eff (toy)':>12} | {'T_int (eff)':>12}")
    print("-" * 55)
    for r in results:
        t_iso_str = f"{r['t_iso_Gyr']:.2f}" if not np.isnan(r['t_iso_Gyr']) else "N/A"
        print(f"{r['Omega_Lambda']:>12.2f} | {t_iso_str:>12} | {r['L_eff_toy']:>12.4f} | {r['T_int_eff']:>12.4f}")
    print()

    # =========================================================================
    # Step 5: Generate plots
    # =========================================================================
    print("-" * 70)
    print("Step 5: Generating plots")
    print("-" * 70)

    # Extract data for plotting
    OL_array = np.array([r['Omega_Lambda'] for r in results])
    t_iso_array = np.array([r['t_iso_Gyr'] for r in results])
    T_int_eff_array = np.array([r['T_int_eff'] for r in results])
    T_int_5Gyr_array = np.array([r['T_int_5Gyr'] for r in results])
    L_eff_toy_array = np.array([r['L_eff_toy'] for r in results])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # -------------------------------------------------------------------------
    # Plot 1: T_int vs Omega_Lambda
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.semilogy(OL_array, T_int_eff_array, 'bo-', markersize=10, linewidth=2,
                 label='$T_{int}$ (avg flux to $t_{iso}$)')
    ax1.semilogy(OL_array, T_int_5Gyr_array, 'rs--', markersize=8, linewidth=1.5,
                 alpha=0.7, label='$T_{int}$ (flux at t=5 Gyr)')
    ax1.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=12)
    ax1.set_ylabel('Interior Lifetime $T_{int}$ [toy units]', fontsize=12)
    ax1.set_title('Effective Interior Lifetime vs $\\Omega_\\Lambda$', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # -------------------------------------------------------------------------
    # Plot 2: Isolation time vs Omega_Lambda
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    valid_iso = ~np.isnan(t_iso_array)
    ax2.plot(OL_array[valid_iso], t_iso_array[valid_iso], 'go-', markersize=10,
             linewidth=2)
    ax2.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=12)
    ax2.set_ylabel('Isolation Time $t_{iso}$ [Gyr]', fontsize=12)
    ax2.set_title('Black Hole Isolation Time vs $\\Omega_\\Lambda$\n(from Program 1)',
                  fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    # Add annotation
    ax2.text(0.05, 0.95,
             'Higher $\\Omega_\\Lambda$ → faster expansion\n→ earlier isolation',
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # -------------------------------------------------------------------------
    # Plot 3: Effective L_in (toy units) vs Omega_Lambda
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.semilogy(OL_array, L_eff_toy_array, 'mo-', markersize=10, linewidth=2)
    ax3.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=12)
    ax3.set_ylabel('Effective $L_{in}$ [toy units]', fontsize=12)
    ax3.set_title('Effective Perturbation Flux vs $\\Omega_\\Lambda$', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)

    # -------------------------------------------------------------------------
    # Plot 4: Flux histories for different Omega_Lambda
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results)))

    for i, r in enumerate(results):
        label = f"$\\Omega_\\Lambda = {r['Omega_Lambda']}$"
        # Normalize flux for comparison
        L_norm = r['L_array'] / L_ref
        ax4.semilogy(r['t_array'], L_norm, color=colors[i], linewidth=2, label=label)

        # Mark isolation time
        if not np.isnan(r['t_iso_Gyr']) and r['t_iso_Gyr'] < 150:
            ax4.axvline(r['t_iso_Gyr'], color=colors[i], linestyle='--', alpha=0.5)

    ax4.axhline(1e-15, color='gray', linestyle=':', linewidth=2, label='$L_{crit}/L_{ref}$')
    ax4.set_xlabel('Cosmic Time $t$ [Gyr]', fontsize=12)
    ax4.set_ylabel('$L_{in} / L_{ref}$', fontsize=12)
    ax4.set_title('Normalized Flux Histories for Different $\\Omega_\\Lambda$', fontsize=14)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.5, 150)

    plt.tight_layout()
    plt.savefig('T_int_vs_omega_lambda.png', dpi=150, bbox_inches='tight')
    print("  Saved: T_int_vs_omega_lambda.png")

    # -------------------------------------------------------------------------
    # Additional plot: Combined T_int and t_iso on same axes (normalized)
    # -------------------------------------------------------------------------
    fig2, ax = plt.subplots(figsize=(10, 7))

    # Normalize both quantities to their values at Omega_Lambda = 0.7
    idx_07 = np.argmin(np.abs(OL_array - 0.7))
    T_int_norm = T_int_eff_array / T_int_eff_array[idx_07]
    t_iso_norm = t_iso_array / t_iso_array[idx_07]

    ax.plot(OL_array, T_int_norm, 'bo-', markersize=12, linewidth=2.5,
            label='Interior Lifetime $T_{int}$ (normalized)')
    ax.plot(OL_array[valid_iso], t_iso_norm[valid_iso], 'g^--', markersize=10,
            linewidth=2, label='Isolation Time $t_{iso}$ (normalized)')

    ax.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant Parameter)', fontsize=14)
    ax.set_ylabel('Normalized Value (relative to $\\Omega_\\Lambda = 0.7$)', fontsize=14)
    ax.set_title('Toy Model: Interior Lifetime and Isolation Time vs $\\Omega_\\Lambda$\n'
                 '(both normalized to their values at $\\Omega_\\Lambda = 0.7$)',
                 fontsize=14)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Add physics explanation
    ax.text(0.98, 0.02,
            'Higher $\\Omega_\\Lambda$:\n'
            '• Faster cosmic expansion\n'
            '• Earlier isolation (lower $t_{iso}$)\n'
            '• Lower effective flux\n'
            '• Longer interior lifetime ($T_{int}$)',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('T_int_vs_omega_lambda_combined.png', dpi=150, bbox_inches='tight')
    print("  Saved: T_int_vs_omega_lambda_combined.png")

    # Show plots
    plt.show()

    # =========================================================================
    # Final Summary
    # =========================================================================
    print()
    print("=" * 70)
    print("Physical Interpretation (Toy Model)")
    print("=" * 70)
    print("""
This toy model demonstrates the connection between cosmology and
black hole interior dynamics:

1. COSMOLOGICAL CONSTANT EFFECT (Program 1):
   - Higher Omega_Lambda leads to faster cosmic acceleration
   - This causes faster dilution of the radiation bath (rho_r ~ a^-4)
   - The black hole becomes "isolated" earlier (lower t_iso)
   - The effective perturbation flux L_eff is lower

2. INTERIOR LIFETIME EFFECT (Program 2):
   - Lower perturbation flux L_in leads to slower mass inflation
   - The curvature blowup is delayed
   - Interior lifetime T_int is longer

3. COMBINED RESULT:
   - Higher Omega_Lambda -> lower L_eff -> longer T_int
   - The scaling is: T_int ~ L_eff^(-p) where p ≈ {:.2f}
   - This creates an inverse relationship between cosmic expansion
     and black hole interior lifetime

CAVEATS:
- This is a highly simplified toy model
- Real black hole interiors require full GR treatment
- The flux-to-interior connection is qualitative only
- The L_in normalization is arbitrary
""".format(p))

    print("Done!")


if __name__ == "__main__":
    main()
