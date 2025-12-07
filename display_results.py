#!/usr/bin/env python3
"""
Display the T_int vs Omega_Lambda results in a GUI window.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend for display
import matplotlib.pyplot as plt

# Add module directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '2'))

from cosmo_flux_lambda import (
    compute_flux_history_and_isolation,
    get_default_L_crit,
    M_sun
)
from bh_interior_toy import fit_scaling_law

print("Computing results...")

# Fit scaling law
A, p, L_fit, T_fit = fit_scaling_law(
    L_in_values=[0.01, 0.02, 0.05, 0.1, 0.2],
    N_u=200, N_v=200, U_max=10.0, V_max=10.0
)

print(f"Scaling law: T_int = {A:.4f} * L_in^(-{p:.4f})")

# Setup
M_bh = 1e8 * M_sun
L_crit, L_ref = get_default_L_crit(M_bh)
omega_lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]

results = []
for OL in omega_lambda_values:
    print(f"  Omega_Lambda = {OL}...", end=" ", flush=True)
    t_arr, L_arr, t_iso = compute_flux_history_and_isolation(
        OL, M_bh, L_crit, t_min_Gyr=0.5, t_max_Gyr=200.0, n_t=500
    )

    if t_iso is not None:
        mask = t_arr <= t_iso
        L_eff = np.nanmean(L_arr[mask])
    else:
        mask = t_arr <= 50.0
        L_eff = np.nanmean(L_arr[mask])
        t_iso = np.nan

    scale_factor = 0.1
    L_eff_toy = L_eff / L_ref * scale_factor
    T_int_eff = A * (L_eff_toy ** (-p))

    results.append({
        'OL': OL, 't_iso': t_iso, 'L_eff_toy': L_eff_toy,
        'T_int': T_int_eff, 't_arr': t_arr, 'L_arr': L_arr / L_ref
    })
    print(f"done")

# Extract arrays
OL_arr = np.array([r['OL'] for r in results])
t_iso_arr = np.array([r['t_iso'] for r in results])
T_int_arr = np.array([r['T_int'] for r in results])

print("\nLaunching display window...")

# Create figure
fig = plt.figure(figsize=(14, 8))
fig.suptitle('Toy Model: Black Hole Interior Lifetime vs Cosmological Constant\n'
             f'Scaling Law: $T_{{int}} = {A:.3f} \\times L_{{in}}^{{-{p:.2f}}}$',
             fontsize=14, fontweight='bold')

# Plot 1: T_int vs Omega_Lambda
ax1 = fig.add_subplot(2, 2, 1)
ax1.semilogy(OL_arr, T_int_arr, 'bo-', markersize=12, linewidth=2.5,
             markerfacecolor='royalblue', markeredgecolor='black', markeredgewidth=1.5)
ax1.set_xlabel('$\\Omega_\\Lambda$', fontsize=12)
ax1.set_ylabel('Interior Lifetime $T_{int}$ [toy units]', fontsize=12)
ax1.set_title('Interior Lifetime vs $\\Omega_\\Lambda$', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)

# Plot 2: Isolation time vs Omega_Lambda
ax2 = fig.add_subplot(2, 2, 2)
valid = ~np.isnan(t_iso_arr)
ax2.plot(OL_arr[valid], t_iso_arr[valid], 'g^-', markersize=12, linewidth=2.5,
         markerfacecolor='limegreen', markeredgecolor='black', markeredgewidth=1.5)
ax2.set_xlabel('$\\Omega_\\Lambda$', fontsize=12)
ax2.set_ylabel('Isolation Time $t_{iso}$ [Gyr]', fontsize=12)
ax2.set_title('Isolation Time vs $\\Omega_\\Lambda$', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)

# Plot 3: Flux histories
ax3 = fig.add_subplot(2, 2, 3)
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))
for i, r in enumerate(results):
    ax3.semilogy(r['t_arr'], r['L_arr'], color=colors[i], linewidth=2,
                 label=f"$\\Omega_\\Lambda = {r['OL']}$")
ax3.axhline(1e-15, color='gray', linestyle=':', linewidth=2)
ax3.set_xlabel('Cosmic Time $t$ [Gyr]', fontsize=12)
ax3.set_ylabel('$L_{in} / L_{ref}$', fontsize=12)
ax3.set_title('Flux Histories', fontsize=12)
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0.5, 150)

# Plot 4: Combined normalized plot
ax4 = fig.add_subplot(2, 2, 4)
idx_07 = np.argmin(np.abs(OL_arr - 0.7))
T_norm = T_int_arr / T_int_arr[idx_07]
t_iso_norm = t_iso_arr / t_iso_arr[idx_07]

ax4.plot(OL_arr, T_norm, 'bo-', markersize=10, linewidth=2, label='$T_{int}$ (normalized)')
ax4.plot(OL_arr[valid], t_iso_norm[valid], 'g^--', markersize=10, linewidth=2,
         label='$t_{iso}$ (normalized)')
ax4.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
ax4.set_xlabel('$\\Omega_\\Lambda$', fontsize=12)
ax4.set_ylabel('Normalized (to $\\Omega_\\Lambda=0.7$)', fontsize=12)
ax4.set_title('Comparison (Normalized)', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 1)

plt.tight_layout()
plt.subplots_adjust(top=0.88)

# Show the plot
plt.show()

print("Window closed.")
