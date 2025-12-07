#!/usr/bin/env python3
"""
Create the T_int vs Omega_Lambda plot and open it on Windows host.
"""

import sys
import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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

print("=" * 60)
print("  BLACK HOLE INTERIOR LIFETIME vs COSMOLOGICAL CONSTANT")
print("=" * 60)
print()

# Fit scaling law
print("Step 1: Fitting T_int(L_in) scaling law...")
A, p, L_fit, T_fit = fit_scaling_law(
    L_in_values=[0.01, 0.02, 0.05, 0.1, 0.2],
    N_u=200, N_v=200, U_max=10.0, V_max=10.0
)
print(f"  Result: T_int = {A:.4f} * L_in^(-{p:.4f})")
print()

# Setup
M_bh = 1e8 * M_sun
L_crit, L_ref = get_default_L_crit(M_bh)
omega_lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]

print("Step 2: Computing T_int for each Omega_Lambda...")
results = []
for OL in omega_lambda_values:
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
    t_iso_str = f"{t_iso:.1f}" if not np.isnan(t_iso) else "N/A"
    print(f"  Ω_Λ = {OL:.1f}  →  t_iso = {t_iso_str:>6} Gyr  →  T_int = {T_int_eff:.2f}")

print()

# Extract arrays
OL_arr = np.array([r['OL'] for r in results])
t_iso_arr = np.array([r['t_iso'] for r in results])
T_int_arr = np.array([r['T_int'] for r in results])

print("Step 3: Creating visualization...")

# Create a nice combined figure
fig = plt.figure(figsize=(16, 10), facecolor='white')
fig.suptitle('Toy Model: Black Hole Interior Lifetime vs Cosmological Constant\n'
             f'Scaling Law: $T_{{int}} = {A:.3f} \\times L_{{in}}^{{-{p:.2f}}}$',
             fontsize=16, fontweight='bold', y=0.98)

# Color scheme
main_blue = '#2E86AB'
main_green = '#28A745'
main_purple = '#6F42C1'
main_orange = '#FD7E14'

# Plot 1: T_int vs Omega_Lambda (main result)
ax1 = fig.add_subplot(2, 2, 1)
ax1.semilogy(OL_arr, T_int_arr, 'o-', markersize=14, linewidth=3,
             color=main_blue, markerfacecolor=main_blue,
             markeredgecolor='white', markeredgewidth=2)
for i, (x, y) in enumerate(zip(OL_arr, T_int_arr)):
    ax1.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
ax1.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant)', fontsize=13)
ax1.set_ylabel('Interior Lifetime $T_{int}$ [toy units]', fontsize=13)
ax1.set_title('Interior Lifetime vs $\\Omega_\\Lambda$', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(-0.02, 1.02)
ax1.set_facecolor('#f8f9fa')

# Plot 2: Isolation time vs Omega_Lambda
ax2 = fig.add_subplot(2, 2, 2)
valid = ~np.isnan(t_iso_arr)
ax2.plot(OL_arr[valid], t_iso_arr[valid], '^-', markersize=14, linewidth=3,
         color=main_green, markerfacecolor=main_green,
         markeredgecolor='white', markeredgewidth=2)
for i, (x, y) in enumerate(zip(OL_arr[valid], t_iso_arr[valid])):
    ax2.annotate(f'{y:.0f}', (x, y), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
ax2.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant)', fontsize=13)
ax2.set_ylabel('Isolation Time $t_{iso}$ [Gyr]', fontsize=13)
ax2.set_title('Isolation Time vs $\\Omega_\\Lambda$', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(-0.02, 1.02)
ax2.set_facecolor('#f8f9fa')

# Add annotation box
ax2.text(0.95, 0.95, 'Higher $\\Omega_\\Lambda$\n→ faster expansion\n→ earlier isolation',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5',
         facecolor='lightyellow', edgecolor='orange', alpha=0.9))

# Plot 3: Flux histories
ax3 = fig.add_subplot(2, 2, 3)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))
for i, r in enumerate(results):
    ax3.semilogy(r['t_arr'], r['L_arr'], color=colors[i], linewidth=2.5,
                 label=f"$\\Omega_\\Lambda = {r['OL']}$")
    # Mark isolation time
    if not np.isnan(r['t_iso']) and r['t_iso'] < 180:
        ax3.axvline(r['t_iso'], color=colors[i], linestyle='--', alpha=0.4, linewidth=1.5)

ax3.axhline(1e-15, color='red', linestyle=':', linewidth=2.5, label='$L_{crit}/L_{ref}$')
ax3.set_xlabel('Cosmic Time $t$ [Gyr]', fontsize=13)
ax3.set_ylabel('Normalized Flux $L_{in} / L_{ref}$', fontsize=13)
ax3.set_title('Flux Histories for Different $\\Omega_\\Lambda$', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xlim(0.5, 180)
ax3.set_ylim(1e-25, 10)
ax3.set_facecolor('#f8f9fa')

# Plot 4: Combined normalized comparison
ax4 = fig.add_subplot(2, 2, 4)
idx_07 = np.argmin(np.abs(OL_arr - 0.7))
T_norm = T_int_arr / T_int_arr[idx_07]
t_iso_norm = t_iso_arr / t_iso_arr[idx_07]

ax4.plot(OL_arr, T_norm, 'o-', markersize=12, linewidth=2.5, color=main_blue,
         label='Interior Lifetime $T_{int}$', markeredgecolor='white', markeredgewidth=1.5)
ax4.plot(OL_arr[valid], t_iso_norm[valid], '^--', markersize=12, linewidth=2.5,
         color=main_green, label='Isolation Time $t_{iso}$',
         markeredgecolor='white', markeredgewidth=1.5)
ax4.axhline(1.0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
ax4.set_xlabel('$\\Omega_\\Lambda$ (Cosmological Constant)', fontsize=13)
ax4.set_ylabel('Normalized Value (rel. to $\\Omega_\\Lambda=0.7$)', fontsize=13)
ax4.set_title('Comparison: $T_{int}$ and $t_{iso}$ (Normalized)', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.set_xlim(-0.02, 1.02)
ax4.set_facecolor('#f8f9fa')

# Add physics summary box
textstr = ('Physical Picture:\n'
           '• Higher $\\Omega_\\Lambda$ → faster cosmic expansion\n'
           '• Faster expansion → earlier isolation (↓ $t_{iso}$)\n'
           '• Earlier isolation → lower effective flux\n'
           '• Lower flux → longer interior lifetime (↑ $T_{int}$)')
props = dict(boxstyle='round,pad=0.8', facecolor='lightcyan',
             edgecolor='steelblue', alpha=0.95)
ax4.text(0.98, 0.02, textstr, transform=ax4.transAxes, fontsize=10,
         verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.subplots_adjust(top=0.91, hspace=0.28, wspace=0.25)

# Save the figure
output_path = os.path.join(os.path.dirname(__file__), 'T_int_vs_OmegaLambda_RESULT.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"  Saved: {output_path}")

# Convert to Windows path and open
win_path = subprocess.run(['wslpath', '-w', output_path], capture_output=True, text=True).stdout.strip()
print(f"  Windows path: {win_path}")
print()
print("Step 4: Opening image on Windows host...")
subprocess.run(['/mnt/c/Windows/explorer.exe', win_path])

print()
print("=" * 60)
print("  DONE! The plot should now be displayed on your screen.")
print("=" * 60)
