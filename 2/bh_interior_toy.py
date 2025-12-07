#!/usr/bin/env python3
"""
bh_interior_toy.py

A toy 1+1D model of black hole interior dynamics inspired by mass inflation.

This is NOT a full Einstein-Maxwell solver. It is a simplified numerical PDE
system that qualitatively captures how stronger perturbation flux leads to
faster interior blowup and hence shorter interior lifetime.

Model:
------
- Double-null-like grid in coordinates (u, v)
- Mass function m(u, v) evolves via: dm/du = F(m, v; L_in)
- F(m, v; L_in) = a * L_in * exp(κ * v) * (1 + b * m)
- The exponential term mimics mass inflation near the inner horizon
- Curvature proxy K = m^2; blowup when K > K_crit
- Interior lifetime T_int = u value when blowup occurs

Author: Toy model for pedagogical purposes
"""

import numpy as np
import matplotlib.pyplot as plt


def build_grid(N_u, N_v, U_max, V_max):
    """
    Build uniform (u, v) grids.

    Parameters
    ----------
    N_u : int
        Number of grid points in u direction
    N_v : int
        Number of grid points in v direction
    U_max : float
        Maximum u coordinate
    V_max : float
        Maximum v coordinate

    Returns
    -------
    u : ndarray
        1D array of u coordinates
    v : ndarray
        1D array of v coordinates
    du : float
        Grid spacing in u
    dv : float
        Grid spacing in v
    """
    u = np.linspace(0, U_max, N_u)
    v = np.linspace(0, V_max, N_v)
    du = u[1] - u[0] if N_u > 1 else U_max
    dv = v[1] - v[0] if N_v > 1 else V_max
    return u, v, du, dv


def mass_inflation_rhs(m, v, L_in, kappa=0.5, a=1.0, b=0.1):
    """
    Compute the RHS of the mass evolution equation: dm/du = F(m, v; L_in)

    This is a simplified mass-inflation-inspired growth term:
    F(m, v; L_in) = a * L_in * exp(κ * v) * (1 + b * m)

    Parameters
    ----------
    m : ndarray
        Mass function at current u, shape (N_v,)
    v : ndarray
        v coordinates, shape (N_v,)
    L_in : float
        Ingoing perturbation flux parameter
    kappa : float
        Inner horizon surface gravity parameter (controls exponential growth)
    a : float
        O(1) constant
    b : float
        O(1) constant controlling nonlinear feedback

    Returns
    -------
    F : ndarray
        RHS of evolution equation, shape (N_v,)
    """
    return a * L_in * np.exp(kappa * v) * (1.0 + b * m)


def evolve_mass_function(N_u, N_v, U_max, V_max, L_in, m0=1.0,
                         kappa=0.5, a=1.0, b=0.1, K_crit=1e6):
    """
    Evolve the mass function m(u, v) using explicit Euler in u.

    Parameters
    ----------
    N_u, N_v : int
        Grid sizes
    U_max, V_max : float
        Domain sizes
    L_in : float
        Perturbation flux parameter
    m0 : float
        Initial mass value at u=0
    kappa, a, b : float
        Model parameters for mass inflation RHS
    K_crit : float
        Curvature threshold for blowup detection

    Returns
    -------
    m_final : ndarray
        Final mass function m(u, v), shape (N_u, N_v)
    u_blowup : float
        u coordinate where blowup occurred (or U_max if none)
    u : ndarray
        u coordinates
    v : ndarray
        v coordinates
    """
    # Build grid
    u, v, du, dv = build_grid(N_u, N_v, U_max, V_max)

    # Initialize mass function: m(u=0, v) = m0
    m = np.full((N_u, N_v), m0, dtype=np.float64)

    # Track blowup
    u_blowup = U_max  # Default: no blowup within domain
    blowup_occurred = False

    # Evolve in u using explicit Euler
    for i in range(N_u - 1):
        if blowup_occurred:
            # Fill rest with last valid values (for visualization)
            m[i+1, :] = m[i, :]
            continue

        # Compute RHS
        F = mass_inflation_rhs(m[i, :], v, L_in, kappa, a, b)

        # Euler step
        m[i+1, :] = m[i, :] + du * F

        # Compute curvature proxy K = m^2
        K = m[i+1, :]**2
        max_K = np.max(K)

        # Check for blowup
        if max_K > K_crit:
            u_blowup = u[i+1]
            blowup_occurred = True

    return m, u_blowup, u, v


def compute_interior_lifetime(L_in, N_u=200, N_v=200, U_max=10.0, V_max=10.0,
                               m0=1.0, kappa=0.5, a=1.0, b=0.1, K_crit=1e6):
    """
    Compute the interior lifetime T_int for a given perturbation flux L_in.

    T_int is defined as the u coordinate where the curvature proxy K = m^2
    first exceeds the threshold K_crit.

    Parameters
    ----------
    L_in : float
        Perturbation flux parameter
    Other parameters: see evolve_mass_function

    Returns
    -------
    T_int : float
        Interior lifetime (u value at blowup)
    m : ndarray
        Final mass function for diagnostics
    u, v : ndarray
        Grid coordinates
    """
    m, u_blowup, u, v = evolve_mass_function(
        N_u, N_v, U_max, V_max, L_in, m0, kappa, a, b, K_crit
    )
    return u_blowup, m, u, v


def plot_mass_function(m, u, v, L_in, save_path=None):
    """
    Create a color map of the mass function m(u, v) or curvature proxy K(u, v).

    Parameters
    ----------
    m : ndarray
        Mass function, shape (N_u, N_v)
    u, v : ndarray
        Grid coordinates
    L_in : float
        Perturbation flux (for title)
    save_path : str, optional
        Path to save figure
    """
    # Compute curvature proxy
    K = m**2

    # Use log scale for better visualization of exponential growth
    # Clip to avoid log(0) issues
    K_clipped = np.clip(K, 1e-10, 1e10)
    log_K = np.log10(K_clipped)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mass function m(u, v)
    ax1 = axes[0]
    m_clipped = np.clip(m, 1e-10, 1e5)
    im1 = ax1.imshow(np.log10(m_clipped).T, origin='lower', aspect='auto',
                     extent=[u[0], u[-1], v[0], v[-1]], cmap='inferno')
    ax1.set_xlabel('u (retarded time)', fontsize=12)
    ax1.set_ylabel('v (advanced time)', fontsize=12)
    ax1.set_title(f'Toy mass inflation: log₁₀(m(u,v)), L_in = {L_in}', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('log₁₀(m)', fontsize=10)

    # Plot 2: Curvature proxy K(u, v)
    ax2 = axes[1]
    im2 = ax2.imshow(log_K.T, origin='lower', aspect='auto',
                     extent=[u[0], u[-1], v[0], v[-1]], cmap='hot')
    ax2.set_xlabel('u (retarded time)', fontsize=12)
    ax2.set_ylabel('v (advanced time)', fontsize=12)
    ax2.set_title(f'Toy mass inflation: curvature proxy log₁₀(K(u,v)), L_in = {L_in}', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('log₁₀(K) = log₁₀(m²)', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved mass function plot to {save_path}")

    plt.show()


def plot_lifetime_vs_flux(L_in_values, T_int_values, save_path=None):
    """
    Plot interior lifetime T_int vs perturbation flux L_in on log-log axes.

    Parameters
    ----------
    L_in_values : array-like
        Perturbation flux values
    T_int_values : array-like
        Corresponding interior lifetimes
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(L_in_values, T_int_values, 'bo-', markersize=10, linewidth=2,
              label='Toy model')

    # Add a reference power-law line for comparison
    # T_int ~ L_in^(-alpha) approximately
    L_fit = np.array(L_in_values)
    # Estimate slope from data
    if len(L_in_values) > 1:
        log_L = np.log(L_in_values)
        log_T = np.log(T_int_values)
        # Simple linear fit
        slope, intercept = np.polyfit(log_L, log_T, 1)
        T_fit = np.exp(intercept) * L_fit**slope
        ax.loglog(L_fit, T_fit, 'r--', linewidth=1.5, alpha=0.7,
                  label=f'Power law fit: T_int ∝ L_in^{{{slope:.2f}}}')

    ax.set_xlabel('Perturbation flux L_in', fontsize=12)
    ax.set_ylabel('Interior lifetime T_int', fontsize=12)
    ax.set_title('Interior lifetime T_int vs perturbation flux L_in (toy model)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    # Add annotation explaining the physics
    ax.annotate('Larger L_in → faster mass inflation → shorter lifetime',
                xy=(0.5, 0.02), xycoords='axes fraction',
                fontsize=9, style='italic', ha='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved lifetime plot to {save_path}")

    plt.show()


# =============================================================================
# Wrapper Functions for Module Use
# =============================================================================

def compute_T_int_for_Lin(L_in, N_u=200, N_v=200, U_max=10.0, V_max=10.0,
                           m0=1.0, kappa=0.5, a=1.0, b=0.1, K_crit=1e6):
    """
    Run the toy PDE evolution for a given L_in and return T_int(L_in).

    This is a helper function for single L_in value computation.

    Parameters
    ----------
    L_in : float
        Perturbation flux parameter
    N_u, N_v : int
        Grid sizes (default 200x200)
    U_max, V_max : float
        Domain sizes (default 10.0)
    m0 : float
        Initial mass value (default 1.0)
    kappa : float
        Inner horizon surface gravity parameter (default 0.5)
    a, b : float
        Model parameters (default 1.0, 0.1)
    K_crit : float
        Curvature threshold for blowup (default 1e6)

    Returns
    -------
    T_int : float
        Interior lifetime (u value at blowup, or U_max if no blowup)
    """
    T_int, _, _, _ = compute_interior_lifetime(
        L_in, N_u, N_v, U_max, V_max, m0, kappa, a, b, K_crit
    )
    return T_int


def run_mass_inflation_scan(L_in_values, N_u=200, N_v=200, U_max=10.0, V_max=10.0,
                             m0=1.0, kappa=0.5, a=1.0, b=0.1, K_crit=1e6):
    """
    Run the toy evolution for multiple L_in values and compute T_int for each.

    Parameters
    ----------
    L_in_values : array-like
        List/array of perturbation flux values
    Other parameters: see compute_T_int_for_Lin

    Returns
    -------
    L_in_array : ndarray
        Input L_in values
    T_int_array : ndarray
        Corresponding interior lifetimes
    """
    L_in_array = np.array(L_in_values)
    T_int_array = np.array([
        compute_T_int_for_Lin(L, N_u, N_v, U_max, V_max, m0, kappa, a, b, K_crit)
        for L in L_in_values
    ])
    return L_in_array, T_int_array


def fit_scaling_law(L_in_values=None, U_max=10.0, **kwargs):
    """
    Compute the scaling law T_int = A * L_in^(-p) from simulation data.

    Runs simulations for a set of L_in values, fits a power law on log-log
    scale, and returns the fit parameters.

    Parameters
    ----------
    L_in_values : array-like, optional
        L_in values to use for fitting. Default: [0.01, 0.02, 0.05, 0.1, 0.2]
    U_max : float
        Maximum u domain size (used to filter out "no blowup" cases)
    **kwargs : dict
        Additional parameters passed to run_mass_inflation_scan

    Returns
    -------
    A : float
        Prefactor in T_int = A * L_in^(-p)
    p : float
        Exponent in T_int = A * L_in^(-p) (positive value)
    L_in_used : ndarray
        L_in values used in the fit (after filtering)
    T_int_used : ndarray
        T_int values used in the fit (after filtering)
    """
    if L_in_values is None:
        L_in_values = [0.01, 0.02, 0.05, 0.1, 0.2]

    # Run scan
    L_in_array, T_int_array = run_mass_inflation_scan(
        L_in_values, U_max=U_max, **kwargs
    )

    # Filter out "no blowup" cases (T_int == U_max)
    valid_mask = T_int_array < U_max
    L_valid = L_in_array[valid_mask]
    T_valid = T_int_array[valid_mask]

    if len(L_valid) < 2:
        raise ValueError("Not enough valid data points for power law fit "
                        "(need at least 2 points with blowup)")

    # Log-log linear fit: log10(T) = slope * log10(L) + intercept
    log_L = np.log10(L_valid)
    log_T = np.log10(T_valid)
    slope, intercept = np.polyfit(log_L, log_T, 1)

    # Convert: T = 10^intercept * L^slope
    # We want: T = A * L^(-p), so p = -slope, A = 10^intercept
    A = 10**intercept
    p = -slope  # Make p positive (since slope is negative)

    return A, p, L_valid, T_valid


if __name__ == "__main__":
    print("=" * 70)
    print("Toy 1+1D Black Hole Interior Model (Mass Inflation Inspired)")
    print("=" * 70)
    print()
    print("Model: dm/du = a * L_in * exp(κ*v) * (1 + b*m)")
    print("Curvature proxy: K = m²")
    print("Interior lifetime: T_int = u value when K exceeds threshold")
    print()

    # Model parameters
    N_u = 200
    N_v = 200
    U_max = 10.0
    V_max = 10.0
    m0 = 1.0
    kappa = 0.5
    a = 1.0
    b = 0.1
    K_crit = 1e6

    print(f"Grid: {N_u} × {N_v} points")
    print(f"Domain: u ∈ [0, {U_max}], v ∈ [0, {V_max}]")
    print(f"Parameters: κ = {kappa}, a = {a}, b = {b}")
    print(f"Initial mass: m₀ = {m0}")
    print(f"Curvature threshold: K_crit = {K_crit:.0e}")
    print()

    # Perturbation flux values to scan
    L_in_values = [0.01, 0.02, 0.05, 0.1, 0.2]
    T_int_values = []

    print("-" * 50)
    print(f"{'L_in':>10} | {'T_int':>15} | {'Status':>15}")
    print("-" * 50)

    # Store one result for detailed plotting
    representative_L_in = 0.1
    representative_result = None

    for L_in in L_in_values:
        T_int, m, u, v = compute_interior_lifetime(
            L_in, N_u, N_v, U_max, V_max, m0, kappa, a, b, K_crit
        )
        T_int_values.append(T_int)

        status = "Blowup" if T_int < U_max else "No blowup"
        print(f"{L_in:>10.3f} | {T_int:>15.4f} | {status:>15}")

        # Store representative result
        if L_in == representative_L_in:
            representative_result = (m, u, v, L_in)

    print("-" * 50)
    print()

    # Convert to numpy array
    T_int_values = np.array(T_int_values)
    L_in_values = np.array(L_in_values)

    # Print summary
    print("Summary:")
    print(f"  - As L_in increases from {L_in_values[0]} to {L_in_values[-1]},")
    print(f"    T_int decreases from {T_int_values[0]:.4f} to {T_int_values[-1]:.4f}")
    print(f"  - This demonstrates: stronger perturbation flux → faster blowup")
    print()

    # Generate plots
    print("Generating plots...")
    print()

    # Plot 1: Mass function / curvature proxy for representative L_in
    if representative_result is not None:
        m, u, v, L_in = representative_result
        plot_mass_function(m, u, v, L_in, save_path="mass_function_colormap.png")

    # Plot 2: Interior lifetime vs perturbation flux
    plot_lifetime_vs_flux(L_in_values, T_int_values, save_path="lifetime_vs_flux.png")

    print()
    print("Done! Generated plots saved as PNG files.")
