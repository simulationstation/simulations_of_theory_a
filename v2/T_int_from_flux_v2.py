#!/usr/bin/env python3
"""
T_int_from_flux_v2.py

Compute interior lifetime T_int(Omega_Lambda) using:
1. Cosmological flux history L_in(t; Omega_Lambda) from v1 module
2. Effective perturbation amplitude Phi_eff via time integral
3. Mass-inflation-motivated logarithmic mapping to T_int

Physical picture:
- The perturbation flux L_in(t) incident on a BH determines the
  strength of mass inflation near the inner horizon.
- We integrate L_in over time (up to isolation) to get an effective
  cumulative perturbation Phi_eff.
- The interior lifetime scales as T_int ~ (1/kappa) * log(C / Phi_eff),
  where larger Phi_eff means faster blowup and shorter lifetime.

References:
- Poisson & Israel (1990) for mass inflation
- Ori (1991) for interior structure
"""

import numpy as np
import sys
import os

# Add v1 module path for flux computation
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '1'))

from cosmo_flux_lambda import (
    compute_flux_history_and_isolation,
    get_default_L_crit,
    M_sun,
    Gyr_to_s
)


# =============================================================================
# Effective Perturbation Amplitude
# =============================================================================

def Phi_eff_from_flux(omega_lambda, M_bh=1e8 * M_sun, L_crit=None,
                      t_min_Gyr=0.5, t_max_Gyr=200.0, n_t=500,
                      use_isolation_cutoff=True):
    """
    Compute effective perturbation amplitude Phi_eff from flux history.

    Phi_eff(Omega_Lambda) = integral_0^{t_iso} L_in(t; Omega_Lambda) dt

    This represents the cumulative radiation perturbation experienced
    by the black hole interior.

    Parameters
    ----------
    omega_lambda : float
        Cosmological constant density parameter
    M_bh : float
        Black hole mass [kg] (default 10^8 M_sun)
    L_crit : float, optional
        Critical flux threshold for isolation [W].
        If None, uses default from v1 module.
    t_min_Gyr : float
        Start time for flux history [Gyr]
    t_max_Gyr : float
        End time for flux history [Gyr]
    n_t : int
        Number of time points
    use_isolation_cutoff : bool
        If True, truncate integral at isolation time t_iso.
        If False, integrate over full time range.

    Returns
    -------
    Phi_eff : float
        Effective perturbation amplitude [W * Gyr]
    t_iso_Gyr : float or None
        Isolation time [Gyr], or None if not found
    """
    # Get default L_crit if not provided
    if L_crit is None:
        L_crit, _ = get_default_L_crit(M_bh)

    # Compute flux history
    t_array_Gyr, L_in_array_W, t_iso_Gyr = compute_flux_history_and_isolation(
        omega_lambda, M_bh, L_crit,
        t_min_Gyr=t_min_Gyr, t_max_Gyr=t_max_Gyr, n_t=n_t
    )

    # Handle NaN values in flux array
    valid_mask = np.isfinite(L_in_array_W)
    if not np.any(valid_mask):
        return 0.0, t_iso_Gyr

    # Determine integration limit
    if use_isolation_cutoff and t_iso_Gyr is not None:
        # Truncate at isolation time
        cutoff_mask = t_array_Gyr <= t_iso_Gyr
        integration_mask = valid_mask & cutoff_mask
    else:
        integration_mask = valid_mask

    t_int = t_array_Gyr[integration_mask]
    L_int = L_in_array_W[integration_mask]

    if len(t_int) < 2:
        return 0.0, t_iso_Gyr

    # Trapezoidal integration: Phi_eff = integral L_in dt
    # Units: [W] * [Gyr] = [W * Gyr]
    Phi_eff = np.trapz(L_int, t_int)

    return Phi_eff, t_iso_Gyr


def Phi_eff_from_flux_simple(omega_lambda, M_bh=1e8 * M_sun,
                              t_avg_Gyr=50.0, n_t=200):
    """
    Simplified Phi_eff: average flux times characteristic time.

    Phi_eff ~ <L_in> * t_char

    This is faster than full integration and gives similar scaling.

    Parameters
    ----------
    omega_lambda : float
        Cosmological constant density parameter
    M_bh : float
        Black hole mass [kg]
    t_avg_Gyr : float
        Characteristic averaging time [Gyr]
    n_t : int
        Number of time points

    Returns
    -------
    Phi_eff : float
        Effective perturbation amplitude [W * Gyr]
    """
    L_crit, _ = get_default_L_crit(M_bh)

    t_array_Gyr, L_in_array_W, t_iso_Gyr = compute_flux_history_and_isolation(
        omega_lambda, M_bh, L_crit,
        t_min_Gyr=0.5, t_max_Gyr=t_avg_Gyr, n_t=n_t
    )

    # Use mean flux
    valid = np.isfinite(L_in_array_W)
    if not np.any(valid):
        return 0.0

    L_mean = np.mean(L_in_array_W[valid])

    # Characteristic time (use isolation time if available, else t_avg)
    t_char = t_iso_Gyr if t_iso_Gyr is not None else t_avg_Gyr

    return L_mean * t_char


# =============================================================================
# Interior Lifetime from Phi_eff
# =============================================================================

def T_int_from_Phi_eff(Phi_eff, kappa_minus=1.0, C=None):
    """
    Compute interior lifetime from effective perturbation using
    mass-inflation-motivated logarithmic formula.

    T_int = (1 / kappa_minus) * log(C / Phi_eff)

    This captures the key physics: larger Phi_eff (stronger perturbation)
    leads to faster mass inflation blowup and shorter interior lifetime.

    Parameters
    ----------
    Phi_eff : float
        Effective perturbation amplitude [W * Gyr]
    kappa_minus : float
        Inner horizon surface gravity parameter (default 1.0).
        Larger kappa_minus -> faster blowup -> smaller T_int.
    C : float
        Normalization constant [W * Gyr]. Must satisfy C > Phi_eff
        for T_int > 0. If None, raises ValueError.

    Returns
    -------
    T_int : float
        Interior lifetime [dimensionless, in units of 1/kappa_minus]

    Raises
    ------
    ValueError
        If C is None or if C <= Phi_eff (would give non-positive T_int)
    """
    if C is None:
        raise ValueError("C must be provided (use scan function to determine appropriate value)")

    if Phi_eff <= 0:
        return np.inf  # No perturbation -> infinite lifetime (limiting case)

    if C <= Phi_eff:
        # Would give negative or zero T_int
        return 0.0

    T_int = (1.0 / kappa_minus) * np.log(C / Phi_eff)

    return T_int


# =============================================================================
# Scan Over Omega_Lambda
# =============================================================================

def scan_T_int_vs_omega_lambda(omega_lambdas, kappa_minus=1.0,
                                M_bh=1e8 * M_sun,
                                use_isolation_cutoff=True,
                                C_factor=10.0,
                                verbose=True):
    """
    Scan T_int over a range of Omega_Lambda values.

    Pipeline:
    1. Compute Phi_eff(Omega_Lambda) for each value
    2. Determine C = C_factor * min(Phi_eff) to ensure T_int > 0
    3. Compute T_int for each Omega_Lambda
    4. Normalize to T_int_norm = T_int / max(T_int)

    Parameters
    ----------
    omega_lambdas : array-like
        Array of Omega_Lambda values to scan
    kappa_minus : float
        Inner horizon surface gravity parameter
    M_bh : float
        Black hole mass [kg]
    use_isolation_cutoff : bool
        Whether to truncate flux integral at isolation time
    C_factor : float
        Multiplier for setting C relative to min(Phi_eff).
        C = C_factor * min(Phi_eff). Larger C_factor -> larger T_int range.
    verbose : bool
        Print progress messages

    Returns
    -------
    omega_array : ndarray
        Omega_Lambda values
    Phi_eff_array : ndarray
        Effective perturbation amplitudes [W * Gyr]
    T_int_array : ndarray
        Interior lifetimes (raw)
    T_int_norm : ndarray
        Normalized interior lifetimes (max = 1)
    t_iso_array : ndarray
        Isolation times [Gyr]
    C : float
        The constant C used in the T_int formula
    """
    omega_array = np.asarray(omega_lambdas)
    n_points = len(omega_array)

    Phi_eff_array = np.zeros(n_points)
    t_iso_array = np.zeros(n_points)

    if verbose:
        print(f"Computing Phi_eff for {n_points} Omega_Lambda values...")

    # Step 1: Compute Phi_eff for all Omega_Lambda
    for i, ol in enumerate(omega_array):
        if verbose and (i % 5 == 0 or i == n_points - 1):
            print(f"  Processing Omega_Lambda = {ol:.3f} ({i+1}/{n_points})...",
                  end='\r')

        Phi_eff, t_iso = Phi_eff_from_flux(
            ol, M_bh=M_bh,
            use_isolation_cutoff=use_isolation_cutoff
        )
        Phi_eff_array[i] = Phi_eff
        t_iso_array[i] = t_iso if t_iso is not None else np.nan

    if verbose:
        print()  # Newline after progress

    # Step 2: Determine C from the data
    # Use C = C_factor * min(Phi_eff) where Phi_eff > 0
    valid_Phi = Phi_eff_array[Phi_eff_array > 0]
    if len(valid_Phi) == 0:
        raise ValueError("All Phi_eff values are zero or negative")

    Phi_min = np.min(valid_Phi)
    Phi_max = np.max(valid_Phi)
    C = C_factor * Phi_max  # Use max so that all T_int > 0

    if verbose:
        print(f"  Phi_eff range: [{Phi_min:.4e}, {Phi_max:.4e}] W*Gyr")
        print(f"  Using C = {C_factor} * Phi_max = {C:.4e} W*Gyr")

    # Step 3: Compute T_int for each Omega_Lambda
    T_int_array = np.zeros(n_points)
    for i in range(n_points):
        T_int_array[i] = T_int_from_Phi_eff(Phi_eff_array[i], kappa_minus, C)

    # Step 4: Normalize
    T_max = np.max(T_int_array[np.isfinite(T_int_array)])
    T_int_norm = T_int_array / T_max if T_max > 0 else T_int_array

    if verbose:
        print(f"  T_int range: [{np.min(T_int_array):.4f}, {np.max(T_int_array):.4f}]")
        print()

    return omega_array, Phi_eff_array, T_int_array, T_int_norm, t_iso_array, C


# =============================================================================
# Main: Test the functions
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("T_int from Flux - v2 Module Test")
    print("=" * 60)
    print()

    # Test single Omega_Lambda
    test_omegas = [0.3, 0.5, 0.7, 0.9]

    print("Single-value tests:")
    print("-" * 50)
    for ol in test_omegas:
        Phi, t_iso = Phi_eff_from_flux(ol)
        print(f"  Omega_Lambda = {ol}: Phi_eff = {Phi:.4e} W*Gyr, t_iso = {t_iso:.1f} Gyr")

    print()

    # Test scan
    print("Running mini scan...")
    omega_test = np.linspace(0.1, 0.9, 9)
    omega, Phi, T_int, T_norm, t_iso, C = scan_T_int_vs_omega_lambda(
        omega_test, kappa_minus=1.0, verbose=True
    )

    print("Results:")
    print("-" * 60)
    print(f"{'Omega_Lambda':>12} | {'Phi_eff':>12} | {'T_int':>10} | {'T_norm':>10}")
    print("-" * 60)
    for i in range(len(omega)):
        print(f"{omega[i]:>12.3f} | {Phi[i]:>12.4e} | {T_int[i]:>10.4f} | {T_norm[i]:>10.4f}")

    print()
    print("Done!")
