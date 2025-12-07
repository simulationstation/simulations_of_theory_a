#!/usr/bin/env python3
"""
T_int_v3_from_formation_history.py

Compute interior lifetime T_int(Omega_Lambda) using:
1. BH formation redshift distribution from v3 formation rate
2. Post-formation flux history from v1/v2 flux module
3. Power-law scaling T_int ~ L_eff^{-p} from mass inflation physics

Physical picture:
- Black holes form at different redshifts depending on Omega_Lambda
- After formation, they experience external flux that drives mass inflation
- The effective post-formation flux L_eff determines interior lifetime
- T_int scales inversely with flux: T_int ~ L_eff^{-p}, p ~ 0.95

Key insight: Higher Omega_Lambda means:
- Later average BH formation (less time for structure)
- BUT also faster flux decay (cosmic acceleration)
- These effects compete to determine net L_eff and thus T_int

References:
- Poisson & Israel (1990) for mass inflation
- Ori (1991) for interior structure
"""

import numpy as np
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), '1'))

from cosmology_utils import dt_dz
from N_BH_v3_press_schechter_sfr import bh_formation_rate_density

# Import v1 flux module
from cosmo_flux_lambda import (
    compute_flux_history_and_isolation,
    get_default_L_crit,
    M_sun
)


# =============================================================================
# Step 1: Formation Redshift and Time
# =============================================================================

def bh_formation_pdf(z_grid, omega_lambda):
    """
    Compute the normalized probability density for BH formation redshift.

    p_BH(z | Omega_Lambda) is proportional to:
        bh_formation_rate_density(z, Omega_Lambda) * |dt/dz|

    This represents the probability that a randomly chosen BH formed at
    redshift z, accounting for both the formation rate and the cosmic
    time available at each redshift.

    Parameters
    ----------
    z_grid : array-like
        Redshift grid points
    omega_lambda : float
        Dark energy density parameter

    Returns
    -------
    p_BH : ndarray
        Normalized PDF such that integral p_BH dz = 1
    """
    z_grid = np.asarray(z_grid)
    n_z = len(z_grid)

    p_unnorm = np.zeros(n_z)

    for i, z in enumerate(z_grid):
        if z < 1e-6:
            # Skip z=0 to avoid potential issues
            p_unnorm[i] = 0.0
            continue

        # BH formation rate density at this z and Omega_Lambda
        rate = bh_formation_rate_density(z, omega_lambda)

        # Time weight |dt/dz| converts rate per redshift to rate per time
        # This accounts for how much cosmic time is spent at each redshift
        weight = abs(dt_dz(z, omega_lambda))

        p_unnorm[i] = rate * weight

    # Normalize using trapezoidal integration
    # integral p_BH dz = 1
    norm = np.trapz(p_unnorm, z_grid)

    if norm <= 0:
        # Fallback: return uniform distribution
        return np.ones(n_z) / (z_grid[-1] - z_grid[0])

    p_BH = p_unnorm / norm

    return p_BH


def mean_formation_redshift(omega_lambda, z_min=0.0, z_max=10.0, nz=400):
    """
    Compute the mean BH formation redshift <z_form>(Omega_Lambda).

    <z_form> = integral z * p_BH(z) dz

    This gives the characteristic redshift at which black holes form
    for a given cosmology. Higher Omega_Lambda suppresses early structure
    formation, potentially shifting <z_form> to lower values.

    Parameters
    ----------
    omega_lambda : float
        Dark energy density parameter
    z_min : float
        Minimum redshift for integration
    z_max : float
        Maximum redshift for integration
    nz : int
        Number of grid points

    Returns
    -------
    z_mean : float
        Mean formation redshift
    """
    z_grid = np.linspace(z_min, z_max, nz)

    # Get formation PDF
    p_BH = bh_formation_pdf(z_grid, omega_lambda)

    # Compute <z> = integral z * p_BH(z) dz
    z_mean = np.trapz(z_grid * p_BH, z_grid)

    return z_mean


def cosmic_time_from_z(z, omega_lambda, H0=70.0, z_max_int=20.0, nz_int=500):
    """
    Compute cosmic time t(z) in Gyr by integrating dt/dz from z to z_max.

    t(z) = integral_z^{z_max} |dt/dz'| dz'

    This gives lookback time from redshift z to z_max (approximating
    z_max as "the beginning"). The absolute normalization is approximate,
    but relative trends vs Omega_Lambda are meaningful.

    Physics note: Higher Omega_Lambda means:
    - Faster expansion at late times (larger H(z) for small z)
    - So |dt/dz| is smaller for small z → less cosmic time since z

    Parameters
    ----------
    z : float
        Redshift at which to evaluate cosmic time
    omega_lambda : float
        Dark energy density parameter
    H0 : float
        Hubble constant [km/s/Mpc]
    z_max_int : float
        Upper integration limit (proxy for "the beginning")
    nz_int : int
        Number of integration points

    Returns
    -------
    t_Gyr : float
        Cosmic time [Gyr] from z_max to z
    """
    if z >= z_max_int:
        return 0.0

    # Integration grid from z to z_max
    z_grid = np.linspace(z, z_max_int, nz_int)

    # Compute |dt/dz| at each point
    dt_dz_array = np.array([abs(dt_dz(zp, omega_lambda, H0)) for zp in z_grid])

    # Integrate to get total time
    t_Gyr = np.trapz(dt_dz_array, z_grid)

    return t_Gyr


def formation_time_Gyr(omega_lambda, z_min=0.0, z_max=10.0, nz=400):
    """
    Compute representative BH formation time t_form(Omega_Lambda) in Gyr.

    Defined as t(z = <z_form>), where <z_form> is the mean formation redshift.

    This represents when (on average) black holes form in this cosmology,
    measured from z_max (early universe) to <z_form>.

    Parameters
    ----------
    omega_lambda : float
        Dark energy density parameter
    z_min : float
        Minimum redshift for mean calculation
    z_max : float
        Maximum redshift for mean calculation
    nz : int
        Number of grid points

    Returns
    -------
    t_form : float
        Representative formation time [Gyr]
    """
    # Get mean formation redshift
    z_form = mean_formation_redshift(omega_lambda, z_min, z_max, nz)

    # Convert to cosmic time
    t_form = cosmic_time_from_z(z_form, omega_lambda)

    return t_form


# =============================================================================
# Step 2: Effective Post-Formation Flux
# =============================================================================

def effective_flux_after_formation(omega_lambda, t_max_Gyr=150.0,
                                    M_bh=1e8 * M_sun, flux_kwargs=None):
    """
    Compute effective post-formation flux L_eff(Omega_Lambda).

    Steps:
    1. Find t_form from formation_time_Gyr
    2. Get flux history (t_array, L_in_array, t_iso) from v1 module
    3. Restrict to times t in [t_form, t_end]
    4. Compute time-averaged flux: L_eff = (1/Delta_t) * integral L_in dt

    Physics note: L_eff represents the average radiation flux a BH
    experiences after it forms. This drives mass inflation in the interior.
    Higher L_eff → faster mass inflation → shorter interior lifetime.

    Parameters
    ----------
    omega_lambda : float
        Dark energy density parameter
    t_max_Gyr : float
        Maximum time for flux integration [Gyr]
    M_bh : float
        Black hole mass [kg]
    flux_kwargs : dict, optional
        Additional kwargs for flux computation

    Returns
    -------
    L_eff : float
        Time-averaged post-formation flux [W]
    t_form : float
        Formation time [Gyr]
    t_end : float
        End time for averaging [Gyr]
    """
    if flux_kwargs is None:
        flux_kwargs = {}

    # Get formation time
    t_form = formation_time_Gyr(omega_lambda)

    # Get default L_crit for isolation time calculation
    L_crit, _ = get_default_L_crit(M_bh)

    # Get flux history from v1 module
    t_min_flux = 0.5  # Start flux computation at 0.5 Gyr
    n_t = flux_kwargs.get('n_t', 500)

    t_array_Gyr, L_in_array_W, t_iso_Gyr = compute_flux_history_and_isolation(
        omega_lambda, M_bh, L_crit,
        t_min_Gyr=t_min_flux, t_max_Gyr=t_max_Gyr, n_t=n_t
    )

    # Determine end time for averaging
    # Use min(t_max_Gyr, t_iso) if t_iso exists
    if t_iso_Gyr is not None and t_iso_Gyr < t_max_Gyr:
        t_end = t_iso_Gyr
    else:
        t_end = t_max_Gyr

    # Handle edge case: t_form > t_end
    # This could happen if BHs form very late (unusual cosmology)
    if t_form >= t_end:
        # Fall back to using a small interval near t_end
        # with a warning-level small L_eff
        t_form_adj = max(t_min_flux, t_end - 1.0)  # Use last 1 Gyr
        if t_form_adj >= t_end:
            # Really extreme case: return minimum flux
            valid = np.isfinite(L_in_array_W) & (L_in_array_W > 0)
            if np.any(valid):
                return np.min(L_in_array_W[valid]), t_form, t_end
            else:
                return 1e-50, t_form, t_end  # Tiny fallback
        t_form = t_form_adj

    # Select flux data in [t_form, t_end]
    mask = (t_array_Gyr >= t_form) & (t_array_Gyr <= t_end)
    mask &= np.isfinite(L_in_array_W)

    if np.sum(mask) < 2:
        # Not enough points for integration
        valid = np.isfinite(L_in_array_W) & (L_in_array_W > 0)
        if np.any(valid):
            return np.mean(L_in_array_W[valid]), t_form, t_end
        else:
            return 1e-50, t_form, t_end

    t_sel = t_array_Gyr[mask]
    L_sel = L_in_array_W[mask]

    # Time-averaged flux: L_eff = (1/Delta_t) * integral L_in dt
    Delta_t = t_sel[-1] - t_sel[0]

    if Delta_t <= 0:
        return np.mean(L_sel), t_form, t_end

    integral_L = np.trapz(L_sel, t_sel)
    L_eff = integral_L / Delta_t

    return L_eff, t_form, t_end


# =============================================================================
# Step 3: Interior Lifetime T_int_v3
# =============================================================================

def T_int_v3_from_L_eff(L_eff_array, p=0.95):
    """
    Compute interior lifetime from effective flux using power-law scaling.

    T_int ~ L_eff^{-p}

    Physics motivation: From mass inflation analysis (Poisson & Israel 1990),
    the interior blowup timescale depends on the perturbation amplitude.
    Our earlier toy PDE simulations found T_int scales roughly as
    L_eff^{-0.95}, with p close to 1.

    The power p ~ 1 means T_int is roughly inversely proportional to flux:
    - Double the flux → half the interior lifetime
    - This captures the physical intuition that stronger perturbations
      cause faster mass inflation and earlier singularity formation

    Parameters
    ----------
    L_eff_array : array-like
        Array of effective flux values [W]
    p : float
        Power-law exponent (default 0.95 from numerical fits)

    Returns
    -------
    T_int_raw : ndarray
        Unnormalized interior lifetimes
    T_int_norm : ndarray
        Normalized to max = 1
    """
    L_eff_array = np.asarray(L_eff_array)

    # Handle zero/negative values
    L_safe = np.where(L_eff_array > 0, L_eff_array, 1e-50)

    # T_int ~ L_eff^{-p}
    T_int_raw = L_safe ** (-p)

    # Normalize to max = 1
    T_max = np.max(T_int_raw[np.isfinite(T_int_raw)])
    T_int_norm = T_int_raw / T_max if T_max > 0 else T_int_raw

    return T_int_raw, T_int_norm


def scan_T_int_v3_vs_omega_lambda(omega_lambdas, p=0.95, t_max_Gyr=150.0,
                                   M_bh=1e8 * M_sun, verbose=True):
    """
    Scan T_int_v3 over a range of Omega_Lambda values.

    For each Omega_Lambda:
    1. Compute t_form (when BHs typically form)
    2. Compute L_eff (post-formation flux average)
    3. Apply T_int ~ L_eff^{-p}

    Parameters
    ----------
    omega_lambdas : array-like
        Array of Omega_Lambda values to scan
    p : float
        Power-law exponent for T_int ~ L_eff^{-p}
    t_max_Gyr : float
        Maximum time for flux integration [Gyr]
    M_bh : float
        Black hole mass [kg]
    verbose : bool
        Print progress messages

    Returns
    -------
    omega_array : ndarray
        Omega_Lambda values
    t_form_array : ndarray
        Formation times [Gyr]
    L_eff_array : ndarray
        Effective post-formation fluxes [W]
    T_int_raw : ndarray
        Raw interior lifetimes
    T_int_norm : ndarray
        Normalized interior lifetimes (max = 1)
    """
    omega_array = np.asarray(omega_lambdas)
    n_points = len(omega_array)

    t_form_array = np.zeros(n_points)
    L_eff_array = np.zeros(n_points)

    if verbose:
        print(f"Scanning T_int_v3 for {n_points} Omega_Lambda values...")
        print(f"  Using p = {p} (T_int ~ L_eff^{{-p}})")
        print()

    for i, ol in enumerate(omega_array):
        if verbose and (i % 5 == 0 or i == n_points - 1):
            print(f"  Processing Omega_Lambda = {ol:.3f} ({i+1}/{n_points})...",
                  end='\r')

        L_eff, t_form, t_end = effective_flux_after_formation(
            ol, t_max_Gyr=t_max_Gyr, M_bh=M_bh
        )

        t_form_array[i] = t_form
        L_eff_array[i] = L_eff

    if verbose:
        print()  # Newline after progress

    # Compute T_int from L_eff
    T_int_raw, T_int_norm = T_int_v3_from_L_eff(L_eff_array, p=p)

    if verbose:
        print(f"  t_form range: [{np.min(t_form_array):.2f}, {np.max(t_form_array):.2f}] Gyr")
        print(f"  L_eff range: [{np.min(L_eff_array):.4e}, {np.max(L_eff_array):.4e}] W")
        print(f"  T_int_norm range: [{np.min(T_int_norm):.4f}, {np.max(T_int_norm):.4f}]")
        print()

    return omega_array, t_form_array, L_eff_array, T_int_raw, T_int_norm


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("T_int v3 from Formation History - Module Test")
    print("=" * 65)
    print()

    # Test formation PDF and mean redshift
    print("1. Testing BH formation PDF and mean redshift:")
    print("-" * 50)

    test_ols = [0.3, 0.5, 0.7, 0.9]

    for ol in test_ols:
        z_mean = mean_formation_redshift(ol)
        t_form = formation_time_Gyr(ol)
        print(f"  OL = {ol}: <z_form> = {z_mean:.3f}, t_form = {t_form:.2f} Gyr")

    print()

    # Test effective flux
    print("2. Testing effective post-formation flux:")
    print("-" * 50)

    for ol in test_ols:
        L_eff, t_f, t_e = effective_flux_after_formation(ol)
        print(f"  OL = {ol}: L_eff = {L_eff:.4e} W, t_form = {t_f:.2f}, t_end = {t_e:.2f} Gyr")

    print()

    # Test mini scan
    print("3. Mini scan test:")
    print("-" * 50)

    omega_test = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    omega, t_form, L_eff, T_raw, T_norm = scan_T_int_v3_vs_omega_lambda(
        omega_test, verbose=True
    )

    print("\nResults:")
    print("-" * 70)
    print(f"{'OL':>8} | {'t_form':>10} | {'L_eff':>12} | {'T_int_raw':>12} | {'T_int_norm':>10}")
    print("-" * 70)
    for i in range(len(omega)):
        print(f"{omega[i]:>8.3f} | {t_form[i]:>10.2f} | {L_eff[i]:>12.4e} | {T_raw[i]:>12.4e} | {T_norm[i]:>10.4f}")

    print()
    print("Done!")
