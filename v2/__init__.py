"""
v2 - Refined Semi-Analytic Cosmological Natural Selection Model

This package provides a more physically grounded implementation of the
fertility functional F(Omega_Lambda) = N_BH(Omega_Lambda) * T_int(Omega_Lambda).

Modules:
--------
cosmology/
    structure_formation : Linear growth factor D(a), critical overdensity
    halo_mass_function  : Press-Schechter mass function, sigma(M)

black_holes/
    nbh_vs_lambda      : N_BH(Omega_Lambda) from P-S integral
    interior_lifetime  : T_int(Omega_Lambda) from mass-inflation flux integral

fertility : Combined fertility functional F(Omega_Lambda)
config    : Default parameters and constants
run_scan  : Main driver for 1D parameter scans
"""

__version__ = "2.0.0"
