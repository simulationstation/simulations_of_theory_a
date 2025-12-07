"""
Black holes subpackage for v2 pipeline.

Provides calculations for:
- Black hole abundance N_BH(Omega_Lambda) from structure formation
- Interior lifetime T_int(Omega_Lambda) from mass-inflation physics
"""

from .nbh_vs_lambda import compute_N_BH
from .interior_lifetime import compute_T_int
