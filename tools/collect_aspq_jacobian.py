#!/usr/bin/env python
"""Collect ASPQ action-Jacobian metrics for DuQuant.

This is the production-facing entrypoint for ASPQ calibration. It reuses the
sanity-check collector, but defaults should be provided explicitly by the run
script:

    --metric-all-layers
    --metric-output results/aspq_metrics/aspq_metrics_top64.pt

The saved .pt is a dict keyed by layer name. Each record contains:

    U:       [d_out, k] top action-subspace eigenvectors
    eigvals: [k]        corresponding eigenvalues

DuQuant consumes this file when GR00T_DUQUANT_ASPQ=1 and
GR00T_DUQUANT_ASPQ_PATH points at it.
"""

from tools.aspq_jacobian_sanity import main


if __name__ == "__main__":
    main()
