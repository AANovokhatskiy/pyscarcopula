"""Shared bounded-block helpers for transfer-matrix GoF routines."""

import numpy as np


def forward_block_size(
        K, max_elements=2_000_000, max_rows=512, element_width=1):
    """Return a row block size bounded by an approximate element budget.

    The block routines allocate arrays with shape roughly
    ``(block_size, K, element_width)``.  Keeping the product below
    ``max_elements`` gives predictable memory use while still preserving
    vectorized copula kernels inside each block.
    """
    K = max(1, int(K))
    max_elements = max(1, int(max_elements))
    max_rows = max(1, int(max_rows))
    element_width = max(1, int(element_width))
    return max(1, min(max_rows, max_elements // (K * element_width)))


def iter_forward_weight_blocks(
        grid, u, copula, x_grid=None, block_size=None,
        max_elements=2_000_000, emission_block=None,
        include_block_info=False, element_width=1):
    """Stream predictive weights while evaluating emissions in row blocks.

    By default yields ``(k, local, weights, fi_block)`` where ``fi_block`` has
    shape ``(block_rows, grid.K)`` and ``local`` is the row offset inside that
    block.  With ``include_block_info=True``, yields
    ``(k, local, weights, fi_block, u_block, start, stop)`` so callers can reuse
    block-local caches such as Student-t PPF tables.  The current row's
    ``weights`` are the normalized predictive mass before observing row ``k``.
    The helper never materializes full ``(T, K)`` weights; only one emission
    block and one ``(K,)`` predictive density are retained.  Callers that
    allocate additional block-local arrays can increase ``element_width`` so
    the automatic block size reflects their wider memory footprint.

    ``emission_block`` can be supplied for models whose density depends on the
    absolute time index.  It must accept ``(u_block, x_grid, start, stop)`` and
    return a float array with shape ``(stop - start, grid.K)``.
    """
    u = np.asarray(u, dtype=np.float64)
    n = len(u)
    if x_grid is None:
        x_grid = grid.z + grid.mu
    else:
        x_grid = np.asarray(x_grid, dtype=np.float64)
    if block_size is None:
        block_size = forward_block_size(
            grid.K,
            max_elements=max_elements,
            element_width=element_width,
        )
    block_size = max(1, int(block_size))

    if emission_block is None:
        def emission_block(u_block, x_grid, start, stop):
            return copula.copula_grid_batch(u_block, x_grid)

    phi = grid.p0.copy()
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        u_block = u[start:stop]
        fi_block = np.asarray(
            emission_block(u_block, x_grid, start, stop),
            dtype=np.float64,
        )
        expected_shape = (stop - start, grid.K)
        if fi_block.shape != expected_shape:
            raise ValueError(
                "emission_block returned shape "
                f"{fi_block.shape}, expected {expected_shape}")

        for local, k in enumerate(range(start, stop)):
            weights = grid.predictive_weights_from_phi(phi)
            if include_block_info:
                yield k, local, weights, fi_block, u_block, start, stop
            else:
                yield k, local, weights, fi_block

            if k < n - 1:
                phi = grid.advance_forward_phi(phi, fi_block[local])
