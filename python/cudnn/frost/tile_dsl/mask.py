# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


import cutlass
from cutlass._mlir.dialects import arith

from .constants import MASK_CAUSAL, MASK_NONE, MASK_PADDED, MASK_SWA  # noqa: F401

_NEG_INF_BITS = -3.4028235e38


def apply_mask_chunk(
    reg_S,
    q_abs,
    kv_col_base,
    seq_kv_len,
    window_left: int,
    mask_flags: int,
    N: int = 64,
    bottom_right: int = 0,
    causal_diag=None,
    mask_value: float = _NEG_INF_BITS,
    window_right: int = 0,
):
    # mask_value: what a masked score becomes.  Default is the legacy finite
    # sentinel; the f16 prefill kernels pass float("-inf") so a fully-masked
    # row's max stays -inf under any scale and the canonical
    # `max == -inf -> substitute 0` guard (row_max_for_exp2) applies.
    if cutlass.const_expr(mask_flags == MASK_NONE):
        return reg_S

    neg_inf = cutlass.Float32(mask_value)
    # The whole band shifts with the diagonal: under BOTTOM_RIGHT the SWA
    # lower limit is q + (S_kv - S_q) - W — the same causal_diag offset the
    # upper (causal) limit uses below. Top-left keeps the plain q - W.
    q_minus_w = None
    if mask_flags & MASK_SWA:
        swa_base = (q_abs + causal_diag) if bottom_right else q_abs
        q_minus_w = swa_base - cutlass.Int32(window_left)
    # window_right is the compile-time diagonal-band right bound (cuDNN
    # diagonal_band_right_bound): kv columns up to q + window_right (plus the
    # bottom-right diagonal offset) stay unmasked. 0 = plain causal.
    if cutlass.const_expr((mask_flags & MASK_CAUSAL) and bottom_right):
        q_caus_lim = q_abs + causal_diag
    else:
        q_caus_lim = q_abs
    if cutlass.const_expr((mask_flags & MASK_CAUSAL) and window_right != 0):
        q_caus_lim = q_caus_lim + cutlass.Int32(window_right)

    elems = []
    for i in range(N):
        kv_abs = kv_col_base + cutlass.Int32(i)
        masked = None
        if cutlass.const_expr(mask_flags & MASK_PADDED):
            term = kv_abs >= seq_kv_len
            masked = term if masked is None else (masked | term)
        if cutlass.const_expr(mask_flags & MASK_CAUSAL):
            term = kv_abs > q_caus_lim
            masked = term if masked is None else (masked | term)
        if cutlass.const_expr(mask_flags & MASK_SWA):
            term = kv_abs < q_minus_w
            masked = term if masked is None else (masked | term)
        val = cutlass.Float32(
            arith.select(
                masked.ir_value(),
                neg_inf.ir_value(),
                reg_S[i].ir_value(),
            )
        )
        elems.append(val)
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


def select_causal_boundary_chunk(
    reg_p,
    q_abs,
    kv_col_base,
    *,
    N: int,
    bottom_right: int,
    causal_diag,
    window_right: int,
    mx_block: int = 32,
):
    """Keep only each row boundary MX block.

    reg_p is the high-precision probability vector produced before the MXFP8
    cast. This helper deliberately does not change the score mask or softmax
    denominator: callers use its result for the BF16 boundary MMA and zero the
    same entries only in the MXFP8 P stream.
    """
    q_caus_lim = (
        q_abs
        + (causal_diag if bottom_right else cutlass.Int32(0))
        + cutlass.Int32(window_right)
    )
    zero_i32 = cutlass.Int32(0)
    clamped_lim = cutlass.Int32(
        arith.select(
            (q_caus_lim >= zero_i32).ir_value(),
            q_caus_lim.ir_value(),
            zero_i32.ir_value(),
        )
    )
    block_start = (clamped_lim // cutlass.Int32(mx_block)) * cutlass.Int32(mx_block)
    zero = cutlass.Float32(0.0)
    elems = []
    for i in range(N):
        kv_abs = kv_col_base + cutlass.Int32(i)
        in_boundary = (q_caus_lim >= cutlass.Int32(0)) & (kv_abs >= block_start) & (kv_abs < block_start + cutlass.Int32(mx_block))
        elems.append(cutlass.Float32(arith.select(in_boundary.ir_value(), reg_p[i].ir_value(), zero.ir_value())))
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


def select_causal_boundary_phase(
    reg_p,
    q_abs,
    kv_col_base,
    *,
    bottom_right: int,
    causal_diag,
    window_right: int,
    mx_block: int = 32,
):
    """Return one compact ``mx_block``-wide BF16 P operand.

    ``kv_col_base`` is the start of an exact 32-position K phase.  This is
    intentionally distinct from :func:`select_causal_boundary_chunk`: the
    former preserves the 64-wide MXFP8 P layout for subtraction, while this
    returns a compact 32-value vector for the BF16 boundary MMA.  A row whose
    boundary is another phase produces zeroes, so one shared MMA still serves
    all 128 rows of the Q tile.
    """
    q_caus_lim = (
        q_abs
        + (causal_diag if bottom_right else cutlass.Int32(0))
        + cutlass.Int32(window_right)
    )
    zero_i32 = cutlass.Int32(0)
    clamped_lim = cutlass.Int32(
        arith.select(
            (q_caus_lim >= zero_i32).ir_value(),
            q_caus_lim.ir_value(),
            zero_i32.ir_value(),
        )
    )
    boundary_start = (clamped_lim // cutlass.Int32(mx_block)) * cutlass.Int32(mx_block)
    zero = cutlass.Float32(0.0)
    elems = []
    for i in range(mx_block):
        kv_abs = kv_col_base + cutlass.Int32(i)
        in_boundary = (
            (q_caus_lim >= zero_i32)
            & (kv_abs >= boundary_start)
            & (kv_abs < boundary_start + cutlass.Int32(mx_block))
        )
        elems.append(
            cutlass.Float32(
                arith.select(in_boundary.ir_value(), reg_p[i].ir_value(), zero.ir_value())
            )
        )
    return cutlass.Vector.from_elements(tuple(elems), cutlass.Float32)


def causal_boundary_chunk_active(
    q_tile_base,
    kv_loop,
    chunk_in_tile: int,
    *,
    q_tile_rows: int,
    tile_n: int,
    chunk_size: int,
    bottom_right: int,
    causal_diag,
    window_right: int,
):
    """Whether a uniform Q tile needs this coarse P/V boundary chunk.

    The chunk is a transport unit; the finer 32-position predicate in
    select_causal_boundary_chunk zeros every non-boundary P value.
    """
    offset = (causal_diag if bottom_right else cutlass.Int32(0)) + cutlass.Int32(window_right)
    first_limit = q_tile_base + offset
    last_limit = first_limit + cutlass.Int32(q_tile_rows - 1)
    zero = cutlass.Int32(0)
    first_clamped = cutlass.Int32(arith.select((first_limit >= zero).ir_value(), first_limit.ir_value(), zero.ir_value()))
    last_clamped = cutlass.Int32(arith.select((last_limit >= zero).ir_value(), last_limit.ir_value(), zero.ir_value()))
    boundary_chunk_lo = first_clamped // cutlass.Int32(chunk_size)
    boundary_chunk_hi = last_clamped // cutlass.Int32(chunk_size)
    current_chunk = kv_loop * cutlass.Int32(tile_n // chunk_size) + cutlass.Int32(chunk_in_tile)
    return (last_limit >= zero) & (current_chunk >= boundary_chunk_lo) & (current_chunk <= boundary_chunk_hi)
