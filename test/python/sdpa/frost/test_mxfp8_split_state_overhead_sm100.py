# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Isolated timing probe for the existing MXFP8 partial-state merge path.

This is deliberately not the causal-diagonal implementation.  ``split_kv=2``
partitions each row's visible KV span into two disjoint pieces, writes a
normalized partial O and LSE for each, then invokes ``split_combine_sm100``.
It therefore measures the mandatory partial-state workspace and merge cost of
a frontend two-pipeline design before adding the diagonal-specific producers.
"""

import os
import time

import pytest
import torch

from frost_test_utils import requires_dsl, requires_pre_rubin_blackwell


pytestmark = [requires_pre_rubin_blackwell, requires_dsl]


def _quantized(shape):
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    source = torch.randn(*shape, device="cuda", dtype=torch.bfloat16).float() * 0.5
    data, _dequant, scale, *_ = quantize_to_mxfp8(source, *shape)
    return data.reshape(*shape), scale


def _make_api(q, k, v, sf_q, sf_k, sf_v, split_kv):
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    output = torch.empty(q.shape[0], q.shape[1], q.shape[2], v.shape[3], device="cuda", dtype=torch.bfloat16)
    amax = torch.empty(1, device="cuda", dtype=torch.float32)
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=output,
        dtype_o=torch.bfloat16,
        is_causal=True,
        split_kv=split_kv,
    )
    assert api.check_support()
    api.compile()
    workspace_bytes = api.scratch_workspace_bytes()
    workspace = torch.empty(workspace_bytes, device="cuda", dtype=torch.uint8) if workspace_bytes else None

    def run():
        api.execute(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            o_tensor=output,
            sf_q=sf_q,
            sf_k=sf_k,
            sf_v=sf_v,
            amax_o=amax,
            workspace=workspace,
        )

    return run, output, workspace_bytes


def _time_us(run, repeats=100):
    for _ in range(20):
        run()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        run()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / repeats


@pytest.mark.L0
def test_mxfp8_split_state_merge_overhead_d128_s2048():
    """Report the exact two-state merge overhead on the target GQA shape."""
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("MXFP8 SM100 split-state probe requires cc10.0 or cc10.3")

    torch.manual_seed(17)
    b = 1
    h_q = int(os.environ.get("SPLIT_STATE_HQ", "64"))
    h_kv = int(os.environ.get("SPLIT_STATE_HKV", "8"))
    s = int(os.environ.get("SPLIT_STATE_S", "2048"))
    d = 128
    q, sf_q = _quantized((b, h_q, s, d))
    k, sf_k = _quantized((b, h_kv, s, d))
    v, sf_v = _quantized((b, h_kv, s, d))

    run_one, o_one, ws_one = _make_api(q, k, v, sf_q, sf_k, sf_v, split_kv=1)
    run_two, o_two, ws_two = _make_api(q, k, v, sf_q, sf_k, sf_v, split_kv=2)

    run_one()
    run_two()
    torch.cuda.synchronize()
    max_abs = (o_one.float() - o_two.float()).abs().max().item()
    assert max_abs <= 5e-2, f"split-state result differs from single-state result: {max_abs}"

    one_us = _time_us(run_one)
    two_us = _time_us(run_two)
    print(
        f"MXFP8 split-state overhead D128 Hq={h_q} Hkv={h_kv} S={s} "
        f"split1={one_us:.2f}us split2={two_us:.2f}us "
        f"delta={two_us - one_us:+.2f}us ({(two_us / one_us - 1.0) * 100.0:+.1f}%) "
        f"workspace_split1={ws_one}B workspace_split2={ws_two}B max_abs={max_abs:.6f}",
        flush=True,
    )


def _make_partial_state_api(q, k, v, v_bf16, sf_q, sf_k, sf_v, *, mode, o_bshd, lse, cga=None):
    """Build one producer of the experimental causal two-state decomposition."""
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    amax = torch.empty(1, device="cuda", dtype=torch.float32)
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        # o_bshd is a non-contiguous BHSD view of its final BSHD partial slab,
        # so execute writes the combine input directly with no copy.
        sample_o=o_bshd,
        sample_lse=lse,
        dtype_o=torch.bfloat16,
        is_causal=True,
        prevent_leakage=(mode == 2),
        partial_state_mode=mode,
        cga=cga,
    )
    assert api.check_support()
    print(f"two-state: compiling state{mode} cga={cga if cga is not None else 2}", flush=True)
    api.compile()
    print(f"two-state: compiled state{mode}", flush=True)

    def run():
        api.execute(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            v_bf16=v_bf16 if mode == 2 else None,
            o_tensor=o_bshd,
            lse_tensor=lse,
            sf_q=sf_q,
            sf_k=sf_k,
            sf_v=sf_v,
            amax_o=amax,
        )

    return run


def _pack_diagonal_bshd(tensor, block=32):
    """Materialize each contiguous Q/K/V diagonal block as one BSHD batch item.

    The SM100 public adapter accepts logical BHSD and converts it to compact
    BSHD before binding its TMA descriptors. Giving it a strided BHSD block
    view would induce an implicit copy on every execute. This producer pack is
    explicit and outside the timed kernel; its transpose below lets _to_bshd
    recover the compact buffer as a view.

    This first producer deliberately supports B=1 only. General B requires
    an explicit (B, G) task map because flattening a strided view across B/G
    is not affine when heads are interleaved in the source BHSD allocation.
    """
    b, h, s, d = tensor.shape
    assert b == 1 and s % block == 0
    return tensor.transpose(1, 2).contiguous().reshape(b * (s // block), block, h, d)


def _pack_diagonal_sf(sf, *, b, h, s, block=32):
    """Pack source F8_128x4 scale atoms for independent local tasks.

    S=128 tasks preserve whole source atoms. S=32 tasks rebase one of the
    four source Q32/K32 scale quadrants into the local first quadrant; padded
    rows receive finite duplicate bytes because they are never valid.
    """
    assert b == 1 and s % block == 0 and block in (32, 128)
    if block == 128:
        # A whole source F8_128x4 atom becomes one local attention problem.
        raw = sf.view(torch.int8).reshape(b, h, s // 128, 512)
        return raw.permute(0, 2, 1, 3).contiguous().reshape(b * (s // 128), h, 512)

    # A Q32 task needs its source atom phase rebased to local phase zero.
    assert s % 128 == 0
    groups = s // 32
    raw = sf.view(torch.int8).reshape(b, h, s // 128, 32, 16)
    raw = raw.permute(0, 2, 1, 3, 4)
    packed = torch.empty((b, groups, h, 32, 16), dtype=torch.int8, device=sf.device)
    for phase in range(4):
        packed[:, phase::4] = raw[..., phase * 4 : (phase + 1) * 4].repeat(1, 1, 1, 1, 4)
    return packed.reshape(b * groups, h, 512)


def _make_diagonal_state_api(q, k, v, v_bf16, sf_q, sf_k, sf_v, *, o_bshd, lse):
    """Compile the exact BF16 boundary state over independent local tasks.

    Each S=128 task owns four global Q32/K32 diagonal blocks in one K128
    tile. Its causal local problem contains no preceding global KV tile, so
    state 2 no longer walks the full causal prefix merely to mask it out.
    prevent_leakage retains the exact original-BF16 V contribution.

    """
    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100

    amax = torch.empty(1, device="cuda", dtype=torch.float32)
    api = SdpaFwdDslSm100(
        sample_q=q,
        sample_k=k,
        sample_v=v,
        sample_o=o_bshd,
        sample_lse=lse,
        dtype_o=torch.bfloat16,
        is_causal=True,
        prevent_leakage=True,
        partial_state_mode=2,
    )
    assert api.check_support()
    print(f"diagonal-state: compiling {q.shape[0]} independent S={q.shape[2]} tasks", flush=True)
    api.compile()
    print("diagonal-state: compiled", flush=True)

    def run():
        api.execute(
            q_tensor=q,
            k_tensor=k,
            v_tensor=v,
            v_bf16=v_bf16,
            o_tensor=o_bshd,
            lse_tensor=lse,
            sf_q=sf_q,
            sf_k=sf_k,
            sf_v=sf_v,
            amax_o=amax,
        )

    return run


@pytest.mark.L0
def test_mxfp8_causal_two_state_d128():
    """Validate and time MXFP8 non-boundary + BF16 boundary state merge.

    This is the actual frontend decomposition prototype, unlike the generic
    split_kv=2 probe above.  It deliberately leaves the two producer kernels
    serial on one stream for this first measurement; it separates their state
    math and proves that split_combine reconstructs the current fused-safe
    output before stream overlap is considered.
    """
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("MXFP8 SM100 two-state probe requires cc10.0 or cc10.3")

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from cudnn.sdpa.fwd.kernels import split_combine_sm100
    import cutlass

    torch.manual_seed(23)
    b = 1
    h_q = int(os.environ.get("SPLIT_STATE_HQ", "2"))
    h_kv = int(os.environ.get("SPLIT_STATE_HKV", "2"))
    s = int(os.environ.get("SPLIT_STATE_S", "256"))
    d = 128
    q, sf_q = _quantized((b, h_q, s, d))
    k, sf_k = _quantized((b, h_kv, s, d))
    v_bf16 = torch.randn(b, h_kv, s, d, device="cuda", dtype=torch.bfloat16) * 0.5
    v, sf_v = _quantized(tuple(v_bf16.shape))

    # The comparison target is the previously validated fused-safe path: both
    # use MXFP8 Q/K scores and original BF16 V on precisely the causal K32 block.
    o_fused = torch.empty(b, h_q, s, d, device="cuda", dtype=torch.bfloat16)
    lse_fused = torch.empty(b, h_q, s, device="cuda", dtype=torch.float32)
    amax_fused = torch.empty(1, device="cuda", dtype=torch.float32)
    print("two-state: compiling fused-safe reference", flush=True)
    fused = SdpaFwdDslSm100(
        sample_q=q, sample_k=k, sample_v=v, sample_o=o_fused, sample_lse=lse_fused,
        dtype_o=torch.bfloat16, is_causal=True, prevent_leakage=True,
    )
    assert fused.check_support()
    fused.compile()
    print("two-state: compiled fused-safe reference", flush=True)

    def run_fused():
        fused.execute(
            q_tensor=q, k_tensor=k, v_tensor=v, v_bf16=v_bf16,
            o_tensor=o_fused, lse_tensor=lse_fused,
            sf_q=sf_q, sf_k=sf_k, sf_v=sf_v, amax_o=amax_fused,
        )

    # split_combine consumes BSHD partial O slabs.  Expose BHSD views of each
    # slab to the regular adapter, avoiding an otherwise artificial copy.
    o_partial = torch.empty(2 * b, s, h_q, d, device="cuda", dtype=torch.bfloat16)
    lse_partial = torch.empty(2 * b, h_q, s, device="cuda", dtype=torch.float32)
    o_non_boundary = o_partial[:b].transpose(1, 2)
    o_boundary = o_partial[b:].transpose(1, 2)
    run_non_boundary = _make_partial_state_api(
        q, k, v, v_bf16, sf_q, sf_k, sf_v,
        mode=1, o_bshd=o_non_boundary, lse=lse_partial[:b],
    )
    boundary_cga = int(os.environ.get("SPLIT_STATE_BOUNDARY_CGA", "2"))
    run_boundary = _make_partial_state_api(
        q, k, v, v_bf16, sf_q, sf_k, sf_v,
        mode=2, o_bshd=o_boundary, lse=lse_partial[b:], cga=boundary_cga,
    )
    o_merged = torch.empty(b, s, h_q, d, device="cuda", dtype=torch.bfloat16)
    lse_merged = torch.empty(b, h_q, s, device="cuda", dtype=torch.float32)
    print("two-state: compiling merge", flush=True)
    combine = split_combine_sm100.compile(
        b=b, h=h_q, sq=s, d_v=d, splits=2, dtype_o="bf16", has_lse=True, has_amax=False,
    )
    print("two-state: compiled merge", flush=True)

    def run_combine():
        combine(
            o_partial, lse_partial, o_merged, lse_merged, None,
            (b, h_q, s, d), cutlass.Int32(2),
        )

    def run_split():
        run_non_boundary()
        run_boundary()
        run_combine()

    run_fused()
    run_split()
    torch.cuda.synchronize()
    max_abs_o = (o_fused.float() - o_merged.transpose(1, 2).float()).abs().max().item()
    max_abs_lse = (lse_fused - lse_merged).abs().max().item()
    assert max_abs_o <= 5e-2, f"two-state O differs from fused-safe output: {max_abs_o}"
    assert max_abs_lse <= 5e-2, f"two-state LSE differs from fused-safe output: {max_abs_lse}"

    fused_us = _time_us(run_fused, repeats=30)
    non_boundary_us = _time_us(run_non_boundary, repeats=30)
    boundary_us = _time_us(run_boundary, repeats=30)
    combine_us = _time_us(run_combine, repeats=30)
    split_us = _time_us(run_split, repeats=30)
    print(
        f"MXFP8 causal two-state D128 Hq={h_q} Hkv={h_kv} S={s} "
        f"fused_safe={fused_us:.2f}us non_boundary={non_boundary_us:.2f}us "
        f"boundary_bf16_cga{boundary_cga}={boundary_us:.2f}us merge={combine_us:.2f}us split_serial={split_us:.2f}us "
        f"delta={split_us - fused_us:+.2f}us ({(split_us / fused_us - 1.0) * 100.0:+.1f}%) "
        f"max_abs_o={max_abs_o:.6f} max_abs_lse={max_abs_lse:.6f}",
        flush=True,
    )


@pytest.mark.L0
def test_mxfp8_causal_diagonal_only_state_d128():
    """Evaluate a separate local BF16 diagonal-state producer.

    State 1 remains the MXFP8 non-boundary producer. State 2 is re-indexed
    into compact local tasks, so it does not visit preceding global KV tiles.
    Its O and LSE bind directly into the state-2 merge slabs; the reported
    diagonal time contains no output gather or scatter.
    """
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("MXFP8 SM100 diagonal-state probe requires cc10.0 or cc10.3")

    from cudnn.sdpa.fwd.api_dsl import SdpaFwdDslSm100
    from cudnn.sdpa.fwd.kernels import split_combine_sm100
    import cutlass

    torch.manual_seed(29)
    b = 1
    h_q = int(os.environ.get("SPLIT_STATE_HQ", "2"))
    h_kv = int(os.environ.get("SPLIT_STATE_HKV", "2"))
    s = int(os.environ.get("SPLIT_STATE_S", "256"))
    d = 128
    assert s % 128 == 0, "short diagonal scale rebasing currently requires S divisible by 128"
    local_s = int(os.environ.get("SPLIT_STATE_DIAGONAL_S", "32"))
    assert local_s in (32, 128) and s % local_s == 0
    groups = s // local_s

    q, sf_q = _quantized((b, h_q, s, d))
    k, sf_k = _quantized((b, h_kv, s, d))
    v_bf16 = torch.randn(b, h_kv, s, d, device="cuda", dtype=torch.bfloat16) * 0.5
    v, sf_v = _quantized(tuple(v_bf16.shape))

    # Exact fused-safe reference.
    o_fused = torch.empty(b, h_q, s, d, device="cuda", dtype=torch.bfloat16)
    lse_fused = torch.empty(b, h_q, s, device="cuda", dtype=torch.float32)
    amax_fused = torch.empty(1, device="cuda", dtype=torch.float32)
    fused = SdpaFwdDslSm100(
        sample_q=q, sample_k=k, sample_v=v, sample_o=o_fused, sample_lse=lse_fused,
        dtype_o=torch.bfloat16, is_causal=True, prevent_leakage=True,
    )
    assert fused.check_support()
    print("diagonal-state: compiling fused-safe reference", flush=True)
    fused.compile()

    def run_fused():
        fused.execute(
            q_tensor=q, k_tensor=k, v_tensor=v, v_bf16=v_bf16,
            o_tensor=o_fused, lse_tensor=lse_fused,
            sf_q=sf_q, sf_k=sf_k, sf_v=sf_v, amax_o=amax_fused,
        )

    # State-1 owns fully visible MXFP8 K32 blocks. Both producers write their
    # normalized state directly into split_combine input slabs.
    o_partial = torch.empty(2 * b, s, h_q, d, device="cuda", dtype=torch.bfloat16)
    lse_partial = torch.empty(2 * b, h_q, s, device="cuda", dtype=torch.float32)
    run_non_boundary = _make_partial_state_api(
        q, k, v, v_bf16, sf_q, sf_k, sf_v,
        mode=1, o_bshd=o_partial[:b].transpose(1, 2), lse=lse_partial[:b],
    )

    # Explicit pre-pack, once per attention invocation in this prototype.
    # Passing .transpose(1, 2) restores compact BSHD as a zero-copy adapter
    # view during every measured execute.
    q_diag_bshd = _pack_diagonal_bshd(q, block=local_s)
    k_diag_bshd = _pack_diagonal_bshd(k, block=local_s)
    v_diag_bshd = _pack_diagonal_bshd(v, block=local_s)
    v_bf16_diag_bshd = _pack_diagonal_bshd(v_bf16, block=local_s)
    sf_q_diag = _pack_diagonal_sf(sf_q, b=b, h=h_q, s=s, block=local_s)
    sf_k_diag = _pack_diagonal_sf(sf_k, b=b, h=h_kv, s=s, block=local_s)
    sf_v_diag = _pack_diagonal_sf(sf_v, b=b, h=h_kv, s=s, block=local_s)

    # The O BSHD view is contiguous and aliases the second merged-state slab.
    # The LSE as_strided view maps task g, head h, local row r to the original
    # [head h, global token local_s*g+r] location. It is non-overlapping
    # the short kernel a direct merge-ready destination without a scatter.
    o_diag_bshd = o_partial[b:].view(groups, local_s, h_q, d)
    o_diag_bhsd = o_diag_bshd.transpose(1, 2)
    lse_diag = lse_partial[b:].as_strided((groups, h_q, local_s), (local_s, s, 1))
    run_diagonal = _make_diagonal_state_api(
        q_diag_bshd.transpose(1, 2),
        k_diag_bshd.transpose(1, 2),
        v_diag_bshd.transpose(1, 2),
        v_bf16_diag_bshd.transpose(1, 2),
        sf_q_diag,
        sf_k_diag,
        sf_v_diag,
        o_bshd=o_diag_bhsd,
        lse=lse_diag,
    )

    o_merged = torch.empty(b, s, h_q, d, device="cuda", dtype=torch.bfloat16)
    lse_merged = torch.empty(b, h_q, s, device="cuda", dtype=torch.float32)
    print("diagonal-state: compiling merge", flush=True)
    combine = split_combine_sm100.compile(
        b=b, h=h_q, sq=s, d_v=d, splits=2, dtype_o="bf16", has_lse=True, has_amax=False,
    )

    def run_combine():
        combine(
            o_partial, lse_partial, o_merged, lse_merged, None,
            (b, h_q, s, d), cutlass.Int32(2),
        )

    def run_split():
        run_non_boundary()
        run_diagonal()
        run_combine()

    def run_pack():
        _pack_diagonal_bshd(q, block=local_s)
        _pack_diagonal_bshd(k, block=local_s)
        _pack_diagonal_bshd(v, block=local_s)
        _pack_diagonal_bshd(v_bf16, block=local_s)
        _pack_diagonal_sf(sf_q, b=b, h=h_q, s=s, block=local_s)
        _pack_diagonal_sf(sf_k, b=b, h=h_kv, s=s, block=local_s)
        _pack_diagonal_sf(sf_v, b=b, h=h_kv, s=s, block=local_s)

    run_fused()
    run_split()
    torch.cuda.synchronize()
    max_abs_o = (o_fused.float() - o_merged.transpose(1, 2).float()).abs().max().item()
    max_abs_lse = (lse_fused - lse_merged).abs().max().item()
    assert max_abs_o <= 5e-2, f"diagonal-state O differs from fused-safe output: {max_abs_o}"
    assert max_abs_lse <= 5e-2, f"diagonal-state LSE differs from fused-safe output: {max_abs_lse}"

    fused_us = _time_us(run_fused, repeats=30)
    non_boundary_us = _time_us(run_non_boundary, repeats=30)
    diagonal_us = _time_us(run_diagonal, repeats=30)
    combine_us = _time_us(run_combine, repeats=30)
    split_us = _time_us(run_split, repeats=30)
    pack_us = _time_us(run_pack, repeats=30)
    print(
        f"MXFP8 causal diagonal-only state D128 Hq={h_q} Hkv={h_kv} S={s} "
        f"fused_safe={fused_us:.2f}us non_boundary={non_boundary_us:.2f}us "
        f"diagonal_bf16_s{local_s}={diagonal_us:.2f}us merge={combine_us:.2f}us "
        f"split_serial={split_us:.2f}us pack_explicit={pack_us:.2f}us "
        f"split_plus_pack={split_us + pack_us:.2f}us "
        f"max_abs_o={max_abs_o:.6f} max_abs_lse={max_abs_lse:.6f}",
        flush=True,
    )
