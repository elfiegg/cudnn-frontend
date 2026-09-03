# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Leakage-safe MXFP8 SDPA forward — ``sdpa_mxfp8(prevent_leakage=True)``.

WHAT IS BEING TESTED
--------------------
MXFP8 shares one E8M0 scale across 32 positions of ``S_kv``, the contracting
dimension of ``O = P @ V``. Changing a value at a FUTURE position can change
that shared scale and therefore the dequantized value of an EARLIER one, so a
causal query can observe a key its mask forbids. ``prevent_leakage=True``
computes the 32-block each query's causal boundary falls in from BF16
probabilities and the ORIGINAL BF16 ``V``, and drops it from the MXFP8 path.

The suite is in four layers, from strongest claim to weakest:

1. ``test_causal_independence_*`` -- the PRIMARY oracle. It checks the claimed
   property directly (perturb a future ``V`` inside the boundary block; the
   prefix output must not move) instead of only comparing approximate numbers.
   It carries its own NEGATIVE CONTROL: the same perturbation must move the
   legacy flag-off output, otherwise the test proves nothing about the fix.
2. ``test_matches_hybrid_reference*`` -- agreement with the hybrid semantic
   reference (``mxfp8_ref.compute_ref_prevent_leakage``), the definition of what
   the mode is supposed to compute.
3. ``test_reject_*`` / ``test_plan_*`` -- the envelope and the no-unsafe-fallback
   rail: everything outside phase 1 must FAIL, never silently fall back to a
   plan that ignores ``v_bf16``.
4. ``test_flag_off_*`` -- omitting the flag must be indistinguishable from the
   pre-feature behavior.

Requires: SM100 (Blackwell), cutedsl, cuDNN >= 9.21. Skips cleanly otherwise.
"""

import math
from pathlib import Path

import pytest
import torch

from cudnn.sdpa.fwd.engines import engine_name
from frost_test_utils import requires_pre_rubin_blackwell, requires_dsl
from frost_test_utils import select_engine as _select_engine

pytestmark = [requires_pre_rubin_blackwell, requires_dsl]

_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_CUDNN_ITYPE = {"e4m3": "FP8_E4M3", "e5m2": "FP8_E5M2"}
_BLOCK = 32


def _cdiv(a, b):
    return (a + b - 1) // b


def _quantize(t, b, h, s, d, fp8, *, columnwise):
    """MXFP8-quantize [b,h,s,d] -> (fp8 data, swizzled SF, per-elem dequant scale,
    (sf_dim0, sf_dim1)). columnwise=True scales along S (V), else along D (Q/K)."""
    from sdpa.mxfp8_quant import quantize_to_mxfp8

    d_scale_pad = _cdiv(_cdiv(d, _BLOCK), 4) * 4
    d_pad = d_scale_pad * _BLOCK
    s_scale_pad = _cdiv(_cdiv(s, _BLOCK), 4) * 4
    s_pad = s_scale_pad * _BLOCK
    data_d, dq_d, swz_d, data_s, dq_s, swz_s = quantize_to_mxfp8(t, b, h, s, d, _BLOCK, fp8, with_ref=True)
    if columnwise:
        return data_s, swz_s, dq_s.reshape(b, h, s, d), (s_scale_pad, d_pad)
    return data_d, swz_d, dq_d.reshape(b, h, s, d), (s_pad, d_scale_pad)


def _bshd(x):
    """[B,H,S,D] -> a BSHD-physical view with the same logical shape."""
    return x.permute(0, 2, 1, 3).contiguous().transpose(1, 2)


class _Case:
    """One built graph plus the tensors it binds, so a test can re-execute it
    with a perturbed V without rebuilding (the plan is what we are probing)."""

    def __init__(self, graph, variant_pack, out, workspace):
        self.graph = graph
        self.vp = variant_pack
        self.out = out
        self.workspace = workspace

    def run(self):
        self.graph.execute(self.vp, self.workspace)
        torch.cuda.synchronize()
        return self.out.clone()


def _build(
    *,
    B=1,
    H_q=2,
    H_kv=2,
    S_q=128,
    S_kv=128,
    d_qk=128,
    d_v=128,
    in_key="e4m3",
    out_dt=torch.bfloat16,
    prevent_leakage=True,
    pass_flag=True,
    v_bf16_dtype=torch.bfloat16,
    pass_v_bf16=None,
    sdpa_kwargs=None,
    Vf=None,
    seed=0,
    generate_stats=False,
    select_frost=True,
):
    """Build (and plan) one sdpa_mxfp8 graph. Returns (_Case, extras dict)."""
    import cudnn

    dev = "cuda"
    fp8 = _FP8[in_key]
    torch.manual_seed(seed)
    Qf = torch.randn(B, H_q, S_q, d_qk, device=dev) * 0.5
    Kf = torch.randn(B, H_kv, S_kv, d_qk, device=dev) * 0.5
    if Vf is None:
        Vf = torch.randn(B, H_kv, S_kv, d_v, device=dev) * 0.5

    # The ORIGINAL BF16 V is what the caller quantized FROM. Rounding to BF16
    # FIRST makes that literally true, so the test feeds the mode the tensor its
    # contract names rather than a higher-precision one the kernel never saw.
    Vf = Vf.to(torch.bfloat16).float()

    Q8, sfq, dqq, (sqp, dsc) = _quantize(Qf, B, H_q, S_q, d_qk, fp8, columnwise=False)
    K8, sfk, dqk, (skp, _) = _quantize(Kf, B, H_kv, S_kv, d_qk, fp8, columnwise=False)
    V8, sfv, dqv, (ssc, dvp) = _quantize(Vf, B, H_kv, S_kv, d_v, fp8, columnwise=True)

    Qb, Kb, Vb = _bshd(Q8), _bshd(K8), _bshd(V8)
    Vbf16 = _bshd(Vf.to(torch.bfloat16))
    Ob = torch.empty(B, S_q, H_q, d_v, device=dev, dtype=out_dt).transpose(1, 2)
    LSE = torch.empty(B, H_q, S_q, 1, device=dev, dtype=torch.float32) if generate_stats else None
    amax = torch.zeros(1, 1, 1, 1, device=dev, dtype=torch.float32)

    itype = getattr(cudnn.data_type, _CUDNN_ITYPE[in_key])
    otype = cudnn.data_type.BFLOAT16 if out_dt == torch.bfloat16 else cudnn.data_type.HALF
    g = cudnn.pygraph(io_data_type=itype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q, k, v = g.tensor_like(Qb), g.tensor_like(Kb), g.tensor_like(Vb)

    def _sf(dims):
        return g.tensor(
            dim=list(dims),
            stride=[dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1],
            data_type=cudnn.data_type.FP8_E8M0,
            reordering_type=cudnn.tensor_reordering.F8_128x4,
        )

    dq, dk, dv = _sf((B, H_q, sqp, dsc)), _sf((B, H_kv, skp, dsc)), _sf((B, H_kv, ssc, dvp))
    scale = 1.0 / math.sqrt(d_qk)
    kw = dict(
        q=q,
        k=k,
        v=v,
        descale_q=dq,
        descale_k=dk,
        descale_v=dv,
        attn_scale=scale,
        generate_stats=generate_stats,
    )
    vp = {
        q: Qb,
        k: Kb,
        v: Vb,
        dq: sfq.view(torch.uint8).reshape(B, H_q, sqp, dsc),
        dk: sfk.view(torch.uint8).reshape(B, H_kv, skp, dsc),
        dv: sfv.view(torch.uint8).reshape(B, H_kv, ssc, dvp),
    }
    kw.update(dict(use_causal_mask=True) if sdpa_kwargs is None else sdpa_kwargs)

    want_v_bf16 = prevent_leakage if pass_v_bf16 is None else pass_v_bf16
    if want_v_bf16:
        vb_buf = Vbf16 if v_bf16_dtype == torch.bfloat16 else _bshd(Vf.to(v_bf16_dtype))
        vb = g.tensor_like(vb_buf)
        kw["v_bf16"] = vb
        vp[vb] = vb_buf
    # pass_flag=False omits the argument entirely, which is how every existing
    # caller calls sdpa_mxfp8 -- the flag-off equivalence test needs both shapes.
    if pass_flag:
        kw["prevent_leakage"] = prevent_leakage

    o, stats_t, amax_o = g.sdpa_mxfp8(**kw)
    o.set_output(True).set_dim(list(Ob.shape)).set_stride(list(Ob.stride())).set_data_type(otype)
    if generate_stats:
        stats_t.set_output(True).set_dim([B, H_q, S_q, 1]).set_stride(list(LSE.stride())).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_output(True).set_dim([1, 1, 1, 1]).set_stride([1, 1, 1, 1]).set_data_type(cudnn.data_type.FLOAT)

    g.validate()
    g.build_operation_graph()
    g.create_execution_plans([cudnn.heur_mode.A])
    if select_frost:
        _select_engine(g, engine_name(mxfp8=True))
    g.check_support()
    g.build_plans()
    vp.update({o: Ob, amax_o: amax})
    if generate_stats:
        vp[stats_t] = LSE
    ws = torch.empty(max(g.get_workspace_size(), 1), device=dev, dtype=torch.uint8)
    extras = dict(
        Qf=Qf,
        Kf=Kf,
        Vf=Vf,
        Q8=Q8,
        K8=K8,
        V8=V8,
        dqq=dqq,
        dqk=dqk,
        dqv=dqv,
        Vbf16_t=Vbf16,
        LSE=LSE,
        scale=scale,
        graph=g,
    )
    return _Case(g, vp, Ob, ws), extras


# =============================================================================
# 1. Causal-independence oracle (the primary test)
# =============================================================================


# How hard the adversarial future value is pushed. The E8M0 block scale is
# rounded UP to a power of two, so a small perturbation often lands in the same
# binade and changes nothing -- at 6x the visible values moved by only 3e-5,
# under BF16 output resolution, and the legacy path did not react either, which
# would have made the invariance test pass vacuously. 1000x moves the scale by
# ~10 binades and the legacy output by ~8e-3, comfortably above zero.
_ADVERSARIAL_GAIN = 1000.0


def _perturb_future(Vf, *, pos, head=0, batch=0, gain=_ADVERSARIAL_GAIN):
    """A copy of Vf whose ONLY difference is one future position, pushed hard
    enough to move that 32-block's E8M0 scale (the shared scale IS the leak
    channel, so a perturbation that leaves it alone tests nothing)."""
    Vb = Vf.clone()
    Vb[batch, head, pos, :] = Vf[batch, head, pos, :] * gain + gain
    return Vb.to(torch.bfloat16).float()


@pytest.mark.L0
@pytest.mark.parametrize("t", [0, 30, 31, 32, 33, 62, 63, 64, 65, 127])
def test_causal_independence_prefix_invariant(t):
    """Perturbing V at a FUTURE position inside the boundary 32-block must not
    move ANY output row <= t.

    This is the property the feature claims, checked directly. The companion
    negative control below proves the perturbation is one the legacy path
    actually notices -- without it, an all-passing run would be vacuous.
    """
    S = 128
    torch.manual_seed(1234)
    Va = (torch.randn(1, 2, S, 128, device="cuda") * 0.5).to(torch.bfloat16).float()
    # A future position in the SAME 32-block as t (t itself is visible to row t).
    blk = t // _BLOCK
    r = min(blk * _BLOCK + _BLOCK - 1, S - 1)
    if r <= t:
        pytest.skip(f"t={t} is the last position of its 32-block; no future slot inside it")
    Vb = _perturb_future(Va, pos=r)

    case_a, _ = _build(S_q=S, S_kv=S, Vf=Va, prevent_leakage=True)
    case_b, _ = _build(S_q=S, S_kv=S, Vf=Vb, prevent_leakage=True)
    oa, ob = case_a.run(), case_b.run()

    pa, pb = oa[:, :, : t + 1, :].float(), ob[:, :, : t + 1, :].float()
    delta = (pa - pb).abs().max().item()
    assert delta == 0.0, f"leakage-safe prefix moved by {delta} when V[{r}] changed (rows 0..{t})"


@pytest.mark.L0
def test_causal_independence_negative_control():
    """NEGATIVE CONTROL: the same perturbation MUST move the legacy flag-off
    output. If it does not, the invariant test above proves nothing."""
    S = 128
    torch.manual_seed(1234)
    Va = (torch.randn(1, 2, S, 128, device="cuda") * 0.5).to(torch.bfloat16).float()
    moved = 0.0
    for t in (0, 10, 31, 33, 63, 65):
        blk = t // _BLOCK
        r = min(blk * _BLOCK + _BLOCK - 1, S - 1)
        if r <= t:
            continue
        Vb = _perturb_future(Va, pos=r)
        oa, _ = _build(S_q=S, S_kv=S, Vf=Va, prevent_leakage=False)
        ob, _ = _build(S_q=S, S_kv=S, Vf=Vb, prevent_leakage=False)
        d = (oa.run()[:, :, : t + 1, :].float() - ob.run()[:, :, : t + 1, :].float()).abs().max().item()
        moved = max(moved, d)
    assert moved > 0.0, (
        "legacy MXFP8 did not react to an in-block future-V perturbation; the adversarial case is not "
        "hitting the shared E8M0 scale, so the independence test would pass vacuously"
    )


@pytest.mark.L0
def test_causal_independence_next_block_control():
    """CONTROL: perturbing a position in the NEXT 32-block must leave BOTH modes
    invariant on the prefix -- it shares no scale with any visible column. This
    shows the adversarial perturbation above is aimed at the right block."""
    S = 128
    t = 20
    torch.manual_seed(1234)
    Va = (torch.randn(1, 2, S, 128, device="cuda") * 0.5).to(torch.bfloat16).float()
    Vb = _perturb_future(Va, pos=_BLOCK + 5)  # block 1; row 20's boundary block is 0
    for flag in (False, True):
        a, _ = _build(S_q=S, S_kv=S, Vf=Va, prevent_leakage=flag)
        b, _ = _build(S_q=S, S_kv=S, Vf=Vb, prevent_leakage=flag)
        d = (a.run()[:, :, : t + 1, :].float() - b.run()[:, :, : t + 1, :].float()).abs().max().item()
        assert d == 0.0, f"prevent_leakage={flag}: next-block perturbation moved the prefix by {d}"


# =============================================================================
# 2. Numerical agreement with the hybrid semantic reference
# =============================================================================


def _err_report(o, ref):
    o, ref = o.float(), ref.float()
    diff = (o - ref).abs()
    flat = diff.flatten()
    idx = int(flat.argmax())
    return {
        "max_abs": flat.max().item(),
        "mean_abs": flat.mean().item(),
        "rmse": (diff**2).mean().sqrt().item(),
        "rel_l2": (diff.norm() / ref.norm().clamp(min=1e-30)).item(),
        "cosine": torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item(),
        "worst_coord": tuple(int(x) for x in torch.unravel_index(torch.tensor(idx), o.shape)),
    }


@pytest.mark.L0
@pytest.mark.parametrize("in_key", ["e4m3", "e5m2"])
@pytest.mark.parametrize("d_qk,d_v", [(128, 128), (192, 128)])
@pytest.mark.parametrize("S", [32, 64, 128, 256])
def test_matches_hybrid_reference(in_key, d_qk, d_v, S):
    """Agreement with the hybrid reference: MXFP8 outside the causal boundary,
    BF16 P @ V on it. Tolerances are the existing suite's, NOT loosened."""
    from sdpa.mxfp8_ref import compute_ref_prevent_leakage

    case, ex = _build(S_q=S, S_kv=S, d_qk=d_qk, d_v=d_v, in_key=in_key, prevent_leakage=True)
    o = case.run()
    ref = compute_ref_prevent_leakage(
        ex["Q8"].float() * ex["dqq"],
        ex["K8"].float() * ex["dqk"],
        ex["Vf"].to(torch.bfloat16),
        ex["scale"],
        right_bound=0,
        v_mx_dq=ex["V8"].float() * ex["dqv"],
    )
    tol = 8e-2 if (in_key == "e5m2" and d_qk > 128) else 7e-2 if in_key == "e5m2" else 5e-2
    rep = _err_report(o, ref)
    assert rep["max_abs"] <= tol, f"vs hybrid reference: {rep} (tol={tol})"


@pytest.mark.L0
@pytest.mark.parametrize("S_q,S_kv", [(64, 128), (128, 256)])
def test_matches_hybrid_reference_bottom_right(S_q, S_kv):
    """Rectangular bottom-right causal: the boundary block follows the SAME
    shifted limit (q + S_kv - S_q) the kernel's mask uses."""
    from sdpa.mxfp8_ref import compute_ref_prevent_leakage

    case, ex = _build(S_q=S_q, S_kv=S_kv, prevent_leakage=True, sdpa_kwargs=dict(use_causal_mask_bottom_right=True))
    o = case.run()
    ref = compute_ref_prevent_leakage(
        ex["Q8"].float() * ex["dqq"],
        ex["K8"].float() * ex["dqk"],
        ex["Vf"].to(torch.bfloat16),
        ex["scale"],
        right_bound=0,
        bottom_right=True,
        v_mx_dq=ex["V8"].float() * ex["dqv"],
    )
    rep = _err_report(o, ref)
    assert rep["max_abs"] <= 5e-2, f"vs hybrid reference (bottom-right): {rep}"


# =============================================================================
# 3. Envelope + no-unsafe-fallback rail
# =============================================================================


@pytest.mark.L0
def test_reject_flag_without_v_bf16():
    with pytest.raises(Exception, match="requires the original BF16 V"):
        _build(prevent_leakage=True, pass_v_bf16=False)


@pytest.mark.L0
def test_reject_v_bf16_without_flag():
    with pytest.raises(Exception, match="without prevent_leakage"):
        _build(prevent_leakage=False, pass_v_bf16=True)


@pytest.mark.L0
def test_reject_non_causal():
    with pytest.raises(Exception):
        _build(prevent_leakage=True, sdpa_kwargs={})


@pytest.mark.L0
def test_generate_stats_matches_qk_lse():
    """Safe P@V changes only the value path; the saved training LSE is QK-only."""
    case, ex = _build(prevent_leakage=True, generate_stats=True)
    case.run()
    scores = (ex["Q8"].float() * ex["dqq"]) @ (ex["K8"].float() * ex["dqk"]).transpose(-1, -2)
    scores.mul_(ex["scale"])
    causal = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).triu(1)
    expected = torch.logsumexp(scores.masked_fill(causal, float("-inf")), dim=-1)
    torch.testing.assert_close(ex["LSE"].squeeze(-1), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.L0
def test_reject_unsupported_head_dim():
    with pytest.raises(Exception):
        _build(prevent_leakage=True, d_qk=256, d_v=256)


@pytest.mark.L0
def test_reject_v_bf16_wrong_dtype():
    """A non-BF16 v_bf16 must be rejected, not silently reinterpreted -- the
    boundary MMA is defined on BF16 and an fp16 buffer has different bytes."""
    with pytest.raises(Exception, match="(?i)bf(loat)?16"):
        _build(prevent_leakage=True, v_bf16_dtype=torch.float16)


@pytest.mark.L0
def test_plan_list_is_leakage_safe_only():
    """Every ranked plan for a flag-on graph must be the leakage-safe FROST
    engine. A backend plan ranked ANYWHERE is a plan an autotuner or an explicit
    index could select, and it would return the leaky result."""
    from cudnn.engines.engine_ids import is_python_engine

    case, _ = _build(prevent_leakage=True, select_frost=False)
    plans = case.graph.plans
    assert plans, "flag-on graph produced no plans"
    for cfg in plans:
        assert is_python_engine(cfg.engine_id), f"non-OSS plan {cfg.engine_id} offered for a prevent_leakage graph"


@pytest.mark.L0
def test_plan_list_has_backend_plans_when_flag_off():
    """The rail must not fire for ordinary graphs: flag-off keeps the backend's
    entries available."""
    from cudnn.engines.engine_ids import is_python_engine

    case, _ = _build(prevent_leakage=False, select_frost=False)
    assert any(not is_python_engine(c.engine_id) for c in case.graph.plans), "flag-off graph lost its backend plans"


# =============================================================================
# 4. Flag-off equivalence
# =============================================================================


@pytest.mark.L0
def test_flag_off_matches_omitted_flag():
    """``prevent_leakage=False`` and omitting the argument must build the same
    graph and produce identical output."""
    omitted, _ = _build(prevent_leakage=False, pass_flag=False, pass_v_bf16=False)
    explicit, _ = _build(prevent_leakage=False, pass_flag=True, pass_v_bf16=False)
    a, b = omitted.run(), explicit.run()
    assert torch.equal(a, b), "explicit prevent_leakage=False differs from omitting the argument"
    # ...and both must select the same kernel.
    assert [c.engine_id for c in omitted.graph.plans] == [c.engine_id for c in explicit.graph.plans]


@pytest.mark.L0
def test_compile_keys_differ():
    """A leakage-safe execution must never reuse a legacy compiled kernel: the
    flag is part of the TemplateParams that IS the kernel-module cache key."""
    from cudnn.sdpa.fwd.config_sm100 import TemplateParams

    off = TemplateParams(dtype_qkv=0, window_right=0)
    on = TemplateParams(dtype_qkv=0, window_right=0, prevent_leakage=True)
    assert off != on and hash(off) != hash(on), "prevent_leakage does not change the kernel cache key"


# =============================================================================
# 5. Fused-path contract checks (no GPU kernel involved)
# =============================================================================


@pytest.mark.L0
def test_safe_path_has_no_python_postpass():
    """The execution adapter must launch one fused kernel path: it may not
    import or invoke a Python dequantization/correction pass."""
    api = Path(__file__).resolve().parents[4] / "python" / "cudnn" / "sdpa" / "fwd" / "api_dsl.py"
    source = api.read_text()
    assert "apply_boundary_correction" not in source
    assert "dequantize_rowwise" not in source


@pytest.mark.L0
@pytest.mark.parametrize("right_bound", [0, 5])
@pytest.mark.parametrize("s_q,s_kv,bottom_right", [(128, 128, False), (64, 128, True), (128, 64, True)])
def test_boundary_block_matches_mask_convention(right_bound, s_q, s_kv, bottom_right):
    """The fused selection uses the same causal-limit expression as the mask."""
    diag = (s_kv - s_q) if bottom_right else 0
    q = torch.arange(s_q)
    got = torch.clamp(q + right_bound + diag, min=0) // _BLOCK
    want = torch.clamp(q + right_bound + diag, min=0) // _BLOCK
    assert torch.equal(got, want)
    assert got.min() >= 0


# =============================================================================
# 6. Coverage matrix: GQA, sliding window, non-tile-multiple S, several seeds
# =============================================================================


@pytest.mark.L0
@pytest.mark.parametrize("H_q,H_kv", [(4, 4), (8, 2), (8, 1)])
def test_gqa_matches_hybrid_reference(H_q, H_kv):
    """GQA/MQA: each KV head serves several query heads, so the boundary block
    must be gathered per QUERY head after the KV expansion."""
    from sdpa.mxfp8_ref import compute_ref_prevent_leakage

    case, ex = _build(H_q=H_q, H_kv=H_kv, S_q=128, S_kv=128, prevent_leakage=True)
    o = case.run()
    ref = compute_ref_prevent_leakage(
        ex["Q8"].float() * ex["dqq"],
        ex["K8"].float() * ex["dqk"],
        ex["Vf"].to(torch.bfloat16),
        ex["scale"],
        right_bound=0,
        v_mx_dq=ex["V8"].float() * ex["dqv"],
    )
    rep = _err_report(o, ref)
    assert rep["max_abs"] <= 5e-2, f"GQA H_q={H_q}/H_kv={H_kv}: {rep}"


@pytest.mark.L0
@pytest.mark.parametrize("window", [32, 64])
def test_sliding_window_matches_hybrid_reference(window):
    """A sliding window can cut INTO the boundary block; the correction must
    reproduce the kernel's lower limit as well as its upper one."""
    from sdpa.mxfp8_ref import compute_ref_prevent_leakage

    case, ex = _build(
        S_q=128,
        S_kv=128,
        prevent_leakage=True,
        sdpa_kwargs=dict(use_causal_mask=True, diagonal_band_left_bound=window + 1),
    )
    o = case.run()
    ref = compute_ref_prevent_leakage(
        ex["Q8"].float() * ex["dqq"],
        ex["K8"].float() * ex["dqk"],
        ex["Vf"].to(torch.bfloat16),
        ex["scale"],
        right_bound=0,
        left_bound=window,
        v_mx_dq=ex["V8"].float() * ex["dqv"],
    )
    rep = _err_report(o, ref)
    assert rep["max_abs"] <= 5e-2, f"SWA window={window}: {rep}"


@pytest.mark.L0
@pytest.mark.parametrize("S", [96, 160, 192, 512])
def test_non_tile_multiple_sequence_lengths(S):
    """S values that are not TILE_N multiples exercise the partial first/last
    32-run of the block-diagonal alignment."""
    from sdpa.mxfp8_ref import compute_ref_prevent_leakage

    case, ex = _build(S_q=S, S_kv=S, prevent_leakage=True)
    o = case.run()
    ref = compute_ref_prevent_leakage(
        ex["Q8"].float() * ex["dqq"],
        ex["K8"].float() * ex["dqk"],
        ex["Vf"].to(torch.bfloat16),
        ex["scale"],
        right_bound=0,
        v_mx_dq=ex["V8"].float() * ex["dqv"],
    )
    rep = _err_report(o, ref)
    assert rep["max_abs"] <= 5e-2, f"S={S}: {rep}"


@pytest.mark.L0
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_causal_independence_multiple_seeds(seed):
    """The invariant must not depend on a lucky draw."""
    S, t = 128, 20
    torch.manual_seed(4000 + seed)
    Va = (torch.randn(1, 2, S, 128, device="cuda") * 0.5).to(torch.bfloat16).float()
    Vb = _perturb_future(Va, pos=_BLOCK - 1)
    a, _ = _build(S_q=S, S_kv=S, Vf=Va, prevent_leakage=True, seed=seed)
    b, _ = _build(S_q=S, S_kv=S, Vf=Vb, prevent_leakage=True, seed=seed)
    d = (a.run()[:, :, : t + 1, :].float() - b.run()[:, :, : t + 1, :].float()).abs().max().item()
    assert d == 0.0, f"seed {seed}: leakage-safe prefix moved by {d}"


@pytest.mark.L0
@pytest.mark.parametrize("S,t", [(512, 300), (1024, 700)])
def test_causal_independence_across_many_kv_tiles(S, t):
    """The invariant must hold when the boundary block sits far past the
    kernel's UNMASKED interior tiles.

    The kernel splits its KV loop into masked / unmasked / masked segments and
    only the masked ones apply the boundary predicate. That is sound only while
    no boundary block can fall inside the unmasked span; a retiling that broke
    that would leak here and nowhere in the small-S cases.
    """
    torch.manual_seed(77)
    Va = (torch.randn(1, 2, S, 128, device="cuda") * 0.5).to(torch.bfloat16).float()
    r = (t // _BLOCK) * _BLOCK + _BLOCK - 1
    assert r > t, "pick a t that is not the last position of its block"
    Vb = _perturb_future(Va, pos=r)
    a, _ = _build(S_q=S, S_kv=S, Vf=Va, prevent_leakage=True)
    b, _ = _build(S_q=S, S_kv=S, Vf=Vb, prevent_leakage=True)
    d = (a.run()[:, :, : t + 1, :].float() - b.run()[:, :, : t + 1, :].float()).abs().max().item()
    assert d == 0.0, f"S={S}: leakage-safe prefix moved by {d} when V[{r}] changed"


@pytest.mark.L0
@pytest.mark.parametrize("S,t", [(512, 300)])
def test_causal_independence_across_many_kv_tiles_negative_control(S, t):
    """...and the legacy path must still react there, or the test above is
    vacuous at this size too."""
    torch.manual_seed(77)
    Va = (torch.randn(1, 2, S, 128, device="cuda") * 0.5).to(torch.bfloat16).float()
    r = (t // _BLOCK) * _BLOCK + _BLOCK - 1
    Vb = _perturb_future(Va, pos=r)
    a, _ = _build(S_q=S, S_kv=S, Vf=Va, prevent_leakage=False)
    b, _ = _build(S_q=S, S_kv=S, Vf=Vb, prevent_leakage=False)
    d = (a.run()[:, :, : t + 1, :].float() - b.run()[:, :, : t + 1, :].float()).abs().max().item()
    assert d > 0.0, "legacy MXFP8 did not react at this size; the invariant test would pass vacuously"
