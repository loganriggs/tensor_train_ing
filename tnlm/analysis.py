"""Interpretation diagnostics (brief §10, items 1-6).

Entry point: `analyze(model, batches, out_dir) -> dict` (JSON-serializable). `batches` is a list of
(idx, doc_ids, targets) tuples already on the model's device. Heavy tables go to files in out_dir.

Two edge views are kept separate (brief §3): the *content* view (folded read/write overlaps and
local polynomial factors) and the *normalization* view (explicit normalizer nodes, scopes, and
denominator-mediated sensitivity). Nothing here claims a content overlap is causally used.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import TensorLM
from .normalization import (GroupRMSNorm, LaggedRMS, RMSNorm, SpectralLinear, DynamicNorm, TaperRMS, dynamic_norms, upcast)

QUANTILES = (0.01, 0.05, 0.5, 0.95, 0.99)
POS_BUCKETS = ((0, 16), (16, 64), (64, 256), (256, 1 << 30))


# ----------------------------------------------------------------------------- helpers
def eff(lin: nn.Module) -> torch.Tensor:
    """Effective weight actually used in the forward pass (spectral reparam resolved)."""
    return upcast((lin.effective_weight() if isinstance(lin, SpectralLinear) else lin.weight).detach())


def gain(norm: nn.Module, d: int, device) -> torch.Tensor:
    """Effective fixed-at-checkpoint per-feature map of a norm site for folding into adjacent linear maps.

    RMSNorm/GroupRMSNorm: the gain only (their denominators are dynamic and reported separately).
    LaggedRMS: gain / sqrt(mean_sq + eps), because the frozen divisor is a constant and folding it makes exported
    read vectors and local tensors invariant to the exact gauge g -> c g, mean_sq -> c^2 (mean_sq + eps) - eps.
    TaperRMS: at gate 1 it is an RMSNorm (gain only); frozen at gate 0 it is the fixed map c * gamma_tilde. A gate strictly
    between 0 and 1 mixes a dynamic and a fixed branch and has no single foldable map, so it is refused."""
    if isinstance(norm, TaperRMS):
        g_ = float(norm.gate)
        if g_ == 1.0:
            return upcast(norm.weight.detach())
        if g_ == 0.0 and bool(norm.frozen):
            return float(norm.c) * upcast(norm.weight_tilde.detach())
        raise ValueError(f"TaperRMS at gate {g_} (frozen={bool(norm.frozen)}) has no fixed foldable map")
    w = getattr(norm, "weight", None)
    if w is None:
        return torch.ones(d, device=device)
    g = upcast(w.detach())
    if isinstance(norm, LaggedRMS):
        g = g / torch.sqrt(norm.mean_sq.detach().to(g.dtype) + norm.eps)
    return g


def norm_scope(norm: nn.Module) -> str:
    if isinstance(norm, GroupRMSNorm):
        return f"per-token RMS over {norm.groups} contiguous feature groups"
    if isinstance(norm, RMSNorm):
        return "per-token RMS over all features"
    if isinstance(norm, LaggedRMS):
        return "frozen scalar (constant at checkpoint)"
    if isinstance(norm, TaperRMS):
        if bool(norm.frozen) and float(norm.gate) == 0.0:
            return "fixed diagonal scale c*gamma_tilde (tapered; constant at checkpoint)"
        return f"per-token RMS, taper gate {float(norm.gate):g}"
    return "none"


def _q(x: torch.Tensor) -> dict:
    x = upcast(x.flatten())
    if x.numel() == 0:
        return {}
    if x.numel() > 1_000_000:
        x = x[torch.randperm(x.numel(), device=x.device)[:1_000_000]]
    qs = torch.quantile(x, torch.tensor(QUANTILES, device=x.device, dtype=x.dtype))
    return {f"q{int(p * 100):02d}": float(v) for p, v in zip(QUANTILES, qs)} | {"mean": float(x.mean())}


@contextmanager
def _modes(model: TensorLM):
    """Always restore live normalization and no ablation."""
    try:
        yield
    finally:
        for _, m in dynamic_norms(model):
            m.denom_mode, m.denom_saved, m.denom_const = None, None, None
        for blk in model.blocks:
            blk.ablate = frozenset()


def _set_mode(model, mode):
    for _, m in dynamic_norms(model):
        m.denom_mode = mode


def _token_losses(model, idx, doc, tgt):
    logits = upcast(model(idx, doc))
    loss = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), ignore_index=-100, reduction="none")
    return logits, loss.view_as(tgt)


def _doc_keys(doc, tgt):
    valid = tgt != -100
    B = doc.shape[0]
    key = torch.arange(B, device=doc.device)[:, None] * (doc.max() + 2) + doc.clamp_min(0)
    return key[valid], valid


def _paired_doc_stats(diffs: list[torch.Tensor], keys: list[torch.Tensor]) -> dict:
    """Mean per-document CE difference and its standard error across documents (not seeds)."""
    per_doc = []
    for i, (d, k) in enumerate(zip(diffs, keys)):
        uk, inv = torch.unique(k, return_inverse=True)
        s = torch.zeros(len(uk), device=d.device, dtype=torch.float64).index_add_(0, inv, d.double())
        c = torch.zeros(len(uk), device=d.device, dtype=torch.float64).index_add_(0, inv, torch.ones_like(d, dtype=torch.float64))
        per_doc.append(s / c)
    v = torch.cat(per_doc)
    n = v.numel()
    se = float(v.std(unbiased=True) / math.sqrt(n)) if n > 1 else float("nan")
    return {"mean": float(v.mean()), "doc_se": se, "documents": n}


# ----------------------------------------------------------------------------- 1. local tensors
@torch.no_grad()
def mlp_local_tensor(blk, basis: torch.Tensor, fold_gain: bool = True):
    """Symmetrized coefficient tensor of the MLP restricted to span(basis) (d x k), with optional
    homogeneous coordinate for biases. Returns T: (d_out, k[+1], k[+1]). Never materializes d x d x d.

    With fold_gain, the norm_in gain G is folded into L and R (G is fixed at a checkpoint)."""
    m = blk.mlp
    L, R, D = eff(m.l), eff(m.r), eff(m.d)
    if fold_gain:
        g = gain(blk.mlp_in, L.shape[1], L.device)
        L, R = L * g, R * g
    basis = basis.to(L.dtype)
    LU, RU = L @ basis, R @ basis
    bL, bR, bD = (getattr(lin, "bias", None) for lin in (m.l, m.r, m.d))
    if bL is not None:
        LU = torch.cat([LU, upcast(bL.detach())[:, None]], 1)
        RU = torch.cat([RU, upcast(bR.detach())[:, None]], 1)
        e = torch.zeros(1, LU.shape[1], device=LU.device, dtype=LU.dtype); e[0, -1] = 1
        LU, RU = torch.cat([LU, e]), torch.cat([RU, e])
        D = torch.cat([D, upcast(bD.detach())[:, None]], 1)
    T = torch.einsum("oh,ha,hb->oab", D, LU, RU)
    return 0.5 * (T + T.transpose(1, 2))


@torch.no_grad()
def local_tensor_check(model: TensorLM, hidden: list[torch.Tensor], k: int = 8, n: int = 16) -> dict:
    """(a) restricted-subspace tensor reconstruction vs module on random inputs (exact algebra);
    (b) on actual probe hidden states, B(N(x)) == D((LGx)*(RGx)) / r(x)^2 for bias-free pre-RMS."""
    out = []
    gen = torch.Generator(device="cpu").manual_seed(0)
    for i, blk in enumerate(model.blocks):
        if blk.mlp.kind == "swiglu":
            out.append({"layer": i, "skipped": "SwiGLU is class G"})
            continue
        d = model.cfg.d_model
        dev = blk.mlp.l.weight.device
        dt = upcast(blk.mlp.l.weight).dtype
        U = torch.linalg.qr(torch.randn(d, k, generator=gen, dtype=torch.float64)).Q.to(device=dev, dtype=dt)
        z = torch.randn(n, k, generator=gen, dtype=torch.float64).to(device=dev, dtype=dt)
        T = mlp_local_tensor(blk, U, fold_gain=False)
        za = torch.cat([z, torch.ones(n, 1, device=dev)], 1) if T.shape[1] > k else z
        rec = torch.einsum("oab,na,nb->no", T, za, za)
        ref = upcast(blk.mlp(z @ U.T))
        rel = float((rec - ref).norm() / ref.norm().clamp_min(1e-30))
        row = {"layer": i, "subspace_rank": k, "rel_err_restricted_tensor": rel}
        norm = blk.mlp_in
        if isinstance(norm, RMSNorm) and not isinstance(norm, GroupRMSNorm) and blk.mlp.kind == "bilinear":
            x = upcast(hidden[i][1].reshape(-1, d)[:256])  # input to MLP sublayer (post-attention residual)
            g = gain(norm, d, dev)
            r = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)
            L, R, D = eff(blk.mlp.l), eff(blk.mlp.r), eff(blk.mlp.d)
            folded = (((x * g) @ L.T) * ((x * g) @ R.T)) @ D.T / r ** 2
            live = upcast(blk.mlp(norm(x)))
            row["rel_err_gain_folded_on_probe"] = float((folded - live).norm() / live.norm().clamp_min(1e-30))
        out.append(row)
    return {"per_layer": out,
            "max_rel_err": max((r.get("rel_err_restricted_tensor", 0) for r in out), default=0.0)}


# ----------------------------------------------------------------------------- 2. read/write graph
def _writers(model):
    for i, blk in enumerate(model.blocks):
        for comp, lin, onorm in (("attn", blk.attn.o, blk.attn_out), ("mlp", blk.mlp.d, blk.mlp_out)):
            W = eff(lin)  # (d, m): columns are write directions
            if isinstance(onorm, DynamicNorm):
                W = W * gain(onorm, W.shape[0], W.device)[:, None]
            yield (i, comp), W, norm_scope(onorm)


def _readers(model, after_layer, after_comp, layer_window=None):
    """Read vectors strictly downstream of a writer: same-layer MLP reads the attention output.
    layer_window=k keeps only readers in layers after_layer .. after_layer+k (the unembedding is always kept)."""
    dev = model.unembed.weight.device
    for j, blk in enumerate(model.blocks):
        if layer_window is not None and j > after_layer + layer_window:
            break
        a = blk.attn
        if j > after_layer:
            g = gain(blk.attn_in, model.cfg.d_model, dev)
            for name in ("q1", "k1", "q2", "k2", "v"):
                if hasattr(a, name):
                    yield (j, f"attn.{name}"), eff(getattr(a, name)) * g
        if j > after_layer or (j == after_layer and after_comp == "attn"):
            g = gain(blk.mlp_in, model.cfg.d_model, dev)
            yield (j, "mlp.l"), eff(blk.mlp.l) * g
            yield (j, "mlp.r"), eff(blk.mlp.r) * g
    g = gain(model.final_norm, model.cfg.d_model, dev)
    yield (len(model.blocks), "unembed"), upcast(model.unembed.weight.detach())[: model.cfg.vocab_size] * g


@torch.no_grad()
def read_write_graph(model: TensorLM, top_k: int = 5, threshold: float = 0.2,
                     layer_window: int | None = None) -> tuple[dict, list]:
    """Signed overlaps between upstream write directions and downstream read vectors (gains folded).
    Reports raw values and cosine (scale/gauge-invariant per vector). Unembedding rows are summarized
    only (vocab-sized). The direct residual path is the identity and is listed as such."""
    edges, summary = [], []
    for (i, comp), W, oscope in _writers(model):
        Wn = W / W.norm(dim=0, keepdim=True).clamp_min(1e-30)
        for (j, rname), Rm in _readers(model, i, comp, layer_window):
            raw = Rm @ W
            cos = (Rm / Rm.norm(dim=1, keepdim=True).clamp_min(1e-30)) @ Wn
            a = cos.abs()
            row = {"writer": f"L{i}.{comp}", "reader": f"L{j}.{rname}", "writer_output_norm": oscope,
                   "max_abs_cos": float(a.max()), "mean_abs_cos": float(a.mean()),
                   "frac_abs_cos_gt": float((a > threshold).float().mean()), "threshold": threshold,
                   "raw_fro": float(raw.norm())}
            summary.append(row)
            if rname != "unembed":
                vals, flat = a.flatten().topk(min(top_k, a.numel()))
                for v, f in zip(vals, flat):
                    r_, c_ = divmod(int(f), a.shape[1])
                    edges.append({"writer": row["writer"], "write_col": c_, "reader": row["reader"],
                                  "read_row": r_, "cos": float(cos[r_, c_]), "raw": float(raw[r_, c_])})
    agg = {"pairs": len(summary), "edges_exported": len(edges), "layer_window": layer_window,
           "max_abs_cos": max(s["max_abs_cos"] for s in summary),
           "note": "content view only; overlap != usage. Direct residual path is identity (not listed)."}
    return agg, (summary, edges)


# ----------------------------------------------------------------------------- 3. normalizer sensitivity
@torch.no_grad()
def _capture(model, idx, doc):
    """Record every dynamic norm's input and denominator, and every block's sublayer inputs."""
    inputs, hidden, hooks = {}, [], []
    for name, m in dynamic_norms(model):
        hooks.append(m.register_forward_pre_hook(lambda mod, args, name=name: inputs.__setitem__(name, args[0].detach())))
    for blk in model.blocks:
        hooks.append(blk.register_forward_pre_hook(lambda mod, args: hidden.append([args[0].detach()])))
    # MLP sublayer input = x + attn branch; capture via mlp_in pre-hook when present, else recompute
    mids = []
    for blk in model.blocks:
        tgt = blk.mlp_in if not isinstance(blk.mlp_in, nn.Identity) else blk.mlp
        hooks.append(tgt.register_forward_pre_hook(lambda mod, args: mids.append(args[0].detach())))
    _set_mode(model, "record")
    try:
        logits = model(idx, doc)
    finally:
        for h in hooks:
            h.remove()
    for h, mid in zip(hidden, mids):
        h.append(mid)
    denoms = {name: m.denom_saved for name, m in dynamic_norms(model)}
    return logits, inputs, denoms, hidden


def _analytic_jvp(norm, x, delta):
    """J_N(x) delta from the closed form (per group for GroupRMS): direct term and radial term."""
    g = upcast(norm.weight.detach())
    if isinstance(norm, GroupRMSNorm):
        xs, ds = x.unflatten(-1, (norm.groups, -1)), delta.unflatten(-1, (norm.groups, -1))
    else:
        xs, ds = x[..., None, :], delta[..., None, :]
    dg = xs.shape[-1]
    r = torch.sqrt(xs.pow(2).mean(-1, keepdim=True) + norm.eps)
    direct = (ds / r).flatten(-2) * g
    radial = (-(xs * (xs * ds).sum(-1, keepdim=True)) / (dg * r ** 3)).flatten(-2) * g
    return direct, radial


def normalizer_sensitivity(model, inputs, denoms, doc, max_tokens: int = 512) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(1)
    valid = (doc >= 0)
    pos = torch.arange(doc.shape[1], device=doc.device).expand_as(doc)
    from .attention import doc_positions
    dpos = doc_positions(doc)
    sites = {}
    for name, m in dynamic_norms(model):
        if name not in inputs:
            continue
        x, r = upcast(inputs[name]), upcast(denoms[name])
        rv = r[..., 0] if r.shape[-1] == 1 else r.unflatten(-1, (m.groups, -1))[..., 0]
        rv_valid = rv[valid]
        by_pos = {}
        for lo, hi in POS_BUCKETS:
            sel = valid & (dpos >= lo) & (dpos < hi)
            if sel.any():
                by_pos[f"pos{lo}-{hi if hi < 1 << 29 else 'inf'}"] = float(rv[sel].float().median())
        xs = x[valid][:max_tokens]
        delta = torch.randn(xs.shape, generator=gen).to(xs.device)
        direct, radial = _analytic_jvp(m, xs, delta)
        prev = m.denom_mode
        m.denom_mode = None
        with torch.no_grad():
            _, auto = torch.func.jvp(lambda t: upcast(m(t)), (xs,), (delta,))
        m.denom_mode = prev
        ana = direct + radial
        jvp_err = float((ana - auto).norm() / auto.norm().clamp_min(1e-30))
        ratio = radial.norm(dim=-1) / direct.norm(dim=-1).clamp_min(1e-30)
        sites[name] = {"scope": norm_scope(m), "denominator": _q(rv_valid), "median_by_doc_position": by_pos,
                       "jvp_rel_err_analytic_vs_autograd": jvp_err,
                       "radial_over_direct_random_delta": _q(ratio)}
    return {"sites": sites,
            "note": "radial term = denominator-mediated sensitivity; zero direct content edge does not imply no influence"}


# ----------------------------------------------------------------------------- 4. ablation decomposition
@torch.no_grad()
def ablation_decomposition(model, batches) -> dict:
    """Remove each branch with norms live, and with every dynamic denominator clamped to its value in
    the unablated forward pass. live - clamped is an intervention-specific normalization-mediated part."""
    res = {}
    comps = [(i, c) for i in range(len(model.blocks)) for c in ("attn", "mlp")]
    acc = {ic: {"live": [], "clamped": [], "keys": []} for ic in comps}
    for idx, doc, tgt in batches:
        _set_mode(model, "record")
        _, base = _token_losses(model, idx, doc, tgt)
        keys, valid = _doc_keys(doc, tgt)
        for i, c in comps:
            model.blocks[i].ablate = frozenset({c})
            _set_mode(model, None)
            _, live = _token_losses(model, idx, doc, tgt)
            _set_mode(model, "clamp")
            _, clamped = _token_losses(model, idx, doc, tgt)
            model.blocks[i].ablate = frozenset()
            acc[(i, c)]["live"].append((live - base)[valid])
            acc[(i, c)]["clamped"].append((clamped - base)[valid])
            acc[(i, c)]["keys"].append(keys)
        _set_mode(model, None)
    for (i, c), a in acc.items():
        nm = [l - k for l, k in zip(a["live"], a["clamped"])]
        res[f"L{i}.{c}"] = {"dCE_live": _paired_doc_stats(a["live"], a["keys"]),
                            "dCE_norm_clamped": _paired_doc_stats(a["clamped"], a["keys"]),
                            "norm_mediated(live-clamped)": _paired_doc_stats(nm, a["keys"])}
    has_softmax = model.cfg.attention == "softmax"
    return {"components": res, "uncertainty": "standard error across documents, not across seeds",
            "note": "softmax row normalization is not clamped" if has_softmax else "all dynamic norms clamped"}


# ----------------------------------------------------------------------------- 5. fixed denominators
@torch.no_grad()
def _row_hashes(batches) -> set:
    import hashlib
    out = set()
    for idx, doc, _ in batches:
        for r in range(idx.shape[0]):
            out.add(hashlib.sha256(idx[r].cpu().numpy().tobytes() + doc[r].cpu().numpy().tobytes()).hexdigest())
    return out


@torch.no_grad()
def fixed_denominator(model, batches) -> dict:
    """Replace every dynamic RMS/GroupRMS denominator with its calibrated mean and compare logits (brief §10.5).

    Scope: only the targeted RMS denominators. Other dynamic operations (e.g. softmax row normalization in the
    control) stay live, and the operator class always comes from the model audit, never from this diagnostic."""
    op_class = model.operator_audit()["class"]
    sites = dynamic_norms(model)
    if not sites:
        return {"skipped": "no dynamic RMS/GroupRMS denominators to fix; this intervention is the identity",
                "operator_class": op_class,
                "note": "class comes from the operator audit; other dynamic operations (e.g. softmax) are untouched"}
    n_half = len(batches) // 2
    calib, evalb = (batches, batches) if len(batches) < 2 else (batches[:n_half], batches[n_half:])
    overlap = len(_row_hashes(calib) & _row_hashes(evalb))
    if len(batches) < 2:
        provenance = "in-sample: single probe batch used for both calibration and evaluation"
    elif overlap:
        provenance = f"overlapping: {overlap} identical (tokens, doc layout) rows in calibration and evaluation halves"
    else:
        provenance = "row-disjoint halves (no identical rows); not a document-level held-out guarantee"
    sums = {n: 0 for n, _ in sites}
    cnt = 0
    for idx, doc, tgt in calib:
        _set_mode(model, "record")
        model(idx, doc)
        v = doc >= 0
        for n, m in sites:
            sums[n] = sums[n] + upcast(m.denom_saved)[v].sum(0)
        cnt += int(v.sum())
    _set_mode(model, None)
    n_targets = sum(int((t != -100).sum()) for _, _, t in evalb)
    base = {"operator_class": op_class, "calibration_tokens": cnt, "evaluation_targets": n_targets,
            "fit_eval_provenance": provenance, "targeted_sites": len(sites)}
    if cnt == 0 or n_targets == 0:
        return base | {"skipped": "no calibration tokens" if cnt == 0 else "no scored evaluation targets"}
    for n, m in sites:
        m.denom_const = sums[n] / cnt  # (1,) for RMS, (d,) per-group-constant for GroupRMS
    stats = {"ce_live": [], "ce_fixed": [], "kl": [], "abs_logit_err": [], "top1_agree": []}
    for idx, doc, tgt in evalb:
        _set_mode(model, None)
        ll, cel = _token_losses(model, idx, doc, tgt)
        _set_mode(model, "const")
        lf, cef = _token_losses(model, idx, doc, tgt)
        v = tgt != -100
        lp, lq = F.log_softmax(ll, -1), F.log_softmax(lf, -1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        stats["ce_live"].append(cel[v]); stats["ce_fixed"].append(cef[v]); stats["kl"].append(kl[v])
        stats["abs_logit_err"].append((ll - lf).abs().mean(-1)[v])
        stats["top1_agree"].append((ll.argmax(-1) == lf.argmax(-1)).float()[v])
    _set_mode(model, None)
    metrics = {k: float(torch.cat(v).mean()) for k, v in stats.items()}
    nonfinite = sorted(k for k, x in metrics.items() if not math.isfinite(x))
    out = base | {k: (x if math.isfinite(x) else None) for k, x in metrics.items()}
    if nonfinite:
        out["nonfinite_metrics"] = nonfinite  # e.g. fixed-denominator logits overflowed: a failed approximation, not "null"
    out["note"] = "small average error on this distribution does not establish OOD/long-context equivalence"
    return out


# ----------------------------------------------------------------------------- 6. low-rank structure
def stable_rank(s: torch.Tensor) -> float:
    s = s.abs()
    return float((s ** 2).sum() / (s.max() ** 2).clamp_min(1e-30))


def entropy_rank(s: torch.Tensor) -> float:
    p = s.abs() / s.abs().sum().clamp_min(1e-30)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def low_rank_structure(model, batches, n_dirs: int = 4, max_dense: int = 4096) -> dict:
    """Per-MLP symmetrized quadratic forms Q_u = sym(G L^T diag(D^T u) R G) along the top singular
    output directions of D and along selected logit differences (frequent target pairs); per-head QK
    bilinear-form spectra. Gauge-invariant: l -> c l, r -> r / c leaves every Q_u unchanged."""
    cfg = model.cfg
    d = cfg.d_model
    tg = torch.cat([b[2][b[2] >= 0].flatten() for b in batches])
    top = torch.bincount(tg, minlength=cfg.vocab_size).topk(3).indices.tolist()
    U_out = upcast(model.unembed.weight.detach()) * gain(model.final_norm, d, tg.device)
    logit_dirs = {f"logit[{top[0]}]-logit[{top[1]}]": U_out[top[0]] - U_out[top[1]],
                  f"logit[{top[0]}]-logit[{top[2]}]": U_out[top[0]] - U_out[top[2]]}
    mlps = []
    for i, blk in enumerate(model.blocks):
        if blk.mlp.kind == "swiglu":
            continue
        g = gain(blk.mlp_in, d, tg.device)
        L, R, D = eff(blk.mlp.l) * g, eff(blk.mlp.r) * g, eff(blk.mlp.d)
        Us = torch.linalg.svd(D, full_matrices=False).U[:, :n_dirs].T
        dirs = {f"D_sv{k}": Us[k] for k in range(Us.shape[0])} | logit_dirs
        row = {}
        for nm, u in dirs.items():
            if d > max_dense:
                row[nm] = {"skipped": f"d_model {d} > {max_dense}; use randomized eigensolver"}
                continue
            w = D.T @ u
            Q = L.T @ (w[:, None] * R)
            ev = torch.linalg.eigvalsh(0.5 * (Q + Q.T))
            o = ev.abs().argsort(descending=True)
            row[nm] = {"stable_rank": stable_rank(ev), "entropy_rank": entropy_rank(ev),
                       "top_eigs": [float(x) for x in ev[o[:8]]], "frac_negative_mass": float(ev[ev < 0].abs().sum() / ev.abs().sum())}
        mlps.append({"layer": i, "forms": row})
    heads = []
    for i, blk in enumerate(model.blocks):
        a = blk.attn
        g = gain(blk.attn_in, d, tg.device)
        for pair in (("q1", "k1"), ("q2", "k2")):
            if not hasattr(a, pair[0]):
                continue
            Wq, Wk = eff(getattr(a, pair[0])) * g, eff(getattr(a, pair[1])) * g
            for h in range(cfg.heads):
                sl = slice(h * cfg.d_key, (h + 1) * cfg.d_key)
                Ra = torch.linalg.qr(Wq[sl].T).R
                Rb = torch.linalg.qr(Wk[sl].T).R
                s = torch.linalg.svdvals(Ra @ Rb.T)  # singular values of Wq_h^T Wk_h without d x d
                heads.append({"layer": i, "pair": "".join(pair), "head": h, "stable_rank": stable_rank(s),
                              "top_sv": float(s.max())})
    return {"mlp_quadratic_forms": mlps, "qk_forms": heads,
            "note": "compare across seeds before interpreting; quadratic forms are gauge-invariant, raw factors are not"}


# ----------------------------------------------------------------------------- shared: materialized attention scores
@torch.no_grad()
def probe_attention_scores(model, idx, doc, max_tokens: int = 512):
    """Run the model on the first probe row (<= max_tokens) and return (scores, mlp_inputs, visible).

    scores[i] = {"S1": (H, T, T), "S2": (H, T, T) or absent} with S = q.k / sqrt(dk) after RoPE, for polynomial layers
    ({} for softmax layers). mlp_inputs[i] is the MLP sublayer input. visible is the (T, T) causal same-document mask.
    Shared by `stability_stats` and `tnlm.qk_clip` so both measure exactly the same scores.
    """
    from .attention import apply_rope, rope_tables, doc_positions, visible_mask
    cfg = model.cfg
    idx, doc = idx[:1, :max_tokens], doc[:1, :max_tokens]
    attn_in, mlp_in = {}, {}
    hooks = []
    for i, blk in enumerate(model.blocks):
        hooks.append(blk.attn.register_forward_pre_hook(lambda m, a, i=i: attn_in.__setitem__(i, a[0].detach())))
        hooks.append(blk.mlp.register_forward_pre_hook(lambda m, a, i=i: mlp_in.__setitem__(i, a[0].detach())))
    try:
        model(idx, doc)
    finally:
        for h in hooks:
            h.remove()
    cos, sin = rope_tables(doc_positions(doc), cfg.d_key, cfg.rope_base)
    scores = []
    for i, blk in enumerate(model.blocks):
        a, x = blk.attn, upcast(attn_in[i])
        row = {}
        if a.kind != "softmax":
            heads = lambda lin: upcast(apply_rope(a._heads(x, lin, cfg.d_key), cos, sin))[0]
            row["S1"] = heads(a.q1) @ heads(a.k1).transpose(-1, -2) / cfg.d_key ** 0.5
            if hasattr(a, "q2"):
                row["S2"] = heads(a.q2) @ heads(a.k2).transpose(-1, -2) / cfg.d_key ** 0.5
        scores.append(row)
    return scores, mlp_in, visible_mask(doc)[0, 0]


# ----------------------------------------------------------------------------- 7. stability instrumentation
@torch.no_grad()
def stability_stats(model, idx, doc, max_tokens: int = 512) -> dict:
    """Per-layer magnitude statistics of the polynomial pieces that can blow up at scale (scaling_review rec. 7).

    Attention: max |S1|, |S2|, |S1*S2| and RMS over causally visible same-document pairs (S = q.k/sqrt(dk), after RoPE),
    on the first `max_tokens` positions of the first row (scores are materialized here, so keep it small).
    MLP: RMS, max and excess kurtosis of the bilinear product (Lx)*(Rx). Heavy tails precede FP8/BF16 failures.
    """
    scores, mlp_in, vis = probe_attention_scores(model, idx, doc, max_tokens)
    idx = idx[:1, :max_tokens]
    out = []
    for i, blk in enumerate(model.blocks):
        row = {"layer": i}
        if scores[i]:
            S1 = scores[i]["S1"]
            S2 = scores[i].get("S2", S1)
            P = S1 * S2
            for name, t in (("S1", S1), ("S2", S2), ("S1S2", P)):
                v = t[:, vis]
                row[f"{name}_max_abs"] = float(v.abs().max())
                row[f"{name}_rms"] = float(v.pow(2).mean().sqrt())
        m = blk.mlp
        if m.kind != "swiglu":
            z = upcast(mlp_in[i])
            prod = upcast(m.l(z)) * upcast(m.r(z))
            c = prod - prod.mean()
            row["mlp_prod_rms"] = float(prod.pow(2).mean().sqrt())
            row["mlp_prod_max_abs"] = float(prod.abs().max())
            row["mlp_prod_excess_kurtosis"] = float(c.pow(4).mean() / c.pow(2).mean().clamp_min(1e-30) ** 2 - 3)
        out.append(row)
    worst = {k: max(r.get(k, 0.0) for r in out) for k in ("S1S2_max_abs", "mlp_prod_max_abs", "mlp_prod_excess_kurtosis")}
    return {"per_layer": out, "worst": worst, "tokens": int(idx.shape[1]),
            "note": "single-row probe; track across checkpoints/scales rather than reading absolute values"}


# ----------------------------------------------------------------------------- entry point
def analyze(model: TensorLM, batches, out_dir: str, rw_layer_window: int | None = None) -> dict:
    """rw_layer_window: limit read/write pairs to readers within k layers of each writer (None = all pairs, O(L^2);
    use e.g. 4 beyond ~300M parameters)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    model.eval()
    batches = list(batches)
    if not batches:
        raise ValueError("analyze needs at least one probe batch")
    with _modes(model):
        idx, doc, tgt = batches[0]
        _, inputs, denoms, hidden = _capture(model, idx, doc)
        audit = model.operator_audit()
        result = {
            "operator_class": audit["class"],
            "normalizer_scopes": {n: norm_scope(m) for n, m in model.named_modules()
                                  if isinstance(m, (RMSNorm, GroupRMSNorm, LaggedRMS, TaperRMS))},
            "local_tensor": local_tensor_check(model, hidden),
        }
        rw_agg, (rw_summary, rw_edges) = read_write_graph(model, layer_window=rw_layer_window)
        result["read_write_graph"] = rw_agg
        (out / "read_write_summary.json").write_text(json.dumps(rw_summary))
        (out / "read_write_top_edges.json").write_text(json.dumps(rw_edges))
        result["normalizer_sensitivity"] = normalizer_sensitivity(model, inputs, denoms, doc)
        result["ablation"] = ablation_decomposition(model, batches)
        result["fixed_denominator"] = fixed_denominator(model, batches)
        result["low_rank"] = low_rank_structure(model, batches)
        result["stability"] = stability_stats(model, idx, doc)
        result["files"] = ["read_write_summary.json", "read_write_top_edges.json", "analysis.json"]
    model.train(was_training)
    result = json.loads(json.dumps(result, default=float).replace("NaN", "null"))
    (out / "analysis.json").write_text(json.dumps(result, indent=1))
    return result
