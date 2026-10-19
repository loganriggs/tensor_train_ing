"""Near-duplicate contamination check: how much of each validation document reappears (as 13-token windows) in a training split.

    python scripts/check_contamination.py --train data/fineweb_13b_heldout [--val data/fineweb_confirmation_320m] [--sample-tokens 6e8]

The held-out exclusion removes training documents that match a validation document by ID or content hash; near-duplicates (templated pages,
boilerplate-wrapped copies) pass it. For every validation document (the val split, split at EOS 50256), the fraction of its 13-gram windows that
occur in an evenly spaced sample of the training split is computed. A document whose duplicate lies outside the sample is missed, so counts of
heavily overlapping documents scale up by roughly 1 / (sample fraction). Low overall overlap (common phrases) is expected; the signal is
documents with >= 50% overlap. Reads tokens only via tnlm.token_format.read_split_tokens. CPU only; ~1 min per 100M sampled tokens.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tnlm.token_format import read_split_tokens  # noqa: E402

N, P, EOS = 13, np.uint64(1000003), 50256


def ngram_hashes(t: np.ndarray) -> np.ndarray:
    """Polynomial hash (mod 2^64) of every length-N window."""
    t = t.astype(np.uint64)
    m = len(t) - N + 1
    if m <= 0:
        return np.empty(0, np.uint64)
    h = np.zeros(m, np.uint64)
    with np.errstate(over="ignore"):
        for k in range(N):
            h = h * P + t[k:k + m]
    return h


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=str(ROOT / "data/fineweb_confirmation_320m"))
    ap.add_argument("--sample-tokens", type=float, default=6e8)
    ap.add_argument("--chunks", type=int, default=60)
    a = ap.parse_args(argv)
    val = np.asarray(read_split_tokens(a.val, "val"))
    cuts = np.flatnonzero(val == EOS)
    bounds = list(zip(np.r_[0, cuts + 1], np.r_[cuts, len(val)]))
    docs = [(s, e) for s, e in bounds if e - s >= N]
    val_h = ngram_hashes(val)
    pos_doc = np.full(len(val_h), -1, np.int64)  # window start -> document index (windows crossing EOS stay -1)
    for i, (s, e) in enumerate(docs):
        pos_doc[s:e - N + 1] = i
    keep = pos_doc >= 0
    val_h, pos_doc = val_h[keep], pos_doc[keep]
    uniq = np.unique(val_h)
    seen = np.zeros(len(uniq), bool)
    train = read_split_tokens(a.train, "train")
    per = int(a.sample_tokens / a.chunks)
    starts = np.linspace(0, len(train) - per - 1, a.chunks).astype(np.int64)
    for s in starts:
        h = ngram_hashes(np.asarray(train[s:s + per]))
        seen |= np.isin(uniq, h, assume_unique=False)
    hit = seen[np.searchsorted(uniq, val_h)]
    frac = np.bincount(pos_doc, weights=hit, minlength=len(docs)) / np.maximum(np.bincount(pos_doc, minlength=len(docs)), 1)
    s_frac = per * a.chunks / len(train)
    print(f"validation: {len(docs)} docs, {len(val_h):,} 13-gram windows; train sample {per * a.chunks / 1e6:.0f}M of {len(train) / 1e9:.2f}B tokens "
          f"({s_frac:.1%})")
    print(f"windows found in sample: {hit.mean():.3%}; median per-doc overlap {np.median(frac):.3%}")
    for thr in (0.1, 0.5, 0.9):
        n = int((frac >= thr).sum())
        print(f"docs with >= {thr:.0%} overlap: {n} in sample  (~{min(len(docs), n / s_frac):.0f} expected in the full split if duplicates are spread evenly)")
    top = np.argsort(-frac)[:5]
    print("top overlap docs (index: fraction):", ", ".join(f"{i}: {frac[i]:.2f}" for i in top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
