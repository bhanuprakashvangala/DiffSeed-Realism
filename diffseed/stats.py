"""Statistics for the ablation.

The whole v1 conclusion rests on differences of 0.4-2.5 pp measured once, with
no interval and no test. On a 1103-image test set the standard error of an
accuracy near 0.88 is about 1.0 pp, so a 2.5 pp gap is roughly 1.8 sigma from a
*single* run -- before accounting for seed-to-seed variance in training, which is
usually larger. Every claim of the form "X beats Y" needs one of these.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap mean and CI over repeated runs."""
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        return float(v[0]), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return (
        float(v.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


def metric_ci(
    labels: Sequence[int], preds: Sequence[int], metric="accuracy", n_boot=5000, seed=0
) -> tuple[float, float, float]:
    """Bootstrap CI over *test items* for a single run."""
    fn = _FAST_METRICS[metric]
    y = np.asarray(labels)
    p = np.asarray(preds)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    vals = np.array([fn(y[i], p[i]) for i in idx])
    return float(fn(y, p)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --------------------------------------------------------------------------- #
# paired tests -- the right family, because every condition is scored on the
# same held-out images
# --------------------------------------------------------------------------- #
def mcnemar(labels, preds_a, preds_b, exact_threshold: int = 25) -> dict:
    """McNemar's test on the discordant pairs of two classifiers."""
    from scipy.stats import binomtest, chi2

    y = np.asarray(labels)
    a = np.asarray(preds_a) == y
    b = np.asarray(preds_b) == y
    n01 = int((a & ~b).sum())  # a right, b wrong
    n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0, "test": "degenerate"}
    if n < exact_threshold:
        p = float(binomtest(n01, n, 0.5).pvalue)
        return {"n01": n01, "n10": n10, "statistic": float(n01), "p_value": p, "test": "exact"}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    return {
        "n01": n01,
        "n10": n10,
        "statistic": float(stat),
        "p_value": float(chi2.sf(stat, 1)),
        "test": "chi2_cc",
    }


# --------------------------------------------------------------------------- #
# Vectorised metrics for the bootstrap loops. They return the same values as
# sklearn's accuracy_score, f1_score(average="macro", zero_division=0) and
# balanced_accuracy_score (labels = union of classes present in y_true and
# y_pred), but avoid sklearn's per-call validation overhead, which dominated the
# runtime of the paired bootstrap. The random stream is unchanged, so intervals
# are identical to the sklearn-based implementation.
def _confusion(y, p):
    """Confusion matrix over the classes present in y or p (integer labels)."""
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.int64)
    k = int(max(y.max(), p.max())) + 1
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    present = (cm.sum(0) + cm.sum(1)) > 0
    return cm[np.ix_(present, present)]


def _fast_acc(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


def _fast_macro_f1(y, p):
    cm = _confusion(y, p)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return float((2 * tp / denom).mean())


def _fast_balanced_acc(y, p):
    cm = _confusion(y, p)
    support = cm.sum(1)
    keep = support > 0
    rec = np.diag(cm)[keep] / support[keep]
    return float(rec.mean())


_FAST_METRICS = {
    "accuracy": _fast_acc,
    "macro_f1": _fast_macro_f1,
    "balanced_acc": _fast_balanced_acc,
}


def paired_bootstrap_delta(
    labels, preds_a, preds_b, metric="accuracy", n_boot: int = 10_000, seed: int = 0
) -> dict:
    """CI on (metric_b - metric_a), resampling test items jointly.

    Pairing removes the shared test-set variance, which is what makes small but
    consistent differences detectable at this sample size.
    """
    fn = _FAST_METRICS[metric]
    y = np.asarray(labels)
    pa = np.asarray(preds_a)
    pb = np.asarray(preds_b)
    obs = fn(y, pb) - fn(y, pa)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        deltas[i] = fn(y[idx], pb[idx]) - fn(y[idx], pa[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # two-sided p from the bootstrap null that delta == 0
    p = float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    return {
        "delta": float(obs),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": min(1.0, p),
        "significant": bool(lo > 0 or hi < 0),
    }


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Step-down correction; needed because this study runs many comparisons."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    reject = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject.tolist()


# --------------------------------------------------------------------------- #
# the threshold question
# --------------------------------------------------------------------------- #
@dataclass
class CrossoverEstimate:
    method: str
    crossover_n: float
    ci_low: float
    ci_high: float
    slope: float
    note: str


def estimate_crossover(
    ns: Sequence[int], deltas: Sequence[float], delta_se: Sequence[float] | None = None,
    n_boot: int = 5000, seed: int = 0,
) -> CrossoverEstimate:
    """Where does synthetic augmentation stop hurting and start helping?

    Fits delta(log n) linearly and solves for delta = 0. This is the number the
    threshold framing turns on: below the crossover, generative augmentation is
    a net cost; above it, a net gain.
    """
    x = np.log(np.asarray(ns, dtype=float))
    y = np.asarray(deltas, dtype=float)
    se = np.asarray(delta_se, dtype=float) if delta_se is not None else np.zeros_like(y)

    slope, intercept = np.polyfit(x, y, 1)
    if abs(slope) < 1e-9:
        return CrossoverEstimate("", float("nan"), float("nan"), float("nan"), float(slope),
                                 "no trend in n")
    root = float(np.exp(-intercept / slope))

    rng = np.random.default_rng(seed)
    roots = []
    for _ in range(n_boot):
        yb = y + rng.normal(0, np.maximum(se, 1e-9))
        idx = rng.integers(0, len(x), len(x))
        if len(np.unique(x[idx])) < 2:
            continue
        s, b = np.polyfit(x[idx], yb[idx], 1)
        if abs(s) > 1e-9:
            roots.append(np.exp(-b / s))
    roots = np.asarray(roots)
    roots = roots[np.isfinite(roots) & (roots > 0) & (roots < 1e6)]
    lo, hi = (np.percentile(roots, [2.5, 97.5]) if len(roots) > 10 else (np.nan, np.nan))

    note = "extrapolated beyond tested range" if root > max(ns) else "within tested range"
    return CrossoverEstimate("", root, float(lo), float(hi), float(slope), note)


def summarize_conditions(results, metric: str = "macro_f1", baseline: str = "A. Scarce Only"):
    """Aggregate ClfResult objects into a table with CIs and paired tests."""
    import pandas as pd

    rows = []
    by_cond: dict[str, list] = {}
    for r in results:
        by_cond.setdefault(r.condition, []).append(r)

    base_runs = by_cond.get(baseline, [])
    for cond, runs in by_cond.items():
        vals = [getattr(r, metric) for r in runs]
        mean, lo, hi = bootstrap_ci(vals)
        row = {
            "condition": cond,
            "n_runs": len(runs),
            f"{metric}_mean": mean,
            f"{metric}_ci_low": lo,
            f"{metric}_ci_high": hi,
            f"{metric}_std": float(np.std(vals)),
            "n_train": runs[0].n_train,
            "n_synth": runs[0].n_synth,
        }
        if base_runs and cond != baseline:
            # pair run-for-run by seed where possible
            pb = paired_bootstrap_delta(
                base_runs[0].labels, base_runs[0].preds, runs[0].preds, metric=metric
            )
            mc = mcnemar(base_runs[0].labels, base_runs[0].preds, runs[0].preds)
            row.update(
                delta=pb["delta"],
                delta_ci_low=pb["ci_low"],
                delta_ci_high=pb["ci_high"],
                delta_p=pb["p_value"],
                mcnemar_p=mc["p_value"],
                significant=pb["significant"],
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# all-pairs comparison between conditions
# --------------------------------------------------------------------------- #
def pairwise_conditions(results, metric: str = "macro_f1", alpha: float = 0.05,
                        n_boot: int = 4000):
    """Every condition against every other, not just against the baseline.

    ``ablation_table`` answers "does this generator beat training on the scarce
    set alone". The comparative claims in this study are between *generators* --
    whether a corrected from-scratch DDPM matches latent-diffusion transfer, for
    instance -- and that comparison is never made by a set of vs-baseline tests.
    Deltas are paired on the shared test set and the family of comparisons is
    Holm-corrected, since k conditions give k(k-1)/2 tests.
    """
    import pandas as pd

    by_cond: dict[str, list] = {}
    for r in results:
        cond = r["condition"] if isinstance(r, dict) else r.condition
        by_cond.setdefault(cond, []).append(r)

    def get(r, k):
        return r[k] if isinstance(r, dict) else getattr(r, k)

    conds = sorted(by_cond)
    rows = []
    for i, a in enumerate(conds):
        for b in conds[i + 1:]:
            # pair seed-for-seed where both conditions were run with that seed
            seeds_a = {get(r, "seed"): r for r in by_cond[a]}
            seeds_b = {get(r, "seed"): r for r in by_cond[b]}
            shared = sorted(set(seeds_a) & set(seeds_b))
            if not shared:
                continue
            deltas, ps = [], []
            for s in shared:
                ra, rb = seeds_a[s], seeds_b[s]
                pb = paired_bootstrap_delta(
                    get(ra, "labels"), get(ra, "preds"), get(rb, "preds"),
                    metric=metric, n_boot=n_boot)
                deltas.append(pb)
                ps.append(mcnemar(get(ra, "labels"), get(ra, "preds"),
                                  get(rb, "preds"))["p_value"])
            rows.append({
                "a": a, "b": b, "n_seeds": len(shared),
                "delta_b_minus_a": float(np.mean([d["delta"] for d in deltas])),
                "ci_low": float(np.mean([d["ci_low"] for d in deltas])),
                "ci_high": float(np.mean([d["ci_high"] for d in deltas])),
                "p_bootstrap": float(np.median([d["p_value"] for d in deltas])),
                "p_mcnemar": float(np.median(ps)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["significant_holm"] = holm_bonferroni(df.p_bootstrap.tolist(), alpha=alpha)
    return df.sort_values("p_bootstrap").reset_index(drop=True)
