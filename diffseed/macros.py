"""Emit every reported number as a LaTeX macro.

The machinery -- name construction, formatting, the two output files and the
``??`` fallback contract -- lives in ``_vendor_macros.py``. What stays here is
the study-specific part: which quantities are worth reporting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._vendor_macros import MacroSet as _MacroSet, fmt, make_name


class MacroSet(_MacroSet):
    """Study-scoped macro set with the emitter name pre-filled."""

    def __init__(self, prefix: str = "DS"):
        super().__init__(prefix, emitter="diffseed.analyze")


def _name(*parts) -> str:
    return make_name("DS", *parts)


def _fmt(v, nd: int = 3) -> str:
    return fmt(v, nd)


# --------------------------------------------------------------------------- #
def emit_diffseed_macros(exp, written: dict, out_dir: Path, prefix: str = "DS") -> tuple:
    """Build the macro set from a completed DiffSeed report.

    ``prefix`` namespaces the emitted names so that a secondary corpus does not
    silently overwrite the primary one. Both runs emit the same quantities under
    the same construction rules; only the namespace differs, so a cross-species
    table can cite ``DSMz`` beside ``DS`` and the reader can tell which corpus
    each number came from. Without this the second run to finish wins, and the
    prose around the number keeps describing the first.
    """
    import pandas as pd

    M = MacroSet(prefix)
    cfg = exp.cfg

    # ---- dataset and protocol ----
    counts = exp.data.counts()
    M.add(sum(counts.values()), "n", "images")
    M.add(len(counts), "n", "classes")
    M.add(cfg.headline_scarcity, "headline", "scarcity")
    M.add(cfg.ddpm_steps, "ddpm", "steps")
    M.add(cfg.sd_steps, "sd", "steps")
    M.add(cfg.ddpm_res, "ddpm", "res")
    M.add(cfg.sd_res, "sd", "res")
    M.add(len(cfg.classifier_seeds), "n", "clf", "seeds")
    M.add(cfg.ddpm_cond_dropout, "ddpm", "cond", "dropout", nd=2)
    M.add(cfg.ddpm_guidance, "ddpm", "guidance", nd=1)
    M.add(cfg.ddpm_sample_steps, "ddpm", "sample", "steps")
    M.add(cfg.sd_sample_steps, "sd", "sample", "steps")
    sp = exp.splits(cfg.headline_scarcity)
    M.add(len(sp.test), "n", "test")
    M.add(len(sp.scarce), "n", "scarce")
    M.add(len(sp.train), "n", "train", "full")
    for c, n in counts.items():
        M.add(n, "count", c)

    # ---- generative quality, per method (pooled across classes) ----
    q = written.get("quality")
    if isinstance(q, pd.DataFrame) and not q.empty:
        pooled = q[q.cls == "ALL"]
        for _, r in pooled.iterrows():
            m = r["method"]
            for col, nd in (("fid", 2), ("fid_inf", 2), ("kid_mean", 4),
                            ("cmmd", 3), ("precision", 3), ("recall", 3),
                            ("density", 3), ("coverage", 3)):
                if col in r:
                    M.add(r[col], col, m, nd=nd)
        per = q[q.cls != "ALL"]
        for _, r in per.iterrows():
            M.add(r["fid"], "fid", r["method"], r["cls"], nd=2)

    def _emit_conditions(tab, prefix_parts=()):
        for _, r in tab.iterrows():
            c = r["condition"]
            M.add_pct(r.get("accuracy"), "acc", *prefix_parts, c)
            M.add(r.get("macro_f1"), "f", "one", *prefix_parts, c)
            if "delta" in r and np.isfinite(r.get("delta", np.nan)):
                M.add_pct(r["delta"], "delta", *prefix_parts, c, nd=2)
                M.add_pct(r["delta_ci_low"], "delta", *prefix_parts, c, "lo", nd=2)
                M.add_pct(r["delta_ci_high"], "delta", *prefix_parts, c, "hi", nd=2)
                M.add(r.get("delta_p"), "p", *prefix_parts, c, nd=4)
                M.add("yes" if r.get("significant") else "no", "sig", *prefix_parts, c)

    # ---- downstream ablation ----
    # Only the primary grid comes from the primary table. Tagged selection
    # variants are taken from the ablation's own single-process table below,
    # because a delta against a baseline trained in a different process is not
    # a measurement of the variant.
    ab = written.get("ablation")
    if isinstance(ab, pd.DataFrame) and not ab.empty:
        _emit_conditions(ab[~ab.condition.astype(str).str.contains(":")])

    # ---- selection ablation, single process, own baseline ----
    absel = written.get("ablation_selection")
    if isinstance(absel, pd.DataFrame) and not absel.empty:
        tagged = absel[absel.condition.astype(str).str.contains(":")]
        _emit_conditions(tagged)
        # The ablation's own reference rows are namespaced so they cannot be
        # confused with the primary grid's conditions of the same name.
        ref = absel[~absel.condition.astype(str).str.contains(":")]
        _emit_conditions(ref, prefix_parts=("abl",))

    # ---- does image quality predict usefulness? ----
    # The measurement argument is usually made with the FID floor alone, which
    # shows a comparison is uninterpretable at this sample size. The direct
    # question is whether the ranking a practitioner would actually use tracks
    # the outcome they care about. Rank correlation answers it in one number,
    # and is unaffected by the level of the baseline the deltas are measured
    # against, since a constant offset leaves ranks intact.
    try:
        q = written.get("quality")
        ab2 = written.get("ablation")
        if (isinstance(q, pd.DataFrame) and not q.empty
                and isinstance(ab2, pd.DataFrame) and not ab2.empty):
            qa = q[q.cls == "ALL"][["method", "fid"]].dropna()
            da = ab2[["condition", "delta"]].rename(columns={"condition": "method"})
            j = qa.merge(da, on="method").dropna()
            if len(j) >= 4:
                rho = float(j.fid.rank().corr(j.delta.rank()))
                M.add(rho, "fid", "utility", "rho")
                M.add(len(j), "fid", "utility", "n", nd=0)
                print(f"  FID-vs-utility rank correlation: rho={rho:+.3f} "
                      f"over {len(j)} generators")
    except Exception as e:
        print(f"  fid/utility correlation skipped: {type(e).__name__}: {e}")

    # ---- threshold crossover ----
    cx = written.get("crossovers")
    if isinstance(cx, dict):
        for m, est in cx.items():
            M.add(est.crossover_n, "crossover", m, nd=0)
            M.add(est.ci_low, "crossover", m, "lo", nd=0)
            M.add(est.ci_high, "crossover", m, "hi", nd=0)

    # ---- cost ----
    costs = written.get("costs")
    if isinstance(costs, pd.DataFrame) and not costs.empty:
        g = costs[costs.name.str.startswith("gen::")].copy()
        if not g.empty:
            g["method"] = g.name.str.split("::").str[1]
            for m, sub in g.groupby("method"):
                M.add(sub.gpu_minutes.mean(), "gpumin", m, nd=1)
                M.add(sub.trainable_params_m.mean(), "params", m, nd=2)
                M.add(sub.peak_vram_gb.max(), "vram", m, nd=1)
                # Energy is recorded per stage and is the axis a laboratory
                # actually pays for. It is reported because GPU-minutes are not
                # comparable across accelerators and watt-hours are.
                if "energy_wh" in sub:
                    M.add(sub.energy_wh.mean(), "energy", m, nd=1)
            # Total cost of the whole generative grid, which is the number
            # someone deciding whether to attempt this at all needs.
            M.add(g.gpu_minutes.sum() / 60.0, "gpu", "hours", "total", nd=1)
            if "energy_wh" in g:
                M.add(g.energy_wh.sum() / 1000.0, "energy", "kwh", "total", nd=2)

    # ---- measurement floor: what the metric reports when the two
    # ---- distributions are in fact identical, at each sample size
    import json as _json

    # ---- information-free controls -------------------------------------
    # These carry the study's central claim, so they must be citable as macros
    # rather than transcribed. dup duplicates real images (no new information),
    # dupshuf duplicates them under permuted labels (information destroyed),
    # weighted/oversample correct the class prior while adding no images at all.
    # The difference ddpm - dup is the part of the measured effect that is
    # actually attributable to generated pixels.
    ctrl = written.get("controls")
    if ctrl is not None and not getattr(ctrl, "empty", True):
        nwords = {25: "twentyFive", 50: "fifty", 100: "hundred",
                  200: "twoHundred", 400: "fourHundred"}
        for n in sorted(ctrl.scarcity.unique()):
            w = nwords.get(int(n))
            if not w:
                continue
            sub = ctrl[ctrl.scarcity == n].set_index("condition")
            for cond in ("dup", "dupshuf", "weighted", "oversample",
                         "ddpm_fixed", "none"):
                if cond not in sub.index:
                    continue
                row = sub.loc[cond]
                base = cond.replace("_", "")
                if cond == "none":
                    M.add(row.get("macro_f1"), "ctrl", "base", w, nd=4)
                    continue
                d = row.get("delta")
                if d is None or (isinstance(d, float) and d != d):
                    continue
                M.add(100 * float(d), "ctrl", base, w, nd=2)
                lo, hi = row.get("delta_ci_low"), row.get("delta_ci_high")
                if lo is not None and lo == lo:
                    M.add(100 * float(lo), "ctrl", base, w, "lo", nd=2)
                    M.add(100 * float(hi), "ctrl", base, w, "hi", nd=2)
            # the headline contrast: effect not explained by real duplicates
            if "ddpm_fixed" in sub.index and "dup" in sub.index:
                syn = sub.loc["ddpm_fixed"].get("delta")
                dup = sub.loc["dup"].get("delta")
                if syn is not None and dup is not None and syn == syn and dup == dup:
                    M.add(100 * (float(syn) - float(dup)), "ctrl", "pixels", w, nd=2)

    curves_p = exp.root / "quality" / f"fid_curves_n{cfg.headline_scarcity}.json"
    if curves_p.exists():
        try:
            curves = _json.loads(curves_p.read_text(encoding="utf-8"))
            any_curve = next(iter(curves.values()), [])
            words = {100: "hundred", 200: "twoHundred", 500: "fiveHundred",
                     1000: "thousand", 2000: "twoThousand"}
            for row in any_curve:
                w = words.get(row["n"])
                if w:
                    M.add(row["fid_real_real_mean"], "fid", "floor", w, nd=2)
        except Exception:
            pass

    # ---- final training loss of the repaired pixel model ----
    losses_p = exp.synth_dir("ddpm_fixed", cfg.headline_scarcity) / "losses.json"
    if losses_p.exists():
        try:
            v = _json.loads(losses_p.read_text(encoding="utf-8"))
            seq = v["g"] if isinstance(v, dict) else v
            if seq:
                M.add(seq[-1], "ddpm", "final", "loss", nd=4)
        except Exception:
            pass

    # ---- memorisation audit: closest synthetic-to-real match seen ----
    memo_p = exp.root / "quality" / f"memorisation_n{cfg.headline_scarcity}.json"
    if memo_p.exists():
        try:
            memo = _json.loads(memo_p.read_text(encoding="utf-8"))
            alld = [p[2] for v in memo.values() for p in v]
            if alld:
                M.add(min(alld), "knn", "min", nd=3)
        except Exception:
            pass

    # ---- sampler selection diagnostic (Section: setup_sampler) ----
    # Recorded from the controlled comparison at fixed weights; see
    # diagnostics/sampler_comparison.json for provenance.
    diag_p = exp.root.parent / "sampler_diagnostic.json"
    if not diag_p.exists():
        diag_p = Path(__file__).resolve().parents[2] / "diagnostics" / "sampler_diagnostic.json"
    if diag_p.exists():
        try:
            d = _json.loads(diag_p.read_text(encoding="utf-8"))
            M.add(d.get("real_std"), "real", "std", nd=3)
            M.add(d.get("ddpm_1000_g1"), "std", "ddpm", "thousand", nd=3)
            M.add(d.get("ddpm_1000_g2"), "std", "ddpm", "thousand", "cfg", nd=3)
            M.add(d.get("ddim_100_g1"), "std", "ddim", "hundred", nd=3)
            M.add(d.get("dpmsolver_100_g1"), "std", "dpmsolver", nd=3)
        except Exception:
            pass

    # ---- selection ablation: per-variant rejection statistics ----
    # The downstream delta says whether a variant helps; these say why. The
    # mean confidence of the kept set against the dropped set is the direct
    # measure of whether a variant is selecting for class-correctness at all,
    # and a variant can keep a plausible number of images while selecting
    # against it.
    fdir = exp.root / "filtered" / f"n{cfg.headline_scarcity}"
    if fdir.is_dir():
        for rp in sorted(fdir.glob("filter_report__*.csv")):
            tag = rp.stem.replace("filter_report__", "")
            try:
                fr = pd.read_csv(rp)
                for m, sub in fr.groupby("method"):
                    cand = int(sub.n_candidates.sum())
                    kept = int(sub.n_kept.sum())
                    M.add(kept, "filtkept", m, tag)
                    M.add_pct(kept / cand if cand else float("nan"),
                              "filtkeptpct", m, tag, nd=0)
                    M.add(float(sub.mean_confidence_kept.mean()),
                          "filtconfkept", m, tag)
                    M.add(float(sub.mean_confidence_dropped.mean()),
                          "filtconfdrop", m, tag)
            except Exception:
                pass

    # ---- sampler diagnostic ----
    sd_p = exp.root / "sampler_diagnostic.csv"
    if sd_p.exists():
        try:
            sd = pd.read_csv(sd_p)
            M.add(float(sd.real_std.iloc[0]), "real", "std")
            for r in sd.itertuples():
                M.add(float(r.output_std), "std", r.arm)
                M.add_pct(float(r.saturated_frac), "sat", r.arm, nd=1)
                if "fid" in sd.columns:
                    M.add(float(r.fid), "samplerfid", r.arm, nd=1)
                if "seconds_per_image" in sd.columns:
                    M.add(float(r.seconds_per_image), "samplersecs", r.arm, nd=2)
        except Exception:
            pass

    # ---- quality-aware selection ----
    fr_p = exp.root / "filtered" / f"n{cfg.headline_scarcity}" / "filter_report.csv"
    if fr_p.exists():
        try:
            fr = pd.read_csv(fr_p)
            for m, sub in fr.groupby("method"):
                cand = int(sub.n_candidates.sum())
                kept = int(sub.n_kept.sum())
                M.add(cand, "filtcand", m)
                M.add(kept, "filtkept", m)
                M.add_pct(kept / cand if cand else float("nan"), "filtkeptpct", m, nd=0)
                M.add(int(sub.dropped_low_confidence.sum()), "filtlowconf", m)
                M.add(int(sub.dropped_near_duplicate.sum()), "filtdup", m)
                M.add(float(sub.mean_confidence_kept.mean()), "filtconfkept", m)
                M.add(float(sub.mean_confidence_dropped.mean()), "filtconfdrop", m)
        except Exception:
            pass

    # ---- detectability ----
    det = written.get("detect")
    if isinstance(det, pd.DataFrame) and not det.empty:
        for m, sub in det.groupby("method"):
            M.add(sub.detector_auc.mean(), "detauc", m)
            M.add(sub.fooling_rate.mean(), "fool", m)

    return M.write(out_dir)
