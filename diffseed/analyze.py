"""Turn the raw run artefacts into tables, statistics and figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures as FIG
from .stats import (
    bootstrap_ci,
    estimate_crossover,
    holm_bonferroni,
    mcnemar,
    paired_bootstrap_delta,
)

# Reported for soybean grain imagery by a 2025 Computers and Electronics in
# Agriculture paper; drawn on the sample-size figure so readers can see at what
# N our numbers are and are not comparable to it.
# A literature reference line was drawn here at FID 36.68, labelled as a 2025
# COMPAG soybean result. The source could not be established: the value appears
# in no reference the manuscript carries and in none of the earlier study's
# text. An uncited number a reader would read off a figure is worse than no
# reference line, and the measurement argument does not need one -- the
# real-versus-real floor already shows the comparison is invalid. Restore this
# only together with a citation.
LITERATURE_FID = None


def load_downstream(root: Path, scarcity: int) -> pd.DataFrame:
    p = Path(root) / "downstream" / f"n{scarcity}.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    return _drop_degenerate_synthetic(df, root, scarcity)


def _drop_degenerate_synthetic(df: pd.DataFrame, root: Path,
                               scarcity: int) -> pd.DataFrame:
    """Remove conditions that promise synthetic images and contribute none.

    Selection trains its scoring classifier on the scarce real data, so below
    some data volume the scorer never learns and rejects every candidate. The
    filter then writes an empty manifest, and the "+filtered" condition trains
    on exactly the baseline training set: same images, same count, same split.

    Nothing in the results file marks that. The row looks like an ordinary
    condition, so the delta machinery compares the baseline against itself and
    reports the difference between two runs of one experiment as an effect. On
    this corpus at n=25 that fabricated a 2.05 pp DEFICIT for selection, which
    would have been written up as "selection is harmful when data is scarcest".

    The difference between those runs is a useful quantity -- it is the
    run-to-run noise floor under nondeterministic kernels -- but it is not an
    effect, so it is recorded separately rather than averaged into a table.
    """
    if df.empty or "n_synth" not in df or "condition" not in df:
        return df
    promises = df.condition.str.contains(r"[+]filtered|[+]", regex=True) | ~df.condition.isin(
        ("none", "oracle", "trad", "weighted", "oversample", "dup", "dupshuf"))
    bad = promises & (df.n_synth.fillna(0) == 0)
    if not bad.any():
        return df
    rec = df[bad]
    out = Path(root) / "downstream" / "degenerate_conditions.csv"
    cols = [c for c in ("condition", "model", "seed", "n_real", "n_synth",
                        "n_train", "macro_f1") if c in rec]
    hdr = not out.exists()
    rec = rec[cols].assign(scarcity=scarcity)
    rec.to_csv(out, mode="a", header=hdr, index=False)
    for cond, g in rec.groupby("condition"):
        print(f"  [degenerate] {cond} @ n={scarcity}: n_synth=0 over "
              f"{len(g)} run(s) -- identical to the baseline, excluded from "
              f"delta tables (recorded in {out.name})")
    return df[~bad]


def load_quality(root: Path, scarcity: int) -> pd.DataFrame:
    p = Path(root) / "quality" / f"n{scarcity}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


# --------------------------------------------------------------------------- #
def _seed_level_p(deltas):
    """Two-sided p for "the mean delta is zero", resampling SEEDS.

    A paired t on the per-seed deltas. With three seeds this is a weak test and
    that is the point: it reports the power the design actually has, instead of
    borrowing significance from a test-item bootstrap that holds the training
    run fixed and therefore cannot speak to replication.
    """
    import numpy as np

    d = np.asarray(deltas, dtype=float)
    n = len(d)
    if n < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if d.mean() != 0 else 1.0
    t = d.mean() / (sd / np.sqrt(n))
    try:
        from scipy import stats as _st

        return float(2 * (1 - _st.t.cdf(abs(t), df=n - 1)))
    except Exception:
        # Normal approximation; conservative direction is not guaranteed, so the
        # t statistic and n are reported alongside for anyone checking.
        from math import erf, sqrt

        return float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


def ablation_table(df: pd.DataFrame, metric="macro_f1", baseline="none") -> pd.DataFrame:
    """Per-condition mean with CI, plus paired tests against the scarce baseline."""
    if df.empty:
        return df
    rows = []
    base = df[df.condition == baseline]
    for cond, g in df.groupby("condition"):
        vals = g[metric].tolist()
        mean, lo, hi = bootstrap_ci(vals)
        row = {
            "condition": cond,
            "n_runs": len(g),
            "n_train": int(g.n_train.iloc[0]),
            "n_synth": int(g.n_synth.iloc[0]),
            "accuracy": float(g.accuracy.mean()),
            metric: mean,
            f"{metric}_ci_low": lo,
            f"{metric}_ci_high": hi,
            f"{metric}_sd": float(np.std(vals)),
        }
        if not base.empty and cond != baseline:
            # pair seed-for-seed on the shared test set, then average the deltas
            deltas, ps = [], []
            for seed in sorted(set(g.seed) & set(base.seed)):
                a = base[base.seed == seed].iloc[0]
                b = g[g.seed == seed].iloc[0]
                pb = paired_bootstrap_delta(a.labels, a.preds, b.preds, metric=metric, n_boot=2000)
                deltas.append(pb)
                ps.append(mcnemar(a.labels, a.preds, b.preds)["p_value"])
            if deltas:
                # Two resampling units, reported separately because they answer
                # different questions and are not interchangeable.
                #
                # The item-level interval resamples TEST IMAGES at a fixed seed
                # and averages those intervals over seeds. It therefore contains
                # no between-seed variance at all, and averaging intervals is not
                # itself a valid interval. It says: given this trained model, how
                # precisely is its advantage measured on this test set.
                #
                # The seed-level interval resamples the per-seed deltas and is
                # the one that answers "would this replicate if retrained". In a
                # scarcity study seed variance dominates, so this is the interval
                # a claim should be judged against, and it is what drives
                # significance below.
                per_seed = [d["delta"] for d in deltas]
                s_mean, s_lo, s_hi = bootstrap_ci(per_seed)
                row.update(
                    delta=float(np.mean(per_seed)),
                    delta_item_ci_low=float(np.mean([d["ci_low"] for d in deltas])),
                    delta_item_ci_high=float(np.mean([d["ci_high"] for d in deltas])),
                    delta_item_p=float(np.median([d["p_value"] for d in deltas])),
                    delta_ci_low=float(s_lo),
                    delta_ci_high=float(s_hi),
                    delta_sd=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                    delta_n_seeds=len(per_seed),
                    delta_p=_seed_level_p(per_seed),
                    mcnemar_p=float(np.median(ps)),
                )
        rows.append(row)

    out = pd.DataFrame(rows)
    if "delta_p" in out:
        # Holm within families rather than across every row.
        #
        # The primary grid asks "does this generator help?"; the selection
        # ablation asks "which criterion produces the benefit?". They are
        # different questions, and correcting them jointly makes the primary
        # result's significance depend on how many ablation variants happened to
        # be run in the same session -- run seven and the headline can lose the
        # significance it had with none. That is a bookkeeping artefact, not a
        # statistical judgement, so the families are corrected separately.
        fam = ["ablation" if ":" in str(c) else "primary" for c in out.condition]
        out["family"] = fam
        out["significant_holm"] = False
        for _, sub in out.groupby("family"):
            m = sub.delta_p.notna()
            idx = sub.index[m]
            if len(idx):
                out.loc[idx, "significant_holm"] = holm_bonferroni(
                    sub.loc[m, "delta_p"].tolist())
        out["significant"] = out["significant_holm"].fillna(False)
    return out.sort_values(metric, ascending=False).reset_index(drop=True)


def threshold_table(root: Path, levels, metric="macro_f1", baseline="none",
                    model: str | None = None) -> pd.DataFrame:
    """Downstream effect against scarcity, for ONE architecture.

    ``model`` must be given whenever the downstream cache holds more than one.
    It did not, historically, because only the headline architecture was ever
    written there; the architecture sweep now writes into the same file, and
    without this filter the per-seed pairing inside ``ablation_table`` selects
    whichever row happens to come first, silently averaging four backbones with
    baselines spanning ten points into a single "the effect at n" number.
    """
    rows = []
    for n in levels:
        df = load_downstream(root, n)
        if df.empty:
            continue
        if model is not None and "model" in df and (df.model == model).any():
            df = df[df.model == model]
        elif "model" in df and df.model.nunique() > 1:
            raise ValueError(
                f"threshold_table: {df.model.nunique()} architectures in the n={n} "
                f"cache and no model specified; refusing to average across them")
        tab = ablation_table(df, metric=metric, baseline=baseline)
        for _, r in tab.iterrows():
            if r.condition in (baseline, "oracle") or pd.isna(r.get("delta", np.nan)):
                continue
            rows.append(
                {"method": r.condition, "scarcity": n, "delta": r["delta"],
                 "delta_ci_low": r["delta_ci_low"], "delta_ci_high": r["delta_ci_high"],
                 "p": r.get("delta_p", np.nan)}
            )
    return pd.DataFrame(rows)


def crossovers(thr: pd.DataFrame) -> dict:
    out = {}
    for m, g in thr.groupby("method"):
        g = g.sort_values("scarcity")
        if len(g) < 2:
            continue
        se = ((g.delta_ci_high - g.delta_ci_low) / 3.92).tolist()
        est = estimate_crossover(g.scarcity.tolist(), g.delta.tolist(), se)
        est.method = m
        out[m] = est
    return out


# --------------------------------------------------------------------------- #
def build_report(exp) -> dict:
    """Write every table and figure for a completed (or partial) experiment."""
    root = exp.root
    figs = root / "figures"
    tabs = root / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    FIG.use_paper_style()

    cfg = exp.cfg
    levels = list(cfg.scarcity_levels)
    head = cfg.headline_scarcity if cfg.headline_scarcity in levels else levels[0]
    written = {}

    # ---- ablation at the headline scarcity ----
    df = load_downstream(root, head)
    if not df.empty:
        head_df = df[df.model == cfg.clf_headline] if (df.model == cfg.clf_headline).any() else df
        tab = ablation_table(head_df)
        tab.drop(columns=[c for c in ("preds", "labels") if c in tab], errors="ignore").to_csv(
            tabs / "table_ablation.csv", index=False
        )
        written["ablation"] = tab
        if "delta" in tab:
            written["forest"] = FIG.fig_forest(tab, figs)

        if df.model.nunique() > 1:
            hm = df.rename(columns={"condition": "method"})
            written["heatmap"] = FIG.fig_model_method_heatmap(hm, figs)
            hm.groupby(["model", "method"]).macro_f1.mean().unstack().to_csv(
                tabs / "table_model_method.csv"
            )

    # ---- generator vs generator, not just generator vs baseline ----
    if not df.empty:
        from .stats import pairwise_conditions

        try:
            head_rows = df[df.model == cfg.clf_headline] if (df.model == cfg.clf_headline).any() else df
            pw = pairwise_conditions(head_rows.to_dict("records"))
            if not pw.empty:
                pw.to_csv(tabs / "table_pairwise_conditions.csv", index=False)
                written["pairwise"] = pw
                sig = pw[pw.significant_holm]
                print(f"  pairwise: {len(sig)}/{len(pw)} condition pairs separable "
                      f"after Holm correction")
        except Exception as e:
            print(f"pairwise comparison skipped: {type(e).__name__}: {e}")

    # ---- selection ablation, analysed within its own process ----
    # The ablation carries its own baseline and its own copy of the default
    # rule, all trained in one process, because results shift between processes
    # by more than the differences the ablation is measuring. It is therefore
    # summarised against that baseline and not against the primary grid's.
    abl_p = root / "downstream" / f"n{head}_ablation.jsonl"
    if abl_p.exists():
        try:
            adf = pd.read_json(abl_p, lines=True)
            if (adf.model == cfg.clf_headline).any():
                adf = adf[adf.model == cfg.clf_headline]
            atab = ablation_table(adf, baseline="none")
            atab.to_csv(tabs / "table_selection_ablation.csv", index=False)
            written["ablation_selection"] = atab
            print(f"  selection ablation: {len(atab)} conditions "
                  f"(own baseline, single process)")
        except Exception as e:
            print(f"selection ablation summary skipped: {type(e).__name__}: {e}")

    # ---- information-free controls, every scarcity level ----
    # These decide what the augmentation effect is made of, and nothing read
    # them until now: the stage ran 10 seeds per condition and produced no
    # table, no figure and no macro.
    #
    # dup adds the same number of REAL images already in the training set, so it
    # matches image count, optimiser steps, epochs, schedule length,
    # validation-selection count and label prior, and adds no information.
    # delta(dup) is therefore the part of the effect that is not the pixels.
    # weighted/oversample correct the label prior WITHOUT adding images at all,
    # which separates prior correction from data volume.
    ctrl_rows = []
    for n in levels:
        cp = root / "downstream" / f"n{n}_controls.jsonl"
        if not cp.exists():
            continue
        try:
            cdf = pd.read_json(cp, lines=True)
            if "model" in cdf and (cdf.model == cfg.clf_headline).any():
                cdf = cdf[cdf.model == cfg.clf_headline]
            ct = ablation_table(cdf, baseline="none")
            ct["scarcity"] = n
            ctrl_rows.append(ct)
        except Exception as e:
            print(f"  controls n={n} skipped: {type(e).__name__}: {e}")
    if ctrl_rows:
        ctab = pd.concat(ctrl_rows, ignore_index=True)
        ctab.to_csv(tabs / "table_controls.csv", index=False)
        written["controls"] = ctab
        for n in sorted(ctab.scarcity.unique()):
            sub = ctab[ctab.scarcity == n].set_index("condition")
            def _d(c):
                return float(sub.delta[c]) * 100 if c in sub.index else float("nan")
            syn, dup = _d("ddpm_fixed"), _d("dup")
            print(f"  controls n={n}: ddpm_fixed {syn:+.2f} | dup {dup:+.2f} | "
                  f"attributable to pixels {syn - dup:+.2f} pp")

    # ---- threshold grid ----
    thr = threshold_table(root, levels, model=cfg.clf_headline)
    if not thr.empty:
        thr.to_csv(tabs / "table_threshold.csv", index=False)
        cx = crossovers(thr)
        written["threshold"] = FIG.fig_threshold(thr, figs, crossovers=cx)
        pd.DataFrame(
            [{"method": k, "crossover_n": v.crossover_n, "ci_low": v.ci_low,
              "ci_high": v.ci_high, "slope": v.slope, "note": v.note} for k, v in cx.items()]
        ).to_csv(tabs / "table_crossover.csv", index=False)
        written["crossovers"] = cx

    # ---- generative quality ----
    q = load_quality(root, head)
    if not q.empty:
        q.to_csv(tabs / "table_quality.csv", index=False)
        pooled = q[q.cls == "ALL"].copy()
        written["quality"] = q

        curves_p = root / "quality" / f"fid_curves_n{head}.json"
        if curves_p.exists():
            curves = json.loads(curves_p.read_text(encoding="utf-8"))
            if curves:
                written["fid_bias"] = FIG.fig_fid_bias(curves, figs, reference=LITERATURE_FID)

        # cost-quality pareto
        costs = exp.ledger.to_frame()
        if not costs.empty and not pooled.empty:
            g = costs[costs.name.str.startswith("gen::")].copy()
            if not g.empty:
                g["method"] = g.name.str.split("::").str[1]
                g["scarcity"] = g.name.str.split("::").str[2].str.lstrip("n").astype(int)
                g = g[g.scarcity == head]
                merged = pooled.merge(g[["method", "gpu_minutes", "trainable_params_m"]], on="method")
                if not merged.empty:
                    written["pareto"] = FIG.fig_pareto(merged, figs)

        radar_cols = [c for c in ("fid_inf", "kid_mean", "cmmd", "precision", "recall",
                                  "density", "coverage") if c in pooled and pooled[c].notna().any()]
        if len(radar_cols) >= 3 and len(pooled) > 1:
            hib = {"precision": True, "recall": True, "density": True, "coverage": True,
                   "fid_inf": False, "kid_mean": False, "cmmd": False}
            written["radar"] = FIG.fig_radar(pooled.dropna(subset=radar_cols), figs,
                                             radar_cols, higher_is_better=hib)

    # ---- feature-space embedding (v1 had this; the rewrite had dropped it) ----
    emb_p = root / "quality" / f"embedding_n{head}.npz"
    if emb_p.exists():
        try:
            z = np.load(emb_p, allow_pickle=True)
            emb, source = z["emb"], z["source"].astype(str)
            # the figure signature wants per-point class labels; sources are the
            # meaningful grouping here, so colour by source in both panels
            written["embedding"] = FIG.fig_embedding(
                emb, np.zeros(len(emb), dtype=int), source, ["all classes"], figs)
        except Exception as e:
            print(f"embedding figure skipped: {e}")

    # ---- denoising trajectory (v1 fig 5) ----
    ck_dir = root / "checkpoints"
    traj_methods = [m for m in ("ddpm_fixed", "ddpm_v1", "ddpm_ft")
                    if (ck_dir / f"{m}_n{head}.pt").exists()]
    if traj_methods:
        try:
            written["trajectory"] = _render_trajectory(exp, traj_methods[0], head, figs)
        except Exception as e:
            print(f"trajectory figure skipped: {type(e).__name__}: {e}")

    # ---- memorisation audit ----
    memo_p = root / "quality" / f"memorisation_n{head}.json"
    if memo_p.exists():
        memo = json.loads(memo_p.read_text(encoding="utf-8"))
        best = min(memo.items(), key=lambda kv: min(p[2] for p in kv[1]) if kv[1] else 1e9,
                   default=(None, None))
        if best[0] and best[1]:
            written["memorisation"] = FIG.fig_memorisation(
                [tuple(p) for p in best[1]], figs)
            pd.DataFrame(
                [{"method": m, "min_lpips": min(p[2] for p in v),
                  "mean_lpips": float(np.mean([p[2] for p in v])),
                  "n_below_0.15": sum(1 for p in v if p[2] < 0.15)}
                 for m, v in memo.items() if v]
            ).to_csv(tabs / "table_memorisation.csv", index=False)

    # ---- sample grids ----
    from .metrics import list_images

    sources = {"real": {c: [str(p) for p in list_images(exp.real_ref_dir(head) / c, 4)]
                        for c in exp.class_names}}
    for m in list(cfg.grid_methods) + list(cfg.extra_methods):
        d = exp.synth_dir(m, head)
        if d.exists():
            sources[m] = {c: [str(p) for p in list_images(d / c, 4)] for c in exp.class_names}
    if len(sources) > 1:
        written["grid"] = FIG.fig_sample_grid(sources, exp.class_names, figs)

    # ---- generator training curves ----
    curves = {}
    for m in list(cfg.grid_methods) + list(cfg.extra_methods):
        p = exp.synth_dir(m, head) / "losses.json"
        if p.exists():
            v = json.loads(p.read_text(encoding="utf-8"))
            curves[m] = v["g"] if isinstance(v, dict) else v
    if curves:
        written["curves"] = FIG.fig_training_curves(curves, figs)

    # ---- cost ledger ----
    costs = exp.ledger.to_frame()
    if not costs.empty:
        costs.to_csv(tabs / "table_costs.csv", index=False)
        written["costs"] = costs

    summary = {
        k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else str(v))
        for k, v in written.items()
    }
    # LaTeX macros for the manuscript: every reported number comes from here,
    # so a table can no longer disagree with the data behind it
    try:
        from .macros import emit_diffseed_macros

        written["detect"] = None
        det_p = root / "detectability" / f"n{head}.csv"
        if det_p.exists():
            written["detect"] = pd.read_csv(det_p)
        mac, dfl = emit_diffseed_macros(exp, written, root / "paper_macros",
                                        prefix=getattr(exp.cfg, "macro_prefix", "DS"))
        print(f"  macros: {mac.name}, {dfl.name}")
    except Exception as e:
        print(f"macro emission skipped: {type(e).__name__}: {e}")

    (root / "report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"report written: {len(written)} artefacts -> {root}")
    return written


def _render_trajectory(exp, method: str, scarcity: int, figs):
    """Reverse-diffusion trajectory, one row per class.

    Reconstructed from the saved generator checkpoint rather than captured
    during sampling, so it costs nothing during the long generation pass and can
    be regenerated later. This is why generators are checkpointed before they
    sample.
    """
    import torch

    from .generators import pixel_ddpm as P

    cfg = exp.cfg
    variant = {"ddpm_v1": "v1", "ddpm_fixed": "fixed", "ddpm_ft": "ft"}[method]
    ck = exp.root / "checkpoints" / f"{method}_n{scarcity}.pt"

    from diffusers import DDPMScheduler

    if variant == "v1":
        model = P.LegacyClassConditionedUNet(exp.n_classes, cfg.ddpm_res)
    elif variant == "ft":
        from diffusers import UNet2DModel

        model = UNet2DModel.from_pretrained(cfg.ddpm_ft_model)
        model.register_to_config(sample_size=cfg.ddpm_res)
        model.class_embedding = torch.nn.Embedding(
            exp.n_classes + 1, model.time_embedding.linear_2.out_features)
    else:
        model = P.build_fixed_unet(exp.n_classes, cfg.ddpm_res, cfg.ddpm_channels)
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    model = model.to(cfg.device).eval()

    sched = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    skip = {exp.splits(scarcity).majority_class}
    traj = {}
    for ci, name in enumerate(exp.class_names):
        if ci in skip:
            continue
        _, snaps = P.sample_pixel_ddpm(
            model, sched, exp.n_classes, ci, 1, cfg, variant=variant,
            batch_size=1, steps=cfg.ddpm_sample_steps, seed=cfg.seed,
            return_trajectory=True,
        )
        if snaps:
            traj[name] = snaps
    del model
    torch.cuda.empty_cache()
    if traj:
        return FIG.fig_trajectory(traj, figs, name=f"fig_denoising_{method}")
    return None
