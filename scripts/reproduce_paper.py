#!/usr/bin/env python3
"""Regenerate the manuscript's tables, figures and reported numbers from the
saved run records in ``results/``, and check each number against the value
printed in the manuscript.

No GPU, no images and no model weights are needed; it runs in under a minute on
a laptop CPU.

Usage:
    python scripts/reproduce_paper.py [--results results] [--out outputs]

Outputs:
    outputs/tables/*.csv          Table 2 and the supporting summaries
    outputs/figures/*.pdf|png     Figures 2, 3, 6 and 7
    outputs/verification.csv      claim | paper value | reproduced value | match
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diffseed import figures as FIG  # noqa: E402
from diffseed.analyze import ablation_table  # noqa: E402

HEADLINE_MODEL = "resnet50"
GENERATORS = ["sd_lora_sdedit", "ddpm_v1", "ddpm_ft", "ddpm_fixed", "sd_lora",
              "sd_turbo_lora", "dcgan", "fastgan"]  # Table 2 order (by FID)

# --------------------------------------------------------------------------- #
# Values as printed in the manuscript (Artificial Intelligence in Agriculture
# submission, Sep 2026).
# --------------------------------------------------------------------------- #
PAPER = {
    # Table 2
    "T2 FID sd_lora_sdedit": 31.52, "T2 FID ddpm_v1": 50.67, "T2 FID ddpm_ft": 54.25,
    "T2 FID ddpm_fixed": 63.85, "T2 FID sd_lora": "66.50", "T2 FID sd_turbo_lora": 82.44,
    "T2 FID dcgan": "122.80", "T2 FID fastgan": 275.61,
    "T2 dF1 sd_lora_sdedit (pp)": 6.98, "T2 dF1 ddpm_v1 (pp)": 4.05, "T2 dF1 ddpm_ft (pp)": 4.55,
    "T2 dF1 ddpm_fixed (pp)": 6.68, "T2 dF1 sd_lora (pp)": 5.64, "T2 dF1 sd_turbo_lora (pp)": 5.86,
    "T2 dF1 dcgan (pp)": 3.18, "T2 dF1 fastgan (pp)": 7.11,
    "Spearman rho(FID, utility), 8 generators": 0.071,
    "number of generator configurations": 8,
    "pooled FID generated images (N)": 1600,
    "pooled FID real images (N)": 2000,
    "classifier training runs per condition": 3,
    # Section 3.3
    "min LPIPS to nearest real training image": 0.264,
    "generated images below LPIPS 0.15": 0,
    "Fig 5 closest pair 2": "0.270", "Fig 5 closest pair 3": "0.270", "Fig 5 closest pair 4": 0.301,
    "Fig 5 closest pair 5": 0.308, "Fig 5 closest pair 6": 0.313, "Fig 5 closest pair 7": 0.322,
    "Fig 5 closest pair 8": 0.324,
    "detector AUC ddpm_fixed": "1.000",
    "detector AUC sd_lora_sdedit": 0.999,
    # Section 3.4
    "real-vs-real FID, N=100": 69.05, "real-vs-real FID, N=200": 48.76,
    "real-vs-real FID, N=500": 28.01, "real-vs-real FID, N=1000": "16.70",
    # Section 3.5
    "sampler DDPM 1000 steps FID": 107.7, "sampler DDPM 1000 steps output std": 0.589,
    "sampler DDPM 500 steps FID": 109.6, "sampler DDPM 500 steps output std": 0.576,
    "sampler DDIM 100 steps FID": "148.0",
    "sampler DPM-Solver++ output std": 245.631,
    "sampler DPM-Solver++ saturated pixels (%)": 99.7,
    "sampler DPM-Solver++ FID": 352.1,
    # Section 3.6
    "GPU-minutes FastGAN": 3.7, "GPU-minutes repaired DDPM": 397.4, "GPU-minutes SD LoRA": 57.6,
    "total GPU-hours": 42.6,
    # Section 2 (configuration recorded with the run)
    "repaired DDPM resolution": 128, "repaired DDPM training steps": 8500,
    "condition dropout": "0.10", "guidance strength": 2, "production sampling steps": 500,
    "soybean corpus size (images)": 5513,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    args = ap.parse_args()
    res = Path(args.results) / "soybean"
    out = Path(args.out)
    tabs, figs = out / "tables", out / "figures"
    tabs.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    FIG.use_paper_style()
    got: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    print("[1/6] realism vs utility (Table 2)")
    q = pd.read_csv(res / "quality" / "n100.csv")
    pooled = q[q.cls == "ALL"].copy()
    ds = pd.read_json(res / "downstream" / "n100.jsonl", lines=True)
    ab = ablation_table(ds[ds.model == HEADLINE_MODEL], baseline="none").set_index("condition")
    t2 = pooled.set_index("method").loc[GENERATORS, ["fid", "n_real", "n_fake"]].copy()
    t2["delta_macro_f1_pp"] = [100 * float(ab.loc[m, "delta"]) for m in GENERATORS]
    t2["delta_ci_low_pp"] = [100 * float(ab.loc[m, "delta_ci_low"]) for m in GENERATORS]
    t2["delta_ci_high_pp"] = [100 * float(ab.loc[m, "delta_ci_high"]) for m in GENERATORS]
    t2.round(3).to_csv(tabs / "table2_realism_vs_utility.csv")
    for m in GENERATORS:
        got[f"T2 FID {m}"] = float(t2.loc[m, "fid"])
        got[f"T2 dF1 {m} (pp)"] = float(t2.loc[m, "delta_macro_f1_pp"])
    rho = float(t2.fid.rank().corr(t2.delta_macro_f1_pp.rank()))
    got["Spearman rho(FID, utility), 8 generators"] = rho
    got["number of generator configurations"] = len(t2)
    got["pooled FID generated images (N)"] = int(t2.n_fake.iloc[0])
    got["pooled FID real images (N)"] = int(t2.n_real.iloc[0])
    got["classifier training runs per condition"] = int(ab.loc["none", "n_runs"])
    ab.drop(columns=[c for c in ("preds", "labels") if c in ab], errors="ignore") \
        .to_csv(tabs / "table_downstream_n100.csv")
    q.to_csv(tabs / "table_quality_n100.csv", index=False)

    radar_cols = [c for c in ("fid_inf", "kid_mean", "cmmd", "precision", "recall",
                              "density", "coverage") if c in pooled and pooled[c].notna().any()]
    hib = {"precision": True, "recall": True, "density": True, "coverage": True,
           "fid_inf": False, "kid_mean": False, "cmmd": False}
    FIG.fig_radar(pooled.dropna(subset=radar_cols), figs, radar_cols, higher_is_better=hib)

    # ------------------------------------------------------------------ #
    print("[2/6] memorisation and detectability")
    memo = json.loads((res / "quality" / "memorisation_n100.json").read_text(encoding="utf-8"))
    alld = [p[2] for v in memo.values() for p in v]
    got["min LPIPS to nearest real training image"] = min(alld)
    got["generated images below LPIPS 0.15"] = sum(d < 0.15 for d in alld)
    best = min(memo, key=lambda k: min(p[2] for p in memo[k]))
    closest = sorted(p[2] for p in memo[best])
    for i in range(2, 9):
        got[f"Fig 5 closest pair {i}"] = closest[i - 1]
    pd.DataFrame([{"method": m, "min_lpips": min(p[2] for p in v),
                   "mean_lpips": float(np.mean([p[2] for p in v])),
                   "n_below_0.15": sum(p[2] < 0.15 for p in v)} for m, v in memo.items()]
                 ).to_csv(tabs / "table_memorisation.csv", index=False)
    det = pd.read_csv(res / "detectability" / "n100.csv")
    dm = det.groupby("method")[["detector_auc", "detector_acc"]].mean()
    dm.to_csv(tabs / "table_detectability.csv")
    got["detector AUC ddpm_fixed"] = float(dm.loc["ddpm_fixed", "detector_auc"])
    got["detector AUC sd_lora_sdedit"] = float(dm.loc["sd_lora_sdedit", "detector_auc"])

    # ------------------------------------------------------------------ #
    print("[3/6] finite-sample FID (Figure 6)")
    curves = json.loads((res / "quality" / "fid_curves_n100.json").read_text(encoding="utf-8"))
    floor = {r["n"]: r["fid_real_real_mean"] for r in next(iter(curves.values()))}
    for n in (100, 200, 500, 1000):
        got[f"real-vs-real FID, N={n}"] = floor[n]
    pd.DataFrame([{"method": m, **r} for m, rs in curves.items() for r in rs]) \
        .to_csv(tabs / "table_fid_sample_size.csv", index=False)
    FIG.fig_fid_bias(curves, figs, reference=None)

    # ------------------------------------------------------------------ #
    print("[4/6] sampler diagnostic")
    sd = pd.read_csv(res / "sampler_diagnostic.csv").set_index("arm")
    sd.to_csv(tabs / "table_sampler.csv")
    got["sampler DDPM 1000 steps FID"] = sd.loc["ddpm_thousand", "fid"]
    got["sampler DDPM 1000 steps output std"] = sd.loc["ddpm_thousand", "output_std"]
    got["sampler DDPM 500 steps FID"] = sd.loc["ddpm_prod", "fid"]
    got["sampler DDPM 500 steps output std"] = sd.loc["ddpm_prod", "output_std"]
    got["sampler DDIM 100 steps FID"] = sd.loc["ddim_hundred", "fid"]
    got["sampler DPM-Solver++ output std"] = sd.loc["dpmsolver", "output_std"]
    got["sampler DPM-Solver++ saturated pixels (%)"] = 100 * sd.loc["dpmsolver", "saturated_frac"]
    got["sampler DPM-Solver++ FID"] = sd.loc["dpmsolver", "fid"]

    # ------------------------------------------------------------------ #
    print("[5/6] cost (Figure 7) and training curves (Figure 3)")
    costs = pd.DataFrame(json.loads((res / "costs.json").read_text(encoding="utf-8")))
    gen = costs[costs.name.str.startswith("gen::")].copy()
    gen["method"] = gen.name.str.split("::").str[1]
    gen["scarcity"] = gen.name.str.split("::").str[2].str.lstrip("n").astype(int)
    per = gen.groupby("method").agg(gpu_minutes=("gpu_minutes", "mean"),
                                    trainable_params_m=("trainable_params_m", "mean"),
                                    peak_vram_gb=("peak_vram_gb", "max"),
                                    runs=("gpu_minutes", "size"))
    per.to_csv(tabs / "table_generator_cost.csv")
    got["GPU-minutes FastGAN"] = per.loc["fastgan", "gpu_minutes"]
    got["GPU-minutes repaired DDPM"] = per.loc["ddpm_fixed", "gpu_minutes"]
    got["GPU-minutes SD LoRA"] = per.loc["sd_lora", "gpu_minutes"]
    got["total GPU-hours"] = gen.gpu_minutes.sum() / 60
    g100 = gen[gen.scarcity == 100]
    merged = pooled.merge(g100[["method", "gpu_minutes", "trainable_params_m"]], on="method")
    FIG.fig_pareto(merged, figs)
    losses = {}
    for m in GENERATORS:
        p = res / "generator_losses" / "n100" / f"{m}.json"
        if p.exists():
            v = json.loads(p.read_text(encoding="utf-8"))
            losses[m] = v["g"] if isinstance(v, dict) else v
    FIG.fig_training_curves({m: v for m, v in losses.items() if v}, figs)

    # ------------------------------------------------------------------ #
    prov = json.loads((res / "provenance.json").read_text(encoding="utf-8"))
    cfg = prov["config"]
    got["repaired DDPM resolution"] = cfg["ddpm_res"]
    got["repaired DDPM training steps"] = cfg["ddpm_steps"]
    got["condition dropout"] = cfg["ddpm_cond_dropout"]
    got["guidance strength"] = cfg["ddpm_guidance"]
    got["production sampling steps"] = cfg["ddpm_sample_steps"]
    got["soybean corpus size (images)"] = prov["dataset"]["n_files"]

    # ------------------------------------------------------------------ #
    print("[6/6] verification against the manuscript")
    ver = []
    for k, pv in PAPER.items():
        rv = got.get(k)
        nd = _decimals(pv)
        match = rv is not None and round(float(rv), nd) == round(float(pv), nd)
        ver.append({"claim": k, "paper": f"{float(pv):.{nd}f}",
                    "reproduced": "" if rv is None else f"{float(rv):.{nd}f}",
                    "match": "yes" if match else "NO"})
    vdf = pd.DataFrame(ver)
    vdf.to_csv(out / "verification.csv", index=False)
    print(vdf.to_string(index=False))
    print(f"\n{(vdf.match == 'yes').sum()}/{len(vdf)} manuscript numbers reproduced from saved results")
    for _, r in vdf[vdf.match != "yes"].iterrows():
        print(f"  not reproduced: {r.claim}: paper {r.paper}, recomputed {r.reproduced}")
    print(f"tables -> {tabs}\nfigures -> {figs}")
    return 0


def _decimals(v) -> int:
    """Decimal places printed in the manuscript (strings keep trailing zeros)."""
    if isinstance(v, int):
        return 0
    s = v if isinstance(v, str) else repr(float(v))
    return len(s.split(".")[1]) if "." in s else 0


if __name__ == "__main__":
    raise SystemExit(main())
