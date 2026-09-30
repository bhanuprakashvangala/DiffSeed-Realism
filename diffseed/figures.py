"""Publication figures.

The headline figures are the three that answer the reviewer directly:

``fig_fid_bias``     FID against sample size, with the real-vs-real floor and
                     the literature reference line drawn in. Shows how much of a
                     78.8-vs-36.7 gap is sample size rather than image quality.
``fig_threshold``    Downstream delta against real images per class, one line
                     per generator, with the zero line and the fitted crossover.
                     This is the threshold result.
``fig_pareto``       Quality against GPU-minutes. Shows the transfer-learning
                     run is not simply buying its win with compute.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# house style
# --------------------------------------------------------------------------- #
from .printsafe import (
    HATCHES,
    LINESTYLES,
    MARKERS,
    PALETTE,
    apply_print_rcparams,
    hatch_bars,
    hatch_grouped,
    save_grayscale_proof,
    tidy_log_axis,
)
from .printsafe import LINE_PALETTE

# Stable ordering so a method keeps the same colour AND the same hatch/dash
# across every figure. The redundant encodings are what make these readable in
# the black-and-white printed manual; colour alone is not a reliable channel.
METHOD_ORDER = (
    "real", "none", "trad", "dcgan", "fastgan", "ddpm_v1", "ddpm_fixed",
    "ddpm_ft", "sd_lora", "sd_lora_sdedit", "sd_turbo_lora", "oracle",
)
_IDX = {m: i for i, m in enumerate(METHOD_ORDER)}


def _slot(m: str) -> int:
    return _IDX.get(m, abs(hash(m)) % len(PALETTE))


METHOD_COLORS = {m: PALETTE[_slot(m) % len(PALETTE)] for m in METHOD_ORDER}


def mhatch(m: str) -> str:
    return HATCHES[_slot(m) % len(HATCHES)]


def mline(m: str):
    return LINESTYLES[_slot(m) % len(LINESTYLES)]


def mmark(m: str) -> str:
    return MARKERS[_slot(m) % len(MARKERS)]


def mlinec(m: str) -> str:
    """Line colour: darker subset, since thin strokes on white wash out."""
    return LINE_PALETTE[_slot(m) % len(LINE_PALETTE)]
METHOD_LABELS = {
    "none": "Scarce only",
    "trad": "Traditional aug.",
    "dcgan": "DCGAN",
    "fastgan": "FastGAN + DiffAug",
    "ddpm_v1": "DDPM scratch (v1)",
    "ddpm_fixed": "DDPM scratch (fixed)",
    "ddpm_ft": "DDPM fine-tuned",
    "sd_lora": "SD1.5 + LoRA",
    "sd_lora_sdedit": "SD1.5 + LoRA (SDEdit)",
    "sd_turbo_lora": "SD-Turbo + LoRA",
    "oracle": "Full data (oracle)",
}


def use_paper_style():
    apply_print_rcparams(mpl)
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.8,
        }
    )


def _c(m):
    return METHOD_COLORS.get(m, PALETTE[_slot(m) % len(PALETTE)])


def _l(m):
    return METHOD_LABELS.get(m, m)


def _save(fig, out: Path, name: str, gray_proof: bool = True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    png = out / f"{name}.png"
    if gray_proof:
        # desaturated copy next to the figure: this is what the printed manual
        # shows, and the only way to confirm the hatch/dash encodings worked
        proof = out / "grayscale_proofs"
        proof.mkdir(exist_ok=True)
        g = save_grayscale_proof(png)
        if g is not None:
            g.replace(proof / g.name)
    return png


# --------------------------------------------------------------------------- #
# 1. the FID sample-size bias figure
# --------------------------------------------------------------------------- #
def fig_fid_bias(curves: dict[str, list[dict]], out: Path, reference: dict | None = None,
                 name="fig_fid_sample_size_bias"):
    """``curves``: method -> rows from ``metrics.fid_vs_n``."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    first = next(iter(curves.values()))
    ns = [r["n"] for r in first]
    floor = [r["fid_real_real_mean"] for r in first]
    floor_sd = [r["fid_real_real_std"] for r in first]

    ax.plot(ns, floor, color="#1a1a1a", ls=(0, (6, 2)), marker="o", ms=5,
            markerfacecolor="white", markeredgewidth=1.1,
            label="Real vs real (measurement floor)")
    ax.errorbar(ns, floor, yerr=floor_sd, fmt="none", ecolor="#1a1a1a",
                elinewidth=1.0, capsize=2.6, capthick=1.0, zorder=2)

    for m, rows in curves.items():
        x = [r["n"] for r in rows]
        y = [r["fid_gen_mean"] for r in rows]
        s = [r["fid_gen_std"] for r in rows]
        ax.errorbar(x, y, yerr=s, fmt="none", ecolor=mlinec(m), elinewidth=1.0,
                    capsize=2.6, capthick=1.0, alpha=0.9, zorder=2)
        ax.plot(x, y, marker=mmark(m), ms=5.5, color=mlinec(m), linestyle=mline(m),
                markeredgecolor="#1a1a1a", markeredgewidth=0.5, zorder=3, label=_l(m))

    if reference:
        ax.axhline(reference["value"], color="#c62828", ls=":", lw=1.6)
        ax.annotate(
            f"{reference['label']}\nFID = {reference['value']:.2f}",
            xy=(max(ns), reference["value"]), xytext=(-8, 8),
            textcoords="offset points", ha="right", fontsize=8.5, color="#c62828",
        )

    ax.set_xscale("log")
    tidy_log_axis(ax, ns)
    ax.set_xlabel("Samples per set (N)")
    ax.set_ylabel("FID")
    ax.set_title("FID is a function of sample size")
    ax.legend(loc="upper right", fontsize=8)

    # right panel: the same data as excess over the floor
    for m, rows in curves.items():
        x = [r["n"] for r in rows]
        y = [r["fid_gen_mean"] - r["fid_real_real_mean"] for r in rows]
        ax2.plot(x, y, marker=mmark(m), ms=5, color=mlinec(m), linestyle=mline(m),
                 markeredgecolor="#1a1a1a", markeredgewidth=0.5, label=_l(m))
    ax2.axhline(0, color="#1b1b1b", ls="--", lw=1.2)
    ax2.set_xscale("log")
    tidy_log_axis(ax2, ns)
    ax2.set_xlabel("Samples per set (N)")
    ax2.set_ylabel("FID above the real-vs-real floor")
    ax2.set_title("Excess FID: the part that is actually image quality")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------- #
# 2. the threshold figure
# --------------------------------------------------------------------------- #
def fig_threshold(df, out: Path, metric="macro_f1", name="fig_threshold_grid",
                  crossovers: dict | None = None):
    """``df`` columns: method, scarcity, delta, delta_ci_low, delta_ci_high."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Confidence intervals as capped error bars, not filled bands. Overlapping
    # translucent bands from four methods stack into an undifferentiated grey in
    # black-and-white print; capped bars stay readable and a small horizontal
    # jitter keeps coincident intervals from hiding each other.
    methods = list(df.groupby("method").groups)
    for k, (m, g) in enumerate(df.groupby("method")):
        g = g.sort_values("scarcity")
        jitter = 1.0 + 0.018 * (k - (len(methods) - 1) / 2)
        xs = g["scarcity"] * jitter
        if "delta_ci_low" in g:
            ax.errorbar(xs, g["delta"],
                        yerr=[g["delta"] - g["delta_ci_low"],
                              g["delta_ci_high"] - g["delta"]],
                        fmt="none", ecolor=mlinec(m), elinewidth=1.1, capsize=3.0,
                        capthick=1.1, alpha=0.9, zorder=2)
        ax.plot(xs, g["delta"], marker=mmark(m), ms=6.5, color=mlinec(m),
                linestyle=mline(m), markeredgecolor="#1a1a1a", markeredgewidth=0.6,
                zorder=3, label=_l(m))

    ax.axhline(0, color="#1b1b1b", lw=1.3)
    ax.text(df["scarcity"].min(), 0, "  no benefit", va="bottom", fontsize=8, color="#1b1b1b")

    if crossovers:
        for m, est in crossovers.items():
            if np.isfinite(est.crossover_n) and est.crossover_n <= df["scarcity"].max() * 3:
                ax.axvline(est.crossover_n, color=_c(m), ls=":", lw=1.2, alpha=0.8)
                ax.annotate(f"n*≈{est.crossover_n:.0f}", xy=(est.crossover_n, ax.get_ylim()[1]),
                            xytext=(2, -10), textcoords="offset points",
                            fontsize=8, color=_c(m), rotation=90, va="top")

    ax.set_xscale("log")
    tidy_log_axis(ax, sorted(df["scarcity"].unique()))
    ax.set_xlabel("Real images per minority class")
    ax.set_ylabel(f"Δ {metric.replace('_', ' ')} vs scarce baseline")
    ax.set_title("When does synthetic augmentation start to pay?")
    ax.legend(loc="best", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------- #
# 3. quality vs compute
# --------------------------------------------------------------------------- #
def fig_pareto(df, out: Path, x="gpu_minutes", y="fid", size="trainable_params_m",
               name="fig_quality_cost_pareto", ylabel="FID (N matched)", invert_y=True):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    smax = max(df[size].max(), 1e-6)
    for _, r in df.iterrows():
        ax.scatter(r[x], r[y], s=60 + 620 * (r[size] / smax), color=_c(r["method"]),
                   marker=mmark(r["method"]), alpha=0.85, edgecolor="#1a1a1a",
                   linewidth=0.9, zorder=3)
        ax.annotate(_l(r["method"]), (r[x], r[y]), xytext=(7, 5),
                    textcoords="offset points", fontsize=8.5, color=_c(r["method"]))

    # Pareto frontier
    d = df.sort_values(x)
    best, front = np.inf, []
    for _, r in d.iterrows():
        if r[y] < best:
            best = r[y]
            front.append((r[x], r[y]))
    if len(front) > 1:
        ax.plot(*zip(*front), color="#9e9e9e", ls="--", lw=1.2, zorder=1)

    ax.set_xscale("log")
    ax.set_xlabel("Generator training cost (GPU-minutes, log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title("Quality per unit of compute\n(bubble area ∝ trainable parameters)")
    if invert_y:
        ax.invert_yaxis()
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------- #
# 4. metric radar
# --------------------------------------------------------------------------- #
def fig_radar(df, out: Path, metrics: Sequence[str], name="fig_metric_radar",
              higher_is_better: dict | None = None):
    higher_is_better = higher_is_better or {}
    labels = [m.replace("_", " ") for m in metrics]
    n = len(metrics)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang += ang[:1]

    norm = df.copy()
    for m in metrics:
        v = df[m].astype(float)
        rng = v.max() - v.min()
        z = (v - v.min()) / rng if rng > 1e-12 else np.ones_like(v) * 0.5
        norm[m] = z if higher_is_better.get(m, True) else 1 - z

    fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"polar": True})
    for _, r in norm.iterrows():
        vals = [float(r[m]) for m in metrics]
        vals += vals[:1]
        ax.plot(ang, vals, color=mlinec(r["method"]), label=_l(r["method"]), lw=1.9,
                linestyle=mline(r["method"]), marker=mmark(r["method"]), ms=4)
        ax.fill(ang, vals, color=_c(r["method"]), alpha=0.08)

    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.set_title("Generator profile (normalised, outer = better)", pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------- #
# 5. sample grids
# --------------------------------------------------------------------------- #
def fig_sample_grid(sources: dict[str, dict[str, list]], class_names: Sequence[str],
                    out: Path, n_per_class: int = 4, name="fig_sample_grid"):
    """``sources``: method -> class_name -> list of image paths. Rows are methods."""
    from PIL import Image

    methods = list(sources)
    ncols = len(class_names) * n_per_class
    fig, axes = plt.subplots(len(methods), ncols,
                             figsize=(0.92 * ncols, 1.05 * len(methods)))
    axes = np.atleast_2d(axes)

    for r, m in enumerate(methods):
        for ci, cname in enumerate(class_names):
            paths = sources[m].get(cname, [])[:n_per_class]
            for k in range(n_per_class):
                ax = axes[r, ci * n_per_class + k]
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if k < len(paths):
                    ax.imshow(Image.open(paths[k]).convert("RGB"))
                if r == 0 and k == 0:
                    ax.set_title(cname.replace(" soybeans", ""), fontsize=8,
                                 loc="left", pad=6)
            if ci == 0:
                axes[r, 0].set_ylabel(_l(m), rotation=0, ha="right", va="center",
                                      fontsize=8.5, labelpad=8, color=_c(m))
    fig.suptitle("Real and synthetic seeds by generator", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------- #
# 6. memorisation audit
# --------------------------------------------------------------------------- #
def fig_memorisation(pairs, out: Path, name="fig_memorisation_audit", threshold=0.15):
    """``pairs``: list of (synth_path, nearest_real_path, lpips_distance)."""
    from PIL import Image

    pairs = sorted(pairs, key=lambda p: p[2])[:8]
    fig, axes = plt.subplots(2, len(pairs), figsize=(1.25 * len(pairs), 3.0))
    axes = np.atleast_2d(axes)
    for i, (s, r, d) in enumerate(pairs):
        axes[0, i].imshow(Image.open(s).convert("RGB"))
        axes[1, i].imshow(Image.open(r).convert("RGB"))
        axes[0, i].set_title(f"{d:.3f}", fontsize=8,
                             color="#c62828" if d < threshold else "#2e7d32")
        for row in (0, 1):
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])
    axes[0, 0].set_ylabel("synthetic", fontsize=8.5, rotation=0, ha="right", va="center")
    axes[1, 0].set_ylabel("nearest real", fontsize=8.5, rotation=0, ha="right", va="center")
    fig.suptitle(
        f"Closest synthetic-to-real matches (LPIPS; red < {threshold} = possible copy)",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------- #
# 7. model x method heatmap
# --------------------------------------------------------------------------- #
def fig_model_method_heatmap(df, out: Path, value="macro_f1", name="fig_model_method_heatmap"):
    piv = df.pivot_table(index="model", columns="method", values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(1.15 * len(piv.columns) + 2.4, 0.62 * len(piv) + 2.0))
    # RdYlGn is a diverging map whose ends collapse to the same grey in print;
    # cividis is monotonic in luminance and colour-vision-safe
    im = ax.imshow(piv.values, cmap="cividis", aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([_l(c) for c in piv.columns], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=8.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                lo, hi = np.nanmin(piv.values), np.nanmax(piv.values)
                frac = (v - lo) / (hi - lo) if hi > lo else 0.5
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if frac < 0.55 else "#1a1a1a")
    ax.set_title(f"{value.replace('_', ' ')} across classifier architectures")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label=value.replace("_", " "))
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------- #
# 8. paired-delta forest plot
# --------------------------------------------------------------------------- #
def fig_forest(df, out: Path, name="fig_delta_forest", metric="macro_f1"):
    """``df``: condition, delta, delta_ci_low, delta_ci_high, significant."""
    d = df.dropna(subset=["delta"]).sort_values("delta")
    fig, ax = plt.subplots(figsize=(6.6, 0.46 * len(d) + 1.9))
    y = np.arange(len(d))
    for i, (_, r) in enumerate(d.iterrows()):
        col = "#2e7d32" if r["delta"] > 0 else "#c62828"
        if not r.get("significant", False):
            col = "#9e9e9e"
        ls = "-" if r.get("significant", False) else (0, (3, 2))
        ax.plot([r["delta_ci_low"], r["delta_ci_high"]], [i, i], color=col, lw=2.4,
                linestyle=ls)
        ax.plot(r["delta"], i, marker=mmark(r["condition"]), color=col, ms=7,
                markeredgecolor="#1a1a1a", markeredgewidth=0.6,
                markerfacecolor=col if r.get("significant", False) else "white")
    ax.axvline(0, color="#1b1b1b", lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([_l(c) if c in METHOD_LABELS else c for c in d["condition"]], fontsize=8.5)
    ax.set_xlabel(f"Δ {metric.replace('_', ' ')} vs scarce baseline (95% paired bootstrap CI)")
    ax.set_title("Grey = interval crosses zero (no detectable effect)")
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------- #
# 9. feature-space embedding
# --------------------------------------------------------------------------- #
def fig_embedding(emb, labels, sources, class_names, out: Path, name="fig_feature_space"):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    cmap = plt.get_cmap("tab10")

    for ci, cname in enumerate(class_names):
        m = np.asarray(labels) == ci
        axes[0].scatter(emb[m, 0], emb[m, 1], s=9, alpha=0.62, color=cmap(ci),
                        label=cname.replace(" soybeans", ""))
    axes[0].legend(fontsize=8, markerscale=1.6)
    axes[0].set_title("By class")

    src = np.asarray(sources)
    for s in dict.fromkeys(src):
        m = src == s
        axes[1].scatter(emb[m, 0], emb[m, 1], s=16, alpha=0.7, color=_c(s),
                        marker=mmark(s), linewidths=0.3, edgecolors="#1a1a1a",
                        label=_l(s))
    axes[1].legend(fontsize=8, markerscale=1.6)
    axes[1].set_title("By source")

    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
        a.set_xlabel("dim 1")
        a.set_ylabel("dim 2")
    fig.suptitle("Inception feature space: real vs synthetic", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, name)


# --------------------------------------------------------------------------- #
# 10. training curves + denoising trajectory
# --------------------------------------------------------------------------- #
def fig_training_curves(curves: dict[str, Sequence[float]], out: Path,
                        name="fig_training_curves", xlabel="Epoch", ylabel="Loss"):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for m, v in curves.items():
        ax.plot(np.arange(1, len(v) + 1), v, color=mlinec(m), label=_l(m),
                linestyle=mline(m))
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Generator training")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return _save(fig, out, name)


def fig_trajectory(traj_by_class: dict[str, list], out: Path, name="fig_denoising"):
    rows = len(traj_by_class)
    cols = max(len(v) for v in traj_by_class.values())
    fig, axes = plt.subplots(rows, cols, figsize=(1.05 * cols, 1.12 * rows))
    axes = np.atleast_2d(axes)
    for r, (cname, snaps) in enumerate(traj_by_class.items()):
        for c in range(cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(snaps):
                ax.imshow(snaps[c].squeeze(0).permute(1, 2, 0).numpy())
            if c == 0:
                ax.set_ylabel(cname.replace(" soybeans", ""), rotation=0, ha="right",
                              va="center", fontsize=8)
        axes[r, 0].set_title("noise" if r == 0 else "", fontsize=8)
        axes[r, -1].set_title("sample" if r == 0 else "", fontsize=8)
    fig.suptitle("Reverse diffusion trajectory", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, name)
