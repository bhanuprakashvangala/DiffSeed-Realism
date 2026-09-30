"""Quality-aware selection of synthetic images before augmentation.

Every arm of this study so far adds *all* generated images to the training set.
That is the condition the recent synthetic-augmentation literature identifies as
the one that fails: a generator fitted to a hundred images produces a long tail
of samples that are off-distribution, ambiguous between classes, or near-copies
of a training image, and adding them unfiltered injects label noise that a
classifier cannot recover from. Reported gains from generative augmentation
concentrate where a selection step is applied.

This module scores each synthetic image on three axes and keeps a subset. The
axes are chosen to fail in different ways, so a sample has to be reasonable on
all of them rather than merely extreme on one:

``confidence``  A classifier trained on the *real* scarce data is asked for the
                probability of the class the image was generated to depict. Low
                probability means the generator did not produce that class,
                whatever it looks like. This is the label-noise filter.

``typicality``  Cosine similarity to the real class centroid in feature space.
                Catches images that are confidently classified but sit far from
                the real distribution, which confidence alone will not.

``novelty``     Distance to the *nearest* real training image. This one is
                two-sided: too far is off-distribution, but too close is a
                near-copy that adds no information while inflating apparent
                dataset size. Both tails are dropped.

The scoring classifier is trained on real data only, so selection never sees the
test split.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class FilterReport:
    method: str
    scarcity: int
    cls: str
    n_candidates: int
    n_kept: int
    dropped_low_confidence: int
    dropped_atypical: int
    dropped_near_duplicate: int
    mean_confidence_kept: float
    mean_confidence_dropped: float

    def as_dict(self):
        from dataclasses import asdict

        return asdict(self)


# --------------------------------------------------------------------------- #
def _feature_extractor(cfg):
    """Penultimate features from an ImageNet backbone, for typicality/novelty."""
    import timm

    m = timm.create_model("resnet50.a1_in1k", pretrained=True, num_classes=0)
    return m.eval().to(cfg.device)


@torch.no_grad()
def _embed(model, paths: Sequence[Path], cfg, batch: int = 32) -> np.ndarray:
    from PIL import Image
    from torchvision import transforms as T

    from .data import IMAGENET_MEAN, IMAGENET_STD

    tf = T.Compose([
        T.Resize((cfg.clf_res, cfg.clf_res)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    out = []
    for i in range(0, len(paths), batch):
        xs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i : i + batch]])
        out.append(model(xs.to(cfg.device)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 2048), dtype=np.float32)


@torch.no_grad()
def _class_probs(clf, paths: Sequence[Path], cfg, batch: int = 32) -> np.ndarray:
    from PIL import Image
    from torchvision import transforms as T

    from .data import IMAGENET_MEAN, IMAGENET_STD

    tf = T.Compose([
        T.Resize((cfg.clf_res, cfg.clf_res)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    out = []
    for i in range(0, len(paths), batch):
        xs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i : i + batch]])
        out.append(torch.softmax(clf(xs.to(cfg.device)).float(), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


# --------------------------------------------------------------------------- #
def select_synthetic(
    synth_dir: Path,
    real_paths_by_class: dict[int, list[str]],
    class_names: Sequence[str],
    scoring_clf,
    cfg,
    keep_frac: float = 0.6,
    min_confidence: float = 0.5,
    novelty_low_pct: float = 2.0,
    novelty_high_pct: float = 95.0,
    use_confidence: bool = True,
    use_typicality: bool = True,
    use_novelty: bool = True,
    skip_classes: Sequence[int] = (),
    method: str = "",
    scarcity: int = 0,
) -> tuple[dict[int, list[str]], list[FilterReport]]:
    """Score and select synthetic images per class.

    Returns the kept paths per class and a per-class report. Selection is a
    *fraction* of candidates rather than a fixed count so that a generator
    producing uniformly poor samples contributes proportionally fewer, rather
    than being topped up with its own worst output to hit a quota.
    """
    from .metrics import list_images

    feat = _feature_extractor(cfg)
    kept: dict[int, list[str]] = {}
    reports: list[FilterReport] = []

    for ci, cname in enumerate(class_names):
        if ci in skip_classes:
            continue
        cand = list_images(Path(synth_dir) / cname)
        real = [Path(p) for p in real_paths_by_class.get(ci, [])]
        if not cand or not real:
            continue

        probs = _class_probs(scoring_clf, cand, cfg)
        conf = probs[:, ci] if probs.shape[1] > ci else np.zeros(len(cand))

        fs = _embed(feat, cand, cfg)
        fr = _embed(feat, real, cfg)
        fs_n = fs / (np.linalg.norm(fs, axis=1, keepdims=True) + 1e-9)
        fr_n = fr / (np.linalg.norm(fr, axis=1, keepdims=True) + 1e-9)
        centroid = fr_n.mean(0, keepdims=True)
        centroid /= np.linalg.norm(centroid) + 1e-9
        typicality = (fs_n @ centroid.T).ravel()

        sim_to_real = fs_n @ fr_n.T
        nearest = sim_to_real.max(1)  # 1.0 = identical direction

        lo_nov = np.percentile(nearest, novelty_low_pct)
        hi_nov = np.percentile(nearest, novelty_high_pct)
        typ_floor = np.percentile(typicality, 10)

        # Each criterion can be switched off so that its individual contribution
        # is measurable. Selection is the study's only positive result and a
        # single combined number does not say which of the three does the work,
        # nor whether any of them is doing harm.
        drop_conf = (conf < min_confidence) if use_confidence else np.zeros(len(cand), bool)
        drop_typ = ((~drop_conf) & (typicality < typ_floor)) if use_typicality             else np.zeros(len(cand), bool)
        drop_dup = ((~drop_conf) & (~drop_typ) & ((nearest > hi_nov) | (nearest < lo_nov)))             if use_novelty else np.zeros(len(cand), bool)

        eligible = ~(drop_conf | drop_typ | drop_dup)
        # rank the survivors by confidence, break ties by typicality
        # Ranking follows the criteria in use: with confidence disabled the
        # ranking must not silently keep using it, or "no confidence" would
        # still be a confidence-selected set.
        score = (conf if use_confidence else np.zeros(len(cand)))
        if use_typicality:
            score = score + 0.25 * typicality
        order = np.argsort(-score)
        order = [i for i in order if eligible[i]]
        n_keep = max(1, int(round(keep_frac * len(cand))))
        chosen = order[:n_keep]

        kept[ci] = [str(cand[i]) for i in chosen]
        reports.append(FilterReport(
            method=method, scarcity=scarcity, cls=cname,
            n_candidates=len(cand), n_kept=len(chosen),
            dropped_low_confidence=int(drop_conf.sum()),
            dropped_atypical=int(drop_typ.sum()),
            dropped_near_duplicate=int(drop_dup.sum()),
            mean_confidence_kept=float(conf[chosen].mean()) if chosen else float("nan"),
            mean_confidence_dropped=float(conf[~eligible].mean()) if (~eligible).any() else float("nan"),
        ))

    del feat
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return kept, reports


def write_manifest(path: Path, kept: dict[int, list[str]], class_names: Sequence[str]) -> Path:
    """Persist the selection so the downstream stage can consume it verbatim."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {class_names[ci]: paths for ci, paths in sorted(kept.items())}, indent=2
    ), encoding="utf-8")
    return path


def load_manifest(path: Path, class_names: Sequence[str]) -> dict[int, list[str]]:
    path = Path(path)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    idx = {n: i for i, n in enumerate(class_names)}
    return {idx[k]: v for k, v in raw.items() if k in idx}
