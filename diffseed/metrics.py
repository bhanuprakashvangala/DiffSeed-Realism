"""Generative-quality metrics.

v1 reported FID on 200-vs-200 images and a "LPIPS" number computed by pairing
real image *i* with synthetic image *i* and averaging the distance. Neither is
interpretable:

* FID is strongly biased upward at small N. Comparing a 200-sample FID of 78.8
  against a literature number computed on thousands of samples is not a
  comparison of image quality, it is a comparison of sample sizes. ``fid_vs_n``
  and ``fid_infinity`` quantify that bias directly.
* Arbitrarily paired LPIPS measures nothing -- two unrelated *real* images score
  about the same. ``intra_lpips`` (diversity) and ``knn_lpips`` (fidelity to the
  nearest real neighbour) are the meaningful decompositions.

Added here and absent from v1: KID (unbiased for small N), CMMD (CLIP-MMD,
Jayasumana et al. CVPR 2024, designed to be reliable at small N), improved
precision/recall (Kynkaanniemi et al.), density/coverage (Naeem et al.), and a
nearest-neighbour memorisation audit.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def list_images(d: Path, limit: int | None = None) -> list[Path]:
    files = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)
    return files[:limit] if limit else files


# --------------------------------------------------------------------------- #
# feature extractors
# --------------------------------------------------------------------------- #
class InceptionFeatures:
    """Pool3 features (2048-d) from the clean-fid Inception, with a fallback.

    Using clean-fid's own network keeps FID here numerically comparable to the
    values other papers report with ``clean-fid``.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.mode = "clean"
        try:
            from cleanfid.features import build_feature_extractor

            self.model = build_feature_extractor("clean", device=torch.device(device))
            self.resize = 299
            self.backend = "cleanfid"
        except Exception:  # pragma: no cover - fallback path
            import torchvision

            net = torchvision.models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
            net.fc = torch.nn.Identity()
            self.model = net.eval().to(device)
            self.resize = 299
            self.backend = "torchvision"

    @torch.no_grad()
    def __call__(self, paths: Sequence[Path], batch_size: int = 32) -> np.ndarray:
        feats = []
        for i in range(0, len(paths), batch_size):
            batch = []
            for p in paths[i : i + batch_size]:
                img = Image.open(p).convert("RGB").resize(
                    (self.resize, self.resize), Image.BICUBIC
                )
                batch.append(torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float())
            x = torch.stack(batch).to(self.device)
            if self.backend == "cleanfid":
                # clean-fid's extractor expects uint8-valued float in [0, 255]
                out = self.model(x)
            else:
                out = self.model((x / 255.0 - 0.5) / 0.5)
            feats.append(out.float().cpu().numpy())
        return np.concatenate(feats, 0)


class ClipFeatures:
    """CLIP ViT-L/14 image embeddings, used for CMMD and CLIP-FID."""

    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.proc = CLIPImageProcessor.from_pretrained(model_name)
        self.model = (
            CLIPVisionModelWithProjection.from_pretrained(model_name).eval().to(device)
        )
        self.device = device

    @torch.no_grad()
    def __call__(self, paths: Sequence[Path], batch_size: int = 32) -> np.ndarray:
        feats = []
        for i in range(0, len(paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
            px = self.proc(images=imgs, return_tensors="pt").pixel_values.to(self.device)
            emb = self.model(pixel_values=px).image_embeds
            feats.append(emb.float().cpu().numpy())
        return np.concatenate(feats, 0)


# --------------------------------------------------------------------------- #
# distribution distances
# --------------------------------------------------------------------------- #
def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(f1: np.ndarray, f2: np.ndarray) -> float:
    """FID between two feature sets (symmetric eigendecomposition, no scipy)."""
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1 = np.cov(f1, rowvar=False)
    s2 = np.cov(f2, rowvar=False)
    diff = mu1 - mu2
    # tr(sqrt(s1 s2)) computed stably as tr(sqrt(sqrt(s1) s2 sqrt(s1)))
    a = _sqrtm_psd(s1)
    covmean = _sqrtm_psd(a @ s2 @ a)
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kernel_distance(
    f1: np.ndarray, f2: np.ndarray, n_subsets: int = 100, subset_size: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """KID: unbiased MMD^2 with a degree-3 polynomial kernel.

    Unlike FID, KID has no small-sample bias, so it is the honest metric when
    only a few hundred images per class exist.
    """
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(f1), len(f2))
    d = f1.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = f1[rng.choice(len(f1), m, replace=False)]
        y = f2[rng.choice(len(f2), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        np.fill_diagonal(kxx, 0)
        np.fill_diagonal(kyy, 0)
        vals.append(
            kxx.sum() / (m * (m - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean()
        )
    v = np.asarray(vals)
    return float(v.mean()), float(v.std())


def cmmd(f1: np.ndarray, f2: np.ndarray, sigma: float = 10.0, scale: float = 1000.0) -> float:
    """CLIP-MMD (Jayasumana et al., CVPR 2024) on L2-normalised CLIP embeddings.

    Reliable at sample sizes where FID is meaningless, which is exactly the
    regime this study operates in.
    """
    x = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-12)
    y = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-12)
    gamma = 1.0 / (2 * sigma**2)

    def _k(a, b):
        d2 = (
            (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T
        )
        return np.exp(-gamma * np.clip(d2, 0, None))

    return float(scale * (_k(x, x).mean() + _k(y, y).mean() - 2 * _k(x, y).mean()))


# --------------------------------------------------------------------------- #
# fidelity / diversity decomposition
# --------------------------------------------------------------------------- #
def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_t = torch.from_numpy(a).float()
    b_t = torch.from_numpy(b).float()
    return torch.cdist(a_t, b_t).numpy()


def prdc(real: np.ndarray, fake: np.ndarray, k: int = 5) -> dict[str, float]:
    """Precision, recall (Kynkaanniemi 2019) and density, coverage (Naeem 2020).

    precision -> fraction of synthetic samples inside the real manifold (fidelity)
    recall    -> fraction of real samples inside the synthetic manifold (coverage
                 of the real distribution)
    density   -> precision, but robust to real-manifold outliers
    coverage  -> fraction of real samples with a synthetic neighbour nearby
    """
    d_rr = _pairwise(real, real)
    d_ff = _pairwise(fake, fake)
    d_rf = _pairwise(real, fake)

    def radii(d):
        return np.sort(d, axis=1)[:, k]  # k-th NN, column 0 is the point itself

    r_real = radii(d_rr)
    r_fake = radii(d_ff)

    precision = float((d_rf < r_real[:, None]).any(0).mean())
    recall = float((d_rf < r_fake[None, :]).any(1).mean())
    density = float((1.0 / k) * (d_rf < r_real[:, None]).sum(0).mean())
    coverage = float((d_rf.min(1) < r_real).mean())
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


# --------------------------------------------------------------------------- #
# the small-N bias curve -- the direct answer to "your FID is 78, theirs is 36"
# --------------------------------------------------------------------------- #
def fid_vs_n(
    real_feats: np.ndarray,
    fake_feats: np.ndarray,
    ns: Sequence[int],
    repeats: int = 5,
    seed: int = 0,
) -> list[dict]:
    """FID as a function of sample size, plus a real-vs-real control.

    The control is the key number: FID between two disjoint halves of the *real*
    data at the same N. Any FID below that floor is unmeasurable, and the floor
    at N=200 is large.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        if n > min(len(real_feats), len(fake_feats)):
            continue
        gen_vals, ctrl_vals = [], []
        for _ in range(repeats):
            ri = rng.choice(len(real_feats), n, replace=False)
            fi = rng.choice(len(fake_feats), n, replace=False)
            gen_vals.append(frechet_distance(real_feats[ri], fake_feats[fi]))
            if 2 * n <= len(real_feats):
                perm = rng.permutation(len(real_feats))
                ctrl_vals.append(
                    frechet_distance(real_feats[perm[:n]], real_feats[perm[n : 2 * n]])
                )
        rows.append(
            {
                "n": n,
                "fid_gen_mean": float(np.mean(gen_vals)),
                "fid_gen_std": float(np.std(gen_vals)),
                "fid_real_real_mean": float(np.mean(ctrl_vals)) if ctrl_vals else float("nan"),
                "fid_real_real_std": float(np.std(ctrl_vals)) if ctrl_vals else float("nan"),
            }
        )
    return rows


def fid_infinity(
    real_feats: np.ndarray, fake_feats: np.ndarray, points: int = 15, repeats: int = 3, seed: int = 0
) -> float:
    """FID extrapolated to infinite samples (Chong & Forsyth, CVPR 2020).

    FID(N) is approximately linear in 1/N; the intercept is the bias-free value.
    """
    rng = np.random.default_rng(seed)
    n_max = min(len(real_feats), len(fake_feats))
    if n_max < 50:
        return float("nan")  # the extrapolation is meaningless below this
    ns = np.unique(np.linspace(max(20, n_max // 10), n_max, points).astype(int))
    xs, ys = [], []
    for n in ns:
        for _ in range(repeats):
            ri = rng.choice(len(real_feats), n, replace=False)
            fi = rng.choice(len(fake_feats), n, replace=False)
            xs.append(1.0 / n)
            ys.append(frechet_distance(real_feats[ri], fake_feats[fi]))
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(intercept)


# --------------------------------------------------------------------------- #
# perceptual metrics done correctly
# --------------------------------------------------------------------------- #
class Lpips:
    def __init__(self, device="cuda", net="alex"):
        import lpips as _lpips

        self.fn = _lpips.LPIPS(net=net).to(device).eval()
        self.device = device

    def _load(self, p: Path, res: int) -> torch.Tensor:
        img = Image.open(p).convert("RGB").resize((res, res), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
        return (x * 2 - 1).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def intra_lpips(self, paths: Sequence[Path], n_pairs: int = 500, res: int = 256, seed: int = 0) -> float:
        """Mean LPIPS between random pairs *within* a set = diversity.

        Mode collapse shows up here as a value far below the real set's."""
        rng = np.random.default_rng(seed)
        if len(paths) < 2:
            return float("nan")
        vals = []
        for _ in range(n_pairs):
            i, j = rng.choice(len(paths), 2, replace=False)
            vals.append(self.fn(self._load(paths[i], res), self._load(paths[j], res)).item())
        return float(np.mean(vals))

    @torch.no_grad()
    def knn_lpips(
        self, fake: Sequence[Path], real: Sequence[Path], res: int = 256, max_fake: int = 200
    ) -> tuple[float, list[float]]:
        """LPIPS from each synthetic image to its closest real image = fidelity.

        Also the memorisation audit: values near zero mean the generator copied
        a training image rather than sampling a new one.
        """
        real_t = torch.cat([self._load(p, res) for p in real])
        dists, argmins = [], []
        chunk = 32  # broadcast each synthetic image against a block of real ones
        for p in fake[:max_fake]:
            x = self._load(p, res)
            best, best_i = float("inf"), -1
            for s in range(0, len(real_t), chunk):
                block = real_t[s : s + chunk]
                d = self.fn(x.expand(block.size(0), -1, -1, -1), block).flatten()
                v, i = torch.min(d, 0)
                if v.item() < best:
                    best, best_i = v.item(), s + int(i)
            dists.append(best)
            argmins.append(best_i)
        self.last_argmins = argmins
        return float(np.mean(dists)), dists


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #
@dataclass
class QualityReport:
    method: str
    cls: str
    n_real: int
    n_fake: int
    fid: float
    fid_inf: float
    kid_mean: float
    kid_std: float
    cmmd: float
    clip_fid: float
    precision: float
    recall: float
    density: float
    coverage: float
    intra_lpips_fake: float
    intra_lpips_real: float
    knn_lpips: float

    def as_dict(self):
        return asdict(self)


def evaluate_generator(
    real_dir: Path,
    fake_dir: Path,
    method: str,
    cls: str,
    inception: InceptionFeatures,
    clip: ClipFeatures | None = None,
    lpips_fn: "Lpips | None" = None,
    n_max: int = 1000,
    prdc_k: int = 5,
) -> QualityReport:
    real = list_images(real_dir, n_max)
    fake = list_images(fake_dir, n_max)
    fr = inception(real)
    ff = inception(fake)

    kid_m, kid_s = kernel_distance(fr, ff, subset_size=min(100, len(fr), len(ff)))
    pr = prdc(fr, ff, k=prdc_k)

    if clip is not None:
        cr, cf = clip(real), clip(fake)
        cmmd_v = cmmd(cr, cf)
        clip_fid_v = frechet_distance(cr, cf)
    else:
        cmmd_v = clip_fid_v = float("nan")

    if lpips_fn is not None:
        il_f = lpips_fn.intra_lpips(fake, n_pairs=200)
        il_r = lpips_fn.intra_lpips(real, n_pairs=200)
        knn, _ = lpips_fn.knn_lpips(fake, real[:200], max_fake=100)
    else:
        il_f = il_r = knn = float("nan")

    return QualityReport(
        method=method,
        cls=cls,
        n_real=len(real),
        n_fake=len(fake),
        fid=frechet_distance(fr, ff),
        fid_inf=fid_infinity(fr, ff),
        kid_mean=kid_m,
        kid_std=kid_s,
        cmmd=cmmd_v,
        clip_fid=clip_fid_v,
        precision=pr["precision"],
        recall=pr["recall"],
        density=pr["density"],
        coverage=pr["coverage"],
        intra_lpips_fake=il_f,
        intra_lpips_real=il_r,
        knn_lpips=knn,
    )
