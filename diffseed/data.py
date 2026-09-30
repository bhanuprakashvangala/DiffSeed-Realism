"""Dataset discovery, the scarcity protocol, and dataloaders.

Two things here differ from v1 and both matter for the conclusions:

1. v1 built its scarce set from a single ``np.random.seed(SEED)`` draw and never
   held out a validation split, so every classifier reported its *final-epoch*
   test score. Here scarcity draws are indexed by ``draw`` so a level can be
   resampled, and a validation split is carved out of train for model selection.

2. v1 added synthetic images to *every* class including the majority one, which
   re-inflates the imbalance it claims to fix. ``MixedDataset`` only augments
   classes that were actually subsampled.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def find_image_root(base: Path) -> Path:
    """Find the directory whose immediate subdirectories hold the images."""
    base = Path(base)
    best, best_n = base, -1
    for dirpath, dirnames, _ in os.walk(base):
        if not dirnames:
            continue
        n = 0
        for d in dirnames:
            sub = Path(dirpath) / d
            if not sub.is_dir():
                continue
            try:
                n += sum(1 for f in os.listdir(sub) if f.lower().endswith(IMG_EXT))
            except OSError:
                pass
        if n > best_n:
            best, best_n = Path(dirpath), n
    return best


@dataclass
class SeedData:
    root: Path
    dataset: ImageFolder
    class_names: list[str]
    targets: np.ndarray

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def counts(self) -> dict[str, int]:
        c = Counter(self.targets.tolist())
        return {self.class_names[i]: c[i] for i in range(self.num_classes)}


def load_seed_data(data_root: Path) -> SeedData:
    root = find_image_root(Path(data_root))
    ds = ImageFolder(root)
    if len(ds) == 0:
        raise RuntimeError(f"no images found under {root}")
    targets = np.array([s[1] for s in ds.samples])
    return SeedData(root=root, dataset=ds, class_names=list(ds.classes), targets=targets)


# --------------------------------------------------------------------------- #
# splits + scarcity
# --------------------------------------------------------------------------- #
@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    scarce: np.ndarray
    majority_class: int
    scarce_per_class: int

    def scarce_counts(self, targets: np.ndarray, n_classes: int) -> list[int]:
        t = targets[self.scarce]
        return [int((t == i).sum()) for i in range(n_classes)]


def _stratified_split(
    targets: np.ndarray, idx: np.ndarray, frac: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``idx`` into (rest, held) with ``frac`` of each class held out."""
    held: list[int] = []
    for c in np.unique(targets[idx]):
        cls_idx = idx[targets[idx] == c]
        cls_idx = rng.permutation(cls_idx)
        k = max(1, int(round(frac * len(cls_idx))))
        held.extend(cls_idx[:k].tolist())
    held_arr = np.array(sorted(held))
    rest = np.array(sorted(set(idx.tolist()) - set(held)))
    return rest, held_arr


def make_splits(
    data: SeedData,
    scarce_per_class: int,
    seed: int = 42,
    test_frac: float = 0.2,
    val_frac: float = 0.1,
    draw: int = 0,
) -> Splits:
    """Build test/val/train splits and the scarce training subset.

    The test and val splits are a function of ``seed`` only, so every scarcity
    level and every generator is scored on the exact same held-out images.
    """
    targets = data.targets
    all_idx = np.arange(len(targets))

    rng_split = np.random.default_rng(seed)
    trainval, test = _stratified_split(targets, all_idx, test_frac, rng_split)
    train, val = _stratified_split(targets, trainval, val_frac, rng_split)

    counts = Counter(targets[train].tolist())
    majority = max(counts, key=lambda k: counts[k])

    rng_draw = np.random.default_rng(seed + 10_000 * (draw + 1))
    scarce: list[int] = []
    for c in range(data.num_classes):
        cls_idx = train[targets[train] == c]
        if c == majority:
            scarce.extend(cls_idx.tolist())
        else:
            k = min(scarce_per_class, len(cls_idx))
            scarce.extend(rng_draw.choice(cls_idx, size=k, replace=False).tolist())

    return Splits(
        train=train,
        val=val,
        test=test,
        scarce=np.array(sorted(scarce)),
        majority_class=int(majority),
        scarce_per_class=scarce_per_class,
    )


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def diffusion_transform(res: int, train: bool = True) -> T.Compose:
    """Images normalised to [-1, 1] for pixel-space diffusion."""
    ops: list = [T.Resize((res, res), interpolation=T.InterpolationMode.BICUBIC)]
    if train:
        ops += [T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)]
    ops += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
    return T.Compose(ops)


def latent_cache_transform(res: int) -> T.Compose:
    """Augmentation applied *before* VAE encoding when caching latents.

    Cached latents freeze augmentation, so ``sd_latent_variants`` different
    draws of this transform are cached per image to keep some diversity.
    """
    return T.Compose(
        [
            T.Resize(int(res * 1.15), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomCrop(res),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )


def classifier_transform(res: int, train: bool, heavy: bool = False) -> T.Compose:
    if not train:
        return T.Compose(
            [
                T.Resize((res, res), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    if heavy:  # condition B: strong geometric + photometric augmentation
        return T.Compose(
            [
                T.Resize((res, res), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(0.5),
                T.RandomVerticalFlip(0.5),
                T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                T.ColorJitter(0.4, 0.4, 0.3, 0.1),
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ]
        )
    return T.Compose(
        [
            T.Resize((res, res), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.3),
            T.RandomRotation(15),
            T.ColorJitter(0.2, 0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
class IndexedImageDataset(Dataset):
    """A subset of an ImageFolder addressed by sample index."""

    def __init__(self, base: ImageFolder, indices: Sequence[int], transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        path, label = self.base.samples[self.indices[i]]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class MixedDataset(Dataset):
    """Real images plus synthetic images, augmenting only subsampled classes.

    ``synth_per_class`` synthetic images are added to every class *except* the
    majority class, which was never subsampled and therefore needs no top-up.
    """

    def __init__(
        self,
        base: ImageFolder,
        real_indices: Sequence[int],
        synth_dir: Path | None,
        class_names: Sequence[str],
        transform=None,
        synth_per_class: int = 0,
        skip_classes: Sequence[int] = (),
        synth_seed: int = 0,
        allowed: dict | None = None,
    ):
        self.base = base
        self.real_indices = list(real_indices)
        self.transform = transform
        self.synth: list[tuple[str, int]] = []

        if synth_dir is not None and synth_per_class > 0:
            rng = np.random.default_rng(synth_seed)
            synth_dir = Path(synth_dir)
            for ci, name in enumerate(class_names):
                if ci in skip_classes:
                    continue
                d = synth_dir / name
                if not d.is_dir():
                    continue
                files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
                if allowed is not None:
                    # quality-aware selection: only images the filter kept
                    keep = set(allowed.get(ci, []))
                    files = [f for f in files if str(f) in keep]
                if len(files) > synth_per_class:
                    pick = rng.choice(len(files), size=synth_per_class, replace=False)
                    files = [files[i] for i in sorted(pick)]
                self.synth.extend((str(f), ci) for f in files)

    @property
    def labels(self) -> list[int]:
        """Label of every item, real then synthetic, in ``__getitem__`` order.

        Needed by the class-balancing baselines, which have to weight or resample
        by class without loading a single image.
        """
        real = [int(self.base.targets[i]) for i in self.real_indices]
        return real + [ci for _, ci in self.synth]

    @property
    def n_real(self) -> int:
        return len(self.real_indices)

    @property
    def n_synth(self) -> int:
        return len(self.synth)

    def __len__(self) -> int:
        return len(self.real_indices) + len(self.synth)

    def __getitem__(self, i: int):
        if i < len(self.real_indices):
            path, label = self.base.samples[self.real_indices[i]]
        else:
            path, label = self.synth[i - len(self.real_indices)]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int, drop_last=False):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=workers > 0,
    )


def save_manifest(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
