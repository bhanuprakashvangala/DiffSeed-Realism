"""Downstream classification.

Differences from v1 that change what the numbers mean:

* **Model selection on a validation split.** v1 trained a fixed 30 epochs and
  reported the final-epoch test score. With ~600 training images that score is
  dominated by where the run happened to stop. Here the best-val checkpoint is
  restored before touching the test set.
* **Repeats.** Every condition is trained under several seeds so a 2.5 pp
  difference can be checked against its own run-to-run spread. v1's headline
  claim -- diffusion augmentation *hurts* by 2.53 pp -- rests on one seed.
* **Balanced metrics.** Accuracy on this dataset is dominated by the untouched
  majority class. Macro-F1, balanced accuracy and MCC are reported alongside.
* **A model zoo.** ResNet-50 alone cannot distinguish "synthetic data is bad"
  from "synthetic data is bad *for a ResNet*".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from tqdm.auto import tqdm

TIMM_NAMES = {
    "resnet50": "resnet50.a1_in1k",
    "efficientnetv2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "vit_b16": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "deit3_small": "deit3_small_patch16_224.fb_in22k_ft_in1k",
    "maxvit_tiny": "maxvit_tiny_tf_224.in1k",
}


def build_classifier(name: str, num_classes: int, cfg):
    import timm

    model = timm.create_model(
        TIMM_NAMES.get(name, name), pretrained=True, num_classes=num_classes
    )
    model = model.to(cfg.device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.grad_checkpoint and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    return model


@dataclass
class ClfResult:
    condition: str
    model: str
    seed: int
    scarcity: int
    n_train: int
    n_real: int
    n_synth: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_acc: float
    mcc: float
    per_class_f1: list[float]
    preds: list[int] = field(default_factory=list, repr=False)
    labels: list[int] = field(default_factory=list, repr=False)
    best_val_f1: float = 0.0
    seconds: float = 0.0

    def as_row(self):
        d = asdict(self)
        d.pop("preds")
        d.pop("labels")
        return d


@torch.no_grad()
def _evaluate(model, loader, cfg, amp: bool):
    model.eval()
    preds, labs = [], []
    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        if cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=cfg.torch_dtype, enabled=amp):
            out = model(x)
        preds.append(out.float().argmax(1).cpu())
        labs.append(y)
    return torch.cat(preds).numpy(), torch.cat(labs).numpy()


def train_classifier(
    train_ds,
    val_loader,
    test_loader,
    num_classes: int,
    cfg,
    condition: str,
    model_name: str | None = None,
    seed: int = 0,
    scarcity: int = 0,
    epochs: int | None = None,
    cost=None,
    save_to=None,
    balance: str | None = None,
) -> ClfResult:
    import time

    from .data import make_loader

    model_name = model_name or cfg.clf_headline
    epochs = epochs or cfg.clf_epochs
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_classifier(model_name, num_classes, cfg)

    # Class-balancing baselines. A paper claiming that selected synthetic data
    # helps under imbalance has to say what it beats, and the honest comparison
    # is against the standard remedies rather than against doing nothing. Both
    # are classifier-side: no images are generated and nothing else changes.
    class_w = None
    sampler = None
    if balance in ("weighted", "oversample"):
        lbl = np.asarray(getattr(train_ds, "labels", []), dtype=int)
        if lbl.size:
            counts = np.bincount(lbl, minlength=num_classes).astype(float)
            counts[counts == 0] = 1.0
            if balance == "weighted":
                # inverse frequency, normalised to mean 1 so the effective
                # learning rate is unchanged relative to the unweighted run
                w = counts.sum() / (num_classes * counts)
                class_w = torch.tensor(w / w.mean(), dtype=torch.float32,
                                       device=cfg.device)
            else:
                from torch.utils.data import WeightedRandomSampler

                per = (1.0 / counts)[lbl]
                sampler = WeightedRandomSampler(
                    torch.as_tensor(per, dtype=torch.double),
                    num_samples=len(lbl), replacement=True)

    if sampler is not None:
        from torch.utils.data import DataLoader

        loader = DataLoader(train_ds, batch_size=cfg.clf_batch, sampler=sampler,
                            num_workers=cfg.num_workers, drop_last=False,
                            pin_memory=(cfg.device == "cuda"))
    else:
        loader = make_loader(train_ds, cfg.clf_batch, True, cfg.num_workers,
                             drop_last=False)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.clf_lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.clf_lr, total_steps=epochs * max(1, len(loader)), pct_start=0.25
    )
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.clf_label_smoothing,
                               weight=class_w)
    amp = cfg.device == "cuda" and cfg.amp_dtype != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=amp and cfg.amp_dtype == "fp16")

    best_f1, best_state = -1.0, None
    t0 = time.perf_counter()
    for _ in tqdm(range(epochs), desc=f"{condition}|{model_name}|s{seed}", leave=False):
        model.train()
        for x, y in loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)
            if cfg.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=cfg.torch_dtype, enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

        vp, vl = _evaluate(model, val_loader, cfg, amp)
        vf1 = f1_score(vl, vp, average="macro", zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    preds, labels = _evaluate(model, test_loader, cfg, amp)
    secs = time.perf_counter() - t0

    # The best-val weights are discarded below. Any caller that needs the fitted
    # model -- the synthetic-image scorer, for one -- must ask for them here;
    # rebuilding the architecture afterwards yields a random network, which
    # scores nothing and fails silently.
    if save_to is not None:
        from pathlib import Path as _P

        _P(save_to).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_to)

    n_real = getattr(train_ds, "n_real", len(train_ds))
    n_synth = getattr(train_ds, "n_synth", 0)

    if cost is not None:
        from .bench import count_params

        cost.trainable_params_m, cost.total_params_m = count_params(model)
        cost.steps = epochs * max(1, len(loader))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ClfResult(
        condition=condition,
        model=model_name,
        seed=seed,
        scarcity=scarcity,
        n_train=len(train_ds),
        n_real=n_real,
        n_synth=n_synth,
        accuracy=float(accuracy_score(labels, preds)),
        macro_f1=float(f1_score(labels, preds, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(labels, preds, average="weighted", zero_division=0)),
        balanced_acc=float(balanced_accuracy_score(labels, preds)),
        mcc=float(matthews_corrcoef(labels, preds)),
        per_class_f1=[float(v) for v in f1_score(labels, preds, average=None, zero_division=0)],
        preds=preds.tolist(),
        labels=labels.tolist(),
        best_val_f1=float(best_f1),
        seconds=secs,
    )


# --------------------------------------------------------------------------- #
# real-vs-synthetic detectability
# --------------------------------------------------------------------------- #
def detectability(real_paths, fake_paths, cfg, epochs: int = 8, seed: int = 0) -> dict:
    """Train a ResNet-18 to tell real from synthetic on a *held-out* split.

    v1 reported 100% discriminator accuracy but drew its real images with
    ``sorted(glob)[:40]`` from the full dataset, so the same files could appear
    in both this test and the generator's training set. Here the split is
    explicit and the reported number is AUC on unseen images of both kinds.
    """
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset
    from PIL import Image
    from torchvision import transforms as T

    from .data import IMAGENET_MEAN, IMAGENET_STD

    tf = T.Compose(
        [
            T.Resize((cfg.clf_res, cfg.clf_res)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    n = min(len(real_paths), len(fake_paths))
    rng = np.random.default_rng(seed)
    real_paths = [real_paths[i] for i in rng.permutation(len(real_paths))[:n]]
    fake_paths = [fake_paths[i] for i in rng.permutation(len(fake_paths))[:n]]

    X = torch.stack([tf(Image.open(p).convert("RGB")) for p in real_paths + fake_paths])
    y = torch.cat([torch.zeros(n), torch.ones(n)]).long()
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    X, y = X[perm], y[perm]
    cut = int(0.7 * len(X))

    tr = DataLoader(TensorDataset(X[:cut], y[:cut]), batch_size=16, shuffle=True)
    te = DataLoader(TensorDataset(X[cut:], y[cut:]), batch_size=32)

    import timm

    model = timm.create_model("resnet18", pretrained=True, num_classes=2).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb.to(cfg.device)), yb.to(cfg.device)).backward()
            opt.step()

    model.eval()
    probs, labs = [], []
    with torch.no_grad():
        for xb, yb in te:
            probs.append(torch.softmax(model(xb.to(cfg.device)), 1)[:, 1].cpu())
            labs.append(yb)
    probs = torch.cat(probs).numpy()
    labs = torch.cat(labs).numpy()
    acc = float(((probs > 0.5).astype(int) == labs).mean())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "detector_acc": acc,
        "detector_auc": float(roc_auc_score(labs, probs)),
        # 1.0 when the detector is at chance, 0.0 when it is perfect
        "fooling_rate": float(max(0.0, 1.0 - 2 * (acc - 0.5))),
        "n_per_side": n,
    }
