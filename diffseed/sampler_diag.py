"""Controlled comparison of samplers at matched generator weights.

The repaired pixel-space model produced pure noise under DPM-Solver++ while its
training loss looked healthy, which is the failure mode this stage exists to
document: the sampler, not the model, was broken, and nothing in the training
signal said so. The paper reports the comparison rather than only its
conclusion, so the numbers behind that paragraph have to come from a stage
someone else can re-run instead of from a diagnostic typed into a terminal once.

Sampling here is deliberately reimplemented rather than routed through
``sample_pixel_ddpm``. That function fixes the scheduler internally, which is
correct for production sampling and useless for this measurement: the whole
point is to vary the scheduler while holding weights, initial noise, class,
and guidance fixed. A local loop makes the controlled variable the only
variable.

Two statistics are reported, because they answer different questions. The pixel
standard deviation against that of the real images is a blunt instrument and
that is why it is used: a diverged sampler saturates towards the corners of the
value range and its spread runs away from the real one, which is visible without
any learned metric and so cannot itself be blamed on a mis-specified metric.
Spread cannot, however, rank two samplers that both work -- a blurred output and
a sharp one can share a standard deviation -- so each arm is also scored by
Frechet distance and by wall-clock cost per image. That is what makes the stage
answer the practical question as well as the diagnostic one: sampling dominates
the cost of every pixel-space arm in this study, so whether a 100-step sampler
matches a 500-step one is worth a measurement rather than an assumption.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class SamplerArm:
    """One sampler configuration in the comparison."""

    key: str
    label: str
    kind: str          # ddpm | ddim | dpmpp
    steps: int
    guidance: float


def default_arms(cfg) -> list[SamplerArm]:
    """The four configurations the manuscript reports.

    Guidance is held at 1.0 (off) everywhere except the one arm that exists to
    show guidance is not the culprit; otherwise a divergence could be blamed on
    the guidance scale rather than on the solver.
    """
    prod = int(getattr(cfg, "ddpm_sample_steps", 500))
    return [
        SamplerArm("ddpm_thousand", "Ancestral DDPM, 1000 steps", "ddpm", 1000, 1.0),
        SamplerArm("ddpm_thousand_cfg", "Ancestral DDPM, 1000 steps, CFG",
                   "ddpm", 1000, float(getattr(cfg, "ddpm_guidance", 2.0))),
        # The setting the study actually samples with. Without it the comparison
        # justifies "ancestral rather than the alternatives" but says nothing
        # about the step count it is cited to justify, which is the one decision
        # a reader would want evidence for.
        SamplerArm("ddpm_prod", f"Ancestral DDPM, {prod} steps (used here)",
                   "ddpm", prod, 1.0),
        SamplerArm("ddim_hundred", "DDIM, 100 steps", "ddim", 100, 1.0),
        SamplerArm("dpmsolver", "DPM-Solver++, 100 steps", "dpmpp", 100, 1.0),
    ]


def _scheduler(kind: str):
    """A freshly constructed scheduler on the model's training schedule.

    Constructed rather than adapted from the training scheduler's config: the
    first version of this comparison did it both ways to rule out configuration
    translation as the cause, and the two agreed, so the simpler path is kept.
    """
    from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler

    common = dict(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    if kind == "ddpm":
        return DDPMScheduler(**common)
    if kind == "ddim":
        return DDIMScheduler(**common)
    if kind == "dpmpp":
        return DPMSolverMultistepScheduler(**common, algorithm_type="dpmsolver++")
    raise ValueError(f"unknown sampler kind: {kind}")


@torch.no_grad()
def _sample(model, arm: SamplerArm, n_classes: int, class_idx: int, n: int,
            cfg, seed: int) -> torch.Tensor:
    """Sample ``n`` images under one arm, from identical starting noise.

    The initial noise is drawn on the CPU from a seeded generator so that every
    arm starts from the same tensor regardless of device RNG behaviour. Without
    that, an arm could differ because it got different noise.
    """
    device = cfg.device
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(n, 3, cfg.ddpm_res, cfg.ddpm_res, generator=gen).to(device)

    sched = _scheduler(arm.kind)
    sched.set_timesteps(arm.steps)

    labels = torch.full((n,), class_idx, device=device, dtype=torch.long)
    null = torch.full((n,), n_classes, device=device, dtype=torch.long)
    use_cfg = arm.guidance and abs(arm.guidance - 1.0) > 1e-6

    for t in sched.timesteps:
        if use_cfg:
            out = model(torch.cat([x, x]), t,
                        class_labels=torch.cat([null, labels])).sample
            uncond, cond = out.chunk(2)
            eps = uncond + arm.guidance * (cond - uncond)
        else:
            eps = model(x, t, class_labels=labels).sample
        x = sched.step(eps, t, x).prev_sample
    return x


def _real_std(exp, scarcity: int) -> float:
    """Pixel standard deviation of the real training images, same normalisation."""
    from .data import IndexedImageDataset, diffusion_transform

    cfg = exp.cfg
    sp = exp.splits(scarcity)
    ds = IndexedImageDataset(exp.data.dataset, sp.scarce,
                             diffusion_transform(cfg.ddpm_res, False))
    take = min(len(ds), 256)
    batch = torch.stack([ds[i][0] for i in range(take)])
    return float(batch.std().item())


@torch.no_grad()
def _arm_fid(x: torch.Tensor, real_dir, cfg, n_real: int = 200) -> float:
    """Frechet distance between one arm's samples and the real class images.

    Spread alone separates a diverged sampler from a working one, but it cannot
    rank two working samplers: a blurred output and a sharp one can share a
    standard deviation. That distinction is the whole question when the cheap
    sampler is being considered as a replacement for the expensive one, so the
    arms are scored on a distributional metric as well.

    The value is not comparable with the FIDs elsewhere in the study -- far
    fewer samples, one class -- and exists only to compare arms with each other
    at matched sample count.
    """
    import numpy as np
    from PIL import Image

    from . import metrics as M

    real_paths = M.list_images(real_dir, n_real)
    if len(real_paths) < 10:
        return float("nan")

    # Samples are in [-1, 1]; the feature extractor takes uint8 images.
    arr = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
    arr = np.transpose(arr, (0, 2, 3, 1))

    inc = M.InceptionFeatures(cfg.device)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i, im in enumerate(arr):
            Image.fromarray(im).save(tmp / f"{i:04d}.png")
        ff = inc(M.list_images(tmp))
    fr = inc(real_paths)
    return float(M.frechet_distance(fr, ff))


def run_sampler_diagnostic(exp, method: str = "ddpm_fixed",
                           scarcity: int | None = None,
                           n: int | None = None,
                           with_fid: bool = True) -> "object":
    """Sample one class under every arm and tabulate spread and distance.

    Returns a dataframe and writes it beside the other artefacts. Missing
    checkpoints are not an error: this is a diagnostic, and it should not be
    able to fail a run whose generators trained fine.
    """
    import pandas as pd

    from .generators import pixel_ddpm as P

    cfg = exp.cfg
    scarcity = cfg.headline_scarcity if scarcity is None else scarcity
    ck = exp.root / "checkpoints" / f"{method}_n{scarcity}.pt"
    if not ck.exists():
        print(f"  sampler diagnostic: no checkpoint at {ck}, skipping")
        return None

    model = P.build_fixed_unet(exp.n_classes, cfg.ddpm_res, cfg.ddpm_channels)
    model.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    model = model.to(cfg.device).eval()

    # A minority class, so the arm is measured where the study actually needs
    # the generator to work rather than on the class it has the most data for.
    skip = {exp.splits(scarcity).majority_class}
    class_idx = next(i for i in range(exp.n_classes) if i not in skip)

    # Enough samples for a usable Frechet distance when one is asked for; the
    # divergence check alone needs far fewer, and 1000-step arms are not cheap.
    n = n if n is not None else (100 if with_fid else 16)

    real = _real_std(exp, scarcity)
    real_dir = exp.real_ref_dir(scarcity) / exp.class_names[class_idx]
    rows = []
    for arm in default_arms(cfg):
        t0 = time.time()
        x = _sample(model, arm, exp.n_classes, class_idx, n, cfg, seed=cfg.seed)
        secs = time.time() - t0
        arr = x.float().cpu().numpy()

        fid = float("nan")
        if with_fid:
            try:
                fid = _arm_fid(x, real_dir, cfg)
            except Exception as e:
                print(f"    (fid unavailable for {arm.key}: {type(e).__name__}: {e})")

        rows.append({
            "arm": arm.key,
            "label": arm.label,
            "kind": arm.kind,
            "steps": arm.steps,
            "guidance": arm.guidance,
            "n_samples": n,
            "output_std": float(arr.std()),
            "output_mean": float(arr.mean()),
            "saturated_frac": float(np.mean(np.abs(arr) > 0.99)),
            "real_std": real,
            "fid": fid,
            "sample_seconds": secs,
            "seconds_per_image": secs / max(1, n),
        })
        print(f"  sampler diagnostic: {arm.label:34s} std={rows[-1]['output_std']:.3f} "
              f"(real {real:.3f})  fid={fid:.1f}  {secs / max(1, n):.2f}s/img")

    del model
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = exp.root / "sampler_diagnostic.csv"
    df.to_csv(out, index=False)
    print(f"  sampler diagnostic -> {out}")
    return df


def load_sampler_diagnostic(root: Path):
    import pandas as pd

    p = Path(root) / "sampler_diagnostic.csv"
    return pd.read_csv(p) if p.exists() else None
