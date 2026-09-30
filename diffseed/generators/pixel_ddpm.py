"""Pixel-space class-conditional DDPMs.

Three variants, which together separate *implementation* failure from
*paradigm* failure:

``ddpm_v1``     Faithful reproduction of the original notebook, kept so the
                published 78-97 FID numbers stay reproducible. It has three
                defects, all preserved deliberately and all documented below.

``ddpm_fixed``  Same budget, same resolution, same data. Fixes only the
                implementation. If this alone closes most of the gap, the
                original conclusion ("diffusion cannot do seeds") was really a
                statement about the code, not about diffusion.

``ddpm_ft``     Fine-tunes a *pretrained* pixel DDPM instead of training from
                scratch. Isolates the transfer-learning variable from the
                latent-space variable that Stable Diffusion changes at the same
                time.

Defects reproduced in ``ddpm_v1`` and fixed in ``ddpm_fixed``:

1. Conditioning. v1 learns ``nn.Embedding(n_classes, H*W)`` and concatenates the
   reshaped 16384-dim vector as a fourth input channel. That spends 80k
   parameters per class on a static spatial map the network sees only at the
   input layer, where the first conv immediately mixes it into the image. The
   standard mechanism -- ``num_class_embeds`` on ``UNet2DModel``, which adds the
   class embedding to the timestep embedding and therefore reaches *every*
   residual block -- is used instead.

2. Classifier-free guidance. v1's sampler builds its "unconditional" branch by
   passing a *random class label*, and the model was never trained with label
   dropout, so no unconditional branch exists. The guidance formula was
   therefore extrapolating along the difference between two arbitrary
   conditional predictions, which injects noise proportional to the guidance
   scale. Here a real null token is trained via ``cond_dropout``.

3. No EMA. Diffusion sample quality is famously dominated by the EMA weights;
   v1 sampled from the raw training weights and additionally selected the
   checkpoint by *training* loss, which for a diffusion objective is essentially
   uncorrelated with sample quality.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision.utils import save_image
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
# v1 architecture, preserved verbatim
# --------------------------------------------------------------------------- #
class LegacyClassConditionedUNet(nn.Module):
    """The original notebook's model. Do not 'fix' this -- it is the control."""

    def __init__(self, num_classes: int, image_size: int = 128, in_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes, image_size * image_size)
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=in_channels + 1,
            out_channels=in_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def forward(self, x, t, class_labels):
        bs, _, h, w = x.shape
        cond = self.class_emb(class_labels).view(bs, 1, h, w)
        return self.unet(torch.cat([x, cond], dim=1), t).sample


# --------------------------------------------------------------------------- #
# fixed architecture
# --------------------------------------------------------------------------- #
def build_fixed_unet(num_classes: int, image_size: int, channels: Sequence[int]) -> UNet2DModel:
    """Class embedding injected into the timestep pathway, plus a null class.

    The extra class index ``num_classes`` is the unconditional token, which is
    what makes classifier-free guidance well defined.

    Self-attention sits only at the deepest level, matching the reference DDPM
    configurations, which attach attention at 16x16 and below. Measured effect
    of dropping the 32x32 attention level: 0.767 -> 0.723 s/step at batch 32,
    about 6%. It is the conventional placement rather than a meaningful speedup;
    the step cost here is dominated by the convolutional trunk, not attention.
    """
    n_levels = len(channels)
    down = tuple(
        "AttnDownBlock2D" if i == n_levels - 1 else "DownBlock2D" for i in range(n_levels)
    )
    up = tuple("AttnUpBlock2D" if i == 0 else "UpBlock2D" for i in range(n_levels))
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=tuple(channels),
        down_block_types=down,
        up_block_types=up,
        num_class_embeds=num_classes + 1,  # +1 = null token for CFG
        attention_head_dim=8,
    )


class EMA:
    """Exponential moving average of weights, with a warmup ramp."""

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self.step = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_pixel_ddpm(
    loader,
    num_classes: int,
    cfg,
    variant: str = "fixed",
    log_every: int = 20,
    cost=None,
):
    """Train one of the pixel-space variants. Returns (sampler_model, losses)."""
    device = cfg.device
    legacy = variant == "v1"

    # Budget is specified in optimiser steps, not epochs. v1 quoted 100 epochs
    # at batch 16; running "100 epochs" at batch 32 would silently halve the
    # number of updates and hand the from-scratch baseline a rigged comparison.
    steps_per_epoch = max(1, len(loader))
    target_steps = getattr(cfg, "ddpm_steps", None) or cfg.ddpm_epochs * steps_per_epoch

    if legacy:
        model = LegacyClassConditionedUNet(num_classes, cfg.ddpm_res).to(device)
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        epochs, lr, use_ema = math.ceil(target_steps / steps_per_epoch), 1e-4, False
    elif variant == "ft":
        model = UNet2DModel.from_pretrained(cfg.ddpm_ft_model)
        # Retarget a pretrained unconditional 256px UNet to our resolution and
        # add the class-embedding pathway it was never trained with.
        model.register_to_config(sample_size=cfg.ddpm_res)
        model.config.num_class_embeds = num_classes + 1
        emb_dim = model.time_embedding.linear_2.out_features
        model.class_embedding = nn.Embedding(num_classes + 1, emb_dim)
        nn.init.zeros_(model.class_embedding.weight)  # start as the pretrained prior
        model = model.to(device)
        scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
        epochs = max(1, math.ceil(target_steps / (4 * steps_per_epoch)))
        lr, use_ema = cfg.ddpm_lr / 2, True
    else:
        model = build_fixed_unet(num_classes, cfg.ddpm_res, cfg.ddpm_channels).to(device)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        epochs = math.ceil(target_steps / steps_per_epoch)
        lr, use_ema = cfg.ddpm_lr, True

    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.grad_checkpoint and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()

    ema = EMA(model, cfg.ddpm_ema_decay) if use_ema else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6, betas=(0.9, 0.999))
    total_steps = epochs * max(1, len(loader))
    lr_sched = get_cosine_schedule_with_warmup(opt, min(500, total_steps // 10), total_steps)

    amp = cfg.device == "cuda" and cfg.amp_dtype != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=amp and cfg.amp_dtype == "fp16")

    losses = []
    pbar = tqdm(range(epochs), desc=f"DDPM[{variant}]")
    for epoch in pbar:
        model.train()
        running, n = 0.0, 0
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if cfg.channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            noise = torch.randn_like(images)
            t = torch.randint(0, scheduler.config.num_train_timesteps, (images.size(0),), device=device)
            noisy = scheduler.add_noise(images, noise, t)

            if not legacy and cfg.ddpm_cond_dropout > 0:
                # Train the null token so guidance has a real unconditional branch.
                drop = torch.rand(labels.shape, device=device) < cfg.ddpm_cond_dropout
                labels = torch.where(drop, torch.full_like(labels, num_classes), labels)

            with torch.autocast("cuda", dtype=cfg.torch_dtype, enabled=amp):
                if legacy:
                    pred = model(noisy, t, labels)
                else:
                    pred = model(noisy, t, class_labels=labels).sample
                loss = F.mse_loss(pred.float(), noise.float())

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            lr_sched.step()
            if ema is not None:
                ema.update(model)

            running += loss.item() * images.size(0)
            n += images.size(0)

        losses.append(running / max(1, n))
        if (epoch + 1) % log_every == 0 or epoch == 0:
            pbar.set_postfix(loss=f"{losses[-1]:.4f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

    if cost is not None:
        from ..bench import count_params

        cost.trainable_params_m, cost.total_params_m = count_params(model)
        cost.steps = total_steps

    sampler_model = ema.shadow if ema is not None else model
    return sampler_model.eval(), scheduler, losses


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_pixel_ddpm(
    model,
    train_scheduler,
    num_classes: int,
    class_idx: int,
    n: int,
    cfg,
    variant: str = "fixed",
    batch_size: int = 16,
    guidance: float | None = None,
    steps: int | None = None,
    seed: int = 0,
    return_trajectory: bool = False,
):
    """Sample images for one class.

    ``v1`` reproduces the original broken guidance (random label as the
    "unconditional" branch); every other variant uses the trained null token.

    All variants sample with an ancestral DDPM. DPM-Solver++ was used here at
    first and diverges on these models; see ``sampler_diag`` for the controlled
    comparison and ``fresh_scheduler`` below for the measured numbers.
    """
    device = cfg.device
    legacy = variant == "v1"
    guidance = cfg.ddpm_guidance if guidance is None else guidance
    steps = cfg.ddpm_sample_steps if steps is None else steps

    def fresh_scheduler():
        # Ancestral DDPM sampling for every variant.
        #
        # DPM-Solver++ was tried here first, as the "modern" fast sampler, and it
        # diverges outright on these from-scratch cosine-schedule models: a
        # controlled comparison at matched weights gave output std 0.499 (pure
        # saturated noise) against 0.139 for 1000-step ancestral DDPM and 0.065
        # for 100-step DDIM, on a real-image reference of 0.275. Both
        # from_config and a cleanly constructed solver failed identically, so it
        # is the solver and not config translation. DDIM at 100 steps stayed
        # structured but washed out.
        #
        # A scheduler is still rebuilt per batch: multistep solvers carry
        # ``step_index`` and an output history across calls, so a reused one
        # resumes mid-trajectory and walks off the end of its sigma table.
        s = DDPMScheduler.from_config(train_scheduler.config)
        s.set_timesteps(steps, device=device)
        return s

    gen = torch.Generator(device=device).manual_seed(seed + 1000 * class_idx)
    out, traj = [], []
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        sched = fresh_scheduler()
        x = torch.randn(b, 3, cfg.ddpm_res, cfg.ddpm_res, device=device, generator=gen)
        cond = torch.full((b,), class_idx, dtype=torch.long, device=device)
        null = torch.full((b,), num_classes, dtype=torch.long, device=device)

        snap_at = set(np.linspace(0, len(sched.timesteps) - 1, 8).astype(int).tolist())
        for i, t in enumerate(sched.timesteps):
            tb = t.expand(b) if t.dim() == 0 else t
            if legacy:
                e_c = model(x, tb, cond)
                if guidance > 1.0:
                    rnd = torch.randint(0, num_classes, (b,), device=device, generator=gen)
                    e_u = model(x, tb, rnd)
                    eps = e_u + guidance * (e_c - e_u)
                else:
                    eps = e_c
            else:
                if guidance > 1.0:
                    # one batched forward pass for both branches
                    xin = torch.cat([x, x])
                    lin = torch.cat([cond, null])
                    tin = tb.repeat(2) if tb.dim() > 0 else tb
                    e = model(xin, tin, class_labels=lin).sample
                    e_c, e_u = e.chunk(2)
                    eps = e_u + guidance * (e_c - e_u)
                else:
                    eps = model(x, tb, class_labels=cond).sample
            x = sched.step(eps, t, x).prev_sample
            if return_trajectory and i in snap_at:
                traj.append(((x[:1].clamp(-1, 1) + 1) / 2).cpu())

        out.append(((x.clamp(-1, 1) + 1) / 2).cpu())
        done += b

    images = torch.cat(out)[:n]
    return (images, traj) if return_trajectory else images


def generate_to_dir(
    model,
    scheduler,
    class_names: Sequence[str],
    out_dir: Path,
    n_per_class: int,
    cfg,
    variant: str = "fixed",
    seed: int = 0,
    skip_classes: Sequence[int] = (),
):
    out_dir = Path(out_dir)
    for ci, name in enumerate(class_names):
        if ci in skip_classes:
            continue
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        have = len(list(d.glob("*.png")))
        if have >= n_per_class:
            continue
        imgs = sample_pixel_ddpm(
            model, scheduler, len(class_names), ci, n_per_class - have, cfg,
            variant=variant, seed=seed + have,
        )
        for i, img in enumerate(imgs):
            save_image(img, d / f"{variant}_{have + i:05d}.png")
    return out_dir
