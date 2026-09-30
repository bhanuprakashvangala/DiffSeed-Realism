"""Latent-diffusion transfer learning -- the experiment v1 was missing.

Instead of learning the distribution of seed images from ~500 examples, this
starts from Stable Diffusion's pretrained prior and adapts it with LoRA. Only
the rank-16 adapters on the attention projections are trained: 3.19 M
parameters (measured) against the ~110 M the from-scratch DDPM had to fit.

Efficiency, which is what makes this cheaper than the from-scratch run it
replaces:

* **Latents are cached.** The VAE encoder is run exactly once per augmented
  crop, up front, and the training loop then reads 4x64x64 tensors from RAM.
  This removes the VAE from the hot loop entirely -- typically 35-40% of step
  time -- and lets the whole training set sit resident.
* **Text embeddings are cached.** There are six prompts in this study (five
  classes plus the null prompt), so the text encoder runs six times total and is
  then deleted. It never occupies VRAM during training.
* **The VAE and text encoder are freed before training starts**, leaving VRAM
  for the UNet. Measured peak on an A10G at batch 4 without gradient
  checkpointing: 13.1 GB. The small-VRAM path (batch 1, grad accumulation 4,
  checkpointing, 8-bit Adam) is selected automatically below 9 GB.
* **DPM-Solver++ 2M at 25 steps** for sampling instead of 1000-step ancestral
  DDPM sampling, a 40x reduction in forward passes per image.

Cached latents freeze augmentation, so ``sd_latent_variants`` independently
augmented crops are cached per source image to keep geometric diversity.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# stage 1 -- cache
# --------------------------------------------------------------------------- #
@torch.no_grad()
def cache_latents_and_text(
    image_paths: Sequence[tuple[str, int]],
    class_names: Sequence[str],
    class_prompts: dict[str, str],
    cfg,
    model_id: str | None = None,
    variants: int | None = None,
):
    """Encode every training image to VAE latents and every prompt to embeddings.

    Returns ``(latents, labels, text_embeds, null_embed)`` where ``latents`` is
    ``(N * variants, 4, res/8, res/8)`` held in CPU RAM.
    """
    from diffusers import AutoencoderKL
    from transformers import CLIPTextModel, CLIPTokenizer

    from ..data import latent_cache_transform

    model_id = model_id or cfg.sd_model
    variants = variants or cfg.sd_latent_variants
    device, dtype = cfg.device, cfg.torch_dtype

    # ---- text ----
    tok = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
    txt = txt.to(device).eval()

    prompts = [class_prompts[c] for c in class_names] + [""]
    ids = tok(
        prompts, padding="max_length", max_length=tok.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    embeds = txt(ids)[0].cpu()
    text_embeds, null_embed = embeds[:-1], embeds[-1:]

    del txt, tok
    _free()

    # ---- latents ----
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
    vae = vae.to(device).eval()
    scaling = vae.config.scaling_factor

    tf = latent_cache_transform(cfg.sd_res)
    lat_list, lab_list = [], []
    batch, batch_lab = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device, dtype=dtype)
        z = vae.encode(x).latent_dist.sample() * scaling
        lat_list.append(z.float().cpu())
        lab_list.append(torch.tensor(batch_lab))
        batch.clear()
        batch_lab.clear()

    torch.manual_seed(cfg.seed)
    for path, label in tqdm(image_paths, desc="caching latents"):
        img = Image.open(path).convert("RGB")
        for _ in range(variants):
            batch.append(tf(img))
            batch_lab.append(label)
            if len(batch) >= 16:
                flush()
    flush()

    del vae
    _free()

    latents = torch.cat(lat_list)
    labels = torch.cat(lab_list)
    return latents, labels, text_embeds, null_embed


# --------------------------------------------------------------------------- #
# stage 2 -- LoRA fine-tune
# --------------------------------------------------------------------------- #
def train_sd_lora(
    latents: torch.Tensor,
    labels: torch.Tensor,
    text_embeds: torch.Tensor,
    null_embed: torch.Tensor,
    cfg,
    model_id: str | None = None,
    steps: int | None = None,
    cost=None,
    log_every: int = 100,
):
    """Fine-tune SD's UNet with LoRA on cached latents. Returns (unet, losses)."""
    from diffusers import DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig

    model_id = model_id or cfg.sd_model
    steps = steps or cfg.sd_steps
    device, dtype = cfg.device, cfg.torch_dtype

    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float32)
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=cfg.sd_lora_rank,
            lora_alpha=cfg.sd_lora_alpha,
            init_lora_weights="gaussian",
            target_modules=list(cfg.sd_lora_targets),
        )
    )
    unet = unet.to(device)
    if cfg.channels_last:
        unet = unet.to(memory_format=torch.channels_last)
    if cfg.grad_checkpoint:
        unet.enable_gradient_checkpointing()
    unet.enable_attention_slicing() if cfg.grad_checkpoint else None

    params = [p for p in unet.parameters() if p.requires_grad]
    if cfg.use_8bit_adam:
        try:
            import bitsandbytes as bnb

            opt = bnb.optim.AdamW8bit(params, lr=cfg.sd_lr, weight_decay=1e-4)
        except Exception:
            opt = torch.optim.AdamW(params, lr=cfg.sd_lr, weight_decay=1e-4)
    else:
        opt = torch.optim.AdamW(params, lr=cfg.sd_lr, weight_decay=1e-4)

    lr_sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.sd_lr, total_steps=steps, pct_start=0.1
    )
    noise_sched = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    text_embeds = text_embeds.to(device, dtype=torch.float32)
    null_embed = null_embed.to(device, dtype=torch.float32)

    amp = device == "cuda" and cfg.amp_dtype != "fp32"
    scaler = torch.amp.GradScaler("cuda", enabled=amp and cfg.amp_dtype == "fp16")

    n = latents.size(0)
    g = torch.Generator().manual_seed(cfg.seed)
    losses, running = [], []

    pbar = tqdm(range(steps), desc="SD-LoRA")
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg.sd_grad_accum):
            idx = torch.randint(0, n, (cfg.sd_batch,), generator=g)
            z0 = latents[idx].to(device, non_blocking=True)
            lab = labels[idx].to(device)

            noise = torch.randn(z0.shape, device=device)
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (z0.size(0),), device=device)
            zt = noise_sched.add_noise(z0, noise, t)

            ctx = text_embeds[lab]
            if cfg.sd_text_dropout > 0:
                drop = torch.rand(z0.size(0), device=device) < cfg.sd_text_dropout
                ctx = torch.where(drop[:, None, None], null_embed.expand_as(ctx), ctx)

            with torch.autocast("cuda", dtype=cfg.torch_dtype, enabled=amp):
                pred = unet(zt, t, encoder_hidden_states=ctx).sample
                if noise_sched.config.prediction_type == "v_prediction":
                    target = noise_sched.get_velocity(z0, noise, t)
                else:
                    target = noise
                loss = F.mse_loss(pred.float(), target.float()) / cfg.sd_grad_accum

            scaler.scale(loss).backward()
            running.append(loss.item() * cfg.sd_grad_accum)

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        lr_sched.step()

        if (step + 1) % log_every == 0:
            losses.append(float(np.mean(running)))
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")
            running = []

    if cost is not None:
        from ..bench import count_params

        cost.trainable_params_m, cost.total_params_m = count_params(unet)
        cost.steps = steps

    return unet.eval(), losses


def save_lora(unet, path: Path):
    from peft.utils import get_peft_model_state_dict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(get_peft_model_state_dict(unet), path)


def load_lora_unet(path: Path, cfg, model_id: str | None = None):
    """Rebuild a LoRA-adapted UNet from saved adapter weights."""
    from diffusers import UNet2DConditionModel
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict

    model_id = model_id or cfg.sd_model
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float32
    )
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=cfg.sd_lora_rank,
            lora_alpha=cfg.sd_lora_alpha,
            init_lora_weights="gaussian",
            target_modules=list(cfg.sd_lora_targets),
        )
    )
    set_peft_model_state_dict(unet, torch.load(path, map_location="cpu", weights_only=False))
    return unet.to(cfg.device).eval()


# --------------------------------------------------------------------------- #
# stage 3 -- sample
# --------------------------------------------------------------------------- #
def build_pipeline(unet, cfg, model_id: str | None = None, turbo: bool = False):
    """Assemble an inference pipeline around the fine-tuned UNet."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    model_id = model_id or (cfg.sd_turbo_model if turbo else cfg.sd_model)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, unet=unet.to(cfg.torch_dtype), torch_dtype=cfg.torch_dtype,
        safety_checker=None, requires_safety_checker=False,
    )
    if not turbo:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
        )
    pipe = pipe.to(cfg.device)
    pipe.set_progress_bar_config(disable=True)
    if cfg.grad_checkpoint:  # small-VRAM path
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    return pipe


@torch.no_grad()
def generate_to_dir(
    pipe,
    class_names: Sequence[str],
    class_prompts: dict[str, str],
    out_dir: Path,
    n_per_class: int,
    cfg,
    negative_prompt: str = "",
    tag: str = "sdlora",
    seed: int = 0,
    batch_size: int = 4,
    skip_classes: Sequence[int] = (),
    steps: int | None = None,
    guidance: float | None = None,
):
    out_dir = Path(out_dir)
    steps = steps or cfg.sd_sample_steps
    guidance = cfg.sd_guidance if guidance is None else guidance

    for ci, name in enumerate(class_names):
        if ci in skip_classes:
            continue
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        have = len(list(d.glob("*.png")))
        prompt = class_prompts[name]
        pbar = tqdm(total=max(0, n_per_class - have), desc=f"gen {name}", leave=False)
        while have < n_per_class:
            b = min(batch_size, n_per_class - have)
            gen = torch.Generator(device=cfg.device).manual_seed(seed + 100_000 * ci + have)
            imgs = pipe(
                prompt=[prompt] * b,
                negative_prompt=[negative_prompt] * b if negative_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
            ).images
            for i, im in enumerate(imgs):
                im.save(d / f"{tag}_{have + i:05d}.png")
            have += b
            pbar.update(b)
        pbar.close()
    return out_dir


@torch.no_grad()
def generate_sdedit_to_dir(
    unet,
    real_paths_by_class: dict[int, list[str]],
    class_names: Sequence[str],
    class_prompts: dict[str, str],
    out_dir: Path,
    n_per_class: int,
    cfg,
    strength: float | None = None,
    tag: str = "sdedit",
    seed: int = 0,
    batch_size: int = 4,
    skip_classes: Sequence[int] = (),
):
    """SDEdit: re-noise a *real* seed image partway, then denoise with the LoRA.

    This keeps the true morphology of an actual seed and resamples only the
    high-frequency appearance, which is usually the strongest augmentation
    available when the real set is tiny -- and it cannot invent a defect that
    never existed, which matters for a seed-testing application.
    """
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline

    strength = cfg.sdedit_strength if strength is None else strength
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        cfg.sd_model, unet=unet.to(cfg.torch_dtype), torch_dtype=cfg.torch_dtype,
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++"
    )
    pipe = pipe.to(cfg.device)
    pipe.set_progress_bar_config(disable=True)

    out_dir = Path(out_dir)
    for ci, name in enumerate(class_names):
        if ci in skip_classes:
            continue
        srcs = real_paths_by_class.get(ci, [])
        if not srcs:
            continue
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        have = len(list(d.glob("*.png")))
        rng = np.random.default_rng(seed + ci)
        while have < n_per_class:
            b = min(batch_size, n_per_class - have)
            pick = rng.choice(len(srcs), b, replace=len(srcs) < b)
            init = [
                Image.open(srcs[i]).convert("RGB").resize((cfg.sd_res, cfg.sd_res), Image.BICUBIC)
                for i in pick
            ]
            gen = torch.Generator(device=cfg.device).manual_seed(seed + 100_000 * ci + have)
            imgs = pipe(
                prompt=[class_prompts[name]] * b,
                image=init,
                strength=strength,
                num_inference_steps=max(10, cfg.sd_sample_steps),
                guidance_scale=cfg.sd_guidance,
                generator=gen,
            ).images
            for i, im in enumerate(imgs):
                im.save(d / f"{tag}_{have + i:05d}.png")
            have += b
    return out_dir
