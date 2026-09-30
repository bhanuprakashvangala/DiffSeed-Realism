"""Central configuration for DiffSeed v2 experiments.

Everything that controls cost lives here. Three tiers:
  smoke     -- ~15 min on any GPU, proves the pipeline runs end to end
  standard  -- ~6-10 h on one A10G/L4, enough for the paper
  full      -- ~30 h, the complete scarcity x method grid
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

import torch


# --------------------------------------------------------------------------- #
# class prompts for text-conditioned latent diffusion
# --------------------------------------------------------------------------- #
# Rare-token identifiers keep each defect class from collapsing onto Stable
# Diffusion's own prior for the word "soybean".
CLASS_PROMPTS: dict[str, str] = {
    "Broken soybeans": (
        "a sksbrk macro photograph of a single broken soybean seed with a "
        "cracked split cotyledon, studio lighting, plain background"
    ),
    "Immature soybeans": (
        "a sksimm macro photograph of a single immature green soybean seed, "
        "shrivelled and undersized, studio lighting, plain background"
    ),
    "Intact soybeans": (
        "a sksint macro photograph of a single intact healthy soybean seed "
        "with a smooth uniform seed coat, studio lighting, plain background"
    ),
    "Skin-damaged soybeans": (
        "a sksskn macro photograph of a single soybean seed with a wrinkled "
        "torn damaged seed coat, studio lighting, plain background"
    ),
    "Spotted soybeans": (
        "a sksspt macro photograph of a single spotted soybean seed with dark "
        "brown fungal lesions on the seed coat, studio lighting, plain background"
    ),
}

NEGATIVE_PROMPT = (
    "blurry, low quality, cartoon, illustration, text, watermark, multiple seeds"
)


def prompt_for(class_name: str, crop: str = "seed") -> str:
    """Prompt for a class, falling back to a synthesised one.

    The hand-written prompts above are soybean-specific. Running the same
    pipeline on another dataset (the maize cross-species arm) would otherwise
    raise a KeyError on the first unseen class name, so unknown classes get a
    prompt built from the folder name. Rare-token identifiers are derived from
    the name so each class still gets its own handle on the prior rather than
    collapsing onto Stable Diffusion's idea of "seed".
    """
    if class_name in CLASS_PROMPTS:
        return CLASS_PROMPTS[class_name]
    slug = "".join(ch for ch in class_name.lower() if ch.isalnum())[:8] or "cls"
    readable = class_name.replace("_", " ").replace("-", " ").strip()
    return (
        f"a sks{slug} macro photograph of a single {readable} {crop}, "
        f"studio lighting, plain background"
    )


def prompts_for_classes(class_names, crop: str = "seed") -> dict[str, str]:
    return {c: prompt_for(c, crop) for c in class_names}


@dataclass
class Config:
    # ---------------- paths ----------------
    data_root: Path = Path(os.path.expanduser("~/data/soybean_seeds"))
    out_root: Path = Path("./runs")
    tier: str = "standard"

    # ---------------- reproducibility ----------------
    seed: int = 42
    classifier_seeds: Sequence[int] = (0, 1, 2)  # repeats -> confidence intervals

    # ---------------- scarcity protocol ----------------
    # v1 used a single point (100/class). The threshold question needs a sweep.
    scarcity_levels: Sequence[int] = (25, 50, 100, 200)
    headline_scarcity: int = 100  # level at which every method is compared
    # Namespace for emitted LaTeX macros. A secondary corpus sets this so its
    # numbers sit beside the primary ones instead of overwriting them.
    macro_prefix: str = "DS"
    test_frac: float = 0.2
    val_frac: float = 0.1  # held out of train, used for model selection

    # ---------------- generators ----------------
    # methods run at every scarcity level (the threshold grid)
    # sd_lora first: it is the headline arm, so any problem with it should
    # surface before hours of pixel-DDPM training, not after
    grid_methods: Sequence[str] = ("trad", "sd_lora", "ddpm_fixed")
    # methods run only at headline_scarcity (the breadth comparison)
    extra_methods: Sequence[str] = (
        "dcgan",
        "fastgan",
        "ddpm_v1",
        "ddpm_ft",
        "sd_lora_sdedit",
        "sd_turbo_lora",
    )

    n_synth_per_class: int = 400  # generated pool; ratio study subsamples it
    n_eval_synth: int = 1000  # samples used for FID/KID/CMMD
    ratio_sweep: Sequence[int] = (0, 25, 50, 100, 200, 400)

    # ---------------- pixel-space DDPM ----------------
    ddpm_res: int = 128
    # Budget in optimiser steps, not epochs: v1's "100 epochs" was at batch 16,
    # so quoting epochs while running batch 32 would halve the updates and make
    # the v1-vs-fixed comparison meaningless. v1 saw ~8.5k steps; every
    # pixel-space variant here gets the same count.
    # v1 ran 100 epochs at batch 16 over a 1361-image scarce set with
    # drop_last, i.e. 85 steps/epoch = 8500 optimiser steps. Every pixel-space
    # variant gets exactly that, so ddpm_v1 vs ddpm_fixed differs only in the
    # implementation.
    ddpm_steps: int = 8500
    ddpm_epochs: int = 100  # fallback only, used when ddpm_steps is None
    ddpm_batch: int = 32
    ddpm_lr: float = 2e-4
    ddpm_ema_decay: float = 0.9995
    ddpm_cond_dropout: float = 0.1  # enables *real* classifier-free guidance
    ddpm_channels: tuple = (96, 192, 288, 384)
    # ancestral DDPM steps; DPM-Solver++ diverges on these models (see
    # pixel_ddpm.fresh_scheduler), so few-step solvers are not usable here
    ddpm_sample_steps: int = 500
    ddpm_guidance: float = 2.0
    ddpm_ft_model: str = "google/ddpm-celebahq-256"  # pretrained pixel DDPM to fine-tune

    # ---------------- latent diffusion (transfer learning) ----------------
    sd_model: str = "runwayml/stable-diffusion-v1-5"
    sd_turbo_model: str = "stabilityai/sd-turbo"
    sd_res: int = 512  # SD1.5 is native at 512; 256 degrades badly
    sd_lora_rank: int = 16
    sd_lora_alpha: int = 16
    sd_lora_targets: Sequence[str] = ("to_q", "to_k", "to_v", "to_out.0")
    sd_steps: int = 3000  # optimiser steps, not epochs
    sd_batch: int = 4
    sd_grad_accum: int = 1
    sd_lr: float = 1e-4
    sd_text_dropout: float = 0.1
    sd_latent_variants: int = 8  # cached augmented crops per training image
    sd_sample_steps: int = 25  # DPM-Solver++ 2M
    sd_guidance: float = 5.0
    sdedit_strength: float = 0.6  # img2img noise level for the SDEdit variant

    # ---------------- downstream classifiers ----------------
    clf_models: Sequence[str] = (
        "resnet50",
        "efficientnetv2_s",
        "convnext_tiny",
        "vit_b16",
        "swin_tiny",
    )
    clf_headline: str = "resnet50"
    zoo_conditions: Sequence[str] = ("none", "trad", "ddpm_fixed", "sd_lora", "oracle")
    clf_res: int = 224
    clf_epochs: int = 30
    clf_batch: int = 32
    clf_lr: float = 1e-4
    clf_label_smoothing: float = 0.1

    # ---------------- efficiency ----------------
    amp_dtype: str = "bf16"  # bf16 where supported, else fp16
    channels_last: bool = True
    compile_model: bool = False  # torch.compile; slow first step, big win after
    grad_checkpoint: bool = False  # turn on for <=8 GB cards
    use_8bit_adam: bool = False  # needs bitsandbytes
    num_workers: int = 6
    # fraction of generated images retained by quality-aware selection
    filter_keep_frac: float = 0.6
    strict_repro: bool = False  # deterministic kernels for artefact reproduction

    # ---------------- metrics ----------------
    fid_n_curve: Sequence[int] = (100, 200, 500, 1000, 2000)  # small-N bias curve
    clip_model: str = "openai/clip-vit-large-patch14"
    prdc_k: int = 5

    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ------------------------------------------------------------------ #
    def apply_tier(self) -> "Config":
        if self.tier == "smoke":
            self.scarcity_levels = (100,)
            self.grid_methods = ("trad", "sd_lora", "ddpm_fixed")
            self.extra_methods = ()
            self.classifier_seeds = (0,)
            self.n_synth_per_class = 32
            self.n_eval_synth = 32
            self.ratio_sweep = (0, 32)
            self.ddpm_epochs = 5
            self.ddpm_steps = 60
            self.sd_steps = 60
            self.clf_epochs = 2
            self.clf_models = ("resnet50",)
            self.fid_n_curve = (32,)
        elif self.tier == "standard":
            pass
        elif self.tier == "cross":
            # cross-species arm: does the soybean conclusion hold on maize?
            # One scarcity level and the three decisive generators, rather than
            # the whole grid -- enough to test transfer of the finding, not to
            # re-derive it.
            self.scarcity_levels = (100,)
            self.headline_scarcity = 100
            self.grid_methods = ("trad", "sd_lora", "ddpm_fixed")
            self.extra_methods = ()
            self.classifier_seeds = (0, 1, 2)
            self.clf_models = ("resnet50",)
            self.n_synth_per_class = 300
            self.n_eval_synth = 600
            self.ratio_sweep = (0, 50, 100, 200, 300)
        elif self.tier == "full":
            self.scarcity_levels = (25, 50, 100, 200, 400)
            self.grid_methods = (
                "trad",
                "dcgan",
                "fastgan",
                "ddpm_v1",
                "ddpm_fixed",
                "ddpm_ft",
                "sd_lora",
                "sd_lora_sdedit",
            )
            self.classifier_seeds = (0, 1, 2, 3, 4)
            self.sd_steps = 6000
            self.ddpm_steps = 30000
        else:
            raise ValueError(f"unknown tier {self.tier!r}")
        return self

    def fit_to_gpu(self) -> "Config":
        """Shrink batch sizes so the run survives on a small card."""
        if self.device != "cuda":
            return self
        gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gb < 9:
            self.sd_batch, self.sd_grad_accum = 1, 4
            self.ddpm_batch = 16
            self.clf_batch = 16
            self.grad_checkpoint = True
            self.use_8bit_adam = True
        elif gb < 17:
            self.sd_batch, self.sd_grad_accum = 2, 2
            self.ddpm_batch = 24
        if not torch.cuda.is_bf16_supported():
            self.amp_dtype = "fp16"
        return self

    @property
    def torch_dtype(self):
        return {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[self.amp_dtype]

    def dirs(self, *parts) -> Path:
        p = self.out_root.joinpath(*[str(x) for x in parts])
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self):
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
