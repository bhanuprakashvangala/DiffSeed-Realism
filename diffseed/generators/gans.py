"""GAN baselines.

``dcgan``   The v1 baseline, reproduced. Its conditioning is multiplicative
            (``noise * label_embedding``), which is not a standard cGAN and is
            kept only for comparability with the published numbers.

``fastgan`` FastGAN (Liu et al., ICLR 2021) -- the reference architecture for
            GAN training on a *few hundred* images, which is the regime here. It
            adds a skip-layer excitation generator and a self-supervised
            reconstruction decoder on the discriminator, plus differentiable
            augmentation (DiffAugment, Zhao et al. NeurIPS 2020) so the
            discriminator cannot memorise the tiny real set. Without an adaptive
            augmentation of this kind, any GAN comparison at n=100/class is
            measuring discriminator overfitting rather than the method.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm.auto import tqdm


# --------------------------------------------------------------------------- #
# DiffAugment
# --------------------------------------------------------------------------- #
def _rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)


def _rand_saturation(x):
    m = x.mean(dim=1, keepdim=True)
    f = torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2
    return (x - m) * f + m


def _rand_contrast(x):
    m = x.mean(dim=[1, 2, 3], keepdim=True)
    f = torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5
    return (x - m) * f + m


def _rand_translation(x, ratio=0.125):
    b, _, h, w = x.shape
    sh, sw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-sh, sh + 1, (b, 1, 1), device=x.device)
    ty = torch.randint(-sw, sw + 1, (b, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, device=x.device),
        torch.arange(h, device=x.device),
        torch.arange(w, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    xp = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return xp.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2).contiguous()


def _rand_cutout(x, ratio=0.5):
    b, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ox = torch.randint(0, h + (1 - ch % 2), (b, 1, 1), device=x.device)
    oy = torch.randint(0, w + (1 - cw % 2), (b, 1, 1), device=x.device)
    gb, gx, gy = torch.meshgrid(
        torch.arange(b, device=x.device),
        torch.arange(ch, device=x.device),
        torch.arange(cw, device=x.device),
        indexing="ij",
    )
    gx = torch.clamp(gx + ox - ch // 2, min=0, max=h - 1)
    gy = torch.clamp(gy + oy - cw // 2, min=0, max=w - 1)
    mask = torch.ones(b, h, w, dtype=x.dtype, device=x.device)
    mask[gb.flatten(), gx.flatten(), gy.flatten()] = 0
    return x * mask.unsqueeze(1)


def diff_augment(x, policy="color,translation,cutout"):
    fns = {
        "color": [_rand_brightness, _rand_saturation, _rand_contrast],
        "translation": [_rand_translation],
        "cutout": [_rand_cutout],
    }
    for p in policy.split(","):
        for f in fns.get(p.strip(), []):
            x = f(x)
    return x.contiguous()


# --------------------------------------------------------------------------- #
# DCGAN (v1 control)
# --------------------------------------------------------------------------- #
class DCGANGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3, num_classes=5):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, nz)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        z = (noise * self.label_emb(labels)).unsqueeze(2).unsqueeze(3)
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, num_classes=5, img_size=128):
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        self.main = nn.Sequential(
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
        )

    def forward(self, img, labels):
        c = self.label_emb(labels).view(-1, 1, self.img_size, self.img_size)
        return self.main(torch.cat([img, c], 1)).view(-1)


# --------------------------------------------------------------------------- #
# FastGAN
# --------------------------------------------------------------------------- #
class SLEBlock(nn.Module):
    """Skip-layer excitation: gates a high-res feature map with a low-res one."""

    def __init__(self, c_low, c_high):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(c_low, c_high, 4, 1, 0, bias=False),
            nn.SiLU(),
            nn.Conv2d(c_high, c_high, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, low, high):
        return high * self.main(low)


def _up(cin, cout):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(cin, cout * 2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(cout * 2),
        nn.GLU(dim=1),
    )


class FastGANGenerator(nn.Module):
    def __init__(self, nz=256, ngf=64, nc=3, num_classes=5, img_size=128):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, nz)
        self.img_size = img_size
        self.init = nn.Sequential(
            nn.ConvTranspose2d(nz * 2, ngf * 16 * 2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16 * 2),
            nn.GLU(dim=1),
        )
        self.up8 = _up(ngf * 16, ngf * 8)
        self.up16 = _up(ngf * 8, ngf * 4)
        self.up32 = _up(ngf * 4, ngf * 2)
        self.up64 = _up(ngf * 2, ngf * 2)
        self.up128 = _up(ngf * 2, ngf)
        self.sle_8_64 = SLEBlock(ngf * 8, ngf * 2)
        self.sle_16_128 = SLEBlock(ngf * 4, ngf)
        self.out = nn.Sequential(nn.Conv2d(ngf, nc, 3, 1, 1, bias=False), nn.Tanh())

    def forward(self, z, labels):
        x = torch.cat([z, self.label_emb(labels)], 1).unsqueeze(2).unsqueeze(3)
        f4 = self.init(x)
        f8 = self.up8(f4)
        f16 = self.up16(f8)
        f32 = self.up32(f16)
        f64 = self.sle_8_64(f8, self.up64(f32))
        f128 = self.sle_16_128(f16, self.up128(f64))
        return self.out(f128)


def _down(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, 2, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, True),
    )


class FastGANDiscriminator(nn.Module):
    """Hinge-loss discriminator with a self-supervised reconstruction branch."""

    def __init__(self, ndf=64, nc=3, num_classes=5, img_size=128):
        super().__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        self.stem = nn.Sequential(
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True)
        )
        self.d32 = _down(ndf, ndf * 2)
        self.d16 = _down(ndf * 2, ndf * 4)
        self.d8 = _down(ndf * 4, ndf * 8)
        self.logit = nn.Conv2d(ndf * 8, 1, 8, 1, 0, bias=False)
        # reconstruct a 32x32 thumbnail from the 8x8 feature map -- the
        # self-supervision that keeps a small-data discriminator honest
        self.decoder = nn.Sequential(
            _up(ndf * 8, ndf * 4), _up(ndf * 4, ndf * 2),
            nn.Conv2d(ndf * 2, nc, 3, 1, 1, bias=False), nn.Tanh(),
        )

    def forward(self, img, labels, recon=False):
        c = self.label_emb(labels).view(-1, 1, self.img_size, self.img_size)
        h = self.stem(torch.cat([img, c], 1))
        h = self.d8(self.d16(self.d32(h)))
        logit = self.logit(h).view(-1)
        if recon:
            return logit, self.decoder(h)
        return logit


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_gan(loader, num_classes: int, cfg, variant: str = "fastgan", epochs: int = 300, cost=None):
    device = cfg.device
    res = cfg.ddpm_res  # same resolution as the pixel DDPMs, for a fair comparison

    if variant == "dcgan":
        nz = 100
        netG = DCGANGenerator(nz=nz, num_classes=num_classes).to(device)
        netD = DCGANDiscriminator(num_classes=num_classes, img_size=res).to(device)
        optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
        use_aug, use_recon = False, False
    else:
        nz = 256
        netG = FastGANGenerator(nz=nz, num_classes=num_classes, img_size=res).to(device)
        netD = FastGANDiscriminator(num_classes=num_classes, img_size=res).to(device)
        optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))
        use_aug, use_recon = True, True

    g_hist, d_hist = [], []
    steps = 0
    pbar = tqdm(range(epochs), desc=f"GAN[{variant}]")
    for _ in pbar:
        for images, labels in loader:
            real = images.to(device)
            lab = labels.to(device)
            b = real.size(0)
            z = torch.randn(b, nz, device=device)
            fake = netG(z, lab)

            r_in = diff_augment(real) if use_aug else real
            f_in = diff_augment(fake.detach()) if use_aug else fake.detach()

            # --- discriminator ---
            optD.zero_grad(set_to_none=True)
            if variant == "dcgan":
                lossD = F.binary_cross_entropy_with_logits(
                    netD(r_in, lab), torch.full((b,), 0.9, device=device)
                ) + F.binary_cross_entropy_with_logits(
                    netD(f_in, lab), torch.zeros(b, device=device)
                )
            else:
                if use_recon:
                    d_real, rec = netD(r_in, lab, recon=True)
                    target = F.interpolate(r_in, size=rec.shape[-1], mode="area")
                    lossRec = F.mse_loss(rec, target)
                else:
                    d_real, lossRec = netD(r_in, lab), 0.0
                d_fake = netD(f_in, lab)
                lossD = (
                    F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean() + lossRec
                )
            lossD.backward()
            optD.step()

            # --- generator ---
            optG.zero_grad(set_to_none=True)
            f_in2 = diff_augment(fake) if use_aug else fake
            if variant == "dcgan":
                lossG = F.binary_cross_entropy_with_logits(
                    netD(f_in2, lab), torch.ones(b, device=device)
                )
            else:
                lossG = -netD(f_in2, lab).mean()
            lossG.backward()
            optG.step()
            steps += 1

        g_hist.append(float(lossG.item()))
        d_hist.append(float(lossD.item()))
        pbar.set_postfix(G=f"{g_hist[-1]:.3f}", D=f"{d_hist[-1]:.3f}")

    if cost is not None:
        from ..bench import count_params

        tg, _ = count_params(netG)
        td, _ = count_params(netD)
        cost.trainable_params_m, cost.total_params_m = tg + td, tg + td
        cost.steps = steps

    return netG.eval(), nz, {"g": g_hist, "d": d_hist}


@torch.no_grad()
def generate_to_dir(
    netG, nz: int, class_names: Sequence[str], out_dir: Path, n_per_class: int, cfg,
    tag: str = "gan", seed: int = 0, batch_size: int = 32, skip_classes: Sequence[int] = (),
):
    out_dir = Path(out_dir)
    for ci, name in enumerate(class_names):
        if ci in skip_classes:
            continue
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        have = len(list(d.glob("*.png")))
        gen = torch.Generator(device=cfg.device).manual_seed(seed + 1000 * ci)
        while have < n_per_class:
            b = min(batch_size, n_per_class - have)
            z = torch.randn(b, nz, device=cfg.device, generator=gen)
            lab = torch.full((b,), ci, dtype=torch.long, device=cfg.device)
            imgs = (netG(z, lab).clamp(-1, 1) + 1) / 2
            for i in range(b):
                save_image(imgs[i], d / f"{tag}_{have + i:05d}.png")
            have += b
    return out_dir
