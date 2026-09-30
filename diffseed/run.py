"""Experiment orchestration.

Resumable by construction: every stage writes its artefacts to
``runs/<tier>/...`` and skips itself if they already exist. A spot instance can
die halfway through the grid and the next invocation picks up where it stopped.

Stage order:
    1. splits            -- deterministic, shared by every condition
    2. generators        -- train each method at each scarcity level, sample to disk
    3. quality metrics   -- FID/KID/CMMD/PRDC/LPIPS + the sample-size bias curve
    4. downstream        -- classifier ablation, repeated over seeds
    5. analysis          -- paired statistics, crossover fit, figures
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .bench import CostLedger, Stopwatch
from .config import NEGATIVE_PROMPT, Config, prompts_for_classes
from .data import (
    IndexedImageDataset,
    MixedDataset,
    classifier_transform,
    diffusion_transform,
    load_seed_data,
    make_loader,
    make_splits,
    save_manifest,
)

ALL_METHODS = (
    "trad", "dcgan", "fastgan", "ddpm_v1", "ddpm_fixed", "ddpm_ft",
    "sd_lora", "sd_lora_sdedit", "sd_turbo_lora",
)
DIFFUSION_METHODS = ("ddpm_v1", "ddpm_fixed", "ddpm_ft")
SD_METHODS = ("sd_lora", "sd_lora_sdedit", "sd_turbo_lora")
GAN_METHODS = ("dcgan", "fastgan")

# Selection ablation. The study's only positive downstream result comes from
# quality-aware selection, and a single combined number cannot say which of the
# three criteria produces it or how sensitive it is to how much is discarded.
# Each row is (tag, confidence, typicality, novelty, keep_fraction); the full
# three-criterion setting at the default keep fraction is the untagged
# "+filtered" condition and is not repeated here.
FILTER_ABLATION = (
    ("conf",    True,  False, False, 0.6),   # confidence alone
    ("conftyp", True,  True,  False, 0.6),   # + typicality
    ("confnov", True,  False, True,  0.6),   # + novelty
    ("typnov",  False, True,  True,  0.6),   # confidence removed
    ("k20",     True,  True,  True,  0.2),   # keep-fraction sweep, all criteria
    ("k40",     True,  True,  True,  0.4),
    ("k80",     True,  True,  True,  0.8),
)



def set_seed(seed: int, strict: bool = False):
    """Seed everything; ``strict`` additionally constrains kernel selection.

    Delegates to ``repro.enable_determinism`` so the resulting state is recorded
    in the run's provenance file rather than being an invisible default.
    """
    from .repro import enable_determinism

    return enable_determinism(seed, strict=strict)


class Experiment:
    def __init__(self, cfg: Config):
        self.cfg = cfg.apply_tier().fit_to_gpu()
        self.determinism = set_seed(self.cfg.seed, strict=getattr(cfg, "strict_repro", False))
        self.root = self.cfg.dirs(self.cfg.tier)
        self.ledger = CostLedger(self.root / "costs.json")
        self.data = load_seed_data(self.cfg.data_root)
        self.class_names = self.data.class_names
        self.n_classes = self.data.num_classes
        save_manifest(
            self.root / "config.json",
            {"config": self.cfg.to_dict(), "classes": self.class_names,
             "counts": self.data.counts(), "image_root": str(self.data.root)},
        )
        # Provenance: code hash, dataset digest, environment and determinism
        # state, captured at run time because two of the three cannot be
        # recovered afterwards.
        try:
            from .repro import write_provenance

            write_provenance(self.root / "provenance.json", self.cfg,
                             Path(__file__).parent, self.determinism,
                             self.cfg.data_root)
        except Exception as e:
            print(f"provenance capture failed: {type(e).__name__}: {e}")

        self._splits: dict[tuple, object] = {}
        # soybean classes use the hand-written prompts; anything else (the
        # maize cross-species arm) gets one synthesised from the class name
        self.class_prompts = prompts_for_classes(self.class_names)

    # ------------------------------------------------------------------ #
    def splits(self, scarcity: int, draw: int = 0):
        """Splits for a scarcity level, optionally a different scarce draw.

        ``draw`` selects *which* subset of the training pool plays the role of
        the scarce data. The test split is unaffected, so results across draws
        are measured on the same held-out images and remain comparable.

        This exists because confidence intervals over classifier seeds alone hold
        the scarce set fixed, and in a scarcity study the identity of the hundred
        images is a larger source of variation than the classifier's
        initialisation.
        """
        key = (scarcity, draw)
        if key not in self._splits:
            self._splits[key] = make_splits(
                self.data, scarcity, seed=self.cfg.seed,
                test_frac=self.cfg.test_frac, val_frac=self.cfg.val_frac,
                draw=draw,
            )
        return self._splits[key]

    def synth_dir(self, method: str, scarcity: int) -> Path:
        return self.root / "synthetic" / f"n{scarcity}" / method

    def real_ref_dir(self, scarcity: int) -> Path:
        """Real images written to disk once, for FID against a fixed reference.

        The reference is the *training* pool for that class, not the test set,
        so quality is measured against what the generator was asked to model.

        Images are kept at native resolution rather than downsampled to the
        DDPM's 128 px. Downsampling the reference first would blur it on the way
        back up to Inception's 299 px input while leaving 512 px Stable
        Diffusion output sharp, which inflates the gap between the two
        generators for a reason that has nothing to do with either one. Every
        image, real or synthetic, now takes exactly one resize to 299.
        """
        d = self.root / "real_reference" / f"n{scarcity}"
        if d.exists() and any(d.rglob("*.png")):
            return d
        from PIL import Image

        for ci, name in enumerate(self.class_names):
            cd = d / name
            cd.mkdir(parents=True, exist_ok=True)
            idx = [i for i in self.splits(scarcity).train if self.data.targets[i] == ci]
            for k, i in enumerate(idx[: self.cfg.n_eval_synth]):
                p, _ = self.data.dataset.samples[i]
                img = Image.open(p).convert("RGB")
                if max(img.size) > 512:
                    img.thumbnail((512, 512), Image.BICUBIC)
                img.save(cd / f"real_{k:05d}.png")
        return d

    def _pooled_images(self, root: Path, per_class: int) -> list[Path]:
        """Class-balanced pooled sample.

        A flat sorted listing would fill the quota from the alphabetically first
        classes and leave the rest out of the pooled reference entirely.
        """
        from .metrics import list_images

        out: list[Path] = []
        for name in self.class_names:
            sub = Path(root) / name
            if sub.is_dir():
                out.extend(list_images(sub, per_class))
        return out or list_images(root, per_class * len(self.class_names))

    def scarce_paths_by_class(self, scarcity: int) -> dict[int, list[str]]:
        sp = self.splits(scarcity)
        out: dict[int, list[str]] = {}
        for i in sp.scarce:
            p, y = self.data.dataset.samples[i]
            out.setdefault(int(y), []).append(p)
        return out

    # ------------------------------------------------------------------ #
    # stage 2 -- generators
    # ------------------------------------------------------------------ #
    def run_generator(self, method: str, scarcity: int, force: bool = False) -> Path:
        cfg = self.cfg
        out = self.synth_dir(method, scarcity)
        skip = (self.splits(scarcity).majority_class,)

        if method == "trad":
            return out  # handled by the classifier transform, nothing to generate

        done = out / ".done"
        if done.exists() and not force:
            print(f"[skip] {method} @ n={scarcity} already generated")
            return out

        sp = self.splits(scarcity)
        print(f"[run ] {method} @ n={scarcity} ({len(sp.scarce)} training images)")

        with Stopwatch(f"gen::{method}::n{scarcity}") as cost:
            cost.extra.update(method=method, scarcity=scarcity, stage="generator")

            if method in DIFFUSION_METHODS:
                from .generators import pixel_ddpm as P

                variant = {"ddpm_v1": "v1", "ddpm_fixed": "fixed", "ddpm_ft": "ft"}[method]
                ds = IndexedImageDataset(
                    self.data.dataset, sp.scarce, diffusion_transform(cfg.ddpm_res, True)
                )
                loader = make_loader(ds, cfg.ddpm_batch, True, cfg.num_workers, drop_last=True)
                model, sched, losses = P.train_pixel_ddpm(
                    loader, self.n_classes, cfg, variant=variant, cost=cost
                )
                # Checkpoint before sampling. Sampling is where schedulers and
                # guidance go wrong, and without this every such fix costs a
                # full retrain to re-test.
                ckpt = self.root / "checkpoints" / f"{method}_n{scarcity}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), ckpt)
                P.generate_to_dir(
                    model, sched, self.class_names, out, cfg.n_synth_per_class, cfg,
                    variant=variant, seed=cfg.seed, skip_classes=skip,
                )
                (out / "losses.json").write_text(json.dumps(losses), encoding="utf-8")
                del model
                torch.cuda.empty_cache()

            elif method in GAN_METHODS:
                from .generators import gans as G

                ds = IndexedImageDataset(
                    self.data.dataset, sp.scarce, diffusion_transform(cfg.ddpm_res, True)
                )
                loader = make_loader(ds, cfg.ddpm_batch, True, cfg.num_workers, drop_last=True)
                epochs = max(1, cfg.ddpm_epochs // 2)
                netG, nz, hist = G.train_gan(
                    loader, self.n_classes, cfg, variant=method, epochs=epochs, cost=cost
                )
                ckpt = self.root / "checkpoints" / f"{method}_n{scarcity}.pt"
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"G": netG.state_dict(), "nz": nz}, ckpt)
                G.generate_to_dir(
                    netG, nz, self.class_names, out, cfg.n_synth_per_class, cfg,
                    tag=method, seed=cfg.seed, skip_classes=skip,
                )
                (out / "losses.json").write_text(json.dumps(hist), encoding="utf-8")
                del netG
                torch.cuda.empty_cache()

            elif method in SD_METHODS:
                from .generators import sd_lora as S

                turbo = method == "sd_turbo_lora"
                model_id = cfg.sd_turbo_model if turbo else cfg.sd_model
                paths = [self.data.dataset.samples[i] for i in sp.scarce]

                lat_path = self.root / "cache" / f"latents_n{scarcity}_{'turbo' if turbo else 'sd15'}.pt"
                if lat_path.exists():
                    blob = torch.load(lat_path, map_location="cpu", weights_only=False)
                    latents, labels = blob["latents"], blob["labels"]
                    text_embeds, null_embed = blob["text"], blob["null"]
                    print(f"       reusing cached latents: {tuple(latents.shape)}")
                else:
                    latents, labels, text_embeds, null_embed = S.cache_latents_and_text(
                        paths, self.class_names, self.class_prompts, cfg, model_id=model_id
                    )
                    lat_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"latents": latents, "labels": labels,
                         "text": text_embeds, "null": null_embed}, lat_path
                    )

                # SDEdit is a different *sampling* procedure over the same
                # adapter, so it reuses sd_lora's weights rather than paying to
                # train an identical LoRA a second time.
                share = "sd_lora" if method == "sd_lora_sdedit" else method
                lora_path = self.root / "loras" / f"{share}_n{scarcity}.pt"
                if lora_path.exists():
                    print(f"       reusing LoRA adapter {lora_path.name}")
                    unet, losses = S.load_lora_unet(lora_path, cfg, model_id=model_id), []
                else:
                    unet, losses = S.train_sd_lora(
                        latents, labels, text_embeds, null_embed, cfg,
                        model_id=model_id,
                        steps=cfg.sd_steps // (2 if turbo else 1),
                        cost=cost,
                    )
                    S.save_lora(unet, lora_path)
                (out / "losses.json").parent.mkdir(parents=True, exist_ok=True)
                (out / "losses.json").write_text(json.dumps(losses), encoding="utf-8")

                if method == "sd_lora_sdedit":
                    S.generate_sdedit_to_dir(
                        unet, self.scarce_paths_by_class(scarcity), self.class_names,
                        self.class_prompts, out, cfg.n_synth_per_class, cfg,
                        tag=method, seed=cfg.seed, skip_classes=skip,
                    )
                else:
                    pipe = S.build_pipeline(unet, cfg, model_id=model_id, turbo=turbo)
                    S.generate_to_dir(
                        pipe, self.class_names, self.class_prompts, out, cfg.n_synth_per_class,
                        cfg, negative_prompt="" if turbo else NEGATIVE_PROMPT,
                        tag=method, seed=cfg.seed, skip_classes=skip,
                        steps=4 if turbo else cfg.sd_sample_steps,
                        guidance=0.0 if turbo else cfg.sd_guidance,
                    )
                    del pipe
                del unet, latents
                torch.cuda.empty_cache()
            else:
                raise ValueError(f"unknown method {method!r}")

        self.ledger.add(cost)
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text("ok", encoding="utf-8")
        return out

    # ------------------------------------------------------------------ #
    # stage 3 -- quality
    # ------------------------------------------------------------------ #
    def evaluate_quality(self, methods: Sequence[str], scarcity: int,
                         with_clip: bool = True, with_lpips: bool = True) -> "pd.DataFrame":
        import pandas as pd

        from . import metrics as M

        cache = self.root / "quality" / f"n{scarcity}.csv"
        cache.parent.mkdir(parents=True, exist_ok=True)

        inc = M.InceptionFeatures(self.cfg.device)
        clip = M.ClipFeatures(self.cfg.clip_model, self.cfg.device) if with_clip else None
        lp = M.Lpips(self.cfg.device) if with_lpips else None

        real_root = self.real_ref_dir(scarcity)
        skip = {self.splits(scarcity).majority_class}
        rows, curves = [], {}
        feats_by_source: dict[str, "np.ndarray"] = {}

        for m in methods:
            if m == "trad":
                continue
            sdir = self.synth_dir(m, scarcity)
            if not sdir.exists():
                continue
            # per class
            for ci, cname in enumerate(self.class_names):
                if ci in skip or not (sdir / cname).exists():
                    continue
                rep = M.evaluate_generator(
                    real_root / cname, sdir / cname, m, cname, inc, clip, lp,
                    n_max=self.cfg.n_eval_synth, prdc_k=self.cfg.prdc_k,
                )
                rows.append(rep.as_dict())
            # pooled, plus the sample-size bias curve
            per = max(1, self.cfg.n_eval_synth * 2 // max(1, self.n_classes))
            pooled_real = self._pooled_images(real_root, per)
            pooled_fake = self._pooled_images(sdir, per)
            fr = inc(pooled_real)
            ff = inc(pooled_fake)
            cap = 300
            feats_by_source.setdefault("real", fr[:cap])
            feats_by_source[m] = ff[:cap]
            curves[m] = M.fid_vs_n(fr, ff, self.cfg.fid_n_curve)
            pooled = {
                "method": m, "cls": "ALL", "n_real": len(fr), "n_fake": len(ff),
                "fid": M.frechet_distance(fr, ff), "fid_inf": M.fid_infinity(fr, ff),
                **dict(zip(("kid_mean", "kid_std"), M.kernel_distance(fr, ff))),
                **M.prdc(fr, ff, self.cfg.prdc_k),
            }
            # The pooled row is the one the manuscript reports, and it was built
            # from Inception features only -- so CMMD and CLIP-FID were present
            # per class and empty exactly where they are cited, rendering as ??.
            # CMMD matters here more than anywhere else: it is the metric
            # introduced to be usable at the sample sizes where FID is not, which
            # is the measurement argument this study makes.
            if clip is not None:
                cr, cf = clip(pooled_real), clip(pooled_fake)
                pooled["cmmd"] = M.cmmd(cr, cf)
                pooled["clip_fid"] = M.frechet_distance(cr, cf)
            rows.append(pooled)

        # memorisation audit: nearest real neighbour for each synthetic image.
        # A generator trained on ~100 images per class can win on FID simply by
        # reproducing them, so the closest matches have to be looked at.
        if lp is not None:
            pairs: dict[str, list] = {}
            for m in methods:
                if m == "trad":
                    continue
                sdir = self.synth_dir(m, scarcity)
                cname = next((c for i, c in enumerate(self.class_names) if i not in skip), None)
                if cname is None or not (sdir / cname).exists():
                    continue
                fake = M.list_images(sdir / cname, 60)
                real = M.list_images(real_root / cname, 200)
                if not fake or not real:
                    continue
                _, dists = lp.knn_lpips(fake, real, max_fake=60)
                am = getattr(lp, "last_argmins", list(range(len(dists))))
                pairs[m] = [
                    [str(fake[i]), str(real[am[i]]), float(d)] for i, d in enumerate(dists)
                ]
            (self.root / "quality" / f"memorisation_n{scarcity}.json").write_text(
                json.dumps(pairs, indent=2), encoding="utf-8"
            )

        # feature-space embedding: real vs each generator, in one 2-D map.
        # v1 had this (its t-SNE figure) and the v2 rewrite dropped it. It is
        # the one view that shows *where* a generator's distribution sits
        # relative to the real one rather than summarising the gap as a scalar.
        if feats_by_source:
            from sklearn.manifold import TSNE

            names, mats = zip(*feats_by_source.items())
            counts = [len(m) for m in mats]
            allf = np.vstack(mats)
            per = min(400, len(allf))
            rng = np.random.default_rng(self.cfg.seed)
            try:
                emb = TSNE(n_components=2, perplexity=min(30, max(5, len(allf) // 12)),
                           random_state=self.cfg.seed, init="pca").fit_transform(allf)
                src = np.concatenate([[n] * c for n, c in zip(names, counts)])
                np.savez_compressed(
                    self.root / "quality" / f"embedding_n{scarcity}.npz",
                    emb=emb.astype(np.float32), source=src,
                )
                print(f"  embedding saved: {len(allf)} points, {len(names)} sources")
            except Exception as e:
                print(f"  embedding skipped: {type(e).__name__}: {e}")

        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
        (self.root / "quality" / f"fid_curves_n{scarcity}.json").write_text(
            json.dumps(curves, indent=2), encoding="utf-8"
        )
        return df

    # ------------------------------------------------------------------ #
    # stage 4 -- downstream
    # ------------------------------------------------------------------ #
    def build_condition(self, method: str, scarcity: int, synth_per_class: int,
                        draw: int = 0):
        cfg = self.cfg
        sp = self.splits(scarcity, draw)
        skip = (sp.majority_class,)
        base = self.data.dataset

        if method == "none":
            return MixedDataset(base, sp.scarce, None, self.class_names,
                                classifier_transform(cfg.clf_res, True), 0)
        if method == "trad":
            return MixedDataset(base, sp.scarce, None, self.class_names,
                                classifier_transform(cfg.clf_res, True, heavy=True), 0)
        if method in ("dup", "dupshuf"):
            # Information-free size-matched controls.
            #
            # Every synthetic arm adds the same number of images to the same
            # classes, so at fixed epochs it also gains gradient updates,
            # validation-selection opportunities and a shifted label prior. None
            # of those is information about seeds. These two arms add REAL
            # images already in the training set, matching image count, step
            # count, epoch count, schedule length, selection count and class
            # balance exactly, and differing from a synthetic arm only in what
            # the added pixels contain:
            #   dup     - duplicates, correct labels: adds nothing at all
            #   dupshuf - duplicates, permuted labels: adds label noise
            # delta(dup) is therefore the budget component measured directly,
            # with no arithmetic and no epoch surgery, and
            # delta(synth) - delta(dup) is what the generated pixels contribute.
            rng = np.random.default_rng(cfg.seed)
            by_class: dict[int, list[int]] = {}
            for i in sp.scarce:
                by_class.setdefault(int(self.data.targets[i]), []).append(i)
            picks: list[tuple[str, int]] = []
            for ci in range(self.n_classes):
                if ci in skip or not by_class.get(ci):
                    continue
                chosen = rng.choice(by_class[ci], size=synth_per_class, replace=True)
                picks.extend((self.data.dataset.samples[int(j)][0], ci) for j in chosen)
            if method == "dupshuf":
                labels = [lab for _, lab in picks]
                rng.shuffle(labels)
                picks = [(path, lab) for (path, _), lab in zip(picks, labels)]
            ds = MixedDataset(base, sp.scarce, None, self.class_names,
                              classifier_transform(cfg.clf_res, True), 0)
            ds.synth = picks
            return ds
        if method in ("weighted", "oversample"):
            # Standard imbalance remedies. Same data as "none"; the difference
            # is in the loss weighting or in the sampler, applied in
            # train_classifier via the balance argument.
            return MixedDataset(base, sp.scarce, None, self.class_names,
                                classifier_transform(cfg.clf_res, True), 0)
        if method == "oracle":
            return MixedDataset(base, sp.train, None, self.class_names,
                                classifier_transform(cfg.clf_res, True), 0)
        allowed = None
        base_method = method
        if "+filtered" in method:
            from .filtering import load_manifest

            # "m+filtered" uses the default selection; "m+filtered:tag" uses an
            # ablation variant, so several selections can be compared in one run.
            head, _, ftag = method.partition("+filtered")
            base_method = head
            ftag = ftag.lstrip(":")
            man_name = f"{base_method}.json" if not ftag else f"{base_method}__{ftag}.json"
            if draw:
                man_name = man_name.replace(".json", f"__d{draw}.json")
            allowed = load_manifest(
                self.root / "filtered" / f"n{scarcity}" / man_name,
                self.class_names)
            if not allowed:
                raise FileNotFoundError(
                    f"no filter manifest for {base_method} @ n={scarcity}; run --stage filter")
        return MixedDataset(
            base, sp.scarce, self.synth_dir(base_method, scarcity), self.class_names,
            classifier_transform(cfg.clf_res, True), synth_per_class,
            skip_classes=skip, synth_seed=cfg.seed, allowed=allowed,
        )

    # ------------------------------------------------------------------ #
    # stage 4b -- can a classifier tell real from synthetic?
    # ------------------------------------------------------------------ #
    def run_detectability(self, methods: Sequence[str], scarcity: int) -> "pd.DataFrame":
        """Train a detector to separate real from synthetic, per generator.

        v1 reported a fooling rate of 0.000 (perfect detection), but drew its
        real images with ``sorted(glob)[:40]`` over the *whole* dataset, so the
        same files could sit in both the detector's test set and the generator's
        training set. Here real images come from the generator's own training
        pool and the detector is scored on a held-out split of both sides, with
        AUC alongside accuracy.
        """
        import pandas as pd

        from .classify import detectability
        from .metrics import list_images

        cache = self.root / "detectability" / f"n{scarcity}.csv"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            return pd.read_csv(cache)

        real_root = self.real_ref_dir(scarcity)
        skip = {self.splits(scarcity).majority_class}
        rows = []
        for m in methods:
            if m == "trad":
                continue
            sdir = self.synth_dir(m, scarcity)
            if not sdir.exists():
                continue
            for ci, cname in enumerate(self.class_names):
                if ci in skip or not (sdir / cname).exists():
                    continue
                real = [str(p) for p in list_images(real_root / cname, 300)]
                fake = [str(p) for p in list_images(sdir / cname, 300)]
                if len(real) < 40 or len(fake) < 40:
                    continue
                with Stopwatch(f"detect::{m}::{cname}::n{scarcity}") as cost:
                    cost.extra.update(method=m, scarcity=scarcity, stage="detect")
                    res = detectability(real, fake, self.cfg, seed=self.cfg.seed)
                self.ledger.add(cost)
                rows.append({"method": m, "cls": cname, "scarcity": scarcity, **res})
                print(f"  detect {m:16s} {cname:22s} acc={res['detector_acc']:.3f} "
                      f"auc={res['detector_auc']:.3f} fooling={res['fooling_rate']:.3f}")
        df = pd.DataFrame(rows)
        if not df.empty:
            df.to_csv(cache, index=False)
        return df

    # ------------------------------------------------------------------ #
    # stage 4c -- how much synthetic data is optimal?
    # ------------------------------------------------------------------ #
    def run_ratio_study(self, methods: Sequence[str], scarcity: int,
                        models: Sequence[str] | None = None) -> list:
        """Sweep synthetic-images-per-class and watch downstream performance.

        The practically actionable question for a lab: more synthetic data is
        not monotonically better, and the turning point is what a practitioner
        needs. v1 ran this sweep; v2 had the configuration but never called it.
        """
        from .classify import train_classifier

        cfg = self.cfg
        models = models or (cfg.clf_headline,)
        sp = self.splits(scarcity)

        val_loader = make_loader(
            IndexedImageDataset(self.data.dataset, sp.val, classifier_transform(cfg.clf_res, False)),
            cfg.clf_batch, False, cfg.num_workers,
        )
        test_loader = make_loader(
            IndexedImageDataset(self.data.dataset, sp.test, classifier_transform(cfg.clf_res, False)),
            cfg.clf_batch, False, cfg.num_workers,
        )

        cache = self.root / "ratio" / f"n{scarcity}.jsonl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        seen, results = set(), []
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    seen.add((r["condition"], r["n_synth_target"], r["model"], r["seed"]))
                    results.append(r)

        with cache.open("a", encoding="utf-8") as fh:
            for method in methods:
                if method == "trad" or not self.synth_dir(method, scarcity).exists():
                    continue
                for k in cfg.ratio_sweep:
                    if k > cfg.n_synth_per_class:
                        continue
                    for model_name in models:
                        for seed in cfg.classifier_seeds[:1]:
                            key = (method, k, model_name, seed)
                            if key in seen:
                                continue
                            ds = self.build_condition(method, scarcity, k) if k else \
                                self.build_condition("none", scarcity, 0)
                            res = train_classifier(
                                ds, val_loader, test_loader, self.n_classes, cfg,
                                condition=method, model_name=model_name, seed=seed,
                                scarcity=scarcity,
                            )
                            row = _result_to_dict(res)
                            row["n_synth_target"] = k
                            fh.write(json.dumps(row) + "\n")
                            fh.flush()
                            results.append(row)
                            print(f"  ratio {method:16s} synth/class={k:4d} "
                                  f"acc={res.accuracy:.4f} macroF1={res.macro_f1:.4f}")
        return results

    # ------------------------------------------------------------------ #
    # stage 3b -- quality-aware selection of synthetic images
    # ------------------------------------------------------------------ #
    def run_filter(self, methods: Sequence[str], scarcity: int, force: bool = False,
                   tag: str = "", keep_frac: float | None = None,
                   use_confidence: bool = True, use_typicality: bool = True,
                   use_novelty: bool = True, draw: int = 0):
        """Score and select synthetic images before they reach the classifier.

        Every arm so far adds all generated images. The recent
        synthetic-augmentation literature is consistent that this is the setting
        in which augmentation fails: a generator fitted to ~100 images produces a
        tail of off-distribution, class-ambiguous and near-duplicate samples, and
        adding them unfiltered injects label noise. Running both filtered and
        unfiltered conditions makes that an experimental variable rather than an
        assumption.
        """
        from .classify import train_classifier
        from .filtering import select_synthetic, write_manifest

        cfg = self.cfg
        sp = self.splits(scarcity, draw)
        out_dir = self.root / "filtered" / f"n{scarcity}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # scoring classifier: trained on the REAL scarce data only, so selection
        # never sees the test split
        ck = out_dir / ("scoring_clf.pt" if not draw else f"scoring_clf_d{draw}.pt")
        scorer = None
        if ck.exists() and not force:
            from .classify import build_classifier

            scorer = build_classifier(cfg.clf_headline, self.n_classes, cfg)
            scorer.load_state_dict(torch.load(ck, map_location=cfg.device, weights_only=True))
            scorer.eval()
        else:
            val_loader = make_loader(
                IndexedImageDataset(self.data.dataset, sp.val,
                                    classifier_transform(cfg.clf_res, False)),
                cfg.clf_batch, False, cfg.num_workers)
            ds = MixedDataset(self.data.dataset, sp.scarce, None, self.class_names,
                              classifier_transform(cfg.clf_res, True), 0)
            # The whole selection rule rests on this classifier's confidence, so
            # a collapsed one does not degrade the filter gracefully -- it
            # rejects every candidate and the "+filtered" condition silently
            # becomes the baseline with no synthetic data at all. That happened
            # on soybean: the scorer reached macro-F1 0.241 against a 0.200
            # chance level for five classes, and the filter kept 0 of 1600
            # images per generator while reporting a mean dropped confidence of
            # 0.12-0.20, i.e. chance. Retry on a different seed, and refuse to
            # proceed rather than emit a manifest built from noise.
            chance = 1.0 / max(1, self.n_classes)
            floor = float(os.environ.get("DIFFSEED_SCORER_FLOOR", 2.0)) * chance
            res = None
            for attempt, sd in enumerate((cfg.seed, cfg.seed + 1, cfg.seed + 2)):
                res = train_classifier(ds, val_loader, val_loader, self.n_classes,
                                       cfg, condition="scorer",
                                       model_name=cfg.clf_headline, seed=sd,
                                       scarcity=scarcity, save_to=ck)
                if res.best_val_f1 >= floor:
                    if attempt:
                        print(f"  scoring classifier: recovered on seed {sd}")
                    break
                print(f"  scoring classifier collapsed on seed {sd}: "
                      f"val macro-F1 {res.best_val_f1:.4f} < {floor:.4f} "
                      f"({chance:.3f} is chance for {self.n_classes} classes)")
            else:
                # A scorer that cannot beat chance rejects every candidate, so
                # "+filtered" would silently become the scarce baseline. What to
                # do about it depends on which level we are at.
                #
                # At the headline scarcity this is fatal: the paper's lead claim
                # is the selection delta, and a manifest built from noise would
                # report "selection does nothing" when what happened is that the
                # scorer never trained. Refuse.
                #
                # At the other scarcity levels it is a RESULT, not a failure.
                # Selection bootstraps its scorer from the same scarce data the
                # study is short of, so there is a data volume below which the
                # method cannot be applied at all. On soybean (5 classes) that
                # floor sits between 50 and 100 images per class: the scorer
                # reaches 0.241 at n=25 and 0.393 at n=50 against 0.200 chance,
                # then 0.816 at n=100. Aborting the whole stage over an expected
                # negative would also throw away the healthy headline level, so
                # record the level as inapplicable and carry on.
                msg = (f"scoring classifier never exceeded {floor:.3f} val "
                       f"macro-F1 over 3 seeds (best {res.best_val_f1:.4f}, "
                       f"chance {chance:.3f} for {self.n_classes} classes)")
                if scarcity == getattr(cfg, "headline_scarcity", scarcity):
                    raise RuntimeError(
                        msg + "; selection would reject every candidate at the "
                        "HEADLINE scarcity. Refusing to write a manifest from "
                        "noise.")
                print(f"  [skip] filter @ n={scarcity}: {msg}; selection is not "
                      f"applicable at this data volume")
                with (self.root / "filtered" / "scorer_floor.csv").open("a") as fh:
                    if fh.tell() == 0:
                        fh.write("scarcity,n_classes,chance,floor,best_val_f1,seeds_tried" + chr(10))
                    fh.write(f"{scarcity},{self.n_classes},{chance:.4f},"
                             f"{floor:.4f},{res.best_val_f1:.4f},3" + chr(10))
                del res
                torch.cuda.empty_cache()
                return
            print(f"  scoring classifier: val macro-F1 {res.best_val_f1:.4f}")
            from .classify import build_classifier

            scorer = build_classifier(cfg.clf_headline, self.n_classes, cfg)
            scorer.load_state_dict(torch.load(ck, map_location=cfg.device, weights_only=True))
            scorer.eval()

        reports = []
        for m in methods:
            if m == "trad" or not (self.synth_dir(m, scarcity) / ".done").exists():
                continue
            man = out_dir / (f"{m}.json" if not tag else f"{m}__{tag}.json")
            if draw:
                man = man.with_name(man.name.replace(".json", f"__d{draw}.json"))
            if man.exists() and not force:
                print(f"[skip] filter {m} @ n={scarcity}")
                continue
            kept, reps = select_synthetic(
                self.synth_dir(m, scarcity), self.scarce_paths_by_class(scarcity),
                self.class_names, scorer, cfg,
                keep_frac=(keep_frac if keep_frac is not None
                           else getattr(cfg, "filter_keep_frac", 0.6)),
                use_confidence=use_confidence, use_typicality=use_typicality,
                use_novelty=use_novelty,
                skip_classes=(sp.majority_class,), method=m, scarcity=scarcity)
            write_manifest(man, kept, self.class_names)
            reports.extend(reps)
            tot_in = sum(r.n_candidates for r in reps)
            tot_out = sum(r.n_kept for r in reps)
            print(f"  filter {m:16s} kept {tot_out}/{tot_in} "
                  f"(conf kept {np.nanmean([r.mean_confidence_kept for r in reps]):.3f} "
                  f"vs dropped {np.nanmean([r.mean_confidence_dropped for r in reps]):.3f})")
        if reports:
            import pandas as pd

            rep_df = pd.DataFrame([r.as_dict() for r in reports])
            rep_df["tag"] = tag or "default"
            rep_name = "filter_report.csv" if not tag else f"filter_report__{tag}.csv"
            rep_df.to_csv(out_dir / rep_name, index=False)
        del scorer
        torch.cuda.empty_cache()

    def run_downstream(self, methods: Sequence[str], scarcity: int,
                       models: Sequence[str] | None = None,
                       synth_per_class: int | None = None, draw: int = 0,
                       cache_tag: str = "") -> list:
        from .classify import train_classifier

        cfg = self.cfg
        models = models or (cfg.clf_headline,)
        synth_per_class = synth_per_class or min(cfg.n_synth_per_class, 200)
        sp = self.splits(scarcity, draw)

        val_loader = make_loader(
            IndexedImageDataset(self.data.dataset, sp.val, classifier_transform(cfg.clf_res, False)),
            cfg.clf_batch, False, cfg.num_workers,
        )
        test_loader = make_loader(
            IndexedImageDataset(self.data.dataset, sp.test, classifier_transform(cfg.clf_res, False)),
            cfg.clf_batch, False, cfg.num_workers,
        )

        # A repeat on a different scarce draw is a separate experiment and is
        # cached separately, so it cannot be mistaken for extra seeds of the
        # primary one when the results are pooled.
        # ``cache_tag`` isolates a self-contained experiment in its own file.
        # Appending a retrained condition to the primary file would leave two
        # rows for the same (condition, model, seed) from different processes,
        # and the analysis averages over rows.
        stem = f"n{scarcity}" + (f"_draw{draw}" if draw else "") + cache_tag
        cache = self.root / "downstream" / f"{stem}.jsonl"
        cache.parent.mkdir(parents=True, exist_ok=True)
        seen = set()
        results = []
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                seen.add((r["condition"], r["model"], r["seed"]))
                results.append(_dict_to_result(r))

        with cache.open("a", encoding="utf-8") as fh:
            for method in methods:
                for model_name in models:
                    for seed in cfg.classifier_seeds:
                        key = (method, model_name, seed)
                        if key in seen:
                            continue
                        ds = self.build_condition(method, scarcity, synth_per_class,
                                                  draw=draw)
                        with Stopwatch(f"clf::{method}::{model_name}::s{seed}::n{scarcity}") as cost:
                            cost.extra.update(method=method, scarcity=scarcity, stage="classifier")
                            res = train_classifier(
                                ds, val_loader, test_loader, self.n_classes, cfg,
                                condition=method, model_name=model_name, seed=seed,
                                scarcity=scarcity, cost=cost,
                                balance=(method if method in
                                         ("weighted", "oversample") else None),
                            )
                        self.ledger.add(cost)
                        results.append(res)
                        fh.write(json.dumps(_result_to_dict(res)) + "\n")
                        fh.flush()
                        print(f"  {method:16s} {model_name:16s} s{seed} "
                              f"acc={res.accuracy:.4f} macroF1={res.macro_f1:.4f}")
        return results


def _result_to_dict(r):
    from dataclasses import asdict

    return asdict(r)


def _dict_to_result(d):
    from .classify import ClfResult

    return ClfResult(**d)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="DiffSeed v2 experiment runner")
    ap.add_argument("--tier", default="standard", choices=["smoke", "standard", "cross", "full"])
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-root", default="./runs")
    ap.add_argument("--stage", default="all",
                    choices=["all", "generate", "quality", "filter", "detect",
                             "downstream", "ratio", "zoo", "sampler_diag",
                             "baselines", "filter_ablation", "draw_repeat",
                             "zoo_seeds", "controls", "analyze"])
    ap.add_argument("--methods", default=None, help="comma-separated override")
    ap.add_argument("--scarcity", default=None, help="comma-separated override")
    ap.add_argument("--models", default=None, help="comma-separated classifier override")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--macro-prefix", default=None,
                    help="namespace for emitted LaTeX macros (a secondary corpus "
                         "sets e.g. DSMz so it does not overwrite the primary run)")
    ap.add_argument("--strict-repro", action="store_true",
                    help="deterministic kernels; slower, for artefact reproduction")
    args = ap.parse_args(argv)

    cfg = Config(tier=args.tier, out_root=Path(args.out_root))
    cfg.strict_repro = bool(args.strict_repro)
    if args.macro_prefix:
        cfg.macro_prefix = args.macro_prefix
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    exp = Experiment(cfg)
    cfg = exp.cfg

    levels = [int(x) for x in args.scarcity.split(",")] if args.scarcity else list(cfg.scarcity_levels)
    models = args.models.split(",") if args.models else list(cfg.clf_models)

    def methods_for(n):
        if args.methods:
            return args.methods.split(",")
        m = list(cfg.grid_methods)
        if n == cfg.headline_scarcity:
            m += [x for x in cfg.extra_methods if x not in m]
        return m

    if args.stage in ("all", "generate"):
        # One generator failing must not abort the grid. Several arms
        # (ddpm_ft's retargeted checkpoint, sd_turbo's different text encoder,
        # FastGAN's custom architecture) are structurally distinct from the
        # rest, and an unattended multi-hour sweep should lose one arm rather
        # than all of them. Failures are recorded and re-raised only in the
        # summary at the end.
        failures = []
        for n in levels:
            for m in methods_for(n):
                try:
                    exp.run_generator(m, n, force=args.force)
                except Exception as e:
                    import traceback

                    print(f"[FAIL] generator {m} @ n={n}: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    failures.append({"method": m, "scarcity": n,
                                     "error": f"{type(e).__name__}: {e}"})
                    torch.cuda.empty_cache()
        if failures:
            (exp.root / "generator_failures.json").write_text(
                json.dumps(failures, indent=2), encoding="utf-8")
            print(f"\n{len(failures)} generator(s) failed; see generator_failures.json")

    if args.stage in ("all", "quality"):
        for n in levels:
            try:
                exp.evaluate_quality(methods_for(n), n)
            except Exception as e:
                print(f"[FAIL] quality @ n={n}: {type(e).__name__}: {e}")

    if args.stage in ("all", "filter"):
        for n in levels:
            try:
                exp.run_filter(methods_for(n), n)
            except Exception as e:
                print(f"[FAIL] filter @ n={n}: {type(e).__name__}: {e}")

    if args.stage in ("all", "detect"):
        for n in levels:
            try:
                exp.run_detectability(methods_for(n), n)
            except Exception as e:
                print(f"[FAIL] detect @ n={n}: {type(e).__name__}: {e}")

    if args.stage in ("all", "downstream"):
        # headline architecture, every condition, every seed -> the CIs
        for n in levels:
            gen = [m for m in methods_for(n) if m != "trad"]
            # filtered variants run alongside the unfiltered ones, so "does
            # quality-aware selection change the sign?" is a measured contrast
            filt = [f"{m}+filtered" for m in gen
                    if (exp.root / "filtered" / f"n{n}" / f"{m}.json").exists()]
            conds = ["none", "trad"] + gen + filt + ["oracle"]
            exp.run_downstream(conds, n, models=[cfg.clf_headline])

    if args.stage in ("all", "ratio"):
        try:
            exp.run_ratio_study(
                [m for m in methods_for(cfg.headline_scarcity) if m != "trad"],
                cfg.headline_scarcity)
        except Exception as e:
            print(f"[FAIL] ratio study: {type(e).__name__}: {e}")

    if args.stage in ("all", "zoo"):
        # architecture sweep on a few conditions, one seed -> "is this a
        # ResNet artefact?" without paying for the full cross product
        zoo_models = [m for m in models if m != cfg.clf_headline]
        if zoo_models:
            saved = cfg.classifier_seeds
            cfg.classifier_seeds = (saved[0],)
            exp.run_downstream(list(cfg.zoo_conditions), cfg.headline_scarcity, models=zoo_models)
            cfg.classifier_seeds = saved

    if args.stage == "controls":
        # The decisive comparison, and the cheapest. Seeds are raised because
        # the quantity of interest is a difference of a few points against a
        # seed spread of about 1.5, and three seeds cannot resolve that: the
        # minimum detectable effect at n=3 is larger than any effect anyone
        # would report.
        # Run at EVERY scarcity level, not just the headline one. The effect
        # being explained is largest where the induced class imbalance is most
        # extreme (34.6:1 at n=25 against 4.3:1 at n=200), and adding a fixed
        # 200 images per minority class corrects the label prior by 1 + 200/n --
        # 9x at n=25, 2x at n=200. Running the controls only at n=100 tests the
        # hypothesis exactly where the effect is smallest.
        levels_c = [int(x) for x in os.environ.get(
            "DIFFSEED_CTRL_LEVELS", ",".join(str(n) for n in cfg.scarcity_levels)
        ).split(",")]
        n_seeds = int(os.environ.get("DIFFSEED_CTRL_SEEDS", "10"))
        saved = cfg.classifier_seeds
        cfg.classifier_seeds = tuple(range(n_seeds))
        conds = ["none", "dup", "dupshuf", "ddpm_fixed", "weighted", "oversample"]
        print(f"controls: {conds} at n={levels_c}, {n_seeds} seeds, own cache")
        try:
            for nlev in levels_c:
                print(f"=== controls @ n={nlev} ===")
                exp.run_downstream(conds, nlev, cache_tag="_controls")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[FAIL] controls: {type(e).__name__}: {e}")
        finally:
            cfg.classifier_seeds = saved

    if args.stage == "zoo_seeds":
        # The zoo stage runs one seed per architecture, which is enough to see a
        # pattern and not enough to trust one. The headline architecture's
        # baseline sits well below the others here, so whether the reported
        # benefit is information added or a weak baseline repaired depends on
        # numbers that currently rest on a single run each. Downstream caches on
        # (condition, model, seed), so this adds only the missing seeds.
        zoo_models = [m for m in models if m != cfg.clf_headline]
        if zoo_models:
            print(f"zoo with {len(cfg.classifier_seeds)} seeds over {zoo_models}")
            exp.run_downstream(list(cfg.zoo_conditions), cfg.headline_scarcity,
                               models=zoo_models)
        else:
            print("no non-headline models configured")

    if args.stage == "draw_repeat":
        # Repeat the headline comparison on independent draws of the scarce set.
        # Confidence intervals over classifier seeds hold the hundred images
        # fixed, and which hundred you get is the larger source of variation in a
        # scarcity study. The generator is NOT retrained per draw -- it is a
        # fixed artefact here, as it would be in deployment -- but the scoring
        # classifier, the selection and the downstream classifier all are, so
        # this measures whether the selection benefit survives a different draw.
        head = cfg.headline_scarcity
        avail = [m for m in methods_for(head)
                 if m != "trad" and (exp.synth_dir(m, head) / ".done").exists()]
        conds = ["none", "trad"] + avail + [f"{m}+filtered" for m in avail]
        n_draws = int(os.environ.get("DIFFSEED_DRAWS", "3"))
        print(f"draw repeat: {n_draws} draws over {conds} at n={head}")
        for d in range(1, n_draws + 1):
            print(f"=== scarce draw {d} ===")
            try:
                exp.run_filter(avail, head, draw=d)
                exp.run_downstream(conds, head, draw=d)
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"[FAIL] draw {d}: {type(e).__name__}: {e}")

    if args.stage in ("all", "baselines"):
        # Standard imbalance remedies, so the selection result has something to
        # beat other than doing nothing. Classifier-side only: no generation.
        head = cfg.headline_scarcity
        try:
            exp.run_downstream(["weighted", "oversample"], head)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[FAIL] baselines: {type(e).__name__}: {e}")

    if args.stage == "filter_ablation":
        head = cfg.headline_scarcity
        abl = [m for m in methods_for(head)
               if m != "trad" and (exp.synth_dir(m, head) / ".done").exists()]
        print(f"filter ablation over {abl} at n={head}: "
              f"{len(FILTER_ABLATION)} variants")
        # The baseline and the default rule are retrained inside this process.
        # Results are reproducible given identical data within a process but
        # shift across processes, by more than the effects being compared, so a
        # delta taken against a baseline trained in an earlier process is not a
        # measurement of the variant.
        print("=== in-process reference conditions ===")
        try:
            exp.run_downstream(
                ["none"] + list(abl) + [f"{m}+filtered" for m in abl],
                head, cache_tag="_ablation")
        except Exception as e:
            print(f"[FAIL] in-process reference: {type(e).__name__}: {e}")

        for tag, uc, ut, un, kf in FILTER_ABLATION:
            print(f"=== selection variant {tag} "
                  f"(conf={uc} typ={ut} nov={un} keep={kf}) ===")
            try:
                exp.run_filter(abl, head, tag=tag, keep_frac=kf,
                               use_confidence=uc, use_typicality=ut, use_novelty=un)
                exp.run_downstream([f"{m}+filtered:{tag}" for m in abl], head,
                                   cache_tag="_ablation")
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"[FAIL] selection variant {tag}: {type(e).__name__}: {e}")

    if args.stage in ("all", "sampler_diag"):
        # Cheap, and it documents why pixel-space sampling uses an ancestral
        # DDPM rather than the fast solver the literature would suggest.
        from .sampler_diag import run_sampler_diagnostic

        run_sampler_diagnostic(exp, scarcity=cfg.headline_scarcity)

    if args.stage in ("all", "analyze"):
        from .analyze import build_report

        build_report(exp)

    print(f"\nartifacts under {exp.root}")
    return exp


if __name__ == "__main__":
    main()
