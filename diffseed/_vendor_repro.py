"""Provenance capture and determinism control.

An artefact is reproducible when a reader can establish three things: what code
ran, what data it ran on, and what environment it ran in. None of those were
recorded by the original pipeline, and two of them cannot be recovered after the
fact, so they are captured at run time here.

What this module does *not* claim: bit-exact reproduction of GPU floating-point
results across different hardware. cuDNN algorithm selection, TF32 matmuls and
atomics make that unattainable in general. ``enable_determinism`` gets as far as
is achievable on one machine, and ``environment()`` records the rest so that a
divergence can be attributed rather than merely observed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def enable_determinism(seed: int, strict: bool = False) -> dict:
    """Seed every generator and constrain nondeterministic kernels.

    ``strict=True`` additionally forces deterministic algorithm selection, which
    makes runs repeatable on identical hardware at a measurable cost: cuDNN can
    no longer autotune convolutions, and some operations have no deterministic
    implementation and will raise instead of silently varying. That trade is
    right for an artefact run and wrong for the sweep that produced the results,
    so it is opt-in and its state is recorded either way.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    state = {"seed": seed, "strict": bool(strict)}
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            state["deterministic_algorithms"] = True
        except Exception as e:  # pragma: no cover
            state["deterministic_algorithms"] = f"unavailable: {e}"
    else:
        # autotuning is a real speedup on a fixed input size and is what the
        # reported runs used; recorded so the choice is not invisible
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        state["deterministic_algorithms"] = False

    state.update(
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
    )
    return state


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def _pkg_versions(names) -> dict:
    import importlib.metadata as md

    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except Exception:
            out[n] = None
    return out


TRACKED = (
    "torch", "torchvision", "diffusers", "transformers", "peft", "accelerate",
    "timm", "numpy", "scipy", "scikit-learn", "scikit-image", "pandas",
    "matplotlib", "Pillow", "clean-fid", "lpips", "safetensors", "captum",
    "grad-cam", "lime",
)


def environment() -> dict:
    """Everything needed to explain a numerical divergence between two runs."""
    import torch

    gpu = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {
            "name": p.name,
            "total_memory_gb": round(p.total_memory / 1e9, 2),
            "capability": f"{p.major}.{p.minor}",
            "count": torch.cuda.device_count(),
        }
        try:
            gpu["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=10,
            ).strip().splitlines()[0]
        except Exception:
            pass

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": gpu,
        "packages": _pkg_versions(TRACKED),
        "env_vars": {
            k: os.environ.get(k)
            for k in ("CUBLAS_WORKSPACE_CONFIG", "PYTORCH_CUDA_ALLOC_CONF",
                      "HF_HOME", "OMP_NUM_THREADS")
            if os.environ.get(k)
        },
    }


# --------------------------------------------------------------------------- #
# content hashing
# --------------------------------------------------------------------------- #
def hash_code(package_dir: Path) -> dict:
    """SHA-256 over the package source, file by file plus a combined digest.

    There is no git repository here, so the code version cannot be recorded as a
    commit. Hashing the sources gives the same guarantee for the purpose that
    matters: two runs claiming the same code hash ran the same code.
    """
    package_dir = Path(package_dir)
    files = sorted(p for p in package_dir.rglob("*.py") if "__pycache__" not in p.parts)
    per_file, combined = {}, hashlib.sha256()
    for f in files:
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        per_file[str(f.relative_to(package_dir))] = h[:16]
        combined.update(h.encode())
    return {"combined": combined.hexdigest()[:16], "n_files": len(files), "files": per_file}


def hash_dataset(root: Path, sample_limit: int | None = None) -> dict:
    """Digest of the dataset's file inventory.

    Hashes names and sizes rather than pixel content by default: it is orders of
    magnitude faster on tens of thousands of images and still detects the
    failure that actually occurs, which is a different or partially downloaded
    copy of the corpus. ``sample_limit=None`` additionally hashes content for a
    subset when a stronger guarantee is wanted.
    """
    root = Path(root)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    entries = sorted(
        (str(p.relative_to(root)), p.stat().st_size)
        for p in root.rglob("*") if p.suffix.lower() in exts
    )
    inv = hashlib.sha256()
    for name, size in entries:
        inv.update(f"{name}:{size}".encode())

    out = {
        "root": str(root),
        "n_files": len(entries),
        "total_bytes": sum(s for _, s in entries),
        "inventory_sha256": inv.hexdigest()[:16],
    }
    if sample_limit:
        content = hashlib.sha256()
        for name, _ in entries[:sample_limit]:
            content.update(hashlib.sha256((root / name).read_bytes()).digest())
        out["content_sha256_first_n"] = content.hexdigest()[:16]
        out["content_sample_n"] = min(sample_limit, len(entries))
    return out


# --------------------------------------------------------------------------- #
def write_provenance(out_path: Path, cfg, package_dir: Path,
                     determinism: dict | None = None,
                     data_root: Path | None = None) -> Path:
    """Record code, data, environment and configuration for one run."""
    import time

    rec = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
        "code": hash_code(package_dir),
        "environment": environment(),
        "determinism": determinism or {},
    }
    if data_root is not None:
        try:
            rec["dataset"] = hash_dataset(Path(data_root))
        except Exception as e:
            rec["dataset"] = {"error": str(e)}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    return out_path


def compare_provenance(a: Path, b: Path) -> dict:
    """Diff two provenance records, so a divergence can be attributed."""
    ra = json.loads(Path(a).read_text(encoding="utf-8"))
    rb = json.loads(Path(b).read_text(encoding="utf-8"))
    diffs = {}
    if ra["code"]["combined"] != rb["code"]["combined"]:
        changed = [
            f for f, h in ra["code"]["files"].items()
            if rb["code"]["files"].get(f) != h
        ]
        diffs["code"] = {"changed_files": changed}
    for pkg, va in ra["environment"]["packages"].items():
        vb = rb["environment"]["packages"].get(pkg)
        if va != vb:
            diffs.setdefault("packages", {})[pkg] = [va, vb]
    ga, gb = ra["environment"].get("gpu", {}), rb["environment"].get("gpu", {})
    if ga.get("name") != gb.get("name"):
        diffs["gpu"] = [ga.get("name"), gb.get("name")]
    da, db = ra.get("dataset", {}), rb.get("dataset", {})
    if da.get("inventory_sha256") != db.get("inventory_sha256"):
        diffs["dataset"] = [da.get("inventory_sha256"), db.get("inventory_sha256")]
    ca, cb = ra.get("config", {}), rb.get("config", {})
    cfg_diff = {k: [ca.get(k), cb.get(k)] for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
    if cfg_diff:
        diffs["config"] = cfg_diff
    return diffs
