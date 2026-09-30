"""Cost accounting.

Every generator and classifier run is wrapped in a ``Stopwatch`` so the paper
can report a quality-vs-compute Pareto instead of quality alone. Without this,
"latent diffusion beats a from-scratch DDPM" is unfalsifiable -- it might just
be buying the win with 100x the compute (it is not; that is the point).
"""
from __future__ import annotations

import json
import platform
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return platform.processor() or "cpu"


def _nvml_energy_j() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0  # mJ -> J
    except Exception:
        return None


@dataclass
class CostRecord:
    name: str
    seconds: float = 0.0
    peak_vram_gb: float = 0.0
    energy_wh: float = float("nan")
    trainable_params_m: float = 0.0
    total_params_m: float = 0.0
    steps: int = 0
    device: str = field(default_factory=gpu_name)
    extra: dict = field(default_factory=dict)

    @property
    def gpu_minutes(self) -> float:
        return self.seconds / 60.0

    def as_dict(self):
        d = asdict(self)
        d["gpu_minutes"] = self.gpu_minutes
        return d


class Stopwatch:
    """Times a block and records peak VRAM and (where available) energy."""

    def __init__(self, name: str):
        self.record = CostRecord(name=name)
        self._t0 = 0.0
        self._e0: float | None = None

    def __enter__(self) -> CostRecord:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._e0 = _nvml_energy_j()
        self._t0 = time.perf_counter()
        return self.record

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.record.peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        self.record.seconds = time.perf_counter() - self._t0
        e1 = _nvml_energy_j()
        if self._e0 is not None and e1 is not None:
            self.record.energy_wh = (e1 - self._e0) / 3600.0
        return False


def count_params(module: torch.nn.Module) -> tuple[float, float]:
    total = sum(p.numel() for p in module.parameters()) / 1e6
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6
    return trainable, total


class CostLedger:
    """Accumulates cost records across a whole experiment and writes JSON."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: list[CostRecord] = []
        if self.path.exists():
            try:
                for r in json.loads(self.path.read_text(encoding="utf-8")):
                    r.pop("gpu_minutes", None)
                    self.records.append(CostRecord(**r))
            except Exception:
                pass

    def add(self, record: CostRecord) -> CostRecord:
        self.records = [r for r in self.records if r.name != record.name]
        self.records.append(record)
        self.flush()
        return record

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([r.as_dict() for r in self.records], indent=2), encoding="utf-8"
        )

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([r.as_dict() for r in self.records])


@contextmanager
def timed(ledger: CostLedger, name: str, **extra):
    sw = Stopwatch(name)
    with sw as rec:
        rec.extra.update(extra)
        yield rec
    ledger.add(sw.record)
