"""DiffSeed: generative augmentation benchmark for seed-quality imaging.

Imports are lazy so that the analysis of saved results (``diffseed.analyze``,
``diffseed.stats``, ``diffseed.figures``) works without torch installed.
"""
__version__ = "2.0.0"


def __getattr__(name):
    if name in ("Config", "CLASS_PROMPTS", "NEGATIVE_PROMPT"):
        from . import config
        return getattr(config, name)
    if name in ("Experiment", "main"):
        from . import run
        return getattr(run, name)
    raise AttributeError(name)
