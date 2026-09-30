"""LaTeX macro emission, shared by both studies.

Holds the machinery -- name construction, formatting, the two output files --
while each study keeps its own emitter for the quantities it reports. The split
is deliberate: the naming rules and the ``??`` fallback contract must be
identical across papers, but what counts as a reportable number is not.

Why any of this exists: the earlier DiffSeed manuscript printed a per-class
quality table that disagreed with the figure beside it. Three of five values
appeared nowhere in the experimental output, and the paper then built an
argument on the misattribution. Nothing in the workflow could have caught it,
because the table was typed while the figure was generated. Emitting every
number as a macro removes the opportunity.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
           "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}

# Symbols that carry meaning inside a name and would otherwise be stripped.
# ``gradcam`` and ``gradcam++`` are different methods; deleting the ``+`` maps
# both to one macro and the second silently overwrites the first, so the
# manuscript prints one method's number under the other's name.
#
# Only a doubled ``++`` is spelled. A single ``+`` is a *separator* in this
# codebase -- downstream conditions are named ``sd_lora+filtered`` -- and
# spelling it renames every such macro (``SdLoraFiltered`` becomes
# ``SdLoraPlusFiltered``), which breaks the references a manuscript already
# carries. Separators should keep collapsing to a word boundary.
_SYMBOLS = ((r"\+\+", " plusplus "), (r"#", " sharp "))


def make_name(prefix: str, *parts) -> str:
    """A LaTeX-legal macro name, camel-cased with word boundaries preserved.

    Each part is title-cased *separately* before joining. Title-casing the
    concatenation instead yields ``\\DSFidsdLora`` for ``("fid", "sd_lora")``,
    which is legal but unreadable in the manuscript source; per-part casing
    gives ``\\DSFidSdLora``. LaTeX command names cannot contain digits, so
    digits are spelled out, and meaningful symbols are spelled before the
    remaining punctuation is dropped.
    """
    chunks = []
    for p in parts:
        text = str(p)
        for pat, word in _SYMBOLS:
            text = re.sub(pat, word, text)
        clean = re.sub(r"[^0-9A-Za-z]+", " ", text).title().replace(" ", "")
        chunks.append(clean)
    raw = "".join(chunks)
    return prefix + "".join(_DIGITS.get(c, c) for c in raw)


def fmt(v, nd: int = 3) -> str:
    if v is None:
        return "??"
    if isinstance(v, str):
        return v.replace("_", r"\_")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "??"
    if float(f).is_integer() and abs(f) < 1e7:
        return f"{int(f):,}".replace(",", r"{,}")
    return f"{f:.{nd}f}"


class MacroSet:
    """Collects name -> value pairs and writes the macro and defaults files."""

    def __init__(self, prefix: str, emitter: str = "analyze"):
        self.prefix = prefix
        self.emitter = emitter
        self.values: dict[str, str] = {}
        self.collisions: dict[str, list[str]] = {}
        self._sources: dict[str, tuple] = {}

    def _record(self, name: str, parts: tuple, value: str):
        """Store a value, and notice when two different inputs claim one name.

        A collision is silent by construction -- the second write wins and the
        manuscript then prints one quantity under another's name, with nothing
        in the build to indicate it. That is how ``gradcam++`` came to be
        reported as ``gradcam``. Collisions are recorded rather than raised,
        because an analysis run that has already finished should still emit what
        it can, but they are surfaced loudly at write time.
        """
        prev = self._sources.get(name)
        if prev is not None and prev != parts and self.values.get(name) != value:
            self.collisions.setdefault(name, [" / ".join(map(str, prev))])
            self.collisions[name].append(" / ".join(map(str, parts)))
        self._sources[name] = parts
        self.values[name] = value

    def add(self, value, *parts, nd: int = 3):
        self._record(make_name(self.prefix, *parts), parts, fmt(value, nd))
        return self

    def add_pct(self, value, *parts, nd: int = 1):
        """Percentage, stored without the sign so the text controls spacing."""
        name = make_name(self.prefix, *parts)
        try:
            f = float(value)
            v = "??" if not np.isfinite(f) else f"{f * 100:.{nd}f}"
        except (TypeError, ValueError):
            v = "??"
        self._record(name, parts, v)
        return self

    def write(self, out_dir: Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.collisions:
            print(f"  !! {len(self.collisions)} macro name collision(s) -- one "
                  f"quantity is overwriting another and the manuscript will "
                  f"print the wrong value:")
            for name, sources in sorted(self.collisions.items()):
                print(f"     {self.prefix}{name[len(self.prefix):]} <- {sources}")

        macros = out_dir / f"macros{self.prefix}.tex"
        lines = [f"% auto-generated by {self.emitter} -- do not edit"]
        for k in sorted(self.values):
            lines.append(f"\\newcommand{{\\{k}}}{{{self.values[k]}}}")
        macros.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # A macro the analysis did not emit renders as a loud ?? rather than
        # silently keeping a stale value from a previous run.
        defaults = out_dir / f"defaults{self.prefix}.tex"
        dlines = ["% auto-generated fallbacks -- '??' means analyze did not emit this macro"]
        for k in sorted(self.values):
            dlines.append(f"\\providecommand{{\\{k}}}{{\\textbf{{??}}}}")
        defaults.write_text("\n".join(dlines) + "\n", encoding="utf-8")
        return macros, defaults
