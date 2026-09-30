"""Print-safe plotting helpers (palette, markers, greyscale proofs)."""
from __future__ import annotations

from ._vendor_printsafe import *  # noqa: F401,F403
from ._vendor_printsafe import (  # noqa: F401
    HATCHES, LINESTYLES, MARKERS, PALETTE, LINE_PALETTE, apply_print_rcparams,
    hatch_bars, hatch_grouped, save_grayscale_proof, tidy_log_axis, luminance,
    check_separation, audit_palette, line_color, style_for,
)
