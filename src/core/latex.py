"""Low-level LaTeX string helpers shared by the algorithm layer.

Living in ``core`` (not ``viz``) on purpose: the algorithm owns the maths, so it
also owns the exact rendering of each formula it evaluates.  ``viz.mathfmt``
sits on top and handles *presentation* (layout, worked examples, tables).
"""
from __future__ import annotations

_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape(text: object) -> str:
    """Escape an arbitrary label so it is safe inside a LaTeX ``\\text{}``."""
    out = []
    for ch in str(text):
        out.append(_SPECIAL.get(ch, ch))
    return "".join(out)


def text(label: object) -> str:
    """Render a data label (class name, feature name, ...) as upright text."""
    return r"\text{%s}" % escape(label)


def num(value: float, precision: int = 3) -> str:
    """Format a number the way a textbook would: exact integers stay integers."""
    if value is None:
        return "-"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return escape(value)
    if f != f:  # NaN
        return r"\mathrm{NaN}"
    if f in (float("inf"), float("-inf")):
        return r"\infty" if f > 0 else r"-\infty"
    if abs(f - round(f)) < 1e-12:
        return str(int(round(f)))
    return f"{f:.{precision}f}"


def frac(numerator: float, denominator: float, precision: int = 3) -> str:
    """``\\frac{a}{b}`` — keeps the raw counts visible so the reader can check them."""
    if denominator == 0:
        return r"\frac{%s}{0}" % num(numerator, precision)
    return r"\frac{%s}{%s}" % (num(numerator, precision), num(denominator, precision))


def joined(terms: list[str]) -> str:
    """Join signed terms into a readable sum (``a - b + c`` rather than ``a + -b``)."""
    if not terms:
        return "0"
    out = terms[0]
    for term in terms[1:]:
        stripped = term.lstrip()
        if stripped.startswith("-"):
            out += " - " + stripped[1:].lstrip()
        else:
            out += " + " + stripped
    return out
