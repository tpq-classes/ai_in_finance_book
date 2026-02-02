import re
import cufflinks.colors as cf_colors
import cufflinks.plotlytools as cf_plotlytools
import matplotlib.colors as mcolors

_RGB_RE = re.compile(r"^\s*rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$", re.IGNORECASE)
_RGBA_RE = re.compile(r"^\s*rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)\s*$", re.IGNORECASE)

def fixed_to_rgba(color, alpha=1.0):
    """
    Returns a Plotly-valid 'rgba(r,g,b,a)' string.
    Accepts:
      - plotly 'rgb(r,g,b)' / 'rgba(r,g,b,a)'
      - hex '#RRGGBB'
      - named colors ('orange', etc.)
    Ensures alpha is a plain float (not np.float64 repr).
    """
    a = float(alpha)

    if isinstance(color, str):
        s = color.strip()

        m = _RGB_RE.match(s)
        if m:
            r, g, b = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return f"rgba({r}, {g}, {b}, {a})"

        m = _RGBA_RE.match(s)
        if m:
            r, g, b = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            # ignore the embedded alpha; use the passed alpha for cufflinks behavior
            return f"rgba({r}, {g}, {b}, {a})"

    # fallback: let matplotlib parse (hex, named colors, tuples, etc.)
    r, g, b, _ = mcolors.to_rgba(color)
    return f"rgba({int(round(r*255))}, {int(round(g*255))}, {int(round(b*255))}, {a})"

# Patch cufflinks
cf_colors.to_rgba = fixed_to_rgba
cf_plotlytools.to_rgba = fixed_to_rgba

