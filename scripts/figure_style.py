"""Print sizing for the Rhody thesis/proposal figures.

Figures lay out at the width they will occupy on the page, so the numbers
below ARE the printed point sizes at
\\includegraphics[width=\\textwidth]. Sizing type on an oversized canvas and
letting LaTeX shrink it is what makes figure text illegible, and no amount of
bumping the font size fixes it while the canvas is wide.

Kept out of the figure script so a second plate can pick up the same sizes
rather than being tuned independently -- differing label sizes on facing pages
of one document read as sloppy typesetting.
"""

import textwrap

# Printed point sizes.
PT_PANEL_TITLE = 8.5
PT_KEY_LABEL = 7.8
PT_KEY_STATS = 6.9
PT_SCALE_LABEL = 7.4
PT_CAPTION = 6.9

# A typical dissertation \textwidth. Override per-run when yours differs.
DEFAULT_WIDTH_IN = 6.5
DEFAULT_DPI = 400



def wrap_to_width(text, width_in, pt, margin=0.94, avg_char_em=0.52):
    """Wrap text to fit a figure of the given printed width.

    matplotlib has no way to ask "how many characters fit"; avg_char_em is an
    empirical mean advance for DejaVu Sans running text. Slightly conservative
    on purpose - a caption one line taller is harmless, one that runs off the
    plate is not.
    """
    budget = int(width_in * margin * 72.0 / (avg_char_em * pt))
    return textwrap.fill(text, width=budget)
