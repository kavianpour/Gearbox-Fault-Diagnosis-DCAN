"""
Renders the original schematic figures of this repository into assets/*.svg.

The architecture, results-table and sensitivity figures in assets/ are
reproduced from the published paper and are not regenerated here - see
LICENSE. Only the two explanatory diagrams below (no equivalent in the paper)
are original to this repository.
"""

import os

PALETTE = {
    "input": ("#2b6cb0", "#ffffff"),
    "conv": ("#f6c445", "#20303f"),
    "norm": ("#8ec7f0", "#20303f"),
    "act": ("#f3aa9b", "#20303f"),
    "pool": ("#8fd39a", "#20303f"),
    "drop": ("#cfd6dd", "#20303f"),
    "dense": ("#ef8354", "#ffffff"),
    "grl": ("#ef4b3c", "#ffffff"),
    "coral": ("#7e57c2", "#ffffff"),
    "out": ("#12a05c", "#ffffff"),
    "note": ("#f4f6f8", "#42526b"),
}

HEADER = ('<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
          'viewBox="0 0 {w} {h}" font-family="Inter, Segoe UI, Helvetica, Arial, '
          'sans-serif">\n'
          '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" '
          'refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#42526b"/>'
          '</marker>'
          '<marker id="rev" markerWidth="9" markerHeight="9" refX="8" refY="3" '
          'orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#ef4b3c"/></marker>'
          '</defs>\n'
          '<rect width="{w}" height="{h}" fill="#ffffff"/>\n')


def block(x, y, w, h, lines, kind, font=11, rx=8):
    fill, text = PALETTE[kind]
    out = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
           f'fill="{fill}" stroke="#00000018"/>\n')
    step = font + 3
    start = y + h / 2 - (len(lines) - 1) * step / 2 + font / 3
    for i, line in enumerate(lines):
        out += (f'<text x="{x + w / 2}" y="{start + i * step}" fill="{text}" '
                f'font-size="{font}" text-anchor="middle">{line}</text>\n')
    return out


def arrow(x1, y1, x2, y2, marker="arrow", colour="#42526b", dash=None):
    style = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{colour}" '
            f'stroke-width="1.6" fill="none"{style} '
            f'marker-end="url(#{marker})"/>\n')


def elbow(x1, y1, x2, y2, marker="arrow", colour="#42526b", dash=None):
    mid = (x1 + x2) / 2
    style = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="M{x1},{y1} L{mid},{y1} L{mid},{y2} L{x2},{y2}" '
            f'stroke="{colour}" stroke-width="1.6" fill="none"{style} '
            f'marker-end="url(#{marker})"/>\n')


def label(x, y, text, size=13, weight="600", anchor="start", fill="#20303f"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}">{text}</text>\n')


def write(path, width, height, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(HEADER.format(w=width, h=height) + body + "</svg>\n")
    print(f"wrote {path}")


# --------------------------------------------------------------------------- #
def adaptation_concept():
    w, h = 940, 420
    s = label(30, 36, "Two adaptation modules on one feature layer", 18)
    s += label(30, 58, "the adversarial discriminator closes the global gap; "
                       "CORAL matches second-order statistics", 12, "400",
               fill="#5a6b80")

    # adversarial panel
    s += block(40, 100, 400, 130, [], "note", 11, rx=12)
    s += label(60, 126, "Adversarial domain discriminator", 13)
    s += label(60, 150, "D tries to tell source from target;", 11, "400",
               "start", "#5a6b80")
    s += label(60, 168, "the reversed gradient makes F fool it.", 11, "400",
               "start", "#5a6b80")
    s += block(70, 182, 100, 36, ["F(x)"], "dense", 11)
    s += arrow(170, 200, 196, 200)
    s += block(196, 182, 74, 36, ["GRL"], "grl", 11)
    s += arrow(270, 200, 296, 200)
    s += block(296, 182, 110, 36, ["D  source?"], "grl", 11)
    s += arrow(296, 176, 100, 176, marker="rev", colour="#ef4b3c")
    s += label(150, 168, "&#8722;&#8711;", 11, "700", "start", "#ef4b3c")

    # CORAL panel
    s += block(490, 100, 410, 130, [], "note", 11, rx=12)
    s += label(510, 126, "CORAL", 13)
    s += label(510, 150, "MMD-style criteria align first-order statistics;", 11,
               "400", "start", "#5a6b80")
    s += label(510, 168, "matching covariances captures the relationships", 11,
               "400", "start", "#5a6b80")
    s += label(510, 186, "between features that a mean match leaves alone.", 11,
               "400", "start", "#5a6b80")
    s += block(520, 196, 120, 26, ["C&#8347;  source"], "coral", 10)
    s += label(652, 214, "&#8596;", 15, "700", "middle", "#7e57c2")
    s += block(668, 196, 120, 26, ["C&#8348;  target"], "coral", 10)

    # distribution sketch
    s += label(30, 278, "What each one does to the feature distribution", 13)
    xs = [140, 400, 660]
    titles = ["no adaptation", "adversarial only", "adversarial + CORAL"]
    spreads = [(0, 60), (0, 22), (0, 8)]
    for cx, title, (offset, gapv) in zip(xs, titles, spreads):
        s += (f'<ellipse cx="{cx - gapv / 2}" cy="340" rx="62" ry="34" '
              f'fill="#2b6cb0" opacity="0.35"/>\n')
        s += (f'<ellipse cx="{cx + gapv / 2}" cy="340" rx="62" ry="34" '
              f'fill="#ef8354" opacity="0.45"/>\n')
        s += label(cx, 400, title, 12, "600", "middle", "#42526b")

    s += label(770, 336, "source", 11, "600", "start", "#2b6cb0")
    s += label(770, 354, "target", 11, "600", "start", "#ef8354")
    return w, h, s


def transfer_tasks():
    w, h = 820, 250
    s = label(30, 36, "Cross-condition transfer tasks", 18)
    s += label(30, 58, "the source domain is labelled, the target domain is not",
               12, "400", fill="#5a6b80")

    s += block(70, 100, 200, 84,
               ["<tspan font-weight='700'>Condition A</tspan>",
                "20 Hz  /  0 V", "5 health states"], "input", 12)
    s += block(540, 100, 200, 84,
               ["<tspan font-weight='700'>Condition B</tspan>",
                "30 Hz  /  2 V", "5 health states"], "out", 12)

    s += (f'<path d="M270,126 L536,126" stroke="#42526b" stroke-width="1.8" '
          f'fill="none" marker-end="url(#arrow)"/>\n')
    s += (f'<path d="M540,162 L274,162" stroke="#42526b" stroke-width="1.8" '
          f'fill="none" marker-end="url(#arrow)"/>\n')
    s += label(405, 118, "A &#8594; B", 12, "700", "middle", "#42526b")
    s += label(405, 182, "B &#8594; A", 12, "700", "middle", "#42526b")

    s += label(30, 222, "Health, Chipped, Root, Miss, Surface — SEU gearbox, "
                        "second vibration channel, z-score normalised.", 12,
               "400", fill="#5a6b80")
    return w, h, s


if __name__ == "__main__":
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    for name, builder in (("adaptation_modules", adaptation_concept),
                          ("transfer_tasks", transfer_tasks)):
        width, height, body = builder()
        write(os.path.join(here, f"{name}.svg"), width, height, body)
