"""Draw the desktop-app icon: assets/stancesense.ico (16-256 px) and assets/stancesense.png.

    python scripts/make_app_icon.py

A squatting stick figure on a dark teal tile, in the camera window's green.
Drawn at 1024 px and scaled down, so the small sizes stay smooth.
"""
from __future__ import annotations

import argparse
import os

from PIL import Image, ImageDraw

BG = (15, 82, 87, 255)          # dark teal tile
FIG = (255, 255, 255, 255)      # figure
ACCENT = (80, 220, 80, 255)     # same green as the HUD (BGR 80,220,80 is symmetric)
SIZES = [16, 24, 32, 48, 64, 128, 256]


def draw(size: int = 1024) -> Image.Image:
    s = size / 256.0
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=int(52 * s), fill=BG)

    def P(x, y):
        return (x * s, y * s)

    w = int(20 * s)
    # ground line
    d.line([P(44, 222), P(212, 222)], fill=ACCENT, width=int(12 * s))
    # squat: shoulder -> hip (lean), hip -> knee (thigh ~level), knee -> ankle, foot
    shoulder, hip, knee, ankle, toe = P(118, 94), P(88, 152), P(160, 156), P(150, 208), P(184, 208)
    for a, b in ((shoulder, hip), (hip, knee), (knee, ankle), (ankle, toe)):
        d.line([a, b], fill=FIG, width=w)
    # arms reaching forward for balance
    d.line([shoulder, P(196, 110)], fill=FIG, width=int(16 * s))
    # round the joints
    for x, y in (shoulder, hip, knee, ankle, toe, P(196, 110)):
        r = w / 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=FIG)
    # head
    cx, cy, r = 132 * s, 58 * s, 24 * s
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=FIG)
    # knee marker in the accent colour (the joint the app measures from)
    kx, ky = knee
    r = 13 * s
    d.ellipse([kx - r, ky - r, kx + r, ky + r], fill=ACCENT)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="assets")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    big = draw(1024)
    big.resize((256, 256), Image.LANCZOS).save(os.path.join(args.out_dir, "stancesense.png"))
    big.save(os.path.join(args.out_dir, "stancesense.ico"), sizes=[(n, n) for n in SIZES])
    print(f"wrote {args.out_dir}/stancesense.ico and stancesense.png")


if __name__ == "__main__":
    main()
