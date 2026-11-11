"""
Draw the Gmail extension's toolbar icons (a shield with a tick).

Run:  .venv/bin/python scripts/make_extension_icons.py
Writes extension/icons/icon{16,32,48,128}.png. Drawn at 512 px and scaled
down, so the small sizes stay smooth.
"""

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "extension" / "icons"
BLUE, WHITE = (26, 115, 232, 255), (255, 255, 255, 255)


def draw(size: int = 512) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    s = size / 512
    shield = [(256, 24), (456, 96), (440, 280), (256, 488), (72, 280), (56, 96)]
    d.polygon([(x * s, y * s) for x, y in shield], fill=BLUE)
    d.line([(150 * s, 262 * s), (226 * s, 338 * s), (366 * s, 182 * s)], fill=WHITE, width=int(56 * s), joint="curve")
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    big = draw()
    for px in (16, 32, 48, 128):
        big.resize((px, px), Image.LANCZOS).save(OUT / f"icon{px}.png")
        print(OUT / f"icon{px}.png")


if __name__ == "__main__":
    main()
