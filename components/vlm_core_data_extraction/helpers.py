
import json
from pathlib import Path

import pymupdf
from PIL import Image

from schemas import RawCV


def render_pdf(pdf_path: Path, dpi: int = 120) -> list[Image.Image]:
    images = []

    with pymupdf.open(pdf_path) as document:
        for page in document:
            pix = page.get_pixmap(
                dpi=dpi,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                (pix.width, pix.height),
                pix.samples,
            )

            images.append(image)

    return images


def validate_response(response: str) -> RawCV:
    text = response.strip()

    # Remove Markdown fences if present.
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])

    data = json.loads(text)

    return RawCV.model_validate(data)
