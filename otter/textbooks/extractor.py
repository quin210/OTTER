"""Page-image extraction for scanned textbook PDFs.

The sample textbooks are image-only PDFs (one embedded RGB raster per page, no
text layer). We extract the embedded raster losslessly with ``pdfimages``
(no resampling); pages that do not consist of exactly one embedded image are
rendered with ``pdftoppm`` at a resolution matching the typical native raster.
Images are cached on disk so OCR jobs are resumable.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

EXTRACTOR_VERSION = "otter-extract-0.2"
FALLBACK_RENDER_DPI = 110  # ~1100x1540 px for the A4-ish (715x1005 pt) sample pages


@dataclass
class PageImage:
    pdf_page_index: int          # 0-based
    path: Path
    width: int
    height: int
    method: str                  # "pdfimages_native" | "pdftoppm_render"


def pdf_page_count(pdf: Path) -> int:
    meta = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    if re.search(r"^Encrypted:\s+yes", meta, re.M | re.I):
        raise ValueError("Encrypted PDF is unsupported; do not bypass DRM")
    m = re.search(r"^Pages:\s+(\d+)", meta, re.M)
    if not m:
        raise ValueError(f"Could not read page count of {pdf}")
    return int(m.group(1))


def pdf_has_text_layer(pdf: Path, max_pages: int = 5) -> bool:
    if not shutil.which("pdftotext"):
        return False
    out = subprocess.run(["pdftotext", "-l", str(max_pages), "-enc", "UTF-8", str(pdf), "-"],
                         capture_output=True, check=False).stdout.decode("utf-8", "replace")
    return len(re.sub(r"\s+", "", out)) > 50


def list_images(pdf: Path) -> dict[int, list[tuple[int, int, int]]]:
    """Return {0-based page: [(image_num, width, height), ...]} for embedded images."""
    out = subprocess.run(["pdfimages", "-list", str(pdf)], capture_output=True, text=True, check=True).stdout
    pages: dict[int, list[tuple[int, int, int]]] = {}
    for line in out.splitlines()[2:]:
        cols = line.split()
        if len(cols) < 5 or cols[2] != "image":
            continue  # skip smask / stencil entries
        pages.setdefault(int(cols[0]) - 1, []).append((int(cols[1]), int(cols[3]), int(cols[4])))
    return pages


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        head = f.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def extract_page_images(pdf: Path, out_dir: Path) -> list[PageImage]:
    """Extract/cached page images ``out_dir/p0000.png`` for every page of ``pdf``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = pdf_page_count(pdf)
    imgs = list_images(pdf)
    result: list[PageImage] = []
    missing_native = [i for i in range(n) if not (out_dir / f"p{i:04d}.png").exists() and len(imgs.get(i, [])) == 1]
    if missing_native:
        with tempfile.TemporaryDirectory(dir=out_dir) as td:
            subprocess.run(["pdfimages", "-p", "-png", "-f", str(min(missing_native) + 1), "-l", str(max(missing_native) + 1),
                            str(pdf), str(Path(td) / "img")], check=True, capture_output=True)
            for f in Path(td).glob("img-*.png"):
                m = re.match(r"img-(\d+)-(\d+)\.png", f.name)
                if not m:
                    continue
                page = int(m.group(1)) - 1
                if page in missing_native:
                    f.replace(out_dir / f"p{page:04d}.png")
    for i in range(n):
        target = out_dir / f"p{i:04d}.png"
        method = "pdfimages_native" if len(imgs.get(i, [])) == 1 else "pdftoppm_render"
        if not target.exists():
            with tempfile.TemporaryDirectory(dir=out_dir) as td:
                subprocess.run(["pdftoppm", "-r", str(FALLBACK_RENDER_DPI), "-f", str(i + 1), "-l", str(i + 1), "-png",
                                "-singlefile", str(pdf), str(Path(td) / "r")], check=True, capture_output=True)
                (Path(td) / "r.png").replace(target)
                method = "pdftoppm_render"
        w, h = _png_size(target)
        result.append(PageImage(i, target, w, h, method))
    return result
