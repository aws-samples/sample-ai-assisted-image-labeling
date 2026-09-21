"""Annotation data model and persistence for the segmentation labeler.

Formats written per image (all under the output directory):
    <stem>.txt          YOLO segmentation (accepted masks only) — back-compat,
                        consumed directly by YOLO-seg training.
    boxes/<stem>.txt    YOLO detection `cid cx cy w h` (accepted boxes only).
    .meta/<stem>.json   Full annotation records: provenance, review state,
                        scores, pending autolabels. Authoritative when present.

Loading order: the .meta sidecar wins if it exists; otherwise a bare YOLO
seg txt loads as manual/accepted annotations (so pre-existing label dirs
keep working), with boxes derived from polygons.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Normalized point / box aliases
Point = tuple[float, float]
Box = tuple[float, float, float, float]  # (cx, cy, w, h) normalized [0,1]

SOURCES = ("manual", "sam_box", "sam_point", "autolabel")
REVIEWS = ("accepted", "pending", "rejected")

# Sidecars written before the SAM2 migration used sam3_-prefixed sources;
# normalize them on load so old label directories keep working.
_LEGACY_SOURCES = {"sam3_box": "sam_box", "sam3_point": "sam_point"}


def box_from_polygon(pts: list[Point]) -> Optional[Box]:
    """Axis-aligned (cx, cy, w, h) box enclosing a normalized polygon."""
    if len(pts) < 3:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return None
    return ((x0 + x1) / 2, (y0 + y1) / 2, w, h)


def box_to_corners(box: Box) -> tuple[float, float, float, float]:
    """(cx, cy, w, h) -> (x0, y0, x1, y1), clamped to [0, 1]."""
    cx, cy, w, h = box
    return (
        max(0.0, cx - w / 2),
        max(0.0, cy - h / 2),
        min(1.0, cx + w / 2),
        min(1.0, cy + h / 2),
    )


def corners_to_box(x0: float, y0: float, x1: float, y1: float) -> Box:
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


@dataclass
class Annotation:
    class_id: int
    polygon: Optional[list[Point]] = None   # normalized; None for box-only
    box: Optional[Box] = None               # (cx, cy, w, h) normalized
    source: str = "manual"                  # manual | sam_box | sam_point | autolabel
    review: str = "accepted"                # accepted | pending | rejected
    score: Optional[float] = None
    box_edited: bool = False                # user adjusted the auto-derived box

    def __post_init__(self):
        if self.polygon is not None:
            self.polygon = [(float(x), float(y)) for x, y in self.polygon]
        if self.box is not None:
            self.box = tuple(float(v) for v in self.box)  # type: ignore[assignment]

    @property
    def accepted(self) -> bool:
        return self.review == "accepted"

    @property
    def pending(self) -> bool:
        return self.review == "pending"

    def effective_box(self) -> Optional[Box]:
        """The stored box, or one derived from the polygon."""
        if self.box is not None:
            return self.box
        if self.polygon:
            return box_from_polygon(self.polygon)
        return None

    def sync_box_from_polygon(self) -> None:
        """Refresh the derived box after polygon edits (unless user-edited)."""
        if self.polygon and not self.box_edited:
            self.box = box_from_polygon(self.polygon)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.polygon is not None:
            d["polygon"] = [[x, y] for x, y in self.polygon]
        if self.box is not None:
            d["box"] = list(self.box)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(
            class_id=int(d["class_id"]),
            polygon=[(p[0], p[1]) for p in d["polygon"]] if d.get("polygon") else None,
            box=tuple(d["box"]) if d.get("box") else None,
            source=_LEGACY_SOURCES.get(d.get("source", "manual"), d.get("source", "manual")),
            review=d.get("review", "accepted"),
            score=d.get("score"),
            box_edited=bool(d.get("box_edited", False)),
        )


# ---------------------------------------------------------------------------
# Per-image persistence
# ---------------------------------------------------------------------------

class AnnotationIO:
    """Reads and writes the three per-image label representations."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def seg_path(self, img_path: Path) -> Path:
        return self.output_dir / (img_path.stem + ".txt")

    def box_path(self, img_path: Path) -> Path:
        return self.output_dir / "boxes" / (img_path.stem + ".txt")

    def meta_path(self, img_path: Path) -> Path:
        return self.output_dir / ".meta" / (img_path.stem + ".json")

    def has_labels(self, img_path: Path) -> bool:
        return self.seg_path(img_path).exists() or self.meta_path(img_path).exists()

    def has_pending(self, img_path: Path) -> bool:
        p = self.meta_path(img_path)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        return any(a.get("review") == "pending" for a in data.get("annotations", []))

    # -------------------- load --------------------

    def load(self, img_path: Path) -> list[Annotation]:
        meta = self.meta_path(img_path)
        if meta.exists():
            try:
                data = json.loads(meta.read_text())
                return [Annotation.from_dict(a) for a in data.get("annotations", [])]
            except (json.JSONDecodeError, KeyError, ValueError):
                pass  # fall back to YOLO txt
        return self._load_yolo_seg(img_path)

    def _load_yolo_seg(self, img_path: Path) -> list[Annotation]:
        p = self.seg_path(img_path)
        if not p.exists():
            return []
        out: list[Annotation] = []
        for line in p.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                continue
            try:
                cid = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            pts = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            ann = Annotation(class_id=cid, polygon=pts)
            ann.sync_box_from_polygon()
            out.append(ann)
        return out

    # -------------------- save --------------------

    def save(self, img_path: Path, annotations: list[Annotation]) -> None:
        self._write_yolo_seg(img_path, annotations)
        self._write_yolo_boxes(img_path, annotations)
        self._write_meta(img_path, annotations)

    def _write_yolo_seg(self, img_path: Path, annotations: list[Annotation]) -> None:
        path = self.seg_path(img_path)
        lines = []
        for a in annotations:
            if a.accepted and a.polygon and len(a.polygon) >= 3:
                coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in a.polygon)
                lines.append(f"{a.class_id} {coords}")
        if lines:
            path.write_text("\n".join(lines) + "\n")
        elif path.exists():
            path.unlink()

    def _write_yolo_boxes(self, img_path: Path, annotations: list[Annotation]) -> None:
        path = self.box_path(img_path)
        lines = []
        for a in annotations:
            box = a.effective_box() if a.accepted else None
            if box:
                cx, cy, w, h = box
                lines.append(f"{a.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if lines:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(lines) + "\n")
        elif path.exists():
            path.unlink()

    def _write_meta(self, img_path: Path, annotations: list[Annotation]) -> None:
        path = self.meta_path(img_path)
        if not annotations:
            if path.exists():
                path.unlink()
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"annotations": [a.to_dict() for a in annotations]}, indent=1
        ))


# ---------------------------------------------------------------------------
# COCO export
# ---------------------------------------------------------------------------

def export_coco(
    images: list[Path],
    io_: AnnotationIO,
    classes: list[str],
    image_sizes,
) -> dict:
    """Build a COCO dict from accepted annotations across all images.

    image_sizes: callable(img_path) -> (w, h) or None.
    Box-only annotations are included with empty segmentation.
    """
    import time

    annotations = []
    images_meta = []
    ann_id = 1

    for img_idx, img_path in enumerate(images):
        anns = [a for a in io_.load(img_path) if a.accepted]
        if not anns:
            continue
        size = image_sizes(img_path)
        if size is None:
            continue
        img_w, img_h = size

        images_meta.append({
            "id": img_idx + 1,
            "file_name": img_path.name,
            "width": img_w,
            "height": img_h,
        })

        for a in anns:
            box = a.effective_box()
            if box is None:
                continue
            x0, y0, x1, y1 = box_to_corners(box)
            bbox = [x0 * img_w, y0 * img_h, (x1 - x0) * img_w, (y1 - y0) * img_h]

            segmentation: list[list[float]] = []
            area = bbox[2] * bbox[3]
            if a.polygon and len(a.polygon) >= 3:
                abs_pts = [(x * img_w, y * img_h) for x, y in a.polygon]
                segmentation = [[c for xy in abs_pts for c in xy]]
                n = len(abs_pts)
                area = abs(sum(
                    abs_pts[i][0] * abs_pts[(i + 1) % n][1]
                    - abs_pts[(i + 1) % n][0] * abs_pts[i][1]
                    for i in range(n)
                )) / 2.0

            annotations.append({
                "id": ann_id,
                "image_id": img_idx + 1,
                "category_id": a.class_id,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            })
            ann_id += 1

    return {
        "info": {
            "description": "Exported by Segmentation Labeler",
            "date_created": time.strftime("%Y-%m-%d"),
        },
        "licenses": [],
        "images": images_meta,
        "categories": [{"id": i, "name": name} for i, name in enumerate(classes)],
        "annotations": annotations,
    }
