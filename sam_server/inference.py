"""Segmentation inference handlers — Grounding DINO + SAM2, framework-free.

Drop-in replacement for the SAM implementation (preserved in git history on
the sam-endpoint-release branch), keeping the same request/response contract:

    load_model()     -> ModelBundle
    predict()        one image, one prompt set  -> {"masks": [...]}
    predict_multi()  one image, many categories -> {"results": [...]}

Both models are Apache-2.0 licensed and ungated, unlike SAM's custom license:

    Grounding DINO   text prompt -> bounding boxes (open-vocabulary detection)
    SAM2.1           boxes / points -> masks

How the SAM prompt semantics map onto the two-model stack:

    text prompt        GDINO detects all matching boxes; SAM2 masks each one.
    click points       SAM2's native point prompts (a strict upgrade: SAM's
                       multiplex checkpoint lacked the instance head, so clicks
                       went through concept grounding and needed filtering).
                       Negative points now carve the mask directly.
    box prompt         SAM2 masks the boxed object (one mask). Under SAM a
                       box was a concept exemplar returning all lookalikes;
                       combine box+text to get "all matches inside my box".
    exclude_boxes      post-filter on detections (SAM used negative prompts).
    exemplars          not supported — SAM-only feature; logged and ignored.

Environment variables:

    SAM2_MODEL_DIR   Directory holding locally saved checkpoints under
                     <dir>/sam2 and <dir>/gdino (see download_models.py).
                     If absent, models are pulled from the HuggingFace hub —
                     both repos are public.
    SAM2_MODEL_ID    default facebook/sam2.1-hiera-large
    GDINO_MODEL_ID   default IDEA-Research/grounding-dino-base
    SAM2_DTYPE       Autocast dtype override: bf16 | fp16 | fp32. Default:
                     bf16 if the GPU supports it natively, else fp16 on CUDA,
                     else fp32.
"""
from __future__ import annotations

import base64
import contextlib
import io
import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

DEFAULT_SAM2_ID = "facebook/sam2.1-hiera-large"
DEFAULT_GDINO_ID = "IDEA-Research/grounding-dino-base"

# GDINO's text_threshold (word-level grounding confidence). Fixed rather than
# exposed: the client's confidence slider maps onto the box threshold.
TEXT_THRESHOLD = 0.25

# Detections overlapping an exclude_box beyond this IoU are dropped.
EXCLUDE_IOU = 0.5


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    detector: object          # GroundingDinoForObjectDetection
    detector_processor: object
    sam: object               # Sam2Model
    sam_processor: object
    device: torch.device
    dtype: Optional[torch.dtype]  # None = no autocast (fp32)

    @property
    def dtype_name(self) -> str:
        if self.dtype is torch.bfloat16:
            return "bf16"
        if self.dtype is torch.float16:
            return "fp16"
        return "fp32"


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _bf16_native(device: torch.device) -> bool:
    """True only for GPUs with hardware bfloat16 (Ampere/sm_80 and newer)."""
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device) >= (8, 0)


def resolve_dtype(device: torch.device) -> Optional[torch.dtype]:
    """Pick the autocast dtype, honoring the SAM2_DTYPE override."""
    override = os.environ.get("SAM2_DTYPE", "").strip().lower()
    if override:
        table = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
        if override not in table:
            raise ValueError(f"dtype must be bf16|fp16|fp32, got {override!r}")
        if override == "bf16" and device.type == "cuda" and not _bf16_native(device):
            log.warning("bf16 forced on a GPU without native bfloat16 — "
                        "this will run via emulation and be slow")
        return table[override]

    if device.type != "cuda":
        return None
    return torch.bfloat16 if _bf16_native(device) else torch.float16


def _autocast(bundle: ModelBundle):
    if bundle.dtype is None or bundle.device.type not in ("cuda", "cpu"):
        return contextlib.nullcontext()
    return torch.autocast(bundle.device.type, dtype=bundle.dtype)


def _model_source(model_dir: Optional[str], subdir: str, hub_id: str) -> str:
    """Prefer a locally saved checkpoint dir; fall back to the HF hub."""
    if model_dir:
        local = os.path.join(os.path.expanduser(model_dir), subdir)
        if os.path.isdir(local) and os.listdir(local):
            return local
    return hub_id


def load_model(
    model_dir: Optional[str] = None,
    dtype: Optional[str] = None,
) -> ModelBundle:
    """Load Grounding DINO + SAM2 and return a ModelBundle."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if dtype is not None:
        os.environ["SAM2_DTYPE"] = dtype

    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    device = _get_device()
    resolved = resolve_dtype(device)
    log.info("Using device: %s, autocast dtype: %s", device, resolved or "none (fp32)")

    model_dir = model_dir or os.environ.get("SAM2_MODEL_DIR")
    gdino_src = _model_source(
        model_dir, "gdino", os.environ.get("GDINO_MODEL_ID", DEFAULT_GDINO_ID)
    )
    sam_src = _model_source(
        model_dir, "sam2", os.environ.get("SAM2_MODEL_ID", DEFAULT_SAM2_ID)
    )

    log.info("Loading detector: %s", gdino_src)
    detector_processor = AutoProcessor.from_pretrained(gdino_src)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_src)
    detector.to(device).eval()

    log.info("Loading SAM2: %s", sam_src)
    sam_processor = Sam2Processor.from_pretrained(sam_src)
    sam = Sam2Model.from_pretrained(sam_src)
    sam.to(device).eval()

    log.info("Models loaded.")
    return ModelBundle(
        detector=detector,
        detector_processor=detector_processor,
        sam=sam,
        sam_processor=sam_processor,
        device=device,
        dtype=resolved,
    )


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def decode_image(image_b64: str) -> Image.Image:
    try:
        img_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image: {e}") from e


def mask_to_polygon(
    mask_u8: np.ndarray, img_w: int, img_h: int, min_points: int = 3
) -> list[list[float]] | None:
    """Convert a uint8 mask to a normalized polygon via contour extraction."""
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < min_points:
        return None

    pts = approx.reshape(-1, 2)
    return [[float(x) / img_w, float(y) / img_h] for x, y in pts]


def _norm_box_to_xyxy(box: list[float], img_w: int, img_h: int) -> list[float]:
    """[cx, cy, w, h] normalized -> [x0, y0, x1, y1] absolute pixels."""
    cx, cy, w, h = box
    return [
        (cx - w / 2) * img_w,
        (cy - h / 2) * img_h,
        (cx + w / 2) * img_w,
        (cy + h / 2) * img_h,
    ]


def _xyxy_to_norm_box(xyxy, img_w: int, img_h: int) -> list[float]:
    """[x0, y0, x1, y1] absolute -> [cx, cy, w, h] normalized."""
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    return [
        (x0 + x1) / 2 / img_w,
        (y0 + y1) / 2 / img_h,
        (x1 - x0) / img_w,
        (y1 - y0) / img_h,
    ]


def _iou(a: list[float], b: list[float]) -> float:
    """IoU of two xyxy boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _center_in(box: list[float], region: list[float]) -> bool:
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def _gdino_phrase(text: str) -> str:
    """GDINO wants lowercase phrases terminated by a period."""
    phrase = text.strip().lower().rstrip(".")
    return f"{phrase}."


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------

def _detect(
    bundle: ModelBundle,
    image: Image.Image,
    text: str,
    threshold: float,
) -> list[tuple[list[float], float]]:
    """Grounding DINO: text -> [(xyxy_box, score), ...] above threshold."""
    processor = bundle.detector_processor
    inputs = processor(
        images=image, text=_gdino_phrase(text), return_tensors="pt"
    ).to(bundle.device)

    with _autocast(bundle), torch.no_grad():
        outputs = bundle.detector(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[(image.height, image.width)],
    )[0]

    return [
        ([float(v) for v in box.tolist()], float(score))
        for box, score in zip(results["boxes"], results["scores"])
        if np.isfinite(float(score))
    ]


def _sam2_masks_for_boxes(
    bundle: ModelBundle, image: Image.Image, boxes_xyxy: list[list[float]]
) -> list[np.ndarray]:
    """SAM2: one mask per input box, single batched forward pass."""
    if not boxes_xyxy:
        return []
    processor = bundle.sam_processor
    inputs = processor(
        images=image, input_boxes=[boxes_xyxy], return_tensors="pt"
    ).to(bundle.device)

    with _autocast(bundle), torch.no_grad():
        outputs = bundle.sam(**inputs, multimask_output=False)

    masks = processor.post_process_masks(
        outputs.pred_masks.float().cpu(), inputs["original_sizes"]
    )[0]  # (num_boxes, 1, H, W) bool
    return [
        (masks[i, 0].numpy() > 0).astype(np.uint8) * 255
        for i in range(masks.shape[0])
    ]


def _sam2_mask_for_points(
    bundle: ModelBundle,
    image: Image.Image,
    points_abs: list[list[float]],
    labels: list[int],
) -> tuple[Optional[np.ndarray], float]:
    """SAM2: point prompts -> (best mask, iou score). One object per call."""
    processor = bundle.sam_processor
    inputs = processor(
        images=image,
        input_points=[[points_abs]],
        input_labels=[[labels]],
        return_tensors="pt",
    ).to(bundle.device)

    with _autocast(bundle), torch.no_grad():
        # multimask_output only for a single point, where intent is ambiguous.
        outputs = bundle.sam(**inputs, multimask_output=len(points_abs) == 1)

    ious = outputs.iou_scores.float().cpu().numpy().reshape(-1)
    best = int(np.argmax(ious))
    if not np.isfinite(ious[best]):
        return None, 0.0

    masks = processor.post_process_masks(
        outputs.pred_masks.float().cpu(), inputs["original_sizes"]
    )[0]  # (1, num_masks, H, W)
    mask_u8 = (masks[0, best].numpy() > 0).astype(np.uint8) * 255
    return mask_u8, float(ious[best])


def _mask_entry(
    mask_u8: np.ndarray,
    score: float,
    img_w: int,
    img_h: int,
    box_xyxy: Optional[list[float]] = None,
) -> Optional[dict]:
    """Build one {polygon, score, box} result, or None if the mask is empty."""
    polygon = mask_to_polygon(mask_u8, img_w, img_h)
    if not polygon:
        return None
    if box_xyxy is None:
        ys, xs = np.nonzero(mask_u8)
        if len(xs) == 0:
            return None
        box_xyxy = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return {
        "polygon": polygon,
        "score": score,
        "box": _xyxy_to_norm_box(box_xyxy, img_w, img_h),
    }


def _detect_and_segment(
    bundle: ModelBundle,
    image: Image.Image,
    text: str,
    threshold: float,
    exclude_xyxy: list[list[float]],
    within_xyxy: Optional[list[float]] = None,
) -> list[dict]:
    """Full text pipeline: detect, filter, mask, package."""
    img_w, img_h = image.size
    detections = _detect(bundle, image, text, threshold)

    if within_xyxy is not None:
        detections = [d for d in detections if _center_in(d[0], within_xyxy)]
    if exclude_xyxy:
        detections = [
            d for d in detections
            if all(_iou(d[0], ex) < EXCLUDE_IOU for ex in exclude_xyxy)
        ]
    if not detections:
        return []

    masks = _sam2_masks_for_boxes(bundle, image, [d[0] for d in detections])
    results = []
    for (box, score), mask_u8 in zip(detections, masks):
        entry = _mask_entry(mask_u8, score, img_w, img_h, box_xyxy=box)
        if entry:
            results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Prediction (single prompt set)
# ---------------------------------------------------------------------------

def predict(bundle: ModelBundle, request: dict) -> dict:
    """Handle one /segment-style request.

    Request keys: image_b64, text_prompt, box [cx,cy,w,h] normalized,
    exclude_boxes, points [[nx,ny],...], point_labels [1|0,...],
    instance_mode (ignored: point prompts are always single-instance now),
    confidence_threshold.
    Returns {"masks": [{"polygon", "box", "score"}, ...]}.
    """
    pil_image = decode_image(request["image_b64"])
    img_w, img_h = pil_image.size

    text_prompt = request.get("text_prompt") or ""
    box = request.get("box")
    exclude_boxes = request.get("exclude_boxes") or []
    points = request.get("points") or []
    point_labels = request.get("point_labels") or []

    if not text_prompt and box is None and not points:
        return {"masks": []}
    if box is not None and len(box) != 4:
        raise ValueError("box must be [cx, cy, w, h]")
    if points and len(point_labels) not in (0, len(points)):
        raise ValueError("point_labels must match points length")

    threshold = float(request.get("confidence_threshold", 0.2))
    exclude_xyxy = [
        _norm_box_to_xyxy(eb, img_w, img_h) for eb in exclude_boxes if len(eb) == 4
    ]

    # Click-to-label: SAM2's native single-instance point prompting.
    if points:
        labels = [int(bool(l)) for l in (point_labels or [1] * len(points))]
        points_abs = [[p[0] * img_w, p[1] * img_h] for p in points]
        mask_u8, iou = _sam2_mask_for_points(bundle, pil_image, points_abs, labels)
        if mask_u8 is None:
            return {"masks": []}
        entry = _mask_entry(mask_u8, iou, img_w, img_h)
        return {"masks": [entry] if entry else []}

    # Text (optionally restricted to the drawn box): detect all instances.
    if text_prompt:
        within = _norm_box_to_xyxy(box, img_w, img_h) if box is not None else None
        results = _detect_and_segment(
            bundle, pil_image, text_prompt, threshold, exclude_xyxy, within_xyxy=within
        )
        # Box drawn but nothing detected inside it: fall back to segmenting
        # the boxed object directly so the user still gets a mask.
        if not results and within is not None:
            masks = _sam2_masks_for_boxes(bundle, pil_image, [within])
            if masks:
                entry = _mask_entry(masks[0], 1.0, img_w, img_h)
                if entry:
                    results = [entry]
        return {"masks": results}

    # Box only: segment the boxed object (one mask).
    box_xyxy = _norm_box_to_xyxy(box, img_w, img_h)
    masks = _sam2_masks_for_boxes(bundle, pil_image, [box_xyxy])
    if not masks:
        return {"masks": []}
    entry = _mask_entry(masks[0], 1.0, img_w, img_h)
    return {"masks": [entry] if entry else []}


# ---------------------------------------------------------------------------
# Prediction (multi-category, one image)
# ---------------------------------------------------------------------------

def predict_multi(bundle: ModelBundle, request: dict) -> dict:
    """Handle one /segment_multi request: one image, many categories.

    Request: {image_b64, categories: [{name, text_prompt,
              confidence_threshold?, exemplars?}], exclude_boxes?}
    Returns: {results: [{category, masks: [...]}], exemplars_used: bool}

    Exemplars were a SAM concept-grounding feature with no GDINO+SAM2
    equivalent; they are ignored (logged once per request).
    """
    pil_image = decode_image(request["image_b64"])
    img_w, img_h = pil_image.size

    categories = request.get("categories") or []
    exclude_boxes = request.get("exclude_boxes") or []
    exclude_xyxy = [
        _norm_box_to_xyxy(eb, img_w, img_h) for eb in exclude_boxes if len(eb) == 4
    ]

    if any(cat.get("exemplars") for cat in categories):
        log.warning("Exemplar prompts are not supported by the GDINO+SAM2 "
                    "backend — ignoring them")

    out: list[dict] = []
    for cat in categories:
        text = cat.get("text_prompt") or cat.get("name") or ""
        if not text:
            out.append({"category": cat.get("name", ""), "masks": []})
            continue
        threshold = float(cat.get("confidence_threshold", 0.2))
        masks = _detect_and_segment(
            bundle, pil_image, text, threshold, exclude_xyxy
        )
        out.append({"category": cat.get("name", text), "masks": masks})

    return {"results": out, "exemplars_used": False}
