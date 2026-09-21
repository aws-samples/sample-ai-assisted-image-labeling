#!/usr/bin/env python3
"""Download the Grounding DINO + SAM2 checkpoints for offline serving.

Saves both models under <output_dir>/gdino and <output_dir>/sam2 — the layout
inference.load_model() looks for via SAM2_MODEL_DIR. Both repos are public and
Apache-2.0 licensed, so no HuggingFace token is needed.

Used at Docker build time to bake weights into the image (the endpoint runs
with network isolation, so it cannot download at startup), and useful locally:

    python download_models.py /opt/models
    python download_models.py ~/models --sam2 facebook/sam2.1-hiera-base-plus
"""
from __future__ import annotations

import argparse

DEFAULT_SAM2_ID = "facebook/sam2.1-hiera-large"
DEFAULT_GDINO_ID = "IDEA-Research/grounding-dino-base"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("output_dir", help="Directory to save checkpoints into")
    ap.add_argument("--sam2", default=DEFAULT_SAM2_ID, help="SAM2 model id")
    ap.add_argument("--gdino", default=DEFAULT_GDINO_ID,
                    help="Grounding DINO model id")
    args = ap.parse_args()

    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    print(f"Downloading {args.gdino} ...")
    AutoProcessor.from_pretrained(args.gdino).save_pretrained(
        f"{args.output_dir}/gdino"
    )
    AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino).save_pretrained(
        f"{args.output_dir}/gdino"
    )

    print(f"Downloading {args.sam2} ...")
    Sam2Processor.from_pretrained(args.sam2).save_pretrained(
        f"{args.output_dir}/sam2"
    )
    Sam2Model.from_pretrained(args.sam2).save_pretrained(
        f"{args.output_dir}/sam2"
    )

    print(f"Saved to {args.output_dir}/gdino and {args.output_dir}/sam2")


if __name__ == "__main__":
    main()
