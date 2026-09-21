#!/usr/bin/env python3
"""SageMaker serving shim — implements the bring-your-own-container contract.

SageMaker hosting requires exactly two HTTP routes on port 8080:

    GET  /ping         200 once the container is ready to serve
    POST /invocations  the single inference route

Because there is only one route, requests carry a "task" discriminator
("health" | "segment" | "segment_multi") that maps onto the framework-free
handlers in inference.py. The request/response JSON for each task is the
contract that sam_client.py's SageMakerBackend speaks.

The model is loaded during startup, before uvicorn binds the port. SageMaker
retries /ping until it succeeds, so a slow load simply delays InService
rather than failing it.

Usage (SageMaker runs `docker run <image> serve`):
    python serve_sagemaker.py serve

Environment:
    SAM2_MODEL_DIR         checkpoint directory (weights are baked into the
                           image at /opt/models; see inference.py)
    SAM2_DTYPE             bf16 | fp16 | fp32 (default: auto)
    SAGEMAKER_BIND_TO_PORT port override; defaults to 8080
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

import inference

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("sam.sagemaker")

_bundle: Optional[inference.ModelBundle] = None
_load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bundle, _load_error
    try:
        _bundle = inference.load_model()
    except Exception as e:
        # Surface the reason through /ping and the container logs instead of
        # crashing silently, which SageMaker reports only as a health failure.
        _load_error = f"{type(e).__name__}: {e}"
        log.exception("Model load failed")
    yield


app = FastAPI(title="SAM SageMaker Container", lifespan=lifespan)


@app.get("/ping")
def ping():
    """SageMaker health check: 200 = ready, anything else = not ready."""
    if _bundle is None:
        return PlainTextResponse(_load_error or "loading", status_code=503)
    return PlainTextResponse("", status_code=200)


@app.post("/invocations")
async def invocations(request: Request):
    """Single inference route; dispatches on the payload's "task" field."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Body is not valid JSON: {e}")

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    task = body.get("task", "segment")

    if task == "health":
        if _bundle is None:
            return JSONResponse(
                {"status": _load_error or "loading"}, status_code=503
            )
        return {
            "status": "ok",
            "device": str(_bundle.device),
            "dtype": _bundle.dtype_name,
        }

    if _bundle is None:
        raise HTTPException(status_code=503, detail=_load_error or "Model not loaded")

    handlers = {"segment": inference.predict, "segment_multi": inference.predict_multi}
    handler = handlers.get(task)
    if handler is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task {task!r}; expected health, segment, or segment_multi",
        )

    if "image_b64" not in body:
        raise HTTPException(status_code=400, detail="Missing required field image_b64")

    try:
        return handler(_bundle, body)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Inference failed for task=%s", task)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


def main() -> None:
    # SageMaker invokes the image as `docker run <image> serve`; accept and
    # ignore that argument so the same file runs locally without it.
    if len(sys.argv) > 1 and sys.argv[1] not in ("serve", "--serve"):
        print(f"Unknown argument {sys.argv[1]!r}; expected 'serve'", file=sys.stderr)
        raise SystemExit(2)

    port = int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080"))
    log.info("Starting SAM SageMaker container on port %d", port)
    # SageMaker hosting requires the container to listen on all interfaces on
    # the assigned port; the endpoint is reachable only via the SageMaker data
    # plane (IAM + SigV4), never a public network path. nosec B104.
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, access_log=False)  # nosec B104


if __name__ == "__main__":
    main()
