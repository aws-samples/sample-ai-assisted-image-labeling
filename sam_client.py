"""SAM remote client — SageMaker backend.

  SageMakerBackend   SigV4 InvokeEndpoint against a SageMaker real-time
                     endpoint. One /invocations route, so the payload carries
                     a "task" discriminator that the container dispatches on.

It returns polygons and boxes normalized to [0, 1], which is what makes
client-side downscaling (see `encode_image`) lossless with respect to the
coordinates the labeler stores.
"""
from __future__ import annotations

import base64
import io
import json
from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image

# The segmentation models resize inputs internally (see
# sam_server/HARDWARE.md), so shrinking anything larger costs no mask
# quality while keeping us clear of SageMaker's 6 MB InvokeEndpoint payload
# ceiling. A 1280 px PNG base64-encodes to roughly 1-3 MB.
DEFAULT_MAX_SIDE = 1280
SAGEMAKER_MAX_PAYLOAD = 6 * 1024 * 1024

# SageMaker real-time endpoints hard-cap a single InvokeEndpoint call at 60 s.
SAGEMAKER_INVOKE_TIMEOUT = 60.0


def encode_image(image: Image.Image, max_side: Optional[int] = DEFAULT_MAX_SIDE) -> str:
    """Base64-encode a PIL image as PNG, optionally downscaling first.

    Downscaling is safe because every coordinate the server returns is
    normalized to [0, 1] against the image it was given.
    """
    if max_side:
        longest = max(image.size)
        if longest > max_side:
            ratio = max_side / longest
            new_size = (
                max(1, round(image.width * ratio)),
                max(1, round(image.height * ratio)),
            )
            image = image.resize(new_size, Image.LANCZOS)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class SAMBackend(ABC):
    """Transport-agnostic interface the labeler talks to."""

    #: Longest image edge sent to the server; None disables downscaling.
    max_side: Optional[int] = DEFAULT_MAX_SIDE

    def __init__(self) -> None:
        self.connected = False

    # -- lifecycle ----------------------------------------------------------

    @abstractmethod
    def connect(self, timeout: float = 10.0) -> None:
        """Make the backend ready, raising ConnectionError if it is not."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release any transport resources. Must be safe to call twice."""

    @abstractmethod
    def describe(self) -> str:
        """Short human-readable target, for GUI status lines."""

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass

    # -- inference ----------------------------------------------------------

    @abstractmethod
    def _invoke(self, task: str, payload: dict, request_timeout: float) -> dict:
        """Send one request and return the decoded JSON response."""

    def _encode(self, image: Image.Image) -> str:
        return encode_image(image, self.max_side)

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("Not connected. Call connect() first.")

    def health(self, request_timeout: float = 15.0) -> dict:
        """Return the server's health payload (device, dtype, status)."""
        self._require_connected()
        return self._invoke("health", {}, request_timeout)

    def segment(
        self,
        image: Image.Image,
        text_prompt: str = "",
        box: Optional[list[float]] = None,
        exclude_boxes: Optional[list[list[float]]] = None,
        points: Optional[list[list[float]]] = None,
        point_labels: Optional[list[int]] = None,
        instance_mode: bool = False,
        confidence_threshold: float = 0.2,
        request_timeout: float = 60.0,
    ) -> list[dict]:
        """Segment one image with one prompt set.

        Args:
            image: PIL image to segment.
            text_prompt: Optional text description of the target object.
            box: Optional bounding box [cx, cy, w, h] normalized to [0, 1].
            exclude_boxes: Bounding boxes of already-labeled regions on this
                image to suppress as negative prompts, same [cx, cy, w, h]
                format.
            points: Optional click prompts [[nx, ny], ...] normalized to [0, 1].
            point_labels: Per-point labels, 1 = positive, 0 = negative.
            instance_mode: Use the SAM1-style single-object head (requires a
                checkpoint built with instance interactivity).
            confidence_threshold: Minimum score for a mask to be returned (0-1).
            request_timeout: Per-request timeout in seconds.

        Returns:
            List of dicts: [{"polygon": [[nx, ny], ...], "score": float,
                             "box": [cx, cy, w, h] | None}, ...]
        """
        self._require_connected()

        payload: dict = {
            "image_b64": self._encode(image),
            "text_prompt": text_prompt,
            "confidence_threshold": confidence_threshold,
        }
        if box is not None:
            payload["box"] = box
        if exclude_boxes:
            payload["exclude_boxes"] = exclude_boxes
        if points:
            payload["points"] = points
            payload["point_labels"] = point_labels or [1] * len(points)
            payload["instance_mode"] = instance_mode

        data = self._invoke("segment", payload, request_timeout)
        return data.get("masks", [])

    def segment_multi(
        self,
        image: Image.Image,
        categories: list[dict],
        exclude_boxes: Optional[list[list[float]]] = None,
        request_timeout: float = 120.0,
    ) -> dict:
        """Run several category prompts against one image in a single request.

        Args:
            image: PIL image to segment.
            categories: [{"name": str, "text_prompt": str,
                          "confidence_threshold": float,
                          "exemplars": [{"image_b64", "box"}] | None}, ...]
            exclude_boxes: Regions to suppress, [cx, cy, w, h] normalized.
            request_timeout: Per-request timeout in seconds.

        Returns:
            {"results": [{"category": str, "masks": [...]}, ...],
             "exemplars_used": bool}
        """
        self._require_connected()

        payload: dict = {
            "image_b64": self._encode(image),
            "categories": categories,
        }
        if exclude_boxes:
            payload["exclude_boxes"] = exclude_boxes

        return self._invoke("segment_multi", payload, request_timeout)


class SageMakerBackend(SAMBackend):
    """Invokes a SageMaker real-time endpoint serving the SAM container."""

    def __init__(
        self,
        endpoint_name: str = "sam-labeler",
        region: Optional[str] = None,
        profile: Optional[str] = None,
        max_side: Optional[int] = DEFAULT_MAX_SIDE,
    ):
        super().__init__()
        self.endpoint_name = endpoint_name
        self.region = region
        self.profile = profile
        self.max_side = max_side
        self._session = None
        self._runtime_clients: dict[float, object] = {}

    def describe(self) -> str:
        region = self.region or "default region"
        return f"sagemaker://{self.endpoint_name} ({region})"

    # -- boto3 plumbing -----------------------------------------------------

    def _boto_session(self):
        if self._session is None:
            try:
                import boto3
            except ImportError as e:
                raise RuntimeError(
                    "boto3 is required for the SageMaker backend.\n"
                    "    pip install boto3      (or: uv pip install boto3)"
                ) from e
            self._session = boto3.Session(
                profile_name=self.profile, region_name=self.region
            )
        return self._session

    def _runtime(self, read_timeout: float):
        """Cache one runtime client per read timeout (botocore sets it per client)."""
        key = round(min(read_timeout, SAGEMAKER_INVOKE_TIMEOUT) + 5, 1)
        if key not in self._runtime_clients:
            from botocore.config import Config

            self._runtime_clients[key] = self._boto_session().client(
                "sagemaker-runtime",
                config=Config(
                    connect_timeout=10,
                    read_timeout=key,
                    retries={"max_attempts": 2, "mode": "standard"},
                ),
            )
        return self._runtime_clients[key]

    # -- lifecycle ----------------------------------------------------------

    def connect(self, timeout: float = 10.0) -> None:
        """Verify the endpoint exists and is InService, then ping the model.

        Does not create the endpoint — use `endpoint_ctl.py up` for that, so
        that a stray GUI click can never start a GPU instance.
        """
        self.connected = False
        sm = self._boto_session().client("sagemaker")

        try:
            desc = sm.describe_endpoint(EndpointName=self.endpoint_name)
        except Exception as e:
            if "ValidationException" in type(e).__name__ or "Could not find" in str(e):
                raise ConnectionError(
                    f"No SageMaker endpoint named {self.endpoint_name!r}.\n"
                    f"Bring it up first:\n"
                    f"    python endpoint_ctl.py up --endpoint {self.endpoint_name}"
                ) from e
            raise ConnectionError(f"Could not describe endpoint: {e}") from e

        status = desc.get("EndpointStatus")
        if status != "InService":
            hint = ""
            if status == "Creating":
                hint = (
                    "\nIt is still starting (typically 7-10 min). "
                    "Wait, then click Connect again."
                )
            elif status == "Failed":
                hint = f"\nFailure reason: {desc.get('FailureReason', 'unknown')}"
            raise ConnectionError(
                f"Endpoint {self.endpoint_name!r} is {status}, not InService.{hint}"
            )

        # InService only means the container passed /ping; confirm the model
        # itself finished loading before we report success.
        self.connected = True
        try:
            health = self.health(request_timeout=timeout)
        except Exception as e:
            self.connected = False
            raise ConnectionError(f"Endpoint is up but /invocations failed: {e}") from e

        if health.get("status") != "ok":
            self.connected = False
            raise ConnectionError(
                f"Model not ready on {self.endpoint_name!r}: "
                f"{health.get('status', 'unknown')}"
            )

    def disconnect(self) -> None:
        """Drop cached clients. The endpoint keeps running; use `down` to stop it."""
        self._runtime_clients.clear()
        self.connected = False

    # -- inference ----------------------------------------------------------

    def _invoke(self, task: str, payload: dict, request_timeout: float) -> dict:
        body = json.dumps({"task": task, **payload}).encode("utf-8")
        if len(body) > SAGEMAKER_MAX_PAYLOAD:
            raise ValueError(
                f"Request is {len(body) / 1e6:.1f} MB, over SageMaker's 6 MB limit. "
                f"Lower max_side (currently {self.max_side}) or send fewer "
                f"categories/exemplars per request."
            )

        resp = self._runtime(request_timeout).invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=body,
        )
        return json.loads(resp["Body"].read())
