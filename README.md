# Segmentation Labeler with SAM Auto-Labeling

A local desktop labeling tool that pairs fast manual polygon/box labeling with
SAM auto-labeling served from an **on-demand Amazon SageMaker real-time
endpoint**. The client is pure Python (Tkinter + Pillow + boto3) and runs on
Apple Silicon macOS and Linux without a GPU; all GPU inference happens in a
bring-your-own-container (BYOC) image on SageMaker. Nothing bills by the hour
unless you are actively labeling.

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
  - [Architecture Overview](#architecture-overview)
  - [How It Works](#how-it-works)
  - [Component Descriptions](#component-descriptions)
  - [Container Strategy](#container-strategy)
- [Prerequisites](#prerequisites)
- [Installation (UV)](#installation-uv)
- [Deploying the SageMaker Infrastructure (CDK)](#deploying-the-sagemaker-infrastructure-cdk)
- [Endpoint Lifecycle](#endpoint-lifecycle)
  - [In-App Lifecycle](#in-app-lifecycle)
  - [CLI Pre-Start](#cli-pre-start)
  - [Orphan Recovery](#orphan-recovery)
- [How to Use](#how-to-use)
  - [Launching the Labeler](#launching-the-labeler)
  - [Input Modes: Local and S3](#input-modes-local-and-s3)
  - [Manual Labeling Mode](#manual-labeling-mode)
  - [Auto-Labeling Mode (SAM)](#auto-labeling-mode-sam)
  - [Click-to-Label](#click-to-label)
  - [Batch Autolabeling and Review](#batch-autolabeling-and-review)
  - [Edit Mode](#edit-mode)
  - [Navigating Images](#navigating-images)
  - [Managing Annotations](#managing-annotations)
  - [Exporting Labels](#exporting-labels)
- [Hardware & Cost](#hardware--cost)
- [Output Format](#output-format)
  - [YOLO Format](#yolo-format)
  - [COCO Format](#coco-format)
  - [Directory Structure](#directory-structure)
- [Configuration](#configuration)
- [Security](#security)
- [Notices](#notices)

---

## Project Overview

Segmentation Labeler is a Python desktop GUI for creating segmentation masks
and bounding boxes to train computer-vision models. It supports manual polygon
drawing and AI-assisted auto-labeling powered by two Apache-2.0 models:
[Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-base)
(open-vocabulary text-to-box detection) and Meta's
[SAM2](https://huggingface.co/facebook/sam2.1-hiera-large) (box/point-to-mask
segmentation).

The tool is designed for iterative, efficient labeling:

- Navigate large image directories rapidly using keyboard shortcuts.
- Label **boxes, masks, or both** — in "Both" mode a bounding box is derived
  automatically from each mask (and remains adjustable).
- Draw precise polygon segmentation masks manually, or let SAM auto-detect
  objects using bounding-box, text, or click-point prompts.
- **Click-to-label**: click an object (shift/right-click for negative points)
  and SAM segments it.
- **Batch autolabeling**: run per-category text prompts over many images at
  once; proposals arrive as *pending* annotations in a review queue
  (Accept/Reject/Accept-all, jump to next pending image).
- **Edit mode**: adjust boxes with 8 drag handles; move, insert, and delete
  polygon vertices.
- Labels auto-save on every change: YOLO-seg (`<stem>.txt`), YOLO-detection
  (`boxes/<stem>.txt`), and a full-fidelity JSON sidecar (`.meta/<stem>.json`
  with provenance, scores, and review state); optional COCO JSON export.
- Resume labeling sessions where you left off — existing labels are loaded
  automatically at startup.
- Categories persist across images and sessions.

SAM requires a GPU and has known compatibility issues with Apple Silicon
GPUs, so inference runs remotely. The labeling workflow stays fast and local
while heavy inference is offloaded to a SageMaker endpoint.

This is a **single approved workflow**: a local UV client (Tkinter GUI plus a
SageMaker-only backend) talks to an on-demand SageMaker real-time endpoint that
serves the models from a BYOC image with the weights baked in at build time.
There is no SSH tunnel, no self-managed GPU box, and no
long-running resource to forget about — the operator needs AWS credentials and
nothing else. The endpoint exists only while you are labeling, so idle
inference cost is **$0/hr**.

---

## Architecture

### Architecture Overview

The system has two halves that communicate only over the SageMaker data plane
using SigV4-signed HTTPS:

```
                 LOCAL (Apple Silicon / Linux, no GPU)
  +--------------------------------------------------------------------+
  |  UV project (pyproject.toml, uv.lock)                              |
  |                                                                    |
  |  [project.scripts]                                                 |
  |    labeler            -> labeler:main         (Tkinter GUI)        |
  |    start-sam-endpoint -> endpoint_ctl:start  \  shared lifecycle   |
  |    stop-sam-endpoint  -> endpoint_ctl:stop   /  (endpoint_manager) |
  |                                                                    |
  |  +---------------+   +-----------------+   +------------------+     |
  |  | labeler.py    |   | input_source.py |   | endpoint_manager |     |
  |  | Tkinter GUI   |-->| local vs S3     |   | up/down/status   |     |
  |  | annotations.py|   | sync + upload   |   | orphan detect    |     |
  |  +------+--------+   +--------+--------+   | tag guardrail    |     |
  |         |                     |           +--------+---------+     |
  |         v                     |                    |                |
  |  +---------------+            |                    |                |
  |  | sam_client.py|            | boto3 s3           | boto3 sagemaker|
  |  | SageMaker     |            | sync / upload      | + cloudformation|
  |  | Backend       |            |                    |                |
  |  +------+--------+            |                    |                |
  +---------|--------------------|--------------------|-----------------+
            | InvokeEndpoint     | Get/Put/List       | Create/Delete/
            | (SigV4, HTTPS,     | Object             | Describe/List
            |  <= 6 MB, task=..) |                    | Endpoint, DescribeStacks
            v                    v                    v
  ====================================== AWS ===========================
  |                                                                    |
  |  SageMaker real-time endpoint (on demand)                          |
  |  +--------------------------------------------------+              |
  |  | BYOC container (ECR image, network-isolated)     |              |
  |  |  serve_sagemaker.py /ping + /invocations         |              |
  |  |  inference.py: Grounding DINO (text -> boxes)    |              |
  |  |               + SAM2 (boxes/points -> masks)     |              |
  |  |  weights baked into the image at /opt/models     |              |
  |  +--------------------------------------------------+              |
  |                                               S3 image + annotation |
  |  CloudFormation stack (SamLabelerStack):     prefixes (S3 mode)   |
  |  ECR repo + CodeBuild image build, exec role,                      |
  |  CfnModel, CfnEndpointConfig, operator                             |
  |  ManagedPolicy — NOT the endpoint                                  |
  ====================================================================
```

### How It Works

1. The operator launches `labeler` with a local directory, an `s3://` path, or
   nothing (a startup dialog prompts for input/output). `input_source.resolve`
   classifies the source as **local** or **S3**; an S3 source is synced into a
   local cache first (see [Input Modes](#input-modes-local-and-s3)).
2. Images are displayed one at a time on the canvas. Existing label files in
   the output location are loaded automatically, so sessions resume where they
   left off.
3. At launch the app runs an **orphan sweep**: it queries SageMaker for
   managed endpoints (tagged `ManagedBy=sam-labeler`) left running by a prior
   crashed session and offers to delete them.
4. In **Manual mode**, clicks define polygon vertices; closing the polygon
   commits the mask and writes the YOLO label immediately.
5. In **Auto mode**, the app ensures the endpoint exists (creating it on first
   SAM use if needed), confirms it is `InService`, and sends a `health` task to
   verify the model finished loading. The current image is encoded as base64
   PNG and sent as a `segment` request. Before transmission the image is
   downscaled to a 1280 px longest edge, which stays clear of the 6 MB
   `InvokeEndpoint` payload limit and costs nothing in quality (SAM resizes to
   ~1008 px internally). All returned coordinates are normalized, so the
   downscale is invisible to the labels.
6. The container decodes the image and runs a SAM forward pass using the
   box/text/click prompts. Already-labeled regions can optionally be passed as
   negative prompts to suppress re-detection.
7. Masks are converted to simplified polygons via OpenCV contour extraction and
   returned as normalized `[x, y]` coordinates plus model-native boxes. The
   client appends the masks to the current image and saves to disk.
8. For **batch autolabeling** the client calls `segment_multi`: one request per
   image runs every enabled category against a single image encoding.
9. On save/close (and after COCO export) in **S3 mode**, annotations are
   uploaded to a sibling `annotations/` prefix in the source bucket. On normal
   close the app deletes any endpoint **it created this session** — CLI-created
   or pre-existing endpoints are left running.

### Component Descriptions

| Component | Location | Description |
|---|---|---|
| `labeler.py` | Local machine | Main Tkinter GUI. Image display, manual box/polygon drawing, edit mode, review queue, SAM integration, and endpoint-lifecycle wiring on launch/close. `main()` accepts a local dir or an `s3://` path. |
| `annotations.py` | Local machine | Annotation data model and persistence: YOLO-seg, YOLO-detection boxes, `.meta` JSON sidecars, COCO export. |
| `input_source.py` | Local machine | Resolves the image source to a uniform local view (local dir vs `s3://bucket/prefix`), syncs S3 images into a local cache, and uploads annotations back to a sibling `annotations/` prefix. |
| `sam_client.py` | Local machine | `SAMBackend` base (documents the JSON contract, hosts `encode_image`) plus its single SageMaker-only implementation `SageMakerBackend` (boto3 `InvokeEndpoint`, SigV4/HTTPS, payload-size guard, image downscaling). |
| `endpoint_manager.py` | Local machine | Shared endpoint lifecycle used by both the GUI and the CLI: create/delete/status, orphan detection, `ManagedBy` tag guardrail, and stack-output discovery. |
| `endpoint_ctl.py` | Local machine | Thin CLI over `endpoint_manager`. Exposes `start()`/`stop()` (the `uv run start-sam-endpoint` / `stop-sam-endpoint` entry points) and `up`/`down`/`status` argparse subcommands via `uv run python endpoint_ctl.py`. |
| `cdk/` | Deploy-time | Python CDK app: stack-owned ECR repository, a CodeBuild project (+ short-lived custom resource) that builds and pushes the image on AWS by default, execution role, SageMaker Model and EndpointConfig, and an operator IAM policy. Deliberately does **not** create the endpoint. `-c buildLocal=true` falls back to a local Docker image build. |
| `sam_server/Dockerfile` | Deploy-time | BYOC image: CUDA 12.8 + Python 3.12 + `torch==2.10.0` (cu128) + a pinned SAM checkout. Weights are *not* baked in. |
| `sam_server/serve_sagemaker.py` | SageMaker endpoint | SageMaker's `/ping` + `/invocations` contract over `inference.py`; dispatches on the payload's `task` field. |
| `sam_server/inference.py` | SageMaker endpoint | Framework-free model loading + prediction (dtype auto-fallback bf16→fp16→fp32 via `SAM_DTYPE`, checkpoint dir via `SAM_MODEL_DIR`, T4/fp16 fused-MLP patch). |
| `sam_server/benchmark.py` | Deploy-time | Standalone latency/VRAM/throughput benchmark for instance sizing (see `sam_server/HARDWARE.md`). |
| `pyproject.toml` / `uv.lock` | Local machine | UV project metadata, dependencies, and the three `[project.scripts]` entry points. Replaces the old client `requirements.txt`. |

### Container Strategy

The inference container is a **bring-your-own-container (BYOC)** ECR image:
CUDA 12.8 + a pinned PyTorch (`torch==2.10.0`, cu128) + `transformers`, serving
the `/ping` + `/invocations` contract via `serve_sagemaker.py`. The Grounding
DINO and SAM2 checkpoints are **baked into the image** at build time
(`download_models.py`): both are public, Apache-2.0 licensed, and ungated, so
there is no HuggingFace token, no S3 weights bucket, and no upload step — and
the endpoint runs with full network isolation.

By default this image is built **on AWS**: `cdk deploy` runs a native x86_64
CodeBuild project that builds and pushes it to the stack-owned ECR repository,
so no local Docker daemon (or Apple Silicon emulation) is involved. Passing
`-c buildLocal=true` falls back to building the image locally with Docker.

BYOC keeps the CUDA/torch/transformers pins under our control and lets the
image carry the weights, which is what makes network-isolated serving and the
zero-setup operator experience possible. A managed HuggingFace DLC remains a
possible future simplification.

---

## Prerequisites

### Local Machine (Operator)

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| [UV](https://docs.astral.sh/uv/) | Recent | Manages the environment and the `uv run` entry points |
| Tkinter | Bundled with Python | On Linux, may require `sudo apt install python3-tk` |
| AWS credentials | — | Any standard source (`AWS_PROFILE`, SSO, instance role). Needs the operator policy the stack publishes. |
| AWS region | — | Set `AWS_REGION`/`AWS_DEFAULT_REGION` or pass `--region`. |

The client dependencies (installed by UV) are `pillow`, `pillow-heif`, and
`boto3`. There is no `requests` dependency.

### For Deploying the Infrastructure (Deployer, one-time per account)

Only the person who deploys the stack needs these. Everyone else needs
credentials and the operator policy.

| Requirement | Notes |
|---|---|
| AWS CDK CLI | `npm install -g aws-cdk`. The account/region must be bootstrapped once: `cdk bootstrap`. |
| Docker | Only needed for the `-c buildLocal=true` local build. The default path builds the image on AWS CodeBuild, so **no local Docker is required**. |
| [UV](https://docs.astral.sh/uv/) | Manages the CDK app's own virtual environment (separate from the labeler's). `uv` provisions Python 3.12 for it. |
| SAM checkpoint | Gated on HuggingFace — see below. You upload it to S3 yourself; it is never baked into the image. |
| IAM permissions | Enough to create S3, ECR, IAM, and SageMaker resources. |

> **On Apple Silicon:** the default deploy builds the image on AWS CodeBuild
> (native x86_64), so this concern does not apply — no local build, no
> emulation. It matters **only if you pass `-c buildLocal=true`**: SageMaker GPU
> instances are x86_64, so the local build is a cross-platform (`linux/amd64`)
> build under emulation, and that first `cdk deploy` can take 30–60 minutes
> (most of it installing PyTorch under QEMU; subsequent local deploys reuse
> Docker's cache). Sticking with the default CodeBuild path sidesteps this
> entirely.

---

## Installation (UV)

1. Clone this repository:

```bash
git clone <repo-url>
cd labeleler
```

2. Sync the environment (UV creates a virtualenv and installs the pinned
   dependencies from `uv.lock`):

```bash
uv sync
```

3. Run the tool via the UV entry points (defined in `pyproject.toml`
   `[project.scripts]`):

| Command | Maps to | Purpose |
|---|---|---|
| `uv run labeler <input> [output]` | `labeler:main` | Launch the GUI |
| `uv run start-sam-endpoint` | `endpoint_ctl:start` | Pre-start the SageMaker endpoint |
| `uv run stop-sam-endpoint` | `endpoint_ctl:stop` | Stop (delete) the endpoint |

Then set up the SageMaker infrastructure below (deployer, one-time).

---

## Deploying the SageMaker Infrastructure (CDK)

Deploy once per AWS account. After that, the endpoint is created and deleted on
demand (by the app or by `uv run start-sam-endpoint` / `stop-sam-endpoint`).

There is no weights step: the Grounding DINO and SAM2 checkpoints are public
and Apache-2.0, and the image build downloads and bakes them in automatically.

**1. Deploy the stack.**

By default the stack builds the SAM inference image **on AWS** using a native
x86_64 CodeBuild project during `cdk deploy`, then pushes it to a stack-owned
ECR repository. You do **not** need a local Docker daemon for this default path,
and Apple Silicon deployers avoid the slow emulated cross-build entirely. A
custom resource waits for the CodeBuild build to succeed before the SageMaker
model is created. (A local Docker daemon is only required if you opt into the
legacy local build with `-c buildLocal=true` — see the overrides below.)

```bash
cd cdk
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

cdk bootstrap          # once per account/region
cdk deploy
```

> **Why `uv` here too?** The `cdk` app has its own environment, separate from
> the labeler's. Using `uv venv` avoids a common failure: a `python3.12` that
> is itself a uv-managed (python-build-standalone) interpreter cannot bootstrap
> pip via the stock `python -m venv` (`ensurepip` fails). `uv venv` sidesteps
> that entirely. If you prefer stock tooling, use a non-uv Python 3.12 (e.g.
> `brew install python@3.12`).

Useful overrides (all optional):

```bash
cdk deploy \
  -c instanceType=ml.g5.xlarge \      # default: ml.g4dn.xlarge
  -c endpointName=sam-labeler \      # default: sam-labeler
  -c dtype=fp16                       # default: auto (bf16 -> fp16 -> fp32)
  -c buildLocal=true                  # default: build on AWS CodeBuild
```

> **`-c buildLocal=true`** builds the inference image **locally with Docker**
> (the legacy path) instead of on AWS CodeBuild. It requires a running local
> Docker daemon and, on Apple Silicon, is a slow emulated `linux/amd64`
> cross-build. Omit it to use the default CodeBuild path, which needs no local
> Docker.

**2. Grant other users access.** Attach the managed policy from the
`OperatorPolicyArn` output to any user or role that should run the tool. It
allows exactly create, delete, describe, and invoke on this one endpoint — no
permission to change infrastructure:

```bash
aws iam attach-user-policy --user-name alice --policy-arn <OperatorPolicyArn>
```

**What the stack creates** (Req 10, 13.1 — nothing bills by the hour):

| Resource | Cost when idle | Notes |
|---|---|---|
| ECR repository (image) | Storage only | Owned by the stack; receives the built image. Rebuilt only when the `Dockerfile`, `inference.py`, `serve_sagemaker.py`, or container `requirements.txt` change. It is `DESTROY` + emptied on delete, so `cdk destroy` removes it (no orphaned repo/storage cost) |
| CodeBuild project | Free | Native x86_64 project that builds and pushes the image during `cdk deploy` (default path). No cost when idle |
| Image-build custom resource | Free | Short-lived resource that triggers the CodeBuild build and waits for it to succeed before the model is created |
| SageMaker execution role | Free | Least-privilege: pull image, write endpoint logs |
| Operator managed policy | Free | Scoped to create/delete/describe/invoke on the single named endpoint |
| SageMaker Model + EndpointConfig | **Free** | Definitions only; nothing is running |
| SageMaker Endpoint | — | **Not created by CDK.** Created on demand. |

Stack outputs (`EndpointName`, `EndpointConfigName`, `InstanceType`,
`OperatorPolicyArn`) drive client-side discovery, so a redeployed stack needs
no flag changes.

---

## Endpoint Lifecycle

A SageMaker real-time endpoint is the only resource here that bills by the
hour, so the CDK stack does not create one. It is created on demand and, when
the app created it, deleted automatically.

### In-App Lifecycle

- The app **creates the endpoint on first SAM use** in a session (a launch that
  never touches SAM costs nothing), applying the `ManagedBy=sam-labeler` tag.
- On **normal close** the app **deletes only the endpoint it created this
  session** — so it is **$0/hr when idle** (Req 3.3, 3.4). A CLI-created or
  pre-existing endpoint is left running.

### CLI Pre-Start

The endpoint cold start is multi-minute (pulling a multi-GB image and loading
the models). To avoid waiting inside a labeling
session, pre-start it from the CLI:

```bash
uv run start-sam-endpoint         # create it (waits for InService by default)
uv run start-sam-endpoint --no-wait
uv run stop-sam-endpoint          # delete it — back to $0/hr
```

A CLI-created endpoint is **not** owned by the app, so closing the labeler does
**not** delete it — stop it explicitly with `uv run stop-sam-endpoint`. This
lets a second session or a running batch job survive a window close.

A typical pre-start session:

```bash
uv run start-sam-endpoint
uv run labeler ~/data/images ~/data/labels
#   ... label, then close the window ...
uv run stop-sam-endpoint
```

Both entry points accept the same optional flags (`--endpoint`, `--region`,
`--profile`, `--stack`, `--no-wait`, and `--force` on stop). For a full
subcommand UI including a cost/status readout, the underlying script also
offers `up` / `down` / `status` (run it through `uv run` so it uses the
project environment — note there is no `uv run` shortcut for `status`, only the
`start-sam-endpoint` / `stop-sam-endpoint` entry points):

```bash
uv run python endpoint_ctl.py up
uv run python endpoint_ctl.py status   # exists? InService? billable?
uv run python endpoint_ctl.py down
```

`status` always reports whether a **billable** endpoint is currently running
(Req 13.3).

### Orphan Recovery

If a session crashes or is force-quit, the endpoint it created can be left
running. On launch the app queries SageMaker for endpoints carrying the
`ManagedBy=sam-labeler` tag that this session did not create, and offers to
delete any that are still running (Req 4.1–4.3).

**Tag guardrail.** The tool deletes an endpoint **only** if it carries the
`ManagedBy=sam-labeler` tag, unless you pass an explicit `--force` override
(Req 4.4, 5.4). This prevents a name collision from tearing down someone else's
endpoint.

---

## How to Use

### Launching the Labeler

**With arguments:**

```bash
uv run labeler /path/to/images /path/to/output_labels
uv run labeler s3://my-bucket/datasets/run1/images
```

- `input`: a directory of images **or** an `s3://bucket/prefix` path. Supported
  image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.heic`, `.heif`, `.tif`,
  `.tiff`, `.webp`.
- `output` (optional, local mode): directory where label files are written.
  Defaults to `<input_name>_labels` beside the input. Ignored in S3 mode (see
  below).

**Backend options** (all optional; the tool is locked to the SageMaker backend):

| Flag | Default | Purpose |
|---|---|---|
| `--endpoint NAME` | `sam-labeler` | SageMaker endpoint name |
| `--region REGION` | from environment | AWS region |
| `--profile PROFILE` | from environment | AWS profile |

**Without arguments (startup dialog):**

```bash
uv run labeler
```

A startup dialog appears prompting for the input path (local dir or `s3://`
path) and an optional output directory. If existing label files are found, the
tool opens at the first unlabeled image.

### Input Modes: Local and S3

The tool supports two input modes (Req 2):

**Local mode** — the input is a readable local directory. Images are read in
place, and annotations are written to a **local output directory** only (the
`<input_name>_labels` sibling, or the `output` you supply). Nothing is uploaded.

**S3 mode** — the input is an `s3://bucket/prefix` path. On startup the images
under that prefix are **synced to a local cache**
(`~/.cache/sam-labeler/<bucket>/<hash>/images/`) and the GUI labels from the
cache. The sync is idempotent, so re-opening the same source is cheap and does
not clobber local edits. On **save, on normal close, and after COCO export**,
the annotations are **uploaded** to a sibling `annotations/` prefix in the
**same bucket** that supplied the images. An unresolvable source (neither a
readable directory nor a well-formed `s3://` URI) reports an input-source error
and halts startup.

### Manual Labeling Mode

Manual mode is the default. To draw a segmentation mask:

1. Select or add a **category** from the right panel.
2. Click on the canvas to place the first polygon vertex.
3. Continue clicking to add vertices. A live preview line follows your cursor.
4. When 3 or more points exist, a red circle appears around the first point.
   Click within it to close and commit the polygon.

The mask is saved to disk immediately.

**Keyboard shortcuts in manual mode:**

| Key | Action |
|---|---|
| `z` | Undo the last vertex |
| `Esc` | Cancel the entire in-progress polygon |
| `a` / `A` | Previous image |
| `d` / `D` | Next image |

### Auto-Labeling Mode (SAM)

1. Switch to **Auto (SAM)** mode using the radio button in the right panel.
2. Click **Connect**. The app confirms the endpoint is `InService` and that the
   model finished loading. If the endpoint does not exist yet, it is created on
   first use (this is the multi-minute cold start; pre-start it with
   `uv run start-sam-endpoint` to skip the wait). Connection status is shown
   below the controls.
3. Provide one or both prompts:
   - **Bounding box**: click and drag on the canvas to draw a box around the
     target object.
   - **Text prompt**: enter a description of the object (e.g. `"circuit board"`,
     `"red cell"`).
4. Adjust the **Confidence** slider (0.05–0.95). Lower values return more masks;
   higher values return fewer but more certain masks.
5. Optionally enable **Exclude already-labeled regions** to pass existing masks
   as negative prompts, preventing SAM from re-detecting objects you already
   labeled.
6. Click **Run SAM** or press `Enter`.

Returned masks are added to the current image and saved to disk immediately.

### Click-to-Label

In Auto mode, switch the **Prompt** toggle to **Click**. Left-click the object
you want segmented; shift-click or right-click adds a negative point to exclude
a region. Press **Run SAM** (or `Enter`) to segment. `z` removes the last
point, `Esc` clears all points.

### Batch Autolabeling and Review

1. In Auto mode, click **Autolabel…**.
2. Choose the image scope (all unlabeled, or the next N images), enable
   categories, and edit each category's text prompt (saved to
   `.meta/prompts.json` for next time).
3. Proposals are written as **pending** annotations (dashed outlines, score
   shown). They are stored only in the `.meta` sidecar — never in the YOLO
   training files — until accepted.
4. Review with the strip above the annotation list: **Accept** (`Space`),
   **Reject** (`x`), **Accept all on image**, and **Next pending →** to jump to
   the next image with proposals.

### Edit Mode

Switch to **Edit** mode (or double-click an annotation) to correct geometry:

- Drag the 8 white handles to resize a bounding box (marks it user-edited, so it
  no longer auto-follows the mask).
- Drag yellow vertex squares to move polygon points; click on a polygon edge to
  insert a vertex; right-click a vertex to delete it.
- `Esc` exits editing. All changes save immediately.

### Navigating Images

| Key | Action |
|---|---|
| `d` / `D` | Next image |
| `a` / `A` | Previous image |

Labels are saved automatically. You do not need to manually save before
navigating. Clicking an existing mask on the canvas selects it in the mask list.

### Managing Annotations

The **Annotations** list in the right panel shows every annotation on the
current image: a `◆` glyph for masks and `▣` for box-only annotations, with the
category name, vertex count, source (`[sam box]`, `[sam point]`, `[auto]`),
and review state for pending proposals.

| Action | How |
|---|---|
| Select an annotation | Click on it in the list, or click it in the canvas |
| Delete an annotation | Select it, then click **Delete** |
| Change its category | Select it, then click **Change category** and pick from the dropdown |
| Undo last vertex (manual) | Press `z` |
| Delete last committed annotation | Press `z` when not mid-polygon |

### Exporting Labels

YOLO-format labels are saved automatically after every change. No manual export
is required.

To export in **COCO JSON format**, click the **Export COCO** button in the right
panel. This reads all label files in the output directory (across all images in
the session) and writes a single `labels.json`. In S3 mode the export is
included in the annotation upload set.

---

## Hardware & Cost

Benchmarked on a g4dn (Tesla T4): ~5.3 GB peak VRAM, ~0.5 s per prompt,
~67 images/min for 5-category batch autolabeling in fp16 — roughly
**$0.13 per 1k autolabeled images** at g4dn.xlarge on-demand pricing. See
[`sam_server/HARDWARE.md`](sam_server/HARDWARE.md) for the full measurements,
the benchmark script (`sam_server/benchmark.py`), and the instance comparison.

SageMaker instances run roughly 15–25% above raw EC2 (`ml.g4dn.xlarge` ≈
$0.74/hr), the premium for not administering the box. Because the endpoint only
exists while you are labeling, the figure that usually matters is the
per-session cost: an hour of labeling is well under a dollar, and idle inference
cost is **$0/hr**.

**Verify current pricing before committing** — the rates in `endpoint_ctl.py`'s
cost hints and in `HARDWARE.md` are indicative only.

---

## Output Format

### YOLO Format

One `.txt` file per image, named `<image_stem>.txt`, written to the output
directory. Each line is one segmentation mask:

```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```

All coordinates are normalized to `[0, 1]` relative to the image dimensions.
Compatible with YOLOv11 segmentation training.

**Example:**
```
0 0.312500 0.156250 0.437500 0.093750 0.500000 0.187500 0.375000 0.250000
1 0.650000 0.420000 0.700000 0.380000 0.750000 0.430000 0.690000 0.480000
```

A `classes.txt` file in the output directory maps class indices to names:

```
class_name_0
class_name_1
...
```

### COCO Format

Exported on demand via the **Export COCO** button. Produces `labels.json` in the
output directory, conforming to the COCO instance-segmentation format:

```json
{
  "info": { "description": "Exported by Segmentation Labeler", "date_created": "YYYY-MM-DD" },
  "images": [{ "id": 1, "file_name": "image.jpg", "width": 1920, "height": 1080 }],
  "categories": [{ "id": 0, "name": "category_name" }],
  "annotations": [{
    "id": 1,
    "image_id": 1,
    "category_id": 0,
    "segmentation": [[x1, y1, x2, y2, ...]],
    "area": 12345.6,
    "bbox": [x, y, width, height],
    "iscrowd": 0
  }]
}
```

Segmentation coordinates in the COCO export are in absolute pixels (not
normalized).

### Directory Structure

```
output_labels/
├── classes.txt          # Category index -> name mapping
├── image_001.txt        # YOLO-seg label (accepted masks only)
├── image_002.txt
├── boxes/
│   ├── image_001.txt    # YOLO-detection label: cid cx cy w h (accepted boxes)
│   └── image_002.txt
├── .meta/
│   ├── image_001.json   # Full annotation records: provenance, review state, scores
│   ├── image_002.json
│   └── prompts.json     # Per-category text prompts for batch autolabeling
├── ...
└── labels.json          # COCO export (generated on demand)
```

The `.meta` sidecar is authoritative when present; directories containing only
plain YOLO-seg `.txt` files (from older versions) load seamlessly as manual,
accepted annotations. Pending autolabel proposals live only in the sidecar and
never appear in the YOLO files until accepted.

In S3 mode this same tree lives under the local cache and is uploaded to the
`annotations/` prefix of the source bucket on save/close/export.

---

## Configuration

The following parameters can be adjusted directly in the source files.

**`labeler.py`**

| Parameter | Default | Description |
|---|---|---|
| `MAX_W` | `1100` | Maximum canvas display width in pixels |
| `MAX_H` | `800` | Maximum canvas display height in pixels |
| `CLOSE_RADIUS` | `10` | Pixel radius within which clicking the first point closes a polygon |
| Default endpoint | `sam-labeler` | Pre-filled SageMaker endpoint name (override with `--endpoint`) |

**`sam_client.py`**

| Parameter | Default | Description |
|---|---|---|
| `DEFAULT_MAX_SIDE` | `1280` | Longest image edge sent to SageMaker. SAM resizes to ~1008 px internally, so this costs no quality; raising it risks the 6 MB payload limit |
| `SAGEMAKER_MAX_PAYLOAD` | `6 MB` | AWS hard limit on `InvokeEndpoint`; exceeding it raises a clear error rather than a 413 |
| `SAGEMAKER_INVOKE_TIMEOUT` | `60.0 s` | AWS hard cap on a single real-time invocation; caps the effective `request_timeout` |

**`input_source.py`**

| Parameter | Default | Description |
|---|---|---|
| `CACHE_ROOT` | `~/.cache/sam-labeler` | Root of the per-source S3 image/annotation cache |
| `SUPPORTED_IMAGE_EXTENSIONS` | jpg/png/… | Image suffixes synced from an S3 prefix |

**`endpoint_manager.py` / `endpoint_ctl.py`**

| Parameter | Default | Description |
|---|---|---|
| `DEFAULT_STACK` | `SamLabelerStack` | CloudFormation stack read for endpoint/config names |
| `DEFAULT_ENDPOINT` | `sam-labeler` | Endpoint name when stack lookup is skipped |
| `MANAGED_TAG` | `ManagedBy=sam-labeler` | Tag applied on create and checked before delete |
| create `timeout` | `1800 s` | Seconds to wait for `InService` |

**`cdk/sam_labeler_stack.py`** (override via `cdk deploy -c key=value`)

| Context key | Default | Description |
|---|---|---|
| `instanceType` | `ml.g4dn.xlarge` | Endpoint instance type; see [Hardware & Cost](#hardware--cost) |
| `endpointName` | `sam-labeler` | Endpoint name, also scoped into the operator policy |
| `modelPrefix` | `sam-model` | S3 prefix mounted at `/opt/ml/model` |
| `dtype` | *(auto)* | Sets `SAM_DTYPE` in the container |

**`sam_server/Dockerfile`** (override via `--build-arg`)

| Build arg | Default | Description |
|---|---|---|
| `SAM_REF` | pinned commit | SAM repo commit to build against |
| `TORCH_VERSION` | `2.10.0` | Also pinned as a pip constraint so nothing downgrades it |
| `CUDA_CHANNEL` | `cu128` | PyTorch wheel index matching the base image's CUDA |

**`sam_server/inference.py`**

| Parameter | Default | Description |
|---|---|---|
| `SAM2_MODEL_DIR` (env) | `/opt/models` | Directory with `gdino/` and `sam2/` checkpoints (baked by the Dockerfile via `download_models.py`) |
| `SAM2_MODEL_ID` / `GDINO_MODEL_ID` (env / build args) | `facebook/sam2.1-hiera-large` / `IDEA-Research/grounding-dino-base` | Which checkpoints to bake and load |
| `SAM2_DTYPE` (env) | auto | Autocast dtype override: `bf16` \| `fp16` \| `fp32`. Auto picks bf16 on Ampere+ GPUs, fp16 on older CUDA GPUs (T4), fp32 elsewhere |

---

## Security

Designed for public release with AWS security controls in place (Req 12).

### Encryption

- **In transit.** `InvokeEndpoint` is a SigV4-signed AWS API call over HTTPS,
  and all S3 calls (image/annotation sync) go over HTTPS via botocore.

### Network Posture

The endpoint has no public URL: `InvokeEndpoint` is authorized by IAM and
signed with SigV4, encrypted by TLS. There is no inbound network path to open
and no port to expose. The container is reachable solely through the SageMaker
data plane.

### Least-Privilege IAM

- **Execution role.** Pulls the ECR image and writes endpoint logs — nothing
  else. There is no S3 access at all.
- **Operator policy.** Grants create/delete/describe/invoke scoped to the single
  named endpoint ARN (plus the unavoidable `ListEndpoints` /
  `DescribeEndpointConfig` reads). Handing it to a user does not let them change
  the model, the image, or the instance type.

### Model Weights

The Grounding DINO and SAM2 checkpoints are public and Apache-2.0 licensed.
They are baked into the container image at build time from the official
HuggingFace repos, pinned by model id, and served with `HF_HUB_OFFLINE=1` and
full SageMaker network isolation — the running container makes no outbound
calls. (`*.pt` remains gitignored so no checkpoint is ever committed.)

### Reproducibility

- `uv.lock` pins the full client dependency graph.
- The `Dockerfile` pins the model ids (`SAM2_MODEL_ID`, `GDINO_MODEL_ID`) and the PyTorch version
  (`TORCH_VERSION`), so a rebuild cannot silently change model behavior.

### Data Handling

- Images are sent as base64-encoded PNG over TLS to SageMaker; on the wire they
  are downscaled to a 1280 px longest edge, so full-resolution originals never
  leave the machine.
- No image data is stored server-side — images are decoded in memory for
  inference and discarded after the response.
- Label files are written to the local output directory. In S3 mode they are
  additionally uploaded to the `annotations/` prefix of the source bucket; no
  other label data is transmitted anywhere.
- SageMaker writes container logs to CloudWatch
  (`/aws/sagemaker/Endpoints/<name>`). These contain timing and error
  information, but a request carries base64 image data, so treat the endpoint
  log group as potentially sensitive and avoid logging request bodies.

### Recommendations

- Attach the operator policy to users rather than granting broad `sagemaker:*` —
  it is scoped to one endpoint by design.
- Set a billing alarm or AWS Budget. The endpoint is cheap per hour but is not
  free if a session is left running; `uv run python endpoint_ctl.py status` (or
  `uv run stop-sam-endpoint`) reports/handles a running endpoint, and the orphan
  sweep catches ones left by a crash.
- Keep `SAM_REF` pinned in the `Dockerfile` (it already is) rather than
  tracking `main`.

---

## Troubleshooting

The two Docker-related entries below apply **only to the `-c buildLocal=true`
local build**. The default CodeBuild path builds the image in AWS and needs no
local Docker daemon.

**`cdk deploy` fails building the image: `docker-credential-desktop` (or
`-osxkeychain`) not found.** Your `~/.docker/config.json` has a `credsStore`
pointing at a credential helper that is not on `PATH` — commonly a leftover
from an uninstalled Docker Desktop (the `/usr/local/bin/docker-credential-*`
symlinks dangle). Fix by pointing `credsStore` at a helper you actually have,
or remove the `credsStore` line entirely so Docker stores the ECR token in the
config file:

```json
{ "auths": { "<acct>.dkr.ecr.<region>.amazonaws.com": {} } }
```

**`cdk deploy` fails: cannot connect to the Docker daemon.** No daemon is
running. Start Docker Desktop, or install and start Colima
(`brew install colima && colima start`). Colima provides `linux/amd64`
emulation, which the SAM image cross-build requires on Apple Silicon.

**`python3.12 -m venv` fails with an `ensurepip` non-zero exit.** Your
`python3.12` is a uv-managed (python-build-standalone) interpreter, which can't
bootstrap pip through the stock `venv`. Use `uv venv --python 3.12 .venv`
instead (see the CDK deploy step), or install a non-uv Python 3.12.


---

## Notices

This tool is provided for research and dataset-preparation purposes. Users are
responsible for ensuring that the images they label comply with applicable data
use agreements, privacy regulations, and licensing terms.

The SAM2 model and weights are Apache-2.0 licensed by Meta
([facebook/sam2](https://github.com/facebookresearch/sam2)); Grounding DINO is
Apache-2.0 licensed by IDEA-Research
([IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)).

This software is provided "as is" without warranties of any kind. The authors
make no guarantees regarding the accuracy of automatically generated
segmentation masks. All auto-generated labels should be reviewed and corrected
by a human annotator before use in model training.
