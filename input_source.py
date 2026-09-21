"""Input-source abstraction for the SAM 3 labeler.

The labeler operates against a local directory of images and a local output
directory. This module resolves whatever the operator passed as an image
source into that uniform local view, so the rest of the GUI keeps working
unchanged whether the images live on the local filesystem or in S3.

Two modes are supported (Req 2.1, 2.2, 2.5):

* ``local`` -- the source is a readable local directory. Images are read in
  place and annotations are written to a local output directory (the
  ``<input_name>_labels`` sibling convention from ``labeler.py``, unless the
  operator supplies an explicit output directory).
* ``s3`` -- the source is a well-formed ``s3://bucket/prefix`` URI. Images are
  mirrored into a deterministic per-source cache under
  ``~/.cache/sam-labeler/`` and annotations are written to a sibling cache
  directory; both are later synced against the bucket by ``sync_down`` and
  ``upload_annotations`` (tasks 6.3 / 6.5, not implemented here).

Only ``resolve`` plus its result/error types live in this file. ``resolve`` is
pure and unit-testable: parsing an ``s3://`` URI never touches the network and
does not require ``boto3``. The ``session`` parameter is accepted purely for
signature compatibility with ``sync_down`` / ``upload_annotations`` (which do
need a boto3 session); ``resolve`` itself ignores it.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Root of the per-source image/annotation cache used in S3 mode.
CACHE_ROOT = Path.home() / ".cache" / "sam-labeler"

# Image file extensions the labeler recognizes. Matched case-insensitively
# against object-key suffixes when mirroring an S3 prefix in sync_down.
SUPPORTED_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif", ".tif", ".tiff", ".webp"}
)


class InputSourceError(Exception):
    """Raised when an image source is neither a local directory nor a valid
    ``s3://bucket/prefix`` URI (Req 2.5)."""


@dataclass
class ResolvedSource:
    """A source resolved into the uniform local view the GUI consumes.

    Attributes:
        mode: ``"local"`` or ``"s3"``.
        image_dir: Directory the GUI reads images from. In local mode this is
            the source directory; in S3 mode it is the Local_Cache images dir.
        output_dir: Directory ``AnnotationIO`` writes annotations to.
        s3_bucket: Source bucket in S3 mode, else ``None``.
        s3_image_prefix: Normalized image prefix (no leading slash, no trailing
            slash) in S3 mode, else ``None``.
        s3_annotation_prefix: Sibling ``annotations/`` prefix where annotations
            are uploaded in S3 mode, else ``None``.
    """

    mode: str
    image_dir: Path
    output_dir: Path
    s3_bucket: str | None = None
    s3_image_prefix: str | None = None
    s3_annotation_prefix: str | None = None


def _default_output(input_path: Path) -> Path:
    """Mirror ``labeler._default_output`` for local sources.

    For a directory the default output is ``<name>_labels`` beside it; for a
    single file it is ``<stem>_labels``. resolve() only ever passes a directory
    here, but the file branch is kept to stay faithful to labeler.py.
    """
    if input_path.is_dir():
        return input_path.parent / f"{input_path.name}_labels"
    return input_path.parent / f"{input_path.stem}_labels"


def _parse_s3_uri(source: str) -> tuple[str, str]:
    """Parse ``s3://bucket/prefix[/]`` into ``(bucket, normalized_prefix)``.

    The normalized prefix has no leading or trailing slash. Raises
    InputSourceError for a malformed URI (missing bucket, empty prefix, or an
    otherwise ill-formed value). Does not require boto3 or any network call.
    """
    if not source.startswith("s3://"):
        raise InputSourceError(f"Not an s3:// URI: {source!r}")

    remainder = source[len("s3://"):]
    # Split off the bucket from the key/prefix portion.
    bucket, sep, prefix = remainder.partition("/")

    if not bucket:
        raise InputSourceError(
            f"Malformed S3 URI (missing bucket): {source!r}"
        )
    if not sep or not prefix.strip("/"):
        raise InputSourceError(
            f"Malformed S3 URI (missing image prefix): {source!r}. "
            "Expected s3://bucket/prefix"
        )

    # Normalize: drop leading/trailing slashes and collapse the boundaries.
    normalized = prefix.strip("/")
    return bucket, normalized


def _annotation_prefix(image_prefix: str) -> str:
    """Derive the annotation upload prefix from the image prefix.

    Scheme (per design.md section 2, the ``ResolvedSource`` field comment
    ``"<image_prefix_parent>/annotations/"``): annotations land in a sibling
    ``annotations/`` prefix next to the image prefix. For an image prefix with
    a parent (e.g. ``datasets/run1/images``) the annotation prefix is
    ``datasets/run1/annotations``. For a top-level image prefix with no parent
    (e.g. ``images``) there is no parent to be a sibling of, so we append
    instead, giving ``images/annotations``. Both forms are returned without a
    trailing slash; callers add the ``/`` separator when composing keys.
    """
    parent, sep, _leaf = image_prefix.rpartition("/")
    if sep:
        return f"{parent}/annotations"
    # No parent segment: append a child annotations/ under the prefix.
    return f"{image_prefix}/annotations"


def _cache_dirs(bucket: str, prefix: str) -> tuple[Path, Path]:
    """Return ``(image_dir, output_dir)`` under a deterministic cache path.

    The cache is keyed by a hash of ``bucket + "/" + prefix`` so repeated runs
    against the same source reuse the same directory and a resumed session
    finds its existing ``.meta`` sidecars (Req 6.6). Layout:
        ~/.cache/sam-labeler/<bucket>/<prefix-hash>/images/
        ~/.cache/sam-labeler/<bucket>/<prefix-hash>/labels/
    """
    digest = hashlib.sha256(f"{bucket}/{prefix}".encode("utf-8")).hexdigest()[:16]
    base = CACHE_ROOT / bucket / digest
    return base / "images", base / "labels"


def resolve(source: str, output: str | None, session=None) -> ResolvedSource:
    """Resolve an image ``source`` into a ``ResolvedSource``.

    Args:
        source: A local directory path or an ``s3://bucket/prefix`` URI.
        output: Optional explicit output directory (local mode only). Ignored
            in S3 mode, where the output lives beside the image cache.
        session: Accepted for signature compatibility with ``sync_down`` /
            ``upload_annotations``; unused here (resolve makes no network call).

    Returns:
        A ``ResolvedSource`` describing where to read images and write
        annotations.

    Raises:
        InputSourceError: If ``source`` is neither a readable local directory
            nor a well-formed ``s3://`` URI (Req 2.5).
    """
    # S3 takes precedence: an s3:// string is never a local path.
    if source.startswith("s3://"):
        bucket, image_prefix = _parse_s3_uri(source)
        image_dir, output_dir = _cache_dirs(bucket, image_prefix)
        # Create the cache dirs eagerly so downstream sync/IO can assume they
        # exist. This is a local filesystem operation only.
        image_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        return ResolvedSource(
            mode="s3",
            image_dir=image_dir,
            output_dir=output_dir,
            s3_bucket=bucket,
            s3_image_prefix=image_prefix,
            s3_annotation_prefix=_annotation_prefix(image_prefix),
        )

    path = Path(source).expanduser()
    if path.is_dir():
        image_dir = path.resolve()
        output_dir = (
            Path(output).expanduser().resolve()
            if output
            else _default_output(image_dir)
        )
        return ResolvedSource(
            mode="local",
            image_dir=image_dir,
            output_dir=output_dir,
        )

    # Neither an s3:// URI nor a readable directory (nonexistent path, a file,
    # or otherwise unresolvable).
    if path.exists():
        raise InputSourceError(
            f"Image source is not a directory: {source!r}. "
            "Provide a directory of images or an s3://bucket/prefix URI."
        )
    raise InputSourceError(
        f"Image source does not exist: {source!r}. "
        "Provide a readable local directory or an s3://bucket/prefix URI."
    )


def _is_image_key(key: str) -> bool:
    """True when ``key`` names a supported image (case-insensitive suffix).

    Directory-marker keys (those ending in ``/``) and any key whose extension
    is not in ``SUPPORTED_IMAGE_EXTENSIONS`` are rejected.
    """
    if key.endswith("/"):
        return False
    return Path(key).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def sync_down(src: ResolvedSource, session, progress=None) -> None:
    """Mirror the S3 image prefix into ``src.image_dir`` (Req 2.3, 2.4).

    For an S3-mode source, paginate ``list_objects_v2`` under
    ``s3://{s3_bucket}/{s3_image_prefix}/`` and download every supported image
    object into ``src.image_dir``, preserving each key's path relative to the
    image prefix. Parent directories are created as needed. Directory-marker
    keys (ending in ``/``) and non-image keys are skipped.

    Idempotence (Req 2.4): a key is downloaded only when no local file exists at
    the destination or the existing local file's size differs from the S3
    object's ``Size`` reported by the listing. A re-sync of an unchanged prefix
    therefore performs no downloads and never clobbers locally edited files.

    Args:
        src: The resolved source. In local mode (``mode != "s3"`` or
            ``s3_bucket is None``) this is a no-op: local images are read in
            place.
        session: A boto3 ``Session``; the S3 client is obtained via
            ``session.client("s3")``. No session is created internally.
        progress: Optional callable invoked as ``progress(done, total)`` after
            each image key is processed (downloaded or skipped), where ``total``
            is the number of image keys discovered under the prefix.
    """
    # Local mode: nothing to sync, images are read in place.
    if src.mode != "s3" or src.s3_bucket is None:
        return

    client = session.client("s3")
    bucket = src.s3_bucket
    # Trailing slash bounds the listing to keys under this prefix as a folder,
    # so a sibling prefix sharing a name stem is not accidentally included.
    list_prefix = f"{src.s3_image_prefix}/"

    # Enumerate image keys first so progress has a stable total, and so a bad
    # listing fails before any download begins.
    image_objects: list[tuple[str, int]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if _is_image_key(key):
                image_objects.append((key, obj.get("Size", -1)))

    total = len(image_objects)
    prefix_len = len(list_prefix)
    for done, (key, size) in enumerate(image_objects, start=1):
        relative = key[prefix_len:]
        dest = src.image_dir / relative
        # Skip when an identically sized local file already exists (idempotent
        # re-sync; preserves any local edits).
        if not (dest.exists() and dest.stat().st_size == size):
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
        if progress is not None:
            progress(done, total)


def upload_annotations(src: ResolvedSource, session, io_=None) -> None:
    """Upload every annotation file under ``src.output_dir`` to S3 (Req 2b.2, 2b.3).

    In S3 mode, walk ``src.output_dir`` recursively and upload each regular
    file to ``s3://{s3_bucket}/{s3_annotation_prefix}/`` preserving the file's
    path relative to ``output_dir``. So a local file at
    ``<output_dir>/rel/path`` is written to the key
    ``f"{s3_annotation_prefix}/rel/path"``. This covers everything
    ``AnnotationIO`` produces under the output dir: YOLO-seg ``<stem>.txt``,
    ``boxes/<stem>.txt``, ``.meta/<stem>.json`` (including ``prompts.json``),
    ``classes.txt``, and the COCO labels JSON. Uploads are last-writer-wins;
    there is no merge (a single operator owns the session).

    Args:
        src: The resolved source. In local mode (``mode != "s3"`` or
            ``s3_bucket is None``) this is a no-op: annotations stay local
            (Req 2b.1).
        session: A boto3 ``Session``; the S3 client is obtained via
            ``session.client("s3")``. No session is created internally.
        io_: Optional ``AnnotationIO``. Accepted for signature compatibility
            with the design; the on-disk ``src.output_dir`` is the source of
            truth. When provided, ``io_.output_dir`` (if set) locates the
            output dir; otherwise ``src.output_dir`` is used.
    """
    # Local mode: never upload, annotations remain local only (Req 2b.1).
    if src.mode != "s3" or src.s3_bucket is None:
        return

    # output_dir on disk is authoritative; io_ may point at it but defaults
    # to src.output_dir.
    output_dir = src.output_dir
    if io_ is not None:
        io_dir = getattr(io_, "output_dir", None)
        if io_dir is not None:
            output_dir = io_dir

    client = session.client("s3")
    bucket = src.s3_bucket
    prefix = src.s3_annotation_prefix

    for local in sorted(output_dir.rglob("*")):
        # Only upload regular files; skip directories themselves.
        if not local.is_file():
            continue
        # Relative path from output_dir, expressed with forward slashes
        # regardless of OS so S3 keys are consistent.
        relative = PurePosixPath(local.relative_to(output_dir))
        key = f"{prefix}/{relative}" if prefix else str(relative)
        client.upload_file(str(local), bucket, key)


if __name__ == "__main__":
    # Quick inline self-check (Req 2.1, 2.2, 2.5).
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        local = resolve(td, None)
        assert local.mode == "local", local.mode
        assert local.image_dir == Path(td).resolve()
        assert local.output_dir.name == f"{Path(td).name}_labels"
        assert local.s3_bucket is None

    s3 = resolve("s3://my-bucket/datasets/run1/images/", None)
    assert s3.mode == "s3", s3.mode
    assert s3.s3_bucket == "my-bucket", s3.s3_bucket
    assert s3.s3_image_prefix == "datasets/run1/images", s3.s3_image_prefix
    assert s3.s3_annotation_prefix == "datasets/run1/annotations", s3.s3_annotation_prefix
    assert s3.image_dir.exists() and s3.image_dir.name == "images"
    assert s3.output_dir.exists() and s3.output_dir.name == "labels"

    # Top-level prefix: annotations/ is appended (no parent to be sibling of).
    s3_top = resolve("s3://b/images", None)
    assert s3_top.s3_annotation_prefix == "images/annotations", s3_top.s3_annotation_prefix

    for bad in ("s3://", "s3://bucket", "s3://bucket/", "/no/such/path/xyz"):
        try:
            resolve(bad, None)
        except InputSourceError:
            pass
        else:  # pragma: no cover - self-check only
            raise AssertionError(f"expected InputSourceError for {bad!r}")

    print("input_source self-check passed")
