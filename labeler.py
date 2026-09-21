#!/usr/bin/env python3
"""Segmentation/detection labeler with SAM auto-labeling assistant.

Usage:
    python labeler.py [input_path] [output_dir]

    input_path  — a directory of images or a single image file (optional;
                  if omitted a startup dialog is shown)
    output_dir  — where labels are written (optional;
                  defaults to <input_name>_labels next to the input)
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from typing import Optional

from PIL import Image, ImageOps, ImageTk

from annotations import (
    Annotation,
    AnnotationIO,
    Box,
    box_from_polygon,
    box_to_corners,
    corners_to_box,
    export_coco,
)

# input_source is pure-python (no boto3 at import time) and safe to import at
# top level so local-mode launches never require AWS deps.
import input_source

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif", ".tif", ".tiff", ".webp"}
MAX_W, MAX_H = 1100, 800

# Colors used per class index (cycling)
_CLASS_COLORS = [
    "#00ff88", "#ff6644", "#44aaff", "#ffdd00", "#cc44ff",
    "#ff44bb", "#44ffdd", "#ff9900", "#aaffaa", "#ff4444",
]


def _class_color(cid: int) -> str:
    return _CLASS_COLORS[cid % len(_CLASS_COLORS)]


def _point_in_polygon(px: float, py: float, pts: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (normalized coordinates)."""
    n = len(pts)
    inside = False
    x, y = px, py
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# ---------------------------------------------------------------------------
# Startup dialog (when no CLI args given)
# ---------------------------------------------------------------------------

class _StartupDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Open images")
        self.resizable(False, False)
        self.grab_set()

        self.result_input: Optional[Path] = None
        self.result_output: Optional[Path] = None

        ttk.Label(self, text="Input (directory or single image):").grid(
            row=0, column=0, sticky=tk.W, padx=8, pady=(12, 2)
        )
        self._input_var = tk.StringVar()
        ttk.Entry(self, textvariable=self._input_var, width=40).grid(
            row=1, column=0, padx=8, pady=(0, 2)
        )
        ttk.Button(self, text="Browse…", command=self._browse_input).grid(
            row=1, column=1, padx=(0, 8)
        )

        ttk.Label(self, text="Output directory (optional):").grid(
            row=2, column=0, sticky=tk.W, padx=8, pady=(8, 2)
        )
        self._output_var = tk.StringVar()
        ttk.Entry(self, textvariable=self._output_var, width=40).grid(
            row=3, column=0, padx=8, pady=(0, 8)
        )
        ttk.Button(self, text="Browse…", command=self._browse_output).grid(
            row=3, column=1, padx=(0, 8)
        )

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=(0, 10))
        ttk.Button(btn_frame, text="Open", command=self._ok).pack(side=tk.LEFT, padx=6)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=6)

        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _browse_input(self):
        choice = messagebox.askquestion("Input type", "Browse for a directory?\n(No = single file)")
        if choice == "yes":
            p = filedialog.askdirectory()
        else:
            p = filedialog.askopenfilename(
                filetypes=[("Images", " ".join(f"*{e}" for e in IMAGE_EXTS))]
            )
        if p:
            self._input_var.set(p)

    def _browse_output(self):
        p = filedialog.askdirectory()
        if p:
            self._output_var.set(p)

    def _ok(self):
        inp = self._input_var.get().strip()
        if not inp:
            messagebox.showerror("Error", "Please select an input path.", parent=self)
            return
        self.result_input = Path(inp)
        out = self._output_var.get().strip()
        self.result_output = Path(out) if out else None
        self.destroy()


# ---------------------------------------------------------------------------
# Main labeler
# ---------------------------------------------------------------------------

class SegLabeler:
    CLOSE_RADIUS = 10
    HANDLE_RADIUS = 5

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        backend_config: Optional[dict] = None,
        resolved_source: Optional["input_source.ResolvedSource"] = None,
        session: object = None,
    ):
        self.output_dir = output_dir
        self.io = AnnotationIO(output_dir)
        # SageMaker endpoint settings (endpoint name, region, profile). See
        # sam_client.py for SageMakerBackend.
        self.backend_config = backend_config or {}

        # Input-source wiring (Task 10). When resolved_source is None (e.g.
        # direct construction from a test) we behave as local mode: no S3
        # sync and no annotation uploads. The shared boto3 session (if any) is
        # reused for both annotation uploads and the endpoint manager.
        self._resolved = resolved_source
        self._session = session

        # Endpoint lifecycle wiring (Task 9). Names this session created live
        # in _owned_endpoints; only those are torn down on close. CLI-created
        # or pre-existing endpoints are never added here, so they survive.
        self._owned_endpoints: set[str] = set()
        self._endpoint_mgr = None  # built lazily from self._session

        # Resolve images list
        if input_path.is_dir():
            self.images: list[Path] = sorted(
                p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS
            )
            if not self.images:
                raise SystemExit(f"No images found in {input_path}")
        else:
            self.images = [input_path]

        self.classes: list[str] = self._load_classes()
        self.idx = self._first_unlabeled()

        self.anns: list[Annotation] = []
        self.pil_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.scale = 1.0
        self.img_w = 0
        self.img_h = 0

        # Manual mode polygon in-progress
        self._pending: list[tuple[int, int]] = []

        # Manual/auto box drag: (x0, y0, x1, y1) in display pixels
        self._box_start: Optional[tuple[int, int]] = None
        self._box_end: Optional[tuple[int, int]] = None
        self._box_dragging = False

        # Click-to-label points: (x, y, label) display pixels, label 1/0
        self._click_points: list[tuple[int, int, int]] = []

        # Edit mode state
        self._edit_idx: Optional[int] = None
        self._drag_handle: Optional[tuple[str, int]] = None  # ("box"|"vertex", index)

        # Batch autolabel state
        self._autolabel_cancel = threading.Event()
        self._autolabel_running = False

        # SAM backend (lazy import so the tool works without requests/boto3)
        self._sam: Optional[object] = None
        # Settings the live backend was built from, to detect edits in the UI.
        self._sam_key: Optional[tuple] = None

        self._build_ui()
        self.load_image()

    # -------------------- filesystem --------------------

    def _classes_path(self) -> Path:
        return self.output_dir / "classes.txt"

    def _load_classes(self) -> list[str]:
        p = self._classes_path()
        if p.exists():
            return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
        return []

    def _save_classes(self) -> None:
        self._classes_path().write_text(
            "\n".join(self.classes) + ("\n" if self.classes else "")
        )

    def _first_unlabeled(self) -> int:
        for i, img in enumerate(self.images):
            if not self.io.has_labels(img):
                return i
        return len(self.images) - 1

    def _save(self) -> None:
        self.io.save(self.images[self.idx], self.anns)

    # -------------------- UI --------------------

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Segmentation Labeler")

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # ---- Canvas ----
        self.canvas = tk.Canvas(
            main, bg="#222", width=MAX_W, height=MAX_H, highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Shift-ButtonPress-1>", self._on_canvas_shift_press)
        self.canvas.bind("<ButtonPress-2>", self._on_canvas_right_press)
        self.canvas.bind("<ButtonPress-3>", self._on_canvas_right_press)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double)
        self.canvas.bind("<Motion>", self._on_motion)

        # ---- Right panel ----
        panel = ttk.Frame(main, padding=8, width=300)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        # Status
        self.status_var = tk.StringVar()
        ttk.Label(
            panel, textvariable=self.status_var, font=("TkDefaultFont", 10, "bold")
        ).pack(anchor=tk.W)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Category
        ttk.Label(panel, text="Category:").pack(anchor=tk.W)
        self.cat_var = tk.StringVar()
        self.cat_combo = ttk.Combobox(panel, textvariable=self.cat_var, state="readonly")
        self.cat_combo.pack(fill=tk.X, pady=(2, 4))
        self._refresh_combo()

        add_row = ttk.Frame(panel)
        add_row.pack(fill=tk.X, pady=(0, 4))
        self.new_cat_var = tk.StringVar()
        ttk.Entry(add_row, textvariable=self.new_cat_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(add_row, text="Add", command=self._add_category).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Label type selector
        type_row = ttk.Frame(panel)
        type_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(type_row, text="Label type:").pack(side=tk.LEFT)
        self._label_type_var = tk.StringVar(value="both")
        for text, val in (("Boxes", "boxes"), ("Masks", "masks"), ("Both", "both")):
            ttk.Radiobutton(
                type_row, text=text, variable=self._label_type_var, value=val,
            ).pack(side=tk.LEFT, padx=(6, 0))

        # Mode toggle
        ttk.Label(panel, text="Mode:").pack(anchor=tk.W)
        self._mode_var = tk.StringVar(value="manual")
        mode_row = ttk.Frame(panel)
        mode_row.pack(fill=tk.X, pady=(2, 4))
        for text, val in (("Manual", "manual"), ("Auto (SAM)", "auto"), ("Edit", "edit")):
            ttk.Radiobutton(
                mode_row, text=text, variable=self._mode_var,
                value=val, command=self._on_mode_change,
            ).pack(side=tk.LEFT, padx=(0, 8))

        # Manual-mode hint
        self._manual_hint = ttk.Label(
            panel,
            text="Masks: click points, close at start (◉)\n"
                 "Boxes: click and drag\nz: undo  Esc: cancel",
            foreground="#888",
        )
        self._manual_hint.pack(anchor=tk.W, pady=(0, 4))

        # In-progress indicator (manual)
        self._pending_var = tk.StringVar(value="")
        self._pending_label = ttk.Label(
            panel, textvariable=self._pending_var, foreground="#ffcc00"
        )
        self._pending_label.pack(anchor=tk.W)

        # Edit-mode hint (hidden until edit mode)
        self._edit_hint = ttk.Label(
            panel,
            text="Click an annotation to edit.\nDrag handles/vertices;"
                 " right-click vertex deletes;\nclick edge inserts. Esc: done",
            foreground="#888",
        )

        # Auto panel (hidden until auto mode selected)
        self._auto_frame = ttk.LabelFrame(panel, text="SAM Auto-Label", padding=6)
        self._build_auto_panel()

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Review strip (shown only when current image has pending annotations)
        self._review_frame = ttk.LabelFrame(panel, text="Review pending", padding=6)
        rev_btns = ttk.Frame(self._review_frame)
        rev_btns.pack(fill=tk.X)
        ttk.Button(rev_btns, text="Accept (Space)", command=self._accept_selected).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(rev_btns, text="Reject (x)", command=self._reject_selected).pack(
            side=tk.LEFT, padx=(4, 0), fill=tk.X, expand=True
        )
        rev_btns2 = ttk.Frame(self._review_frame)
        rev_btns2.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(rev_btns2, text="Accept all on image", command=self._accept_all).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(rev_btns2, text="Next pending →", command=self._go_next_pending).pack(
            side=tk.LEFT, padx=(4, 0), fill=tk.X, expand=True
        )

        # Annotation list
        ttk.Label(panel, text="Annotations:").pack(anchor=tk.W)
        list_frame = ttk.Frame(panel)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(2, 4))
        self.seg_list = tk.Listbox(list_frame, activestyle="none", selectmode=tk.SINGLE)
        self.seg_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(list_frame, command=self.seg_list.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.seg_list.config(yscrollcommand=sb.set)
        self.seg_list.bind("<<ListboxSelect>>", lambda _e: self._redraw())

        mask_btn_row = ttk.Frame(panel)
        mask_btn_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Button(mask_btn_row, text="Delete", command=self._delete_selected).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(
            mask_btn_row, text="Change category", command=self._change_category
        ).pack(side=tk.LEFT, padx=(4, 0), fill=tk.X, expand=True)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Export
        export_row = ttk.Frame(panel)
        export_row.pack(fill=tk.X)
        ttk.Button(export_row, text="Export COCO", command=self._export_coco).pack(
            side=tk.LEFT
        )
        ttk.Label(export_row, text="(YOLO auto-saves)", foreground="#888").pack(
            side=tk.LEFT, padx=(8, 0)
        )

        # Hotkeys
        for key in ("a", "A"):
            self.root.bind(f"<Key-{key}>", self._on_hotkey_prev)
        for key in ("d", "D"):
            self.root.bind(f"<Key-{key}>", self._on_hotkey_next)
        for key in ("z", "Z"):
            self.root.bind(f"<Key-{key}>", self._on_hotkey_undo)
        for key in ("x", "X"):
            self.root.bind(f"<Key-{key}>", self._on_hotkey_reject)
        self.root.bind("<space>", self._on_hotkey_accept)
        self.root.bind("<Escape>", lambda _e: self._on_escape())
        self.root.bind("<Return>", self._on_return)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_auto_panel(self) -> None:
        cfg = self.backend_config

        # Connection row: the tool is locked to the SageMaker backend, so the
        # only setting is the endpoint name.
        conn_row = ttk.Frame(self._auto_frame)
        conn_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(conn_row, text="Endpoint:").pack(side=tk.LEFT)
        self._sam_endpoint_var = tk.StringVar(
            value=cfg.get("endpoint_name", "sam-labeler")
        )
        self._target_entry = ttk.Entry(
            conn_row, textvariable=self._sam_endpoint_var, width=16
        )
        self._target_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        ttk.Button(conn_row, text="Connect", command=self._sam_connect).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        self._conn_status_var = tk.StringVar(value="○ Disconnected")
        ttk.Label(
            self._auto_frame, textvariable=self._conn_status_var, foreground="#ff6644"
        ).pack(anchor=tk.W, pady=(0, 4))

        # Prompt style toggle
        style_row = ttk.Frame(self._auto_frame)
        style_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(style_row, text="Prompt:").pack(side=tk.LEFT)
        self._prompt_style_var = tk.StringVar(value="boxtext")
        ttk.Radiobutton(
            style_row, text="Box+Text", variable=self._prompt_style_var,
            value="boxtext", command=self._on_prompt_style_change,
        ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(
            style_row, text="Click", variable=self._prompt_style_var,
            value="click", command=self._on_prompt_style_change,
        ).pack(side=tk.LEFT, padx=(6, 0))

        # Text prompt: a toggle that reuses the selected category as the
        # prompt, instead of a free-text field duplicating it.
        self._use_text_prompt_var = tk.BooleanVar(value=True)
        self._text_prompt_check = ttk.Checkbutton(
            self._auto_frame, variable=self._use_text_prompt_var
        )
        self._text_prompt_check.pack(anchor=tk.W, pady=(2, 6))
        self.cat_var.trace_add("write", lambda *_: self._refresh_text_prompt_label())
        self._refresh_text_prompt_label()

        # Hint
        self._auto_hint_var = tk.StringVar(
            value="Draw a box on the image,\nenable the text prompt, or both."
        )
        ttk.Label(
            self._auto_frame, textvariable=self._auto_hint_var, foreground="#888",
        ).pack(anchor=tk.W, pady=(0, 4))

        # Confidence threshold
        conf_row = ttk.Frame(self._auto_frame)
        conf_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(conf_row, text="Confidence:").pack(side=tk.LEFT)
        self._confidence_var = tk.DoubleVar(value=0.2)
        self._conf_label_var = tk.StringVar(value="0.20")
        def _on_conf_change(val):
            self._conf_label_var.set(f"{float(val):.2f}")
        ttk.Scale(
            conf_row, from_=0.05, to=0.95, orient=tk.HORIZONTAL,
            variable=self._confidence_var, command=_on_conf_change,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        ttk.Label(conf_row, textvariable=self._conf_label_var, width=4).pack(side=tk.LEFT)

        # Exclude already-labeled regions
        prior_row = ttk.Frame(self._auto_frame)
        prior_row.pack(fill=tk.X, pady=(0, 4))
        self._use_priors_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            prior_row, text="Exclude already-labeled regions",
            variable=self._use_priors_var,
        ).pack(side=tk.LEFT)

        # Run / Clear buttons
        btn_row = ttk.Frame(self._auto_frame)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="Run SAM", command=self._run_sam).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(btn_row, text="Clear", command=self._clear_prompts).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        # Batch autolabel
        ttk.Separator(self._auto_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Button(
            self._auto_frame, text="Autolabel…", command=self._open_autolabel_dialog
        ).pack(fill=tk.X)
        self._autolabel_status_var = tk.StringVar(value="")
        ttk.Label(
            self._auto_frame, textvariable=self._autolabel_status_var, foreground="#888"
        ).pack(anchor=tk.W)

    def _on_mode_change(self) -> None:
        mode = self._mode_var.get()
        self._cancel_pending()
        self._clear_box_drag()
        self._exit_edit()
        self._manual_hint.pack_forget()
        self._pending_label.pack_forget()
        self._edit_hint.pack_forget()
        self._auto_frame.pack_forget()
        if mode == "auto":
            self._auto_frame.pack(fill=tk.X, pady=(0, 4), before=self.seg_list.master)
        elif mode == "edit":
            self._edit_hint.pack(anchor=tk.W, pady=(0, 4))
        else:
            self._manual_hint.pack(anchor=tk.W, pady=(0, 4))
            self._pending_label.pack(anchor=tk.W)
        self._redraw()

    def _refresh_text_prompt_label(self) -> None:
        cname = self.cat_var.get().strip()
        label = f'Text prompt: "{cname}"' if cname else "Text prompt (select a category)"
        self._text_prompt_check.config(text=label)

    def _on_prompt_style_change(self) -> None:
        self._clear_prompts()
        if self._prompt_style_var.get() == "click":
            self._auto_hint_var.set(
                "Click an object to segment it.\nShift/right-click: negative point."
            )
        else:
            self._auto_hint_var.set(
                "Draw a box on the image,\nenable the text prompt, or both."
            )

    def _refresh_combo(self) -> None:
        self.cat_combo["values"] = self.classes
        if self.cat_var.get() not in self.classes:
            self.cat_var.set(self.classes[0] if self.classes else "")

    def _add_category(self) -> None:
        name = self.new_cat_var.get().strip()
        if not name or name in self.classes:
            return
        self.classes.append(name)
        self._save_classes()
        self._refresh_combo()
        self.cat_var.set(name)
        self.new_cat_var.set("")

    def _current_cid(self) -> Optional[int]:
        cname = self.cat_var.get()
        if not cname:
            messagebox.showinfo("No category", "Select or add a category first.")
            return None
        if cname not in self.classes:
            self.classes.append(cname)
            self._save_classes()
            self._refresh_combo()
            self.cat_var.set(cname)
        return self.classes.index(cname)

    # -------------------- image / labels --------------------

    def load_image(self) -> None:
        self._cancel_pending()
        self._clear_box_drag()
        self._click_points.clear()
        self._exit_edit()
        path = self.images[self.idx]
        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        self.img_w, self.img_h = pil.size

        self.scale = min(MAX_W / self.img_w, MAX_H / self.img_h, 1.0)
        disp_w = int(self.img_w * self.scale)
        disp_h = int(self.img_h * self.scale)
        disp = pil.resize((disp_w, disp_h), Image.LANCZOS) if self.scale != 1.0 else pil

        self.pil_image = pil
        self.tk_image = ImageTk.PhotoImage(disp)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        self.anns = self.io.load(path)
        self._redraw()
        self._refresh_seg_list()
        self._update_review_strip()
        self._restore_status()

    # -------------------- coordinate helpers --------------------

    def _to_disp(self, nx: float, ny: float) -> tuple[float, float]:
        return nx * self.img_w * self.scale, ny * self.img_h * self.scale

    def _to_norm(self, dx: int, dy: int) -> tuple[float, float]:
        dw = self.img_w * self.scale
        dh = self.img_h * self.scale
        return max(0.0, min(1.0, dx / dw)), max(0.0, min(1.0, dy / dh))

    # -------------------- drawing --------------------

    def _redraw(self) -> None:
        self.canvas.delete("seg")
        self.canvas.delete("handle")
        sel = self._selected_index()
        for i, a in enumerate(self.anns):
            self._draw_annotation(i, a, selected=(i == sel))
        if self._mode_var.get() == "edit" and self._edit_idx is not None:
            self._draw_edit_handles()

    def _draw_annotation(self, i: int, a: Annotation, selected: bool) -> None:
        color = _class_color(a.class_id)
        dash = (4, 4) if a.pending else None
        width = 3 if selected else 2

        if a.polygon and len(a.polygon) >= 3:
            flat = [c for nx, ny in a.polygon for c in self._to_disp(nx, ny)]
            kwargs = dict(outline=color, width=width, tags="seg")
            if dash:
                kwargs["dash"] = dash
            self.canvas.create_polygon(
                flat, fill=color, stipple="gray25", **kwargs
            )

        box = a.effective_box()
        show_box = a.box is not None and (
            not a.polygon or self._label_type_var.get() != "masks"
        )
        if box and show_box:
            x0, y0, x1, y1 = box_to_corners(box)
            dx0, dy0 = self._to_disp(x0, y0)
            dx1, dy1 = self._to_disp(x1, y1)
            kwargs = dict(outline=color, width=width, tags="seg")
            if dash:
                kwargs["dash"] = dash
            self.canvas.create_rectangle(dx0, dy0, dx1, dy1, **kwargs)

        # Label text at the annotation's top-left
        name = self.classes[a.class_id] if 0 <= a.class_id < len(self.classes) else f"class_{a.class_id}"
        if a.pending and a.score is not None:
            name = f"{name} {a.score:.2f}?"
        anchor_pt = None
        if a.polygon:
            anchor_pt = a.polygon[0]
        elif box:
            x0, y0, _, _ = box_to_corners(box)
            anchor_pt = (x0, y0)
        if anchor_pt:
            tx, ty = self._to_disp(*anchor_pt)
            self.canvas.create_text(
                tx + 2, ty + 2,
                anchor=tk.NW,
                text=name,
                fill=color,
                font=("TkDefaultFont", 10, "bold"),
                tags="seg",
            )

    def _redraw_pending(self, cursor: Optional[tuple[int, int]] = None) -> None:
        self.canvas.delete("pending")
        pts = self._pending
        if not pts:
            self._pending_var.set("")
            return
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            self.canvas.create_line(x0, y0, x1, y1, fill="#ffcc00", width=2, tags="pending")
        if cursor is not None:
            x0, y0 = pts[-1]
            self.canvas.create_line(
                x0, y0, cursor[0], cursor[1],
                fill="#ffcc00", width=1, dash=(4, 3), tags="pending",
            )
        for i, (x, y) in enumerate(pts):
            if i == 0 and len(pts) >= 3:
                self.canvas.create_oval(
                    x - self.CLOSE_RADIUS, y - self.CLOSE_RADIUS,
                    x + self.CLOSE_RADIUS, y + self.CLOSE_RADIUS,
                    outline="#ff4444", width=2, tags="pending",
                )
            self.canvas.create_oval(
                x - 4, y - 4, x + 4, y + 4,
                fill="#ffcc00", outline="", tags="pending",
            )
        hint = "click start (◉) to close" if len(pts) >= 3 else ""
        self._pending_var.set(f"In progress: {len(pts)} pts  {hint}")

    def _redraw_box_drag(self) -> None:
        self.canvas.delete("boxdrag")
        if self._box_start is None or self._box_end is None:
            return
        x0, y0 = self._box_start
        x1, y1 = self._box_end
        self.canvas.create_rectangle(
            x0, y0, x1, y1,
            outline="#ffdd00", width=2, dash=(6, 3), tags="boxdrag",
        )

    def _redraw_points(self) -> None:
        self.canvas.delete("points")
        for x, y, label in self._click_points:
            color = "#00ff66" if label else "#ff4444"
            self.canvas.create_oval(
                x - 5, y - 5, x + 5, y + 5,
                fill=color, outline="#ffffff", width=1, tags="points",
            )

    # -------------------- canvas events --------------------

    def _on_canvas_press(self, event) -> None:
        mode = self._mode_var.get()

        if mode == "edit":
            self._on_edit_press(event)
            return

        if mode == "auto":
            if self._prompt_style_var.get() == "click":
                self._click_points.append((event.x, event.y, 1))
                self._redraw_points()
            else:
                self._box_start = (event.x, event.y)
                self._box_end = (event.x, event.y)
                self._box_dragging = True
                self._redraw_box_drag()
            return

        # Manual mode
        label_type = self._label_type_var.get()
        if label_type == "boxes":
            # Clicking inside an existing annotation selects it (no drag start)
            hit = self._hit_test(event.x, event.y)
            if hit is not None and not self._pending:
                self._select_index(hit)
                return
            self._box_start = (event.x, event.y)
            self._box_end = (event.x, event.y)
            self._box_dragging = True
            self._redraw_box_drag()
        else:
            # Masks or Both: polygon drawing; click on an existing annotation
            # selects it when not mid-polygon.
            if not self._pending:
                hit = self._hit_test(event.x, event.y)
                if hit is not None:
                    self._select_index(hit)
                    return
            self._manual_click(event)

    def _on_canvas_shift_press(self, event) -> str:
        if self._mode_var.get() == "auto" and self._prompt_style_var.get() == "click":
            self._click_points.append((event.x, event.y, 0))
            self._redraw_points()
            return "break"
        self._on_canvas_press(event)
        return "break"

    def _on_canvas_right_press(self, event) -> None:
        mode = self._mode_var.get()
        if mode == "auto" and self._prompt_style_var.get() == "click":
            self._click_points.append((event.x, event.y, 0))
            self._redraw_points()
        elif mode == "edit":
            self._on_edit_right_press(event)

    def _on_canvas_double(self, event) -> None:
        # Double-click an annotation to jump into edit mode on it.
        hit = self._hit_test(event.x, event.y)
        if hit is not None:
            self._mode_var.set("edit")
            self._on_mode_change()
            self._enter_edit(hit)

    def _hit_test(self, dx: int, dy: int) -> Optional[int]:
        """Topmost annotation containing (dx, dy): polygon test, else box test."""
        nx, ny = self._to_norm(dx, dy)
        for i in range(len(self.anns) - 1, -1, -1):
            a = self.anns[i]
            if a.polygon and len(a.polygon) >= 3:
                if _point_in_polygon(nx, ny, a.polygon):
                    return i
            elif a.box is not None:
                x0, y0, x1, y1 = box_to_corners(a.box)
                if x0 <= nx <= x1 and y0 <= ny <= y1:
                    return i
        return None

    def _selected_index(self) -> Optional[int]:
        sel = self.seg_list.curselection()
        return sel[0] if sel else None

    def _select_index(self, i: int) -> None:
        self.seg_list.selection_clear(0, tk.END)
        self.seg_list.selection_set(i)
        self.seg_list.see(i)
        self.seg_list.activate(i)
        self._redraw()

    def _on_canvas_drag(self, event) -> None:
        mode = self._mode_var.get()
        if mode == "edit":
            self._on_edit_drag(event)
            return
        if self._box_dragging:
            self._box_end = (event.x, event.y)
            self._redraw_box_drag()

    def _on_canvas_release(self, event) -> None:
        mode = self._mode_var.get()
        if mode == "edit":
            self._on_edit_release(event)
            return
        if not self._box_dragging:
            return
        self._box_dragging = False
        self._box_end = (event.x, event.y)
        self._redraw_box_drag()
        if mode == "manual" and self._label_type_var.get() == "boxes":
            self._commit_manual_box()

    def _on_motion(self, event) -> None:
        if self._mode_var.get() == "manual" and self._pending:
            self._redraw_pending((event.x, event.y))

    def _manual_click(self, event) -> None:
        if self._current_cid() is None:
            return
        if len(self._pending) >= 3:
            x0, y0 = self._pending[0]
            dist = ((event.x - x0) ** 2 + (event.y - y0) ** 2) ** 0.5
            if dist <= self.CLOSE_RADIUS:
                self._commit_pending()
                return
        self._pending.append((event.x, event.y))
        self._redraw_pending()

    def _commit_pending(self) -> None:
        pts_norm = [self._to_norm(x, y) for x, y in self._pending]
        self._pending.clear()
        self.canvas.delete("pending")
        self._pending_var.set("")

        cid = self._current_cid()
        if cid is None:
            return

        ann = Annotation(class_id=cid, polygon=pts_norm, source="manual")
        if self._label_type_var.get() in ("both", "boxes"):
            ann.box = box_from_polygon(pts_norm)
        self.anns.append(ann)
        self._save()
        self._redraw()
        self._refresh_seg_list()

    def _commit_manual_box(self) -> None:
        box = self._current_drag_box()
        self._clear_box_drag()
        if box is None:
            return
        cid = self._current_cid()
        if cid is None:
            return
        self.anns.append(Annotation(class_id=cid, box=box, source="manual", box_edited=True))
        self._save()
        self._redraw()
        self._refresh_seg_list()

    def _current_drag_box(self) -> Optional[Box]:
        if self._box_start is None or self._box_end is None:
            return None
        x0, y0 = self._box_start
        x1, y1 = self._box_end
        if abs(x1 - x0) < 4 or abs(y1 - y0) < 4:
            return None
        nx0, ny0 = self._to_norm(min(x0, x1), min(y0, y1))
        nx1, ny1 = self._to_norm(max(x0, x1), max(y0, y1))
        return corners_to_box(nx0, ny0, nx1, ny1)

    def _cancel_pending(self) -> None:
        self._pending.clear()
        self.canvas.delete("pending")
        self._pending_var.set("")

    def _clear_box_drag(self) -> None:
        self._box_start = None
        self._box_end = None
        self._box_dragging = False
        self.canvas.delete("boxdrag")

    def _clear_prompts(self) -> None:
        self._clear_box_drag()
        self._click_points.clear()
        self.canvas.delete("points")

    # -------------------- edit mode --------------------

    def _enter_edit(self, idx: int) -> None:
        self._edit_idx = idx
        self._select_index(idx)
        self._redraw()

    def _exit_edit(self) -> None:
        self._edit_idx = None
        self._drag_handle = None
        self.canvas.delete("handle")

    def _edit_annotation(self) -> Optional[Annotation]:
        if self._edit_idx is None or not (0 <= self._edit_idx < len(self.anns)):
            return None
        return self.anns[self._edit_idx]

    def _box_handles(self, a: Annotation) -> list[tuple[float, float]]:
        """8 handle positions (display px): corners then edge midpoints."""
        if a.box is None:
            return []
        x0, y0, x1, y1 = box_to_corners(a.box)
        dx0, dy0 = self._to_disp(x0, y0)
        dx1, dy1 = self._to_disp(x1, y1)
        mx, my = (dx0 + dx1) / 2, (dy0 + dy1) / 2
        return [
            (dx0, dy0), (dx1, dy0), (dx1, dy1), (dx0, dy1),  # corners: tl tr br bl
            (mx, dy0), (dx1, my), (mx, dy1), (dx0, my),       # edges: t r b l
        ]

    def _draw_edit_handles(self) -> None:
        self.canvas.delete("handle")
        a = self._edit_annotation()
        if a is None:
            return
        r = self.HANDLE_RADIUS
        for x, y in self._box_handles(a):
            self.canvas.create_rectangle(
                x - r, y - r, x + r, y + r,
                fill="#ffffff", outline="#0088ff", width=1, tags="handle",
            )
        if a.polygon:
            for nx, ny in a.polygon:
                x, y = self._to_disp(nx, ny)
                self.canvas.create_rectangle(
                    x - r, y - r, x + r, y + r,
                    fill="#ffcc00", outline="#333333", width=1, tags="handle",
                )

    def _handle_hit_test(self, dx: int, dy: int) -> Optional[tuple[str, int]]:
        a = self._edit_annotation()
        if a is None:
            return None
        r = self.HANDLE_RADIUS + 3
        if a.polygon:
            for i, (nx, ny) in enumerate(a.polygon):
                x, y = self._to_disp(nx, ny)
                if abs(dx - x) <= r and abs(dy - y) <= r:
                    return ("vertex", i)
        for i, (x, y) in enumerate(self._box_handles(a)):
            if abs(dx - x) <= r and abs(dy - y) <= r:
                return ("box", i)
        return None

    def _edge_hit_test(self, dx: int, dy: int, max_dist: float = 6.0) -> Optional[int]:
        """Index i such that inserting after vertex i is closest to the click."""
        a = self._edit_annotation()
        if a is None or not a.polygon:
            return None
        best_i, best_d = None, max_dist
        pts = [self._to_disp(nx, ny) for nx, ny in a.polygon]
        n = len(pts)
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            # distance from point to segment
            vx, vy = x1 - x0, y1 - y0
            seg_len2 = vx * vx + vy * vy
            if seg_len2 == 0:
                continue
            t = max(0.0, min(1.0, ((dx - x0) * vx + (dy - y0) * vy) / seg_len2))
            px, py = x0 + t * vx, y0 + t * vy
            d = ((dx - px) ** 2 + (dy - py) ** 2) ** 0.5
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def _on_edit_press(self, event) -> None:
        # Grab a handle on the annotation being edited
        if self._edit_idx is not None:
            handle = self._handle_hit_test(event.x, event.y)
            if handle is not None:
                self._drag_handle = handle
                return
            # Click on an edge inserts a vertex there and starts dragging it
            edge_i = self._edge_hit_test(event.x, event.y)
            a = self._edit_annotation()
            if edge_i is not None and a and a.polygon:
                nx, ny = self._to_norm(event.x, event.y)
                a.polygon.insert(edge_i + 1, (nx, ny))
                self._drag_handle = ("vertex", edge_i + 1)
                self._redraw()
                return
        # Otherwise select/enter another annotation
        hit = self._hit_test(event.x, event.y)
        if hit is not None:
            self._enter_edit(hit)
        else:
            self._exit_edit()
            self._redraw()

    def _on_edit_drag(self, event) -> None:
        a = self._edit_annotation()
        if a is None or self._drag_handle is None:
            return
        kind, i = self._drag_handle
        nx, ny = self._to_norm(event.x, event.y)
        if kind == "vertex" and a.polygon and i < len(a.polygon):
            a.polygon[i] = (nx, ny)
            a.sync_box_from_polygon()
        elif kind == "box" and a.box is not None:
            x0, y0, x1, y1 = box_to_corners(a.box)
            # corners: tl tr br bl; edges: t r b l
            if i == 0:
                x0, y0 = nx, ny
            elif i == 1:
                x1, y0 = nx, ny
            elif i == 2:
                x1, y1 = nx, ny
            elif i == 3:
                x0, y1 = nx, ny
            elif i == 4:
                y0 = ny
            elif i == 5:
                x1 = nx
            elif i == 6:
                y1 = ny
            elif i == 7:
                x0 = nx
            a.box = corners_to_box(x0, y0, x1, y1)
            a.box_edited = True
        self._redraw()

    def _on_edit_release(self, _event) -> None:
        if self._drag_handle is not None:
            self._drag_handle = None
            self._save()
            self._refresh_seg_list()

    def _on_edit_right_press(self, event) -> None:
        """Right-click a vertex deletes it."""
        a = self._edit_annotation()
        if a is None or not a.polygon:
            return
        handle = self._handle_hit_test(event.x, event.y)
        if handle is None or handle[0] != "vertex":
            return
        if len(a.polygon) <= 3:
            messagebox.showinfo("Minimum size", "A polygon needs at least 3 vertices.")
            return
        del a.polygon[handle[1]]
        a.sync_box_from_polygon()
        self._save()
        self._redraw()
        self._refresh_seg_list()

    # -------------------- delete / change category --------------------

    def _delete_selected(self) -> None:
        sel = self._selected_index()
        if sel is None:
            return
        if self._edit_idx == sel:
            self._exit_edit()
        del self.anns[sel]
        self._save()
        self._redraw()
        self._refresh_seg_list()
        self._update_review_strip()

    def _delete_last(self) -> None:
        mode = self._mode_var.get()
        if mode == "manual":
            if self._pending:
                self._pending.pop()
                self._redraw_pending()
                return
        if mode == "auto":
            if self._click_points:
                self._click_points.pop()
                self._redraw_points()
                return
            self._clear_box_drag()
            return
        if self.anns:
            self.anns.pop()
            self._exit_edit()
            self._save()
            self._redraw()
            self._refresh_seg_list()
            self._update_review_strip()

    def _change_category(self) -> None:
        sel = self._selected_index()
        if sel is None:
            messagebox.showinfo("No selection", "Select an annotation in the list first.")
            return
        if not self.classes:
            messagebox.showinfo("No categories", "Add categories first.")
            return

        a = self.anns[sel]

        dlg = tk.Toplevel(self.root)
        dlg.title("Change category")
        dlg.resizable(False, False)
        dlg.grab_set()

        ttk.Label(dlg, text="New category:").pack(padx=12, pady=(12, 2))
        var = tk.StringVar(
            value=self.classes[a.class_id] if 0 <= a.class_id < len(self.classes) else ""
        )
        combo = ttk.Combobox(dlg, textvariable=var, values=self.classes, state="readonly")
        combo.pack(padx=12, pady=(0, 8))

        def _apply():
            new_name = var.get()
            if new_name not in self.classes:
                return
            a.class_id = self.classes.index(new_name)
            self._save()
            self._redraw()
            self._refresh_seg_list()
            dlg.destroy()

        ttk.Button(dlg, text="Apply", command=_apply).pack(pady=(0, 10))

    def _refresh_seg_list(self) -> None:
        self.seg_list.delete(0, tk.END)
        for a in self.anns:
            name = self.classes[a.class_id] if 0 <= a.class_id < len(self.classes) else f"class_{a.class_id}"
            glyph = "◆" if a.polygon else "▣"
            parts = [f"{glyph} {name}"]
            if a.polygon:
                parts.append(f"({len(a.polygon)} pts)")
            if a.source != "manual":
                tag = "auto" if a.source == "autolabel" else a.source.replace("sam_", "sam ")
                parts.append(f"[{tag}]")
            if a.pending:
                score = f" {a.score:.2f}" if a.score is not None else ""
                parts.append(f"pending{score}")
            self.seg_list.insert(tk.END, "  ".join(parts))

    # -------------------- review workflow --------------------

    def _update_review_strip(self) -> None:
        has_pending = any(a.pending for a in self.anns)
        if has_pending:
            self._review_frame.pack(fill=tk.X, pady=(0, 4), before=self.seg_list.master)
        else:
            self._review_frame.pack_forget()

    def _accept_selected(self) -> None:
        sel = self._selected_index()
        if sel is None:
            # No selection: accept the first pending annotation
            sel = next((i for i, a in enumerate(self.anns) if a.pending), None)
            if sel is None:
                return
        a = self.anns[sel]
        if not a.pending:
            return
        a.review = "accepted"
        self._save()
        self._redraw()
        self._refresh_seg_list()
        self._update_review_strip()
        nxt = next((i for i, x in enumerate(self.anns) if x.pending), None)
        if nxt is not None:
            self._select_index(nxt)

    def _reject_selected(self) -> None:
        sel = self._selected_index()
        if sel is None:
            sel = next((i for i, a in enumerate(self.anns) if a.pending), None)
            if sel is None:
                return
        a = self.anns[sel]
        if not a.pending:
            return
        a.review = "rejected"
        self._save()
        self._redraw()
        self._refresh_seg_list()
        self._update_review_strip()
        nxt = next((i for i, x in enumerate(self.anns) if x.pending), None)
        if nxt is not None:
            self._select_index(nxt)

    def _accept_all(self) -> None:
        changed = False
        for a in self.anns:
            if a.pending:
                a.review = "accepted"
                changed = True
        if changed:
            self._save()
            self._redraw()
            self._refresh_seg_list()
            self._update_review_strip()

    def _go_next_pending(self) -> None:
        for offset in range(1, len(self.images) + 1):
            i = (self.idx + offset) % len(self.images)
            if self.io.has_pending(self.images[i]):
                self.idx = i
                self.load_image()
                return
        messagebox.showinfo("Review", "No other images with pending annotations.")

    # -------------------- SAM connection --------------------

    def _backend_key(self) -> tuple:
        """Identity of the SageMaker backend the current UI settings describe."""
        cfg = self.backend_config
        return (
            "sagemaker",
            self._sam_endpoint_var.get().strip(),
            cfg.get("region"),
            cfg.get("profile"),
        )

    def _build_backend(self, key: tuple):
        from sam_client import SageMakerBackend

        _, endpoint, region, profile = key
        return SageMakerBackend(
            endpoint_name=endpoint, region=region, profile=profile
        )

    # -------------------- endpoint lifecycle --------------------

    def _get_endpoint_manager(self):
        """Lazily build an EndpointManager from the shared session.

        Returns None when there is no session (pure-local run with no AWS), so
        callers skip all lifecycle behavior. The manager is cached so repeated
        connect/close calls reuse a single resolution.
        """
        if self._session is None:
            return None
        if self._endpoint_mgr is None:
            from endpoint_manager import EndpointManager

            cfg = self.backend_config
            # Prefer the endpoint the operator typed in the auto panel; fall
            # back to the CLI-provided config name.
            endpoint = None
            try:
                endpoint = self._sam_endpoint_var.get().strip() or None
            except Exception:
                endpoint = cfg.get("endpoint_name")
            self._endpoint_mgr = EndpointManager(
                self._session,
                stack=cfg.get("stack", "SamLabelerStack"),
                endpoint=endpoint or cfg.get("endpoint_name"),
            )
        return self._endpoint_mgr

    def _orphan_sweep(self) -> None:
        """Task 9.1: offer teardown of managed endpoints this session did not
        create. Defensive: any AWS/credential error is logged to status and
        swallowed so a launch never blocks on it (Req 4.1, 4.2, 4.3)."""
        mgr = self._get_endpoint_manager()
        if mgr is None:
            return
        try:
            orphans = mgr.find_orphans(self._owned_endpoints)
            running = [o for o in orphans if o.billable]
            if not running:
                return
            names = ", ".join(o.name for o in running)
            if messagebox.askyesno(
                "Orphaned endpoints",
                f"Found managed SageMaker endpoint(s) still running from a "
                f"prior session:\n\n    {names}\n\nThese bill by the hour. "
                f"Delete them now?",
            ):
                for o in running:
                    try:
                        # Managed tag is present (find_orphans only returns
                        # managed endpoints), so no force needed.
                        orphan_mgr = self._orphan_manager(mgr, o.name)
                        orphan_mgr.delete(wait=False)
                    except Exception as e:  # noqa: BLE001
                        self.status_var.set(f"Orphan delete failed: {e}")
                self.status_var.set(f"Requested deletion of: {names}")
        except Exception as e:  # noqa: BLE001
            # Credentials/permissions/no-such-stack: never block launch.
            self.status_var.set(f"Orphan sweep skipped: {e}")

    def _orphan_manager(self, base_mgr, name: str):
        """An EndpointManager targeting a specific orphan endpoint name."""
        from endpoint_manager import EndpointManager

        cfg = self.backend_config
        return EndpointManager(
            self._session,
            stack=cfg.get("stack", "SamLabelerStack"),
            endpoint=name,
            no_stack_lookup=True,
        )

    def _ensure_endpoint(self, mgr) -> bool:
        """Task 9.2: ensure the endpoint exists before connecting.

        Returns True when the endpoint is available (already running or created
        here), False when create failed. On an app-created endpoint the name is
        recorded in _owned_endpoints so close tears it down; a pre-existing/CLI
        endpoint is left un-owned so it survives (Req 3.1, 3.2, 3.5, 3.6, 5.3).
        """
        st = mgr.status()
        if st.exists and st.status in ("InService", "Creating", "Updating", "SystemUpdating"):
            # Pre-existing or CLI-created (or already creating). Do NOT own it.
            return True

        # Absent (or Failed handled inside create): create and take ownership.
        self.root.after(
            0,
            lambda: self._conn_status_var.set("Creating endpoint, ~7-10 min…"),
        )
        created = mgr.create(wait=True)
        # create() succeeded (or the endpoint was already InService). Own it
        # only if we actually created it this session.
        self._owned_endpoints.add(created.name)
        return True

    def _upload_annotations_if_s3(self) -> None:
        """Task 10.2: upload annotations to S3 when in S3 mode (Req 2b.1-2b.3).

        Local mode (no resolved source, or mode != "s3") is a silent no-op.
        Failures are surfaced to the status line and swallowed so the GUI never
        crashes on an upload error; local files remain the source of truth.
        """
        if (
            self._resolved is None
            or getattr(self._resolved, "mode", "local") != "s3"
            or self._session is None
        ):
            return
        try:
            input_source.upload_annotations(self._resolved, self._session, self.io)
            self.status_var.set("Annotations uploaded to S3.")
        except Exception as e:  # noqa: BLE001
            self.status_var.set(f"S3 upload failed: {e}")

    def _get_sam_client(self):
        """Return a backend matching the current UI settings, rebuilding if needed."""
        try:
            import sam_client  # noqa: F401
        except ImportError:
            messagebox.showerror(
                "Missing dependency",
                "sam_client.py not found or 'requests' not installed.\n"
                "    pip install requests",
            )
            return None

        key = self._backend_key()
        if self._sam is None or self._sam_key != key:
            if self._sam is not None:
                self._sam.disconnect()
            try:
                self._sam = self._build_backend(key)
            except Exception as e:
                messagebox.showerror("Backend error", str(e))
                return None
            self._sam_key = key
        return self._sam

    def _sam_connect(self) -> None:
        if not self._sam_endpoint_var.get().strip():
            messagebox.showerror("Error", "Enter a SageMaker endpoint name.")
            return

        client = self._get_sam_client()
        if client is None:
            return

        self._conn_status_var.set("Connecting…")
        self.root.update_idletasks()

        mgr = self._get_endpoint_manager()

        def _do_connect():
            try:
                # Create-on-first-use: ensure the endpoint exists before we
                # try to connect. When there is no AWS session (mgr is None)
                # we skip straight to connect and let connect() surface any
                # "endpoint not InService" error, preserving prior behavior.
                if mgr is not None:
                    if not self._ensure_endpoint(mgr):
                        return  # create failure already reported
                client.connect()
                label = f"● Connected — {client.describe()}"
                self.root.after(0, lambda: self._conn_status_var.set(label))
            except Exception as e:
                msg = str(e)
                self.root.after(
                    0,
                    lambda msg=msg: (
                        self._conn_status_var.set("○ Disconnected"),
                        messagebox.showerror("Connection failed", msg),
                    ),
                )

        threading.Thread(target=_do_connect, daemon=True).start()

    def _connected_client(self):
        client = self._get_sam_client()
        if client is None:
            return None
        if not client.connected:
            hint = (
                "Click Connect first.\n\nIf the endpoint is not running yet:\n"
                "    python endpoint_ctl.py up\n"
                "    uv run start-sam-endpoint"
            )
            messagebox.showerror("Not connected", hint)
            return None
        return client

    # -------------------- SAM run --------------------

    def _run_sam(self) -> None:
        cid = self._current_cid()
        if cid is None:
            return

        style = self._prompt_style_var.get()
        # The selected category doubles as the text prompt when enabled.
        text_prompt = (
            self.cat_var.get().strip() if self._use_text_prompt_var.get() else ""
        )
        box_norm = self._current_drag_box()
        points: list[list[float]] = []
        point_labels: list[int] = []
        if style == "click":
            for x, y, label in self._click_points:
                nx, ny = self._to_norm(x, y)
                points.append([nx, ny])
                point_labels.append(label)
            if not points:
                messagebox.showinfo("Nothing to send", "Click on an object first.")
                return
            text_prompt = ""
            box_norm = None
            source = "sam_point"
        else:
            if not text_prompt and box_norm is None:
                messagebox.showinfo(
                    "Nothing to send",
                    "Draw a bounding box on the image, enable the text prompt, "
                    "or both.",
                )
                return
            source = "sam_box"

        client = self._connected_client()
        if client is None:
            return

        self.root.config(cursor="watch")
        self.status_var.set("Running SAM…")
        self.root.update_idletasks()

        image_copy = self.pil_image.copy() if self.pil_image else None
        confidence = self._confidence_var.get()
        exclude_boxes = self._get_exclude_boxes() if self._use_priors_var.get() else []

        def _do_infer():
            try:
                masks = client.segment(
                    image=image_copy,
                    text_prompt=text_prompt,
                    box=box_norm,
                    exclude_boxes=exclude_boxes,
                    points=points or None,
                    point_labels=point_labels or None,
                    # Grounding path beats the SAM1-style instance path for
                    # clicks on this checkpoint (see sam_server/HARDWARE.md);
                    # the server filters to the clicked object.
                    instance_mode=False,
                    confidence_threshold=confidence,
                )
                self.root.after(0, lambda: self._apply_sam_results(masks, cid, source))
            except Exception as e:
                msg = str(e)
                self.root.after(
                    0,
                    lambda msg=msg: (
                        messagebox.showerror("SAM error", msg),
                        self.root.config(cursor=""),
                        self._restore_status(),
                    ),
                )

        threading.Thread(target=_do_infer, daemon=True).start()

    def _get_exclude_boxes(self) -> list[list[float]]:
        """Boxes (cx,cy,w,h normalized) of already-labeled/rejected regions,
        passed as negative prompts to SAM."""
        boxes: list[list[float]] = []
        for a in self.anns:
            if a.review == "pending":
                continue
            box = a.effective_box()
            if box and box[2] > 0.01 and box[3] > 0.01:
                boxes.append(list(box))
        return boxes

    def _apply_sam_results(self, masks: list[dict], cid: int, source: str) -> None:
        self.root.config(cursor="")
        if not masks:
            self._restore_status()
            messagebox.showinfo("SAM", "No masks returned.")
            return

        want_box = self._label_type_var.get() in ("both", "boxes")
        for m in masks:
            polygon = m.get("polygon", [])
            if len(polygon) < 3:
                continue
            pts = [(float(p[0]), float(p[1])) for p in polygon]
            ann = Annotation(
                class_id=cid, polygon=pts, source=source, score=m.get("score")
            )
            if want_box:
                ann.box = tuple(m["box"]) if m.get("box") else box_from_polygon(pts)
            self.anns.append(ann)

        self._save()
        self._redraw()
        self._refresh_seg_list()
        self._clear_prompts()
        self._restore_status()

    def _restore_status(self) -> None:
        self.status_var.set(
            f"[{self.idx + 1}/{len(self.images)}] {self.images[self.idx].name}"
        )

    # -------------------- batch autolabel --------------------

    def _open_autolabel_dialog(self) -> None:
        if self._autolabel_running:
            messagebox.showinfo("Autolabel", "A batch autolabel run is in progress.")
            return
        if not self.classes:
            messagebox.showinfo("No categories", "Add categories first.")
            return
        client = self._connected_client()
        if client is None:
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Batch autolabel")
        dlg.resizable(False, False)
        dlg.grab_set()

        ttk.Label(dlg, text="Images:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=(10, 2))
        scope_var = tk.StringVar(value="unlabeled")
        ttk.Radiobutton(
            dlg, text="All unlabeled images", variable=scope_var, value="unlabeled"
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=16)
        row2 = ttk.Frame(dlg)
        row2.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=16)
        ttk.Radiobutton(
            row2, text="Next", variable=scope_var, value="next"
        ).pack(side=tk.LEFT)
        count_var = tk.StringVar(value="20")
        ttk.Entry(row2, textvariable=count_var, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Label(row2, text="images from here").pack(side=tk.LEFT)

        ttk.Label(dlg, text="Categories (per-class text prompts):").grid(
            row=3, column=0, sticky=tk.W, padx=10, pady=(8, 2)
        )
        cat_frame = ttk.Frame(dlg)
        cat_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=16)
        cat_vars: dict[int, tuple[tk.BooleanVar, tk.StringVar]] = {}
        prompts = self._load_prompts()
        for cid, name in enumerate(self.classes):
            enabled = tk.BooleanVar(value=True)
            prompt = tk.StringVar(value=prompts.get(name, name.replace("_", " ").lower()))
            row = ttk.Frame(cat_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Checkbutton(row, variable=enabled, text=name, width=18).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=prompt, width=24).pack(side=tk.LEFT)
            cat_vars[cid] = (enabled, prompt)

        conf_row = ttk.Frame(dlg)
        conf_row.grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=10, pady=(8, 2))
        ttk.Label(conf_row, text="Confidence:").pack(side=tk.LEFT)
        conf_var = tk.DoubleVar(value=max(0.3, self._confidence_var.get()))
        ttk.Entry(conf_row, textvariable=conf_var, width=6).pack(side=tk.LEFT, padx=4)

        btns = ttk.Frame(dlg)
        btns.grid(row=6, column=0, columnspan=2, pady=10)

        def _start():
            targets = self._autolabel_targets(scope_var.get(), count_var.get())
            categories = []
            for cid, (enabled, prompt) in cat_vars.items():
                if enabled.get() and prompt.get().strip():
                    categories.append({
                        "cid": cid,
                        "name": self.classes[cid],
                        "text_prompt": prompt.get().strip(),
                        "confidence_threshold": float(conf_var.get()),
                    })
            if not targets:
                messagebox.showinfo("Autolabel", "No images in scope.", parent=dlg)
                return
            if not categories:
                messagebox.showinfo("Autolabel", "Enable at least one category.", parent=dlg)
                return
            self._save_prompts({c["name"]: c["text_prompt"] for c in categories})
            dlg.destroy()
            self._start_autolabel(client, targets, categories)

        ttk.Button(btns, text="Start", command=_start).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=6)

    def _prompts_path(self) -> Path:
        return self.output_dir / ".meta" / "prompts.json"

    def _load_prompts(self) -> dict[str, str]:
        p = self._prompts_path()
        if p.exists():
            try:
                return json.loads(p.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_prompts(self, prompts: dict[str, str]) -> None:
        merged = self._load_prompts()
        merged.update(prompts)
        p = self._prompts_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(merged, indent=1))

    def _autolabel_targets(self, scope: str, count_str: str) -> list[int]:
        if scope == "unlabeled":
            return [i for i, img in enumerate(self.images) if not self.io.has_labels(img)]
        try:
            n = max(1, int(count_str))
        except ValueError:
            n = 20
        return list(range(self.idx, min(self.idx + n, len(self.images))))

    def _start_autolabel(self, client, targets: list[int], categories: list[dict]) -> None:
        self._autolabel_cancel.clear()
        self._autolabel_running = True
        self._autolabel_status_var.set(f"Autolabeling 0/{len(targets)}…")

        def _worker():
            done = 0
            errors = 0
            for i in targets:
                if self._autolabel_cancel.is_set():
                    break
                img_path = self.images[i]
                try:
                    pil = Image.open(img_path)
                    pil = ImageOps.exif_transpose(pil).convert("RGB")
                    existing = self.io.load(img_path)
                    exclude = [
                        list(b) for b in (
                            a.effective_box() for a in existing if a.review != "pending"
                        ) if b and b[2] > 0.01 and b[3] > 0.01
                    ]
                    resp = client.segment_multi(
                        pil,
                        categories=[
                            {k: v for k, v in c.items() if k != "cid"}
                            for c in categories
                        ],
                        exclude_boxes=exclude or None,
                    )
                    new_anns: list[Annotation] = []
                    name_to_cid = {c["name"]: c["cid"] for c in categories}
                    for result in resp.get("results", []):
                        cid = name_to_cid.get(result.get("category"))
                        if cid is None:
                            continue
                        for m in result.get("masks", []):
                            polygon = m.get("polygon", [])
                            if len(polygon) < 3:
                                continue
                            pts = [(float(p[0]), float(p[1])) for p in polygon]
                            ann = Annotation(
                                class_id=cid,
                                polygon=pts,
                                source="autolabel",
                                review="pending",
                                score=m.get("score"),
                            )
                            ann.box = tuple(m["box"]) if m.get("box") else box_from_polygon(pts)
                            new_anns.append(ann)
                    if new_anns:
                        self.io.save(img_path, existing + new_anns)
                except Exception:
                    errors += 1
                done += 1
                self.root.after(0, lambda d=done: self._autolabel_status_var.set(
                    f"Autolabeling {d}/{len(targets)}…"
                ))

            def _finish():
                self._autolabel_running = False
                suffix = f" ({errors} errors)" if errors else ""
                if self._autolabel_cancel.is_set():
                    self._autolabel_status_var.set(f"Cancelled at {done}/{len(targets)}{suffix}")
                else:
                    self._autolabel_status_var.set(f"Done: {done} images{suffix}")
                # Refresh current image in case it was in scope
                self.anns = self.io.load(self.images[self.idx])
                self._redraw()
                self._refresh_seg_list()
                self._update_review_strip()

            self.root.after(0, _finish)

        threading.Thread(target=_worker, daemon=True).start()

    # -------------------- COCO export --------------------

    def _export_coco(self) -> None:
        def _size(img_path: Path):
            try:
                pil = Image.open(img_path)
                pil = ImageOps.exif_transpose(pil)
                return pil.size
            except Exception:
                return None

        coco = export_coco(self.images, self.io, self.classes, _size)
        out_path = self.output_dir / "labels.json"
        out_path.write_text(json.dumps(coco, indent=2))
        # In S3 mode, push the freshly written labels.json (and the rest of the
        # output dir) up to the annotation prefix (Task 10.2, Req 2b.2/2b.3).
        self._upload_annotations_if_s3()
        messagebox.showinfo(
            "Export complete",
            f"COCO JSON written to:\n{out_path}\n\n"
            f"{len(coco['images'])} images, {len(coco['annotations'])} annotations.",
        )

    # -------------------- navigation / hotkeys --------------------

    def _hotkeys_allowed(self, event) -> bool:
        return not isinstance(event.widget, (tk.Entry, ttk.Entry))

    def _on_hotkey_prev(self, event) -> None:
        if self._hotkeys_allowed(event):
            self._go(-1)

    def _on_hotkey_next(self, event) -> None:
        if self._hotkeys_allowed(event):
            self._go(1)

    def _on_hotkey_undo(self, event) -> None:
        if self._hotkeys_allowed(event):
            self._delete_last()

    def _on_hotkey_accept(self, event) -> None:
        if self._hotkeys_allowed(event) and any(a.pending for a in self.anns):
            self._accept_selected()

    def _on_hotkey_reject(self, event) -> None:
        if self._hotkeys_allowed(event) and any(a.pending for a in self.anns):
            self._reject_selected()

    def _on_escape(self) -> None:
        mode = self._mode_var.get()
        if mode == "auto":
            self._clear_prompts()
        elif mode == "edit":
            self._exit_edit()
            self._redraw()
        else:
            self._cancel_pending()

    def _on_return(self, event) -> None:
        # Run SAM with Enter only when not typing in a text entry
        if self._mode_var.get() == "auto" and not isinstance(event.widget, (tk.Entry, ttk.Entry)):
            self._run_sam()

    def _go(self, delta: int) -> None:
        new = self.idx + delta
        if 0 <= new < len(self.images):
            self.idx = new
            self.load_image()

    def _on_close(self) -> None:
        self._autolabel_cancel.set()
        self._save()

        # Task 10.2: flush annotations to S3 on close (no-op in local mode).
        self._upload_annotations_if_s3()

        # Disconnect the live backend, if any.
        if self._sam is not None:
            try:
                self._sam.disconnect()
            except Exception:
                pass

        # Task 9.3: tear down ONLY endpoints this app session created. CLI-
        # created (start-sam-endpoint) and pre-existing endpoints are not in
        # _owned_endpoints and are left running (Req 3.3, 5.3). delete is
        # fire-and-forget (wait=False) so close never blocks on a waiter, and
        # any AWS error is swallowed so close cannot hang or crash.
        if self._owned_endpoints and self._session is not None:
            self.status_var.set("Deleting endpoint…")
            try:
                self.root.update_idletasks()
            except Exception:
                pass
            for name in list(self._owned_endpoints):
                try:
                    mgr = self._orphan_manager(self._get_endpoint_manager(), name)
                    mgr.delete(wait=False)
                except Exception:
                    # Best-effort: a failed delete must not block shutdown.
                    pass

        self.root.destroy()

    def run(self) -> None:
        # Task 9.1: run the launch-time orphan sweep once the event loop is up
        # so the askyesno dialog has a live root. Defensive inside _orphan_sweep.
        self.root.after(0, self._orphan_sweep)
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _default_output(input_path: Path) -> Path:
    if input_path.is_dir():
        return input_path.parent / f"{input_path.name}_labels"
    return input_path.parent / f"{input_path.stem}_labels"


def main() -> None:
    ap = argparse.ArgumentParser(description="Segmentation labeler with SAM auto-labeling")
    ap.add_argument("input_path", type=Path, nargs="?", default=None)
    ap.add_argument("output_dir", type=Path, nargs="?", default=None)

    backend = ap.add_argument_group("SAM inference backend")
    backend.add_argument(
        "--endpoint", default="sam-labeler",
        help="SageMaker endpoint name (default: sam-labeler)",
    )
    backend.add_argument("--region", help="AWS region for the SageMaker backend")
    backend.add_argument("--profile", help="AWS profile for the SageMaker backend")
    args = ap.parse_args()

    # The tool is locked to the SageMaker backend, so the config only carries
    # the endpoint name plus the region/profile used to reach it.
    backend_config = {
        "endpoint_name": args.endpoint,
        "region": args.region,
        "profile": args.profile,
    }

    # Raw input source (str) and optional output override. The source may be a
    # local directory or an s3://bucket/prefix URI; input_source.resolve
    # classifies it below.
    if args.input_path is None:
        # Show startup dialog
        root = tk.Tk()
        root.withdraw()
        dlg = _StartupDialog(root)
        root.wait_window(dlg)
        if dlg.result_input is None:
            sys.exit(0)
        raw_source = str(dlg.result_input)
        raw_output = str(dlg.result_output) if dlg.result_output else None
        root.destroy()
    else:
        raw_source = str(args.input_path)
        raw_output = str(args.output_dir) if args.output_dir else None

    # Build a single shared boto3 Session used for S3 sync/upload and the
    # endpoint manager. boto3 is imported lazily so a pure-local run works even
    # when boto3 is not installed. If the source is S3 and boto3 is missing we
    # cannot continue; for local mode the session may be None.
    is_s3 = raw_source.startswith("s3://")
    session = None
    try:
        import boto3
        session = boto3.Session(
            profile_name=args.profile, region_name=args.region
        )
    except ImportError:
        if is_s3:
            print(
                "boto3 is required for S3 input sources.\n"
                "    uv pip install boto3      (or: pip install boto3)",
                file=sys.stderr,
            )
            sys.exit(2)
        # Local mode with no boto3: fine, session stays None (SAM connect will
        # error later if the operator tries it).

    # Resolve the source into the uniform local view. On any resolution error
    # (bad path, malformed s3:// URI), report and halt startup (Req 2.5).
    try:
        resolved = input_source.resolve(raw_source, raw_output, session)
    except input_source.InputSourceError as e:
        print(f"Input source error: {e}", file=sys.stderr)
        sys.exit(2)

    # S3 mode: mirror the prefix into the local cache before the GUI opens
    # (Req 2.3, 2.4). Print brief progress to the console.
    if resolved.mode == "s3":
        if session is None:  # pragma: no cover - guarded above, defensive
            print("boto3 session required for S3 sync.", file=sys.stderr)
            sys.exit(2)
        print(
            f"Syncing images from s3://{resolved.s3_bucket}/"
            f"{resolved.s3_image_prefix}/ …"
        )

        def _progress(done: int, total: int) -> None:
            print(f"\r  synced {done}/{total} images", end="", flush=True)

        try:
            input_source.sync_down(resolved, session, progress=_progress)
        except Exception as e:  # noqa: BLE001
            print(f"\nS3 sync failed: {e}", file=sys.stderr)
            sys.exit(2)
        print()  # newline after the progress line

    input_path = resolved.image_dir
    output_dir = resolved.output_dir
    if not input_path.exists():
        print(f"Path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    SegLabeler(
        input_path,
        output_dir,
        backend_config=backend_config,
        resolved_source=resolved,
        session=session,
    ).run()


if __name__ == "__main__":
    main()
