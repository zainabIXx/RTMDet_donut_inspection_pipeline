#!/usr/bin/env python3
"""
RTMDet-Ins TensorRT Inference  —  Donut 3-class model  (ROI-constrained)
  Classes  : bad (0), double (1), good (2)
  keep_ratio=True  → letterbox resize to 640×640 with padding
  bgr_to_rgb=False → no channel swap
  anchor offset=0
  mask_thr_binary=0.5, score_thr=0.05

  ROI      : loaded from donut_row_roi.json — inference runs only inside that
             region; masks and boxes are reprojected to full-frame coords before
             drawing so the output video is always full-resolution.

Usage:
    python inference_roi.py --engine donut.engine --input video.mp4
    python inference_roi.py --engine donut.engine --input video.mp4 --output out.mp4
    python inference_roi.py --engine donut.engine --input image.jpg --debug
    python inference_roi.py --engine donut.engine --input video.mp4 --roi donut_row_roi.json
    python inference_roi.py --engine donut.engine --input video.mp4 --mqtt
    python inference_roi.py --engine donut.engine --input video.mp4 --mqtt --mqtt-host 192.168.1.10 --mqtt-port 1883 --mqtt-topic donut/inspection
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms as tv_nms
import tensorrt as trt

try:
    import paho.mqtt.client as mqtt
    _MQTT_AVAILABLE = True
except ImportError:
    _MQTT_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
CLASSES     = ('bad', 'double', 'good')
NUM_CLASSES = 3

# Palette (BGR for OpenCV)
PALETTE = [
    ( 60,  20, 220),   # bad    — red
    ( 32,  11, 119),   # double — dark red
    (142, 200,   0),   # good   — green
]

IMG_H, IMG_W = 640, 640
STRIDES      = [8, 16, 32]
PRIOR_OFFSET = 0
SCORE_THR    = 0.40
IOU_THR      = 0.6
MASK_THR     = 0.5
NMS_PRE      = 1000
MAX_PER_IMG  = 100
MIN_BOX_SIDE = 20

# BGR mean/std (bgr_to_rgb=False)
MEAN    = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD     = np.array([ 57.375,  57.12,  58.395], dtype=np.float32)
PAD_VAL = 114.0

# RTMDet-Ins-s mask-head
WEIGHT_NUMS = [80, 64, 8]
BIAS_NUMS   = [ 8,  8, 1]
IN_CH       = [10,  8, 8]
OUT_CH      = [ 8,  8, 1]

FEATMAP_SIZES = [(IMG_H // s, IMG_W // s) for s in STRIDES]
VIDEO_EXTS    = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

ROI_FILE = "donut_row_roi.json"


# ══════════════════════════════════════════════════════════════════════════════
# MQTT publisher
# ══════════════════════════════════════════════════════════════════════════════
class MQTTPublisher:
    """
    Thin wrapper around paho-mqtt.  Connects once on construction, publishes
    JSON payloads, disconnects on close().

    Topic layout
    ────────────
    <base_topic>/row          — full row payload  (all donuts, published once
                                                    per conveyor pass)
    <base_topic>/donut/<N>    — individual donut   (published alongside the row)

    Example payloads
    ────────────────
    donut/inspection/row →
        {
          "row": 3,
          "timestamp": "2024-05-08T12:34:56.789",
          "donuts": {
            "1": {"class": "good", "diameter_mm": 161.4, "hole_width_mm": 58.2,
                  "centricity_mm": 5.1, "shape_irregularity": 1.033},
            "2": {...}
          }
        }

    donut/inspection/donut/1 →
        {"row": 3, "position": 1, "class": "good", "diameter_mm": 161.4, ...}
    """

    def __init__(self, host: str = 'localhost', port: int = 1883,
                 base_topic: str = 'donut/inspection',
                 keepalive: int = 60):
        if not _MQTT_AVAILABLE:
            raise RuntimeError(
                "paho-mqtt is not installed.  Run:  pip install paho-mqtt")

        self.base  = base_topic.rstrip('/')
        self._client = mqtt.Client(client_id='donut_rtmdet', clean_session=True)
        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_publish    = self._on_publish
        self._connected = False

        print(f"[MQTT] connecting to {host}:{port}  base_topic={self.base}")
        self._client.connect(host, port, keepalive)
        self._client.loop_start()          # background thread

        # Wait up to 3 s for the broker ACK
        deadline = time.time() + 3.0
        while not self._connected and time.time() < deadline:
            time.sleep(0.05)
        if not self._connected:
            print("[MQTT] WARNING: broker not reachable — "
                  "messages will queue until connection is established")

    # ── Callbacks ────────────────────────────────────────────────────────────
    def _on_connect(self, client, userdata, flags, rc):
        codes = {0: 'OK', 1: 'bad protocol', 2: 'bad client-id',
                 3: 'server unavailable', 4: 'bad credentials', 5: 'not authorised'}
        if rc == 0:
            self._connected = True
            print(f"[MQTT] connected  ({codes.get(rc, rc)})")
        else:
            print(f"[MQTT] connection refused: {codes.get(rc, rc)}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            print(f"[MQTT] unexpected disconnect (rc={rc}) — will auto-reconnect")

    def _on_publish(self, client, userdata, mid):
        pass   # uncomment to log every publish: print(f"[MQTT] published mid={mid}")

    # ── Public API ────────────────────────────────────────────────────────────
    def publish_row(self, row_number: int, row_data: dict):
        """
        Publish the full row result plus one message per individual donut.
        row_data format: {"1": {measurements}, "2": {...}, ...}
        """
        ts = time.strftime('%Y-%m-%dT%H:%M:%S') + f'.{int(time.time()*1000)%1000:03d}'

        # ── Full-row message ──────────────────────────────────────────────
        row_payload = json.dumps({
            'row':       row_number,
            'timestamp': ts,
            'donuts':    row_data,
        }, indent=None)
        self._client.publish(f'{self.base}/row', row_payload, qos=1, retain=False)

        # ── Per-donut messages ────────────────────────────────────────────
        for pos, measurements in row_data.items():
            donut_payload = json.dumps({
                'row':      row_number,
                'position': int(pos),
                'timestamp': ts,
                **measurements,
            }, indent=None)
            self._client.publish(
                f'{self.base}/donut/{pos}', donut_payload, qos=1, retain=False)

        # Pretty-print to console
        print(f"\n[MQTT] ── row {row_number} published @ {ts} ──")
        for pos, m in row_data.items():
            status = '✓' if m['class'] == 'good' else '✗'
            print(f"  [{pos}] {status} {m['class']:<8s}"
                  f"  D={m['diameter_mm']:6.1f}mm"
                  f"  H={m['hole_width_mm']:5.1f}mm"
                  f"  C={m['centricity_mm']:5.1f}mm"
                  f"  I={m['shape_irregularity']:.3f}")

    def close(self):
        self._client.loop_stop()
        self._client.disconnect()
        print("[MQTT] disconnected")
def load_roi(path: str) -> dict:
    """Return the roi dict {x, y, width, height} from the JSON file."""
    with open(path) as f:
        return json.load(f)["roi"]


# ══════════════════════════════════════════════════════════════════════════════
# TensorRT engine
# ══════════════════════════════════════════════════════════════════════════════
class TRTEngine:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_name   = None
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        self.context.set_input_shape(self.input_name, (1, 3, IMG_H, IMG_W))

        print(f"[Engine] input    : {self.input_name}")
        print(f"[Engine] outputs  : {self.output_names}")
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            print(f"  {name:<30s}  {shape}")

        self._detect_output_style()

    def _detect_output_style(self):
        names = self.output_names
        if any('cls_scores_0' in n for n in names):
            self.style = 'named'
            print("[Engine] output style: named  (cls_scores_0 / bbox_preds_0 / ...)")
        elif any(n.startswith('output') for n in names):
            self.style = 'indexed'
            self.output_names = sorted(names, key=lambda n: int(n.split('_')[-1]))
            print("[Engine] output style: indexed  (output_0 .. output_N)")
            print("         Assuming order: cls×3, bbox×3, kernel×3, mask_feat")
        else:
            self.style = 'positional'
            print("[Engine] output style: positional — using declaration order")
            print("         Assuming order: cls×3, bbox×3, kernel×3, mask_feat")

    def infer(self, blob: torch.Tensor) -> dict:
        blob = blob.contiguous().cuda()
        self.context.set_tensor_address(self.input_name, blob.data_ptr())
        outputs = {}
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            t = torch.empty(shape, dtype=torch.float32, device='cuda').contiguous()
            self.context.set_tensor_address(name, t.data_ptr())
            outputs[name] = t
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return outputs

    def structured(self, raw: dict) -> dict:
        if self.style == 'named':
            return raw
        vals = [raw[n] for n in self.output_names]
        keys = [
            'cls_scores_0', 'cls_scores_1', 'cls_scores_2',
            'bbox_preds_0', 'bbox_preds_1', 'bbox_preds_2',
            'kernel_preds_0', 'kernel_preds_1', 'kernel_preds_2',
            'mask_feat',
        ]
        if len(vals) != len(keys):
            raise RuntimeError(
                f"Engine has {len(vals)} outputs but expected 10. "
                f"Run with --debug to inspect and update the script."
            )
        return dict(zip(keys, vals))


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing  (keep_ratio=True letterbox)
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(bgr: np.ndarray):
    """
    Letterbox resize keeping aspect ratio, pad with PAD_VAL.
    Returns (blob, ori_h, ori_w, pad_top, pad_left, scale)
    """
    ori_h, ori_w = bgr.shape[:2]
    scale  = min(IMG_W / ori_w, IMG_H / ori_h)
    new_w  = round(ori_w * scale)
    new_h  = round(ori_h * scale)

    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top  = (IMG_H - new_h) // 2
    pad_left = (IMG_W - new_w) // 2
    canvas   = np.full((IMG_H, IMG_W, 3), PAD_VAL, dtype=np.float32)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized.astype(np.float32)

    img  = (canvas - MEAN) / STD
    blob = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).unsqueeze(0)
    return blob, ori_h, ori_w, pad_top, pad_left, scale


# ══════════════════════════════════════════════════════════════════════════════
# Prior generation & box decoding
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def build_priors(device='cuda') -> torch.Tensor:
    all_p = []
    for (fh, fw), stride in zip(FEATMAP_SIZES, STRIDES):
        x = (torch.arange(fw, dtype=torch.float32, device=device) + PRIOR_OFFSET) * stride
        y = (torch.arange(fh, dtype=torch.float32, device=device) + PRIOR_OFFSET) * stride
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        s = torch.full_like(xx, stride)
        all_p.append(torch.stack([xx, yy, s, s], dim=-1).reshape(-1, 4))
    return torch.cat(all_p, dim=0)


def decode_boxes(ltrb: torch.Tensor, priors: torch.Tensor) -> torch.Tensor:
    x1 = priors[:, 0] - ltrb[:, 0]
    y1 = priors[:, 1] - ltrb[:, 1]
    x2 = priors[:, 0] + ltrb[:, 2]
    y2 = priors[:, 1] + ltrb[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)


def unpad_boxes(boxes: torch.Tensor, pad_top, pad_left, scale, ori_h, ori_w) -> torch.Tensor:
    """Convert boxes from padded 640×640 space back to the cropped ROI space."""
    b = boxes.clone()
    b[:, 0] = (b[:, 0] - pad_left) / scale
    b[:, 1] = (b[:, 1] - pad_top)  / scale
    b[:, 2] = (b[:, 2] - pad_left) / scale
    b[:, 3] = (b[:, 3] - pad_top)  / scale
    b[:, 0].clamp_(0, ori_w); b[:, 2].clamp_(0, ori_w)
    b[:, 1].clamp_(0, ori_h); b[:, 3].clamp_(0, ori_h)
    return b


# ══════════════════════════════════════════════════════════════════════════════
# Diameter measurement — maximum caliper diameter  (1 px = 1 mm)
# ══════════════════════════════════════════════════════════════════════════════
def measure_diameter(mask_u8: np.ndarray):
    """
    Computes the TRUE maximum diameter of the donut using the rotating-calipers
    method on its convex hull.  This is the longest straight-line distance
    between any two boundary points — it can be at any angle.

    Returns (diameter_mm, pt1, pt2) where pt1/pt2 are the two endpoints
    (in mask/ROI-crop pixel coords) that define the maximum-diameter axis.

    1 pixel == 1 mm.
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        ys, xs = np.where(mask_u8)
        if len(xs) == 0:
            return 0.0, (0, 0), (0, 0)
        cx, cy = int(xs.mean()), int(ys.mean())
        return float(max(xs.max()-xs.min(), ys.max()-ys.min())), (cx, cy), (cx, cy)

    largest = max(contours, key=cv2.contourArea)

    # Convex hull gives us the outer boundary points efficiently
    hull = cv2.convexHull(largest).reshape(-1, 2)

    if len(hull) < 2:
        _, radius = cv2.minEnclosingCircle(largest)
        cx, cy = int(hull[0][0]), int(hull[0][1])
        return float(radius * 2), (cx, cy), (cx, cy)

    # All-pairs maximum distance on convex hull vertices (guaranteed correct).
    # Hull point counts are small (typically 10-60) so O(n²) is negligible.
    hull_f   = hull.astype(np.float64)
    diff     = hull_f[:, None, :] - hull_f[None, :, :]   # (n,n,2)
    dist_sq  = (diff ** 2).sum(axis=2)                    # (n,n)
    idx      = np.unravel_index(dist_sq.argmax(), dist_sq.shape)
    max_dist = float(np.sqrt(dist_sq[idx]))
    pt1, pt2 = hull[idx[0]], hull[idx[1]]

    return max_dist, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]))


# ══════════════════════════════════════════════════════════════════════════════
# Shape irregularity  (aspect ratio of minimum-area bounding rectangle)
# ══════════════════════════════════════════════════════════════════════════════
def measure_irregularity(mask_u8: np.ndarray) -> float:
    """
    Fits the minimum-area rotated rectangle (cv2.minAreaRect) to the largest
    outer contour of the donut mask.

    Returns  longer_side / shorter_side  (always >= 1.0).
      1.0  → perfectly circular / square — no elongation
      >1.0 → increasingly elongated or deformed

    Returns 1.0 if no valid contour is found.
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0

    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 5:          # need at least 5 points for a reliable rect
        return 1.0

    _, (w, h), _ = cv2.minAreaRect(largest)
    if min(w, h) < 1e-3:          # degenerate rectangle
        return 1.0

    return float(max(w, h) / min(w, h))


# ══════════════════════════════════════════════════════════════════════════════
# Shared hole extraction  (used by both centricity and hole-width)
# ══════════════════════════════════════════════════════════════════════════════
def _extract_hole_blob(mask_u8: np.ndarray, crop_bgr: np.ndarray, box: list,
                       debug: bool = False):
    """
    Finds the inner hole of the donut in the actual image and returns its
    binary mask (in bbox-local coords), or None if not found.

    Strategy:
      1. Erode the dough mask inward by a small margin (~5% of radius, max 8px)
         to strip only the outermost boundary noise — NOT a large 25% erosion
         which would clip the hole blob by eroding the inner dough edge into
         the hole space.
      2. Within inner_zone, find pixels darker than 70% of mean dough brightness.
      3. Largest dark blob = the hole.

    Returns (hole_mask_local, x1, y1, donut_area) or (None, x1, y1, donut_area).
    hole_mask_local is in bbox-local coords; add (x1,y1) to get ROI-crop coords.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, crop_bgr.shape[1]), min(y2, crop_bgr.shape[0])

    roi_img  = crop_bgr[y1:y2, x1:x2]
    roi_mask = mask_u8[y1:y2, x1:x2]
    bh, bw   = roi_img.shape[:2]
    if bh == 0 or bw == 0 or roi_mask.sum() == 0:
        return None, x1, y1, 0.0

    gray        = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    donut_area  = float(roi_mask.sum())
    est_radius  = float(np.sqrt(donut_area / np.pi))

    dough_pixels = gray[roi_mask == 1]
    dough_mean   = float(dough_pixels.mean())
    dark_thr     = dough_mean * 0.70

    # FIX: Use a small outer-boundary strip erosion only (~5% of radius, max 8px).
    # A large 25%-radius erosion incorrectly clips the lower half of the hole
    # because eroding the dough mask shrinks the inner dough edge INTO the hole
    # space, making inner_zone=0 exactly where the hole pixels are.
    erode_px = max(3, min(8, int(est_radius * 0.05)))
    k_erode  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (erode_px * 2 + 1, erode_px * 2 + 1))
    inner_zone = cv2.erode(roi_mask, k_erode, iterations=1)
    if inner_zone.sum() == 0:
        return None, x1, y1, donut_area

    dark_inner = ((gray < dark_thr) & (inner_zone == 1)).astype(np.uint8)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dark_inner = cv2.morphologyEx(dark_inner, cv2.MORPH_OPEN,  k3, iterations=1)
    dark_inner = cv2.morphologyEx(dark_inner, cv2.MORPH_CLOSE, k3, iterations=2)

    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(dark_inner)
    if num_labels < 2:
        return None, x1, y1, donut_area

    best = int(stats[1:, cv2.CC_STAT_AREA].argmax()) + 1
    if stats[best, cv2.CC_STAT_AREA] < donut_area * 0.005:
        return None, x1, y1, donut_area

    hole_blob = (labels_im == best).astype(np.uint8)

    if debug:
        print(f"[HOLE] bbox=({x1},{y1},{x2},{y2})  dough_mean={dough_mean:.1f}  "
              f"dark_thr={dark_thr:.1f}  erode_px={erode_px}  "
              f"hole_px={int(hole_blob.sum())}")

    return hole_blob, x1, y1, donut_area


# ══════════════════════════════════════════════════════════════════════════════
# Centricity  (hole offset from outer centroid, 1 px = 1 mm)
# ══════════════════════════════════════════════════════════════════════════════
def measure_centricity(mask_u8: np.ndarray, crop_bgr: np.ndarray, box: list,
                       debug: bool = False):
    """
    Returns (centricity_mm, outer_cx, outer_cy, hole_cx, hole_cy).
    Centricity = Euclidean distance between outer centroid and hole centroid (mm).
    """
    M_out = cv2.moments(mask_u8)
    if M_out['m00'] == 0:
        return 0.0, 0, 0, 0, 0
    outer_cx = int(M_out['m10'] / M_out['m00'])
    outer_cy = int(M_out['m01'] / M_out['m00'])

    hole_blob, x1, y1, _ = _extract_hole_blob(mask_u8, crop_bgr, box, debug=debug)
    if hole_blob is None:
        return 0.0, outer_cx, outer_cy, outer_cx, outer_cy

    M_hole = cv2.moments(hole_blob)
    if M_hole['m00'] == 0:
        return 0.0, outer_cx, outer_cy, outer_cx, outer_cy

    hole_cx = int(M_hole['m10'] / M_hole['m00']) + x1
    hole_cy = int(M_hole['m01'] / M_hole['m00']) + y1
    centricity = float(np.hypot(hole_cx - outer_cx, hole_cy - outer_cy))

    if debug:
        print(f"[CENT] outer=({outer_cx},{outer_cy})  "
              f"hole=({hole_cx},{hole_cy})  centricity={centricity:.2f} mm")
    return centricity, outer_cx, outer_cy, hole_cx, hole_cy


# ══════════════════════════════════════════════════════════════════════════════
# Hole width — maximum caliper diameter of inner hole  (1 px = 1 mm)
# ══════════════════════════════════════════════════════════════════════════════
def measure_hole_width(mask_u8: np.ndarray, crop_bgr: np.ndarray, box: list):
    """
    Returns (hole_width_mm, pt1, pt2) in ROI-crop coords.
    Uses the EXACT same hole blob as measure_centricity for consistency.
    """
    hole_blob, x1, y1, _ = _extract_hole_blob(mask_u8, crop_bgr, box)
    if hole_blob is None:
        return 0.0, (0, 0), (0, 0)

    contours, _ = cv2.findContours(hole_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, (0, 0), (0, 0)

    hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2)
    if len(hull) < 2:
        return 0.0, (0, 0), (0, 0)

    # All-pairs maximum distance on convex hull vertices (guaranteed correct).
    # Hole hulls are small (~15-40 pts) so O(n²) is negligible.
    hull_f   = hull.astype(np.float64)
    diff     = hull_f[:, None, :] - hull_f[None, :, :]   # (n,n,2)
    dist_sq  = (diff ** 2).sum(axis=2)                    # (n,n)
    idx      = np.unravel_index(dist_sq.argmax(), dist_sq.shape)
    max_dist = float(np.sqrt(dist_sq[idx]))
    pt1, pt2 = hull[idx[0]], hull[idx[1]]

    return (max_dist,
            (int(pt1[0]) + x1, int(pt1[1]) + y1),
            (int(pt2[0]) + x1, int(pt2[1]) + y1))


# ══════════════════════════════════════════════════════════════════════════════
# Mask decoding
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def decode_masks(mask_feat: torch.Tensor,
                 kernels:   torch.Tensor,
                 priors:    torch.Tensor) -> torch.Tensor:
    N = priors.shape[0]
    if N == 0:
        return torch.zeros(0, 80, 80, device=mask_feat.device)

    mh, mw      = mask_feat.shape[-2], mask_feat.shape[-1]
    grid_stride = IMG_H // mh

    x = (torch.arange(mw, dtype=torch.float32, device=mask_feat.device) + PRIOR_OFFSET) * grid_stride
    y = (torch.arange(mh, dtype=torch.float32, device=mask_feat.device) + PRIOR_OFFSET) * grid_stride
    yy, xx  = torch.meshgrid(y, x, indexing='ij')
    coord   = torch.stack([xx.flatten(), yy.flatten()], dim=-1).unsqueeze(0)

    pts     = priors[:, :2].unsqueeze(1)
    strides = priors[:, 2:3]
    rel     = (pts - coord).permute(0, 2, 1) / (strides.unsqueeze(-1) * grid_stride)
    rel     = rel.reshape(N, 2, mh, mw)

    mf   = mask_feat.squeeze(0).unsqueeze(0).expand(N, -1, -1, -1)
    x_in = torch.cat([rel, mf], dim=1)

    weights, biases = [], []
    idx = 0
    for wn in WEIGHT_NUMS:
        weights.append(kernels[:, idx:idx + wn]); idx += wn
    for bn in BIAS_NUMS:
        biases.append(kernels[:, idx:idx + bn]);  idx += bn

    for i in range(len(weights)):
        weights[i] = weights[i].reshape(N, OUT_CH[i], IN_CH[i])
        biases[i]  = biases[i].reshape(N, OUT_CH[i])

    feat = x_in.reshape(N, 10, -1)
    for i, (w, b) in enumerate(zip(weights, biases)):
        feat = torch.bmm(w, feat) + b.unsqueeze(-1)
        if i < len(weights) - 1:
            feat = F.relu(feat, inplace=True)

    return feat.reshape(N, mh, mw)


# ══════════════════════════════════════════════════════════════════════════════
# Postprocessing
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def postprocess(outputs:   dict,
                priors:    torch.Tensor,
                ori_h:     int,
                ori_w:     int,
                pad_top:   int,
                pad_left:  int,
                scale:     float,
                score_thr: float,
                crop_bgr:  np.ndarray = None,
                debug:     bool = False) -> list:

    all_cls, all_bbox, all_kern = [], [], []
    for lvl in range(3):
        cs = outputs[f'cls_scores_{lvl}']
        bp = outputs[f'bbox_preds_{lvl}']
        kp = outputs[f'kernel_preds_{lvl}']
        n  = cs.shape[2] * cs.shape[3]
        all_cls.append(cs.permute(0, 2, 3, 1).reshape(1, n, NUM_CLASSES))
        all_bbox.append(bp.permute(0, 2, 3, 1).reshape(1, n, 4))
        all_kern.append(kp.permute(0, 2, 3, 1).reshape(1, n, -1))

    cls_flat  = torch.cat(all_cls,  dim=1).sigmoid()[0]
    bbox_flat = torch.cat(all_bbox, dim=1)[0]
    kern_flat = torch.cat(all_kern, dim=1)[0]

    scores, labels = cls_flat.max(dim=1)

    if debug:
        print("[DEBUG] raw score stats per class:")
        for ci, cn in enumerate(CLASSES):
            sc = cls_flat[:, ci]
            print(f"  [{ci}] {cn:<10s}  max={sc.max().item():.4f}  "
                  f"mean={sc.mean().item():.4f}  "
                  f"above_thr={(sc > score_thr).sum().item()}")
        print(f"[DEBUG] kernel_preds dim: {kern_flat.shape[1]}")

    keep = scores > score_thr
    if keep.sum() == 0:
        return []

    scores  = scores[keep]
    labels  = labels[keep]
    bboxes  = decode_boxes(bbox_flat[keep], priors[keep])
    kernels = kern_flat[keep]
    kpriors = priors[keep]

    # Drop degenerate/tiny boxes (edge artifacts)
    bw = bboxes[:, 2] - bboxes[:, 0]
    bh = bboxes[:, 3] - bboxes[:, 1]
    valid = (bw >= MIN_BOX_SIDE) & (bh >= MIN_BOX_SIDE)
    if valid.sum() == 0:
        return []
    scores  = scores[valid];  labels  = labels[valid]
    bboxes  = bboxes[valid];  kernels = kernels[valid]; kpriors = kpriors[valid]

    keep_idx = []
    for cls_id in range(NUM_CLASSES):
        m = labels == cls_id
        if m.sum() == 0:
            continue
        idx = torch.where(m)[0]
        sc  = scores[idx]
        if len(sc) > NMS_PRE:
            _, top = sc.topk(NMS_PRE)
            idx = idx[top]; sc = sc[top]
        nk = tv_nms(bboxes[idx], sc, IOU_THR)
        keep_idx.append(idx[nk])

    if not keep_idx:
        return []

    keep_idx = torch.cat(keep_idx)
    order    = scores[keep_idx].argsort(descending=True)
    keep_idx = keep_idx[order][:MAX_PER_IMG]

    scores  = scores[keep_idx]
    labels  = labels[keep_idx]
    bboxes  = bboxes[keep_idx]
    kernels = kernels[keep_idx]
    kpriors = kpriors[keep_idx]

    # Decode masks at 80×80, resize to 640×640 BEFORE sigmoid (smooth edges)
    mask_feat = outputs['mask_feat']
    logits    = decode_masks(mask_feat, kernels, kpriors)
    logits    = F.interpolate(logits.unsqueeze(0), size=(IMG_H, IMG_W),
                              mode='bilinear', align_corners=False).squeeze(0)
    mask_bin    = logits.sigmoid() > MASK_THR
    mask_bin_np = mask_bin.cpu().numpy().astype(np.uint8)

    # Unproject boxes back to the ROI crop's coordinate space
    bboxes_roi = unpad_boxes(bboxes, pad_top, pad_left, scale, ori_h, ori_w)

    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    results = []
    for i in range(len(keep_idx)):
        label    = int(labels[i].item())
        cls_name = CLASSES[label]

        # Crop mask to unpadded region, then resize to ROI crop size
        m_padded = mask_bin_np[i]                          # 640×640
        m_crop   = m_padded[pad_top:pad_top + round(ori_h * scale),
                            pad_left:pad_left + round(ori_w * scale)]
        m = cv2.resize(m_crop, (ori_w, ori_h),
                       interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k7, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k7, iterations=1)

        box  = bboxes_roi[i].cpu().numpy().tolist()

        # Skip donuts that touch the left or right ROI edge — partially visible,
        # measurements would be unreliable
        EDGE_TOL = 3  # px
        if box[0] <= EDGE_TOL or box[2] >= ori_w - EDGE_TOL:
            continue

        diameter_mm, diam_pt1, diam_pt2 = measure_diameter(m)
        irregularity = measure_irregularity(m)
        centricity, o_cx, o_cy, h_cx, h_cy = measure_centricity(
            m, crop_bgr if crop_bgr is not None else np.zeros((ori_h, ori_w, 3), dtype=np.uint8), box,
            debug=debug
        )
        _crop = crop_bgr if crop_bgr is not None else np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
        hole_width_mm, hole_pt1, hole_pt2 = measure_hole_width(m, _crop, box)

        results.append(dict(
            mask           = m.astype(bool),
            box            = box,
            score          = float(scores[i].item()),
            label          = label,
            cls            = cls_name,
            diameter_mm    = diameter_mm,
            diam_pt1       = diam_pt1,
            diam_pt2       = diam_pt2,
            hole_width_mm  = hole_width_mm,
            hole_pt1       = hole_pt1,       # ROI-crop coords of hole-width endpoint 1
            hole_pt2       = hole_pt2,       # ROI-crop coords of hole-width endpoint 2
            irregularity   = irregularity,
            centricity_mm  = centricity,
            outer_centroid = (o_cx, o_cy),
            hole_centroid  = (h_cx, h_cy),
        ))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation  (draws on the full frame using ROI offset)
# ══════════════════════════════════════════════════════════════════════════════
def draw_results(bgr: np.ndarray, results: list,
                 roi_x: int = 0, roi_y: int = 0,
                 alpha: float = 0.45) -> np.ndarray:
    """
    Draw segmentation masks and labels onto `bgr` (full frame).
    roi_x / roi_y are the top-left offsets of the ROI so that masks and boxes
    produced in crop-space are correctly placed on the full frame.
    """
    vis   = bgr.copy()
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fscale, thick = 0.6, 1

    for r in results:
        color   = PALETTE[r['label'] % len(PALETTE)]
        mask_u8 = r['mask'].astype(np.uint8)   # ROI-sized boolean mask

        # ── Place the ROI-sized mask onto a full-frame canvas ──────────────
        full_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        roi_h, roi_w = mask_u8.shape
        full_mask[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = mask_u8

        # Coloured mask overlay
        overlay = vis.copy()
        overlay[full_mask == 1] = (
            overlay[full_mask == 1] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)
        vis = overlay

        # Contour
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2, cv2.LINE_AA)

        # ── Compute full-frame centroid coords from stored ROI-crop values ──
        cx = r['outer_centroid'][0] + roi_x
        cy = r['outer_centroid'][1] + roi_y
        hx = r['hole_centroid'][0]  + roi_x
        hy = r['hole_centroid'][1]  + roi_y

        radius_px = int(round(r['diameter_mm'] / 2))

        # ── Class label — drawn below the donut ───────────────────────────
        cls_text = r['cls']
        (clw, clh), clbl = cv2.getTextSize(cls_text, font, 0.6, 2)
        # Use the bottom of the bounding box as anchor, shifted to full-frame
        x1b = int(r['box'][0]) + roi_x
        x2b = int(r['box'][2]) + roi_x
        y2b = int(r['box'][3]) + roi_y
        cl_x = (x1b + x2b) // 2 - clw // 2
        cl_y = y2b + clh + 6          # 6px gap below the donut bottom
        cv2.putText(vis, cls_text, (cl_x, cl_y),
                    font, 0.6, (0, 0, 0),       2, cv2.LINE_AA)   # black shadow
        cv2.putText(vis, cls_text, (cl_x, cl_y),
                    font, 0.6, (255, 255, 255),  1, cv2.LINE_AA)   # white text

        # ── Max-diameter line (angled, at true longest axis) ──────────────
        # Shift endpoints from ROI-crop space to full-frame space
        dp1 = (r['diam_pt1'][0] + roi_x, r['diam_pt1'][1] + roi_y)
        dp2 = (r['diam_pt2'][0] + roi_x, r['diam_pt2'][1] + roi_y)

        cv2.line(vis, dp1, dp2, (255, 255, 255), 2, cv2.LINE_AA)

        # Small perpendicular tick marks at each endpoint
        dx = dp2[0] - dp1[0]
        dy = dp2[1] - dp1[1]
        length = max(float(np.hypot(dx, dy)), 1.0)
        # Perpendicular unit vector
        px, py = -dy / length, dx / length
        tick = 6
        for ep in (dp1, dp2):
            t1 = (int(ep[0] + px * tick), int(ep[1] + py * tick))
            t2 = (int(ep[0] - px * tick), int(ep[1] - py * tick))
            cv2.line(vis, t1, t2, (255, 255, 255), 2, cv2.LINE_AA)

        # Midpoint of the diameter line — anchor for text
        mid_x = (dp1[0] + dp2[0]) // 2
        mid_y = (dp1[1] + dp2[1]) // 2

        # ── Diameter value beside the line mid-point ──────────────────────
        diam_text       = f"{r['diameter_mm']:.1f} mm"
        (dtw, dth), dbl = cv2.getTextSize(diam_text, font, 0.55, 1)
        # place text to the right of the midpoint; flip left if near frame edge
        text_x = mid_x + 8
        if text_x + dtw > vis.shape[1] - 4:
            text_x = mid_x - dtw - 8
        text_y = mid_y + dth // 2
        cv2.putText(vis, diam_text, (text_x, text_y),
                    font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, diam_text, (text_x, text_y),
                    font, 0.55, color, 1, cv2.LINE_AA)

        # ── Centricity overlay ─────────────────────────────────────────────
        # Dot at outer centroid (white)
        cv2.circle(vis, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 4, color,           1,  cv2.LINE_AA)

        # Dot at hole centroid (yellow)
        cv2.circle(vis, (hx, hy), 4, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (hx, hy), 4, (0, 140, 200),  1, cv2.LINE_AA)

        # Line connecting the two centroids
        if (cx, cy) != (hx, hy):
            cv2.line(vis, (cx, cy), (hx, hy), (0, 220, 255), 1, cv2.LINE_AA)

        # Centricity text — placed below the diameter text, same side
        cent_val  = r['centricity_mm']
        cent_text = f"C: {cent_val:.1f} mm"
        (ctw, cth), cbl = cv2.getTextSize(cent_text, font, 0.50, 1)
        cent_x = text_x
        cent_y = text_y + cth + 6
        cv2.putText(vis, cent_text, (cent_x, cent_y),
                    font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, cent_text, (cent_x, cent_y),
                    font, 0.50, (0, 220, 255),   1, cv2.LINE_AA)

        # ── Shape irregularity (min-area rect aspect ratio) ───────────────
        irr_val  = r['irregularity']
        irr_text = f"I: {irr_val:.3f}"
        (itw, ith), ibl = cv2.getTextSize(irr_text, font, 0.50, 1)
        irr_x = text_x
        irr_y = cent_y + ith + 6
        cv2.putText(vis, irr_text, (irr_x, irr_y),
                    font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, irr_text, (irr_x, irr_y),
                    font, 0.50, (180, 180, 0),   1, cv2.LINE_AA)

        # ── Hole max-width line ────────────────────────────────────────────
        hw = r.get('hole_width_mm', 0.0)
        if hw > 0:
            hp1 = (r['hole_pt1'][0] + roi_x, r['hole_pt1'][1] + roi_y)
            hp2 = (r['hole_pt2'][0] + roi_x, r['hole_pt2'][1] + roi_y)

            # Cyan line for the hole
            cv2.line(vis, hp1, hp2, (255, 220, 0), 2, cv2.LINE_AA)

            # Perpendicular tick marks
            hdx = hp2[0] - hp1[0]
            hdy = hp2[1] - hp1[1]
            hlen = max(float(np.hypot(hdx, hdy)), 1.0)
            hpx, hpy = -hdy / hlen, hdx / hlen
            for ep in (hp1, hp2):
                t1 = (int(ep[0] + hpx * 5), int(ep[1] + hpy * 5))
                t2 = (int(ep[0] - hpx * 5), int(ep[1] - hpy * 5))
                cv2.line(vis, t1, t2, (255, 220, 0), 2, cv2.LINE_AA)

            # Hole width text below irregularity
            hw_text = f"H: {hw:.1f} mm"
            (htw, hth), hbl = cv2.getTextSize(hw_text, font, 0.50, 1)
            hw_x = text_x
            hw_y = irr_y + hth + 6
            cv2.putText(vis, hw_text, (hw_x, hw_y),
                        font, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, hw_text, (hw_x, hw_y),
                        font, 0.50, (255, 220, 0),   1, cv2.LINE_AA)

    return vis


def draw_roi_border(vis: np.ndarray, roi: dict, has_donuts: bool) -> np.ndarray:
    """Draw rectangle around ROI and display donut / no_donut status."""
    x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
    if has_donuts:
        label_text = 'donut'
        label_color = (0, 255, 0)    # green
    else:
        label_text = 'no_donut'
        label_color = (0, 0, 255)    # red
    cv2.rectangle(vis, (x, y), (x + w, y + h), label_color, 2)
    cv2.putText(vis, label_text, (x + 4, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)
    return vis


# ══════════════════════════════════════════════════════════════════════════════
# Frame runner  (ROI-aware)
# ══════════════════════════════════════════════════════════════════════════════
def process_frame(engine, priors, frame: np.ndarray, roi: dict,
                  score_thr: float, debug: bool = False):
    """
    Crop the ROI, run inference on the crop, return results in crop-space.
    """
    rx, ry = roi['x'], roi['y']
    rw, rh = roi['width'], roi['height']
    rx2, ry2 = rx + rw, ry + rh

    # Clamp ROI to actual frame size (safety)
    fh, fw = frame.shape[:2]
    rx2 = min(rx2, fw); ry2 = min(ry2, fh)

    crop = frame[ry:ry2, rx:rx2]

    blob, ori_h, ori_w, pad_top, pad_left, scale = preprocess(crop)
    t0      = time.perf_counter()
    raw     = engine.infer(blob)
    dt      = time.perf_counter() - t0
    outputs = engine.structured(raw)
    results = postprocess(outputs, priors, ori_h, ori_w,
                          pad_top, pad_left, scale, score_thr,
                          crop_bgr=crop, debug=debug)
    return results, dt


# ══════════════════════════════════════════════════════════════════════════════
# JSON helpers
# ══════════════════════════════════════════════════════════════════════════════
def _build_row(results: list) -> dict:
    """
    Convert a list of result dicts (one frame snapshot) into a flat JSON row.
    Donuts are sorted left-to-right and indexed 1..N.

    Output format:
        {
          "1": {"class": "good", "diameter_mm": 161.4, "centricity_mm": 8.0, "shape_irregularity": 1.023},
          "2": {...},
          ...
        }
    """
    sorted_r = sorted(results, key=lambda r: r['box'][0])
    return {
        str(i): {
            "class":              r['cls'],
            "diameter_mm":        round(r['diameter_mm'],    2),
            "hole_width_mm":      round(r['hole_width_mm'],  2),
            "centricity_mm":      round(r['centricity_mm'],  2),
            "shape_irregularity": round(r['irregularity'],   4),
        }
        for i, r in enumerate(sorted_r, start=1)
    }


def _merge_live_with_locked(live: list, locked: list) -> list:
    """
    For each live detection, find the matching locked (best-frame) donut by
    x-centroid proximity and overlay its frozen measurements.

    All line endpoints and centroid dots are translated by the rigid delta
    (live_outer_centroid - locked_outer_centroid) so every annotation moves
    exactly with the donut and nothing drifts independently.
    """
    if not locked:
        return live

    merged = []
    locked_sorted = sorted(locked, key=lambda r: r['box'][0])

    for lr in live:
        live_cx_box = (lr['box'][0] + lr['box'][2]) / 2
        best = min(locked_sorted,
                   key=lambda lk: abs((lk['box'][0] + lk['box'][2]) / 2 - live_cx_box))

        merged_r = dict(lr)

        # ── Freeze all scalar measurements from the locked snapshot ──────
        for field in ('diameter_mm', 'centricity_mm', 'irregularity',
                      'hole_width_mm', 'cls', 'label'):
            if field in best:
                merged_r[field] = best[field]

        # ── Compute single rigid translation vector ───────────────────────
        # Everything moves by the delta between live outer centroid and
        # locked outer centroid.  This keeps all annotations glued together.
        lk_outer = np.array(best['outer_centroid'], dtype=np.float64)
        live_outer = np.array(lr['outer_centroid'],  dtype=np.float64)
        delta = live_outer - lk_outer

        def _shift(pt):
            return (int(round(pt[0] + delta[0])), int(round(pt[1] + delta[1])))

        # ── Translate diameter endpoints rigidly ─────────────────────────
        merged_r['diam_pt1'] = _shift(best['diam_pt1'])
        merged_r['diam_pt2'] = _shift(best['diam_pt2'])

        # ── Translate centroid dots rigidly ──────────────────────────────
        merged_r['outer_centroid'] = _shift(best['outer_centroid'])
        merged_r['hole_centroid']  = _shift(best['hole_centroid'])

        # ── Translate hole-width endpoints rigidly ───────────────────────
        hw = best.get('hole_width_mm', 0.0)
        merged_r['hole_width_mm'] = hw
        if hw > 0:
            merged_r['hole_pt1'] = _shift(best['hole_pt1'])
            merged_r['hole_pt2'] = _shift(best['hole_pt2'])
        else:
            merged_r['hole_pt1'] = (0, 0)
            merged_r['hole_pt2'] = (0, 0)

        merged.append(merged_r)

    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Video runner
# ══════════════════════════════════════════════════════════════════════════════
def run_video(engine, priors, src: str, dst, roi: dict,
              score_thr: float, show: bool, debug: bool = False,
              mqtt_pub: 'MQTTPublisher | None' = None):
    cap   = cv2.VideoCapture(src)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = (cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
              if dst else None)

    # JSON output path: same stem as input video
    json_path  = str(Path(src).with_suffix('.json'))
    json_log   = {}
    row_number = 0
    # Per-donut lock: keyed by stable integer ID assigned on first detection.
    # Once a donut is locked (measurements frozen at peak mask-area frame),
    # its scalar values never change again — only the translation delta moves.
    locked_donuts   = {}   # {stable_id: result_dict}
    locked_area     = {}   # {stable_id: mask_area at lock time}
    locked_frozen   = {}   # {stable_id: True once area has peaked and started falling}
    locked_prev_area= {}   # {stable_id: area on previous frame, to detect peak}
    next_id         = [0]  # mutable int wrapped in list so inner func can write it
    in_row          = False

    def _assign_ids(results):
        """
        Match each live result to an existing stable ID (nearest x-centroid
        within 60px), or create a new ID if no match found.
        Returns list of (stable_id, result) pairs.
        """
        assigned = []
        used_ids = set()
        for r in results:
            live_cx = (r['box'][0] + r['box'][2]) / 2
            # Find closest existing ID not yet used this frame
            best_id, best_dist = None, float('inf')
            for sid, lk in locked_donuts.items():
                if sid in used_ids:
                    continue
                lk_cx = (lk['box'][0] + lk['box'][2]) / 2
                d = abs(live_cx - lk_cx)
                if d < best_dist:
                    best_dist, best_id = d, sid
            if best_id is not None and best_dist < 60:
                used_ids.add(best_id)
                assigned.append((best_id, r))
            else:
                sid = next_id[0]; next_id[0] += 1
                used_ids.add(sid)
                assigned.append((sid, r))
        return assigned

    fps_buf, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if in_row and locked_donuts:
                row_number += 1
                row_data = _build_row(list(locked_donuts.values()))
                json_log[f"row_{row_number}"] = row_data
                if mqtt_pub:
                    mqtt_pub.publish_row(row_number, row_data)
            break

        results, dt = process_frame(engine, priors, frame, roi,
                                    score_thr, debug=(debug and idx == 0))

        # ── Per-donut locking ─────────────────────────────────────────────
        # Assign stable IDs, update lock only while area still growing,
        # freeze permanently once area peaks (donut starts to exit ROI).
        if results:
            if not in_row:
                in_row        = True
                locked_donuts = {}
                locked_area   = {}
                locked_frozen = {}
                locked_prev_area = {}
                next_id[0]    = 0

            for sid, r in _assign_ids(results):
                area = int(r['mask'].sum())
                if sid not in locked_donuts:
                    # First time we see this donut — lock immediately
                    locked_donuts[sid]    = r
                    locked_area[sid]      = area
                    locked_prev_area[sid] = area
                    locked_frozen[sid]    = False
                elif not locked_frozen[sid]:
                    if area >= locked_prev_area[sid]:
                        # Still growing — update lock with better frame
                        locked_donuts[sid]    = r
                        locked_area[sid]      = area
                        locked_prev_area[sid] = area
                    else:
                        # Area just started shrinking — freeze permanently
                        locked_frozen[sid]    = True
                        locked_prev_area[sid] = area
                else:
                    locked_prev_area[sid] = area  # track but don't update
        else:
            if in_row:
                row_number += 1
                row_data = _build_row(list(locked_donuts.values()))
                json_log[f"row_{row_number}"] = row_data
                if mqtt_pub:
                    mqtt_pub.publish_row(row_number, row_data)
                locked_donuts    = {}
                locked_area      = {}
                locked_frozen    = {}
                locked_prev_area = {}
                in_row = False

        # ── Draw: live masks/contours, frozen measurements translated
        #         rigidly by (live_centroid - locked_centroid) ─────────────
        locked_list = list(locked_donuts.values())
        draw_results_locked = _merge_live_with_locked(results, locked_list)
        vis = draw_results(frame, draw_results_locked, roi_x=roi['x'], roi_y=roi['y'])
        vis = draw_roi_border(vis, roi, has_donuts=len(results) > 0)

        fps_buf.append(1.0 / max(dt, 1e-6))
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        avg = sum(fps_buf) / len(fps_buf)
        cv2.putText(vis, f'FPS {avg:.1f}  dets:{len(results)}  {idx}/{total}',
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        if writer:
            writer.write(vis)
        if show:
            cv2.imshow('Donut RTMDet-Ins (ROI)', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if in_row and locked_donuts:
                    row_number += 1
                    row_data = _build_row(list(locked_donuts.values()))
                    json_log[f"row_{row_number}"] = row_data
                    if mqtt_pub:
                        mqtt_pub.publish_row(row_number, row_data)
                break
        idx += 1

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    with open(json_path, 'w') as f:
        json.dump(json_log, f, indent=2)
    print(f"Done. {idx} frames processed.  {row_number} rows logged → {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Image runner
# ══════════════════════════════════════════════════════════════════════════════
def run_image(engine, priors, src: str, dst, roi: dict,
              score_thr: float, show: bool, debug: bool = False):
    frame = cv2.imread(src)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {src}")

    results, dt = process_frame(engine, priors, frame, roi, score_thr, debug=debug)
    print(f"Inference : {dt * 1000:.1f} ms  |  detections: {len(results)}")
    for r in results:
        x1, y1, x2, y2 = [int(v) for v in r['box']]
        print(f"  [{r['label']}] {r['cls']:<10s}  score={r['score']:.3f}  "
              f"diameter={r['diameter_mm']:.1f}mm  "
              f"centricity={r['centricity_mm']:.1f}mm  "
              f"irregularity={r['irregularity']:.3f}  "
              f"box=({x1 + roi['x']},{y1 + roi['y']},{x2 + roi['x']},{y2 + roi['y']})")

    # ── JSON output ───────────────────────────────────────────────────────
    if results:
        json_data  = {"row_1": _build_row(results)}
        json_path  = str(Path(src).with_suffix('.json'))
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON saved → {json_path}")

    vis = draw_results(frame, results, roi_x=roi['x'], roi_y=roi['y'])
    vis = draw_roi_border(vis, roi, has_donuts=len(results) > 0)

    if dst:
        cv2.imwrite(dst, vis)
        print(f"Saved → {dst}")
    if show:
        cv2.imshow('Donut RTMDet-Ins (ROI)', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='Donut RTMDet-Ins TensorRT inference (ROI-constrained)')
    ap.add_argument('--engine',     default='donut.engine',       help='Path to .engine file')
    ap.add_argument('--input',      required=True,                help='Image or video path')
    ap.add_argument('--output',     default=None,                 help='Output image/video path')
    ap.add_argument('--roi',        default=ROI_FILE,             help='ROI JSON file (default: donut_row_roi.json)')
    ap.add_argument('--score-thr',  type=float, default=SCORE_THR)
    ap.add_argument('--no-show',    action='store_true',          help='Disable display window')
    ap.add_argument('--debug',      action='store_true',
                    help='Print score/shape diagnostics on the first frame')
    # ── MQTT args ──────────────────────────────────────────────────────────
    ap.add_argument('--mqtt',       action='store_true',          help='Enable MQTT publishing')
    ap.add_argument('--mqtt-host',  default='localhost',          help='MQTT broker host (default: localhost)')
    ap.add_argument('--mqtt-port',  type=int, default=1883,       help='MQTT broker port (default: 1883)')
    ap.add_argument('--mqtt-topic', default='donut/inspection',   help='MQTT base topic (default: donut/inspection)')
    args = ap.parse_args()

    roi    = load_roi(args.roi)
    print(f"[ROI] x={roi['x']}  y={roi['y']}  w={roi['width']}  h={roi['height']}")

    engine = TRTEngine(args.engine)
    priors = build_priors(device='cuda')

    # ── Optionally start MQTT publisher ───────────────────────────────────
    mqtt_pub = None
    if args.mqtt:
        mqtt_pub = MQTTPublisher(
            host=args.mqtt_host,
            port=args.mqtt_port,
            base_topic=args.mqtt_topic,
        )

    try:
        if Path(args.input).suffix.lower() in VIDEO_EXTS:
            run_video(engine, priors, args.input, args.output, roi,
                      args.score_thr, not args.no_show, debug=args.debug,
                      mqtt_pub=mqtt_pub)
        else:
            run_image(engine, priors, args.input, args.output, roi,
                      args.score_thr, not args.no_show, debug=args.debug)
    finally:
        if mqtt_pub:
            mqtt_pub.close()


if __name__ == '__main__':
    main()