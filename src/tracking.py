"""Extra credit: track cones across video frames with sparse optical flow.

Why bother, on top of a per-frame detector?
  * Identity: "cone #3 is getting closer" needs the SAME object across frames.
  * Gaps: the detector misses small cones in some frames (recall is ~40 % for
    16-32 px boxes). Optical flow carries a track through those misses.
  * Smoother distance: single-frame distance jitters with box noise; a track
    can average it (in log space, because the error is multiplicative).
  * Motion stereo: if the robot drives forward a known distance B between two
    frames, the growth of a static cone's box gives its distance WITHOUT
    knowing the cone's real height (see `distance_from_scale_change` and
    docs/GEOMETRY.md section 6).

How a frame is processed (`OpticalFlowTracker.update`)
  1. For every live track, pick corner features inside its last box
     (cv2.goodFeaturesToTrack) and follow them into the new frame with
     pyramidal Lucas-Kanade (cv2.calcOpticalFlowPyrLK), checked forward AND
     backward so bad matches are dropped.
  2. Move the box by the MEDIAN feature shift and rescale it by the MEDIAN
     ratio of pairwise feature distances (robust to a few wrong features).
  3. Match flow-predicted boxes to this frame's detections by IoU (greedy,
     best pair first). A matched track snaps to the detector's box.
  4. Unmatched tracks coast on flow alone for up to `max_coast` frames, then
     die. Unmatched detections start new tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.types import Detection, DistanceEstimate


@dataclass
class Track:
    track_id: int
    det: Detection                      # current box (detector or flow-propagated)
    hits: int = 1                       # frames where the detector confirmed it
    coast: int = 0                      # consecutive frames on flow alone
    history: list[tuple[float, float]] = field(default_factory=list)  # bottom-centre points
    log_dist: float | None = None       # smoothed log-distance (EMA)

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2


def box_iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def propagate_box(prev_gray: np.ndarray, gray: np.ndarray, det: Detection,
                  max_corners: int = 30, fb_thresh_px: float = 1.0) -> Detection | None:
    """Move one box from prev_gray to gray with LK optical flow. None if tracking failed."""
    h, w = gray.shape
    x1, y1 = int(max(0, det.x1)), int(max(0, det.y1))
    x2, y2 = int(min(w, det.x2)), int(min(h, det.y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    mask = np.zeros_like(prev_gray)
    mask[y1:y2, x1:x2] = 255
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=0.01,
                                 minDistance=2, mask=mask, blockSize=3)
    if p0 is None or len(p0) < 3:
        return None
    lk = dict(winSize=(15, 15), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk)
    p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk)
    # Forward-backward check: a feature must come back to where it started.
    fb_err = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
    good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb_err < fb_thresh_px)
    if good.sum() < 3:
        return None
    a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    shift = np.median(b - a, axis=0)
    # Scale = median ratio of pairwise distances (robust; >1 means the object grew).
    i, j = np.triu_indices(len(a), k=1)
    da = np.linalg.norm(a[i] - a[j], axis=1)
    db = np.linalg.norm(b[i] - b[j], axis=1)
    ok = da > 2.0
    scale = float(np.median(db[ok] / da[ok])) if ok.any() else 1.0
    scale = float(np.clip(scale, 0.7, 1.4))
    cx, cy = (det.x1 + det.x2) / 2 + shift[0], (det.y1 + det.y2) / 2 + shift[1]
    hw, hh = det.width * scale / 2, det.height * scale / 2
    return Detection(det.class_id, det.confidence, cx - hw, cy - hh, cx + hw, cy + hh)


class OpticalFlowTracker:
    """Detector + optical-flow tracker. Call `update(frame, detections)` once per frame."""

    def __init__(self, iou_match: float = 0.3, max_coast: int = 5, ema: float = 0.4,
                 classes: set[int] | None = None):
        self.iou_match, self.max_coast, self.ema = iou_match, max_coast, ema
        self.classes = classes          # None = track every class
        self.tracks: list[Track] = []
        self._prev_gray: np.ndarray | None = None
        self._next_id = 1

    def update(self, frame: np.ndarray, detections: list[Detection]) -> list[Track]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        dets = [d for d in detections if self.classes is None or d.class_id in self.classes]

        # 1-2. Flow-predict every live track into this frame.
        if self._prev_gray is not None:
            alive = []
            for t in self.tracks:
                moved = propagate_box(self._prev_gray, gray, t.det)
                if moved is not None:
                    t.det = moved
                    alive.append(t)
                elif t.coast < self.max_coast:
                    alive.append(t)          # keep the old box; may still match a detection
            self.tracks = alive

        # 3. Greedy IoU matching, best pair first, same class only.
        pairs = sorted(((box_iou(t.det, d), ti, di) for ti, t in enumerate(self.tracks)
                        for di, d in enumerate(dets) if t.det.class_id == d.class_id),
                       reverse=True)
        used_t, used_d = set(), set()
        for iou, ti, di in pairs:
            if iou < self.iou_match:
                break
            if ti in used_t or di in used_d:
                continue
            t = self.tracks[ti]
            t.det, t.hits, t.coast = dets[di], t.hits + 1, 0
            used_t.add(ti)
            used_d.add(di)

        # 4. Coast unmatched tracks; drop the stale ones. New tracks for new detections.
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.coast += 1
        self.tracks = [t for t in self.tracks if t.coast <= self.max_coast]
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks.append(Track(self._next_id, d))
                self._next_id += 1
        for t in self.tracks:
            t.history.append(t.det.bottom_center)
            del t.history[:-30]
        self._prev_gray = gray
        return self.tracks

    def smooth_distance(self, track: Track, est: DistanceEstimate) -> float | None:
        """Exponential moving average of the distance, in log space (errors are multiplicative)."""
        if est.distance_m is None or est.distance_m <= 0:
            return None if track.log_dist is None else float(np.exp(track.log_dist))
        x = float(np.log(est.distance_m))
        track.log_dist = x if track.log_dist is None else (1 - self.ema) * track.log_dist + self.ema * x
        return float(np.exp(track.log_dist))


def distance_from_scale_change(h1_px: float, h2_px: float, moved_m: float) -> float | None:
    """Distance at frame 2 of a STATIC object from box growth and known forward motion.

    Pinhole: h = f·H/Z. Moving forward by B: Z2 = Z1 − B. Then
        h2 / h1 = Z1 / Z2 = (Z2 + B) / Z2   =>   Z2 = B · h1 / (h2 − h1).
    Neither the focal length nor the object's real height appears: this is
    the monocular version of stereo (baseline = distance driven). It is only
    usable when the box grows clearly (h2 − h1 of several pixels), because the
    error goes as 1 / (h2 − h1), the analogue of stereo's 1 / disparity.
    """
    dh = h2_px - h1_px
    if moved_m <= 0 or dh <= 0:
        return None
    return moved_m * h1_px / dh
