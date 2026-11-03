# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/collision-detection-system · MIT licence · sp2532-2be7050d798f
"""Real-time detection on a video or CCTV stream (Section 3).

Frames are sampled every N frames, scored by the model, and an incident is raised when
P(accident) stays above the threshold (0.7 by default) for `confirm_frames` samples in a
row. A cooldown keeps one incident from producing an alert per frame.

    python -m tcd.detect path/to/video.mp4 --model vgg19
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import numpy as np

from . import ROOT, load_config


@dataclass
class Incident:
    frame_index: int
    timestamp_s: float
    confidence: float
    frame: np.ndarray | None = field(default=None, repr=False)


class IncidentDetector:
    """Turns a stream of per-frame accident probabilities into incidents."""

    def __init__(self, threshold=0.7, confirm_frames=3, cooldown_seconds=60.0):
        self.threshold, self.confirm, self.cooldown = threshold, confirm_frames, cooldown_seconds
        self._streak, self._peak, self._last_alert = 0, 0.0, -1e9

    def update(self, p_accident: float, frame_index: int, t: float, frame=None) -> Incident | None:
        if p_accident >= self.threshold:
            self._streak += 1
            self._peak = max(self._peak, p_accident)
        else:
            self._streak, self._peak = 0, 0.0
        if self._streak >= self.confirm and t - self._last_alert >= self.cooldown:
            self._last_alert = t
            inc = Incident(frame_index, t, round(self._peak, 4), frame)
            self._streak, self._peak = 0, 0.0
            return inc
        return None


def scorer(model, input_size: int):
    """Wrap a Keras model so it takes one BGR frame and returns P(accident)."""
    import cv2

    def score(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (input_size, input_size)).astype("float32")[None] / 255.0
        return float(1.0 - model.predict(x, verbose=0).ravel()[0])   # sigmoid = P(non-accident)
    return score


def run_video(path, score, cfg, on_incident=None, max_frames=None):
    import cv2
    d = cfg["detection"]
    det = IncidentDetector(d["threshold"], d["confirm_frames"], d["cooldown_seconds"])
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i, incidents, timings = 0, [], []
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and i >= max_frames):
            break
        if i % d["sample_every_n_frames"] == 0:
            t0 = time.perf_counter()
            p = score(frame)
            timings.append(time.perf_counter() - t0)
            inc = det.update(p, i, i / fps, frame)
            if inc:
                incidents.append(inc)
                if on_incident:
                    on_incident(inc)
        i += 1
    cap.release()
    return incidents, (len(timings) / sum(timings) if timings else 0.0)


def main():
    import tensorflow as tf
    from .alerts import send_all
    from .incidents import IncidentStore
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="vgg19")
    a = ap.parse_args()
    cfg = load_config()
    model = tf.keras.models.load_model(ROOT / "models" / f"{a.model}.keras", safe_mode=False)
    store = IncidentStore()

    def handle(inc):
        rid = store.add(inc, source=a.video, model=a.model)
        print(send_all(inc, cfg, incident_id=rid))

    incidents, fps = run_video(a.video, scorer(model, cfg["models"][a.model]["input"]), cfg, handle)
    print(f"{len(incidents)} incident(s); scoring speed {fps:.1f} frames/s")


if __name__ == "__main__":
    main()
