"""End-to-end smoke test on synthetic images (no download, no GPU, ~1 minute).

Checks that data loading, two-phase training, evaluation, video detection, alerts and
the incident log all run together. The synthetic task is trivial, so the accuracy it
prints says nothing about real-world performance.

    python scripts/smoke_test.py
"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcd import load_config  # noqa: E402
from tcd.alerts import send_all  # noqa: E402
from tcd.detect import run_video, scorer  # noqa: E402
from tcd.incidents import IncidentStore  # noqa: E402
from tcd.train import train_one  # noqa: E402


def frame(accident: bool, rng, size=96):
    img = np.full((size, size, 3), rng.integers(60, 120), np.uint8)
    cv2.line(img, (0, size // 2), (size, size // 2), (200, 200, 200), 2)           # a "road"
    if accident:                                                                    # two overlapping "cars"
        c = rng.integers(30, 66, 2)
        cv2.rectangle(img, tuple(c - 12), tuple(c + 4), (0, 0, 255), -1)
        cv2.rectangle(img, tuple(c - 4), tuple(c + 12), (0, 200, 255), -1)
    return img


def main():
    rng = np.random.default_rng(0)
    cfg = load_config()
    cfg["models"]["custom_cnn"]["input"] = 64
    cfg["training"]["batch_size"] = 16
    cfg["training"]["lr_phase1"] = 1e-3          # faster for the toy task
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for split, n in (("train", 60), ("val", 16), ("test", 16)):
            for lab, acc in (("Accident", True), ("NonAccident", False)):
                d = tmp / "data" / split / lab
                d.mkdir(parents=True)
                for i in range(n):
                    cv2.imwrite(str(d / f"{i}.png"), frame(acc, rng))
        model, hist = train_one("custom_cnn", cfg, data_dir=tmp / "data", epochs=(12, 4))
        print("trained epochs:", len(hist["loss"]), "train acc:", [round(a, 2) for a in hist["accuracy"]], "val acc:", [round(a, 2) for a in hist["val_accuracy"]])

        video = tmp / "clip.mp4"
        vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (96, 96))
        for i in range(250):
            vw.write(frame(100 <= i < 160, rng))           # a 'collision' between 4 s and 6.4 s
        vw.release()
        cfg["detection"]["cooldown_seconds"] = 60
        store = IncidentStore(tmp / "incidents.db")
        incidents, fps = run_video(video, scorer(model, 64), cfg,
                                   lambda inc: print(send_all(inc, cfg, store.add(inc, "clip.mp4", "custom_cnn"))["sms"]))
        print(f"incidents: {len(incidents)}  scoring speed: {fps:.1f} frames/s  logged: {len(store.all())}")


if __name__ == "__main__":
    main()
