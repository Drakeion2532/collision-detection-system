# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/collision-detection-system · MIT licence · sp2532-2be7050d798f
from pathlib import Path

import numpy as np
import pytest

from tcd import load_config
from tcd.alerts import compose, maps_link, send_all
from tcd.data import class_weights, collect, label_from_path, stratified_split
from tcd.detect import Incident, IncidentDetector
from tcd.incidents import IncidentStore


def test_paper_class_weights():
    # Training split in the paper: weights 1.5849 (accident) and 0.7304 (non-accident).
    acc = round(27368 / (2 * 1.5849))
    labels = ["Accident"] * acc + ["NonAccident"] * (27368 - acc)
    w = class_weights(labels)
    assert w[0] == pytest.approx(1.5849, abs=1e-3)
    assert w[1] == pytest.approx(0.7304, abs=1e-3)


@pytest.mark.parametrize("path,label", [
    ("ds/train/Accident/1.jpg", "Accident"),
    ("ds/test/NonAccident/2.jpg", "NonAccident"),
    ("ds/data/non-accident/3.png", "NonAccident"),
    ("ds/Accidents/4.jpg", "Accident"),
    ("ds/misc/5.jpg", None),
])
def test_label_from_folder_names(path, label):
    assert label_from_path(Path(path)) == label


def test_merge_dedups_and_splits_without_leakage(tmp_path):
    import cv2
    rng = np.random.default_rng(0)
    for src in ("a", "b"):
        for lab in ("Accident", "NonAccident"):
            d = tmp_path / src / lab
            d.mkdir(parents=True)
            for i in range(20):
                cv2.imwrite(str(d / f"{i}.png"), rng.integers(0, 255, (8, 8, 3), dtype=np.uint8))
    dup = tmp_path / "b" / "Accident" / "dup.png"          # same bytes as a/Accident/0.png
    dup.write_bytes((tmp_path / "a" / "Accident" / "0.png").read_bytes())
    items = collect([tmp_path / "a", tmp_path / "b"])
    assert len(items) == 80                                # the duplicate is dropped
    s = stratified_split(items, {"train": 0.8, "val": 0.1, "test": 0.1})
    paths = [set(p for p, _ in s[k]) for k in ("train", "val", "test")]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    assert sum(len(x) for x in paths) == 80


def test_detector_needs_consecutive_frames_and_respects_cooldown():
    det = IncidentDetector(threshold=0.7, confirm_frames=3, cooldown_seconds=60)
    seq = [0.9, 0.8, 0.2, 0.9, 0.95, 0.91, 0.99, 0.99, 0.99]
    hits = [det.update(p, i, float(i)) for i, p in enumerate(seq)]
    fired = [h for h in hits if h]
    assert len(fired) == 1                                 # blip at 0-1 ignored; cooldown blocks a 2nd alert
    assert fired[0].frame_index == 5 and fired[0].confidence == 0.95


def test_alert_message_has_location_time_and_confidence():
    cfg = load_config()
    msg = compose(Incident(125, 73.5, 0.934), cfg, incident_id=7)
    assert maps_link(18.4636, 73.8682) in msg["body"]
    assert "93.4%" in msg["body"] and "01:13.50" in msg["body"] and "Incident: 7" in msg["body"]
    assert len(msg["sms"]) <= 160
    assert send_all(Incident(1, 1.0, 0.8), cfg)["dry_run"] is True     # dry run by default


def test_incident_store_roundtrip(tmp_path):
    store = IncidentStore(tmp_path / "inc.db")
    rid = store.add(Incident(10, 2.0, 0.88, np.zeros((8, 8, 3), np.uint8)), source="clip.mp4", model="vgg19")
    store.set_status(rid, "confirmed")
    row = store.all()[0]
    assert row["status"] == "confirmed" and row["confidence"] == 0.88 and Path(row["snapshot_path"]).exists()
