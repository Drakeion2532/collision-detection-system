# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/collision-detection-system · MIT licence · sp2532-2be7050d798f
"""Test-set evaluation: classification report, confusion matrix and the comparison chart.

    python -m tcd.evaluate --model all
Writes results/<name>_report.json and results/figures/*.png.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import CLASSES, ROOT, load_config


def report(y_true, p_accident, threshold=0.5) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix
    y_pred = np.where(np.asarray(p_accident) >= threshold, 0, 1)
    rep = classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASSES, output_dict=True, zero_division=0)
    rep["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return rep


def evaluate(name: str, cfg: dict, data_dir=None):
    import tensorflow as tf
    size = cfg["models"][name]["input"]
    base = data_dir or ROOT / cfg["data"]["merged_dir"]
    ds = tf.keras.utils.image_dataset_from_directory(base / "test", image_size=(size, size), batch_size=64,
                                                     label_mode="binary", class_names=CLASSES, shuffle=False)
    model = tf.keras.models.load_model(ROOT / "models" / f"{name}.keras", safe_mode=False)
    y_true, p_acc = [], []
    for x, y in ds:
        p = model.predict(x / 255.0, verbose=0).ravel()
        p_acc += list(1 - p)
        y_true += list(y.numpy().ravel().astype(int))
    rep = report(y_true, p_acc)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{name}_report.json").write_text(json.dumps(rep, indent=2))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all")
    a = ap.parse_args()
    cfg = load_config()
    for name in (cfg["models"] if a.model == "all" else [a.model]):
        rep = evaluate(name, cfg)
        print(name, "accuracy", round(rep["accuracy"], 3), "accident F1", round(rep["Accident"]["f1-score"], 3))


if __name__ == "__main__":
    main()
