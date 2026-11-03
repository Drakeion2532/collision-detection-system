# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/collision-detection-system · MIT licence · sp2532-2be7050d798f
"""Two-phase transfer learning (Section 4.3).

Phase 1 (epochs 1-60):  backbone frozen, train the new head at lr 1e-4.
Phase 2 (epochs 61-100): unfreeze the top of the backbone, fine-tune at lr 1e-5.
Class-weighted binary cross-entropy, L2 regularisation, Adam, ReduceLROnPlateau and
early stopping. The custom CNN trains end to end in one run of the same length.

    python -m tcd.train --model vgg19
    python -m tcd.train --model all
Outputs: models/<name>.keras, results/<name>_history.json
"""
from __future__ import annotations

import argparse
import json

from . import CLASSES, ROOT, load_config
from .data import class_weights
from .models import build, unfreeze_top


def datasets(cfg, input_size: int, data_dir=None):
    import tensorflow as tf
    tc, aug = cfg["training"], cfg["training"]["augmentation"]
    base = data_dir or ROOT / cfg["data"]["merged_dir"]
    kw = dict(image_size=(input_size, input_size), batch_size=tc["batch_size"], label_mode="binary",
              class_names=CLASSES, seed=cfg["data"]["seed"])
    train = tf.keras.utils.image_dataset_from_directory(base / "train", shuffle=True, **kw)
    val = tf.keras.utils.image_dataset_from_directory(base / "val", shuffle=False, **kw)
    L = tf.keras.layers
    augment = tf.keras.Sequential([
        L.RandomRotation(aug["rotation_deg"] / 360),
        L.RandomTranslation(aug["shift"], aug["shift"]),
        L.RandomZoom(aug["zoom"]),
        L.RandomBrightness((aug["brightness"][0] - 1, aug["brightness"][1] - 1), value_range=(0, 1)),
        L.RandomFlip("horizontal") if aug["horizontal_flip"] else L.Identity(),
    ])
    norm = lambda x, y: (x / 255.0, y)                     # Eq. (1): I / 255
    train = train.map(norm).map(lambda x, y: (augment(x, training=True), y)).prefetch(tf.data.AUTOTUNE)
    val = val.map(norm).prefetch(tf.data.AUTOTUNE)
    return train, val


def train_one(name: str, cfg: dict, data_dir=None, weights="imagenet", epochs=None, out_dir=None):
    import tensorflow as tf
    mc, tc = cfg["models"][name], cfg["training"]
    p1, p2 = epochs or (tc["phase1_epochs"], tc["phase2_epochs"])
    train, val = datasets(cfg, mc["input"], data_dir)
    labels = [CLASSES[int(y)] for _, ys in train.unbatch().take(-1).batch(4096) for y in ys.numpy().ravel()] \
        if data_dir else None
    cw = class_weights(labels) if labels else {0: 1.5849, 1: 0.7304}     # paper's weights for the full set
    model, base = build(name, mc["input"], mc["l2"], weights=None if name == "custom_cnn" else weights)
    cbs = lambda: [tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
                   tf.keras.callbacks.EarlyStopping(patience=tc["early_stopping_patience"], restore_best_weights=True)]
    metrics = ["accuracy", tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.Precision(name="precision")]

    model.compile(optimizer=tf.keras.optimizers.Adam(tc["lr_phase1"]), loss="binary_crossentropy", metrics=metrics)
    h1 = model.fit(train, validation_data=val, epochs=p1, class_weight=cw, callbacks=cbs(), verbose=2)
    hist = {k: list(map(float, v)) for k, v in h1.history.items()}
    if base is not None:
        unfreeze_top(base, mc["unfreeze_last"])
        model.compile(optimizer=tf.keras.optimizers.Adam(tc["lr_phase2"]), loss="binary_crossentropy", metrics=metrics)
    h2 = model.fit(train, validation_data=val, initial_epoch=len(hist["loss"]), epochs=len(hist["loss"]) + p2,
                   class_weight=cw, callbacks=cbs(), verbose=2)
    for k, v in h2.history.items():
        hist[k] = hist.get(k, []) + list(map(float, v))
    hist["phase2_start"] = len(h1.history["loss"])

    out = out_dir or ROOT
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    model.save(out / "models" / f"{name}.keras")
    (out / "results" / f"{name}_history.json").write_text(json.dumps(hist))
    return model, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vgg19", choices=["vgg19", "inceptionv3", "mobilenetv3large", "custom_cnn", "all"])
    a = ap.parse_args()
    cfg = load_config()
    for name in (cfg["models"] if a.model == "all" else [a.model]):
        train_one(name, cfg)


if __name__ == "__main__":
    main()
