# Copyright (c) 2026 Shubhan Punde · https://github.com/Drakeion2532/collision-detection-system · MIT licence · sp2532-2be7050d798f
"""Builds every architecture and runs a two-phase training step on tiny synthetic data.

Uses random initial weights so the test runs offline; real training uses ImageNet weights.
Skipped automatically when TensorFlow is not installed.
"""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from tcd.models import build, unfreeze_top  # noqa: E402


@pytest.mark.parametrize("name,size", [("custom_cnn", 64), ("mobilenetv3large", 96), ("vgg19", 64), ("inceptionv3", 96)])
def test_build_and_two_phase_step(name, size):
    model, base = build(name, size, 1e-4, weights=None)
    x = np.random.rand(4, size, size, 3).astype("float32")
    y = np.array([[0], [1], [0], [1]], "float32")
    model.compile(tf.keras.optimizers.Adam(1e-4), "binary_crossentropy")
    model.fit(x, y, epochs=1, verbose=0, class_weight={0: 1.5849, 1: 0.7304})
    if base is not None:
        frozen = sum(int(not l.trainable) for l in base.layers)
        unfreeze_top(base, 4)
        assert sum(int(l.trainable) for l in base.layers) > 0 and frozen > 0
        model.compile(tf.keras.optimizers.Adam(1e-5), "binary_crossentropy")
        model.fit(x, y, epochs=1, verbose=0)
    p = model.predict(x, verbose=0)
    assert p.shape == (4, 1) and np.all((p >= 0) & (p <= 1))
