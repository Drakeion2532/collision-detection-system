"""The four architectures compared in the paper.

Pretrained backbones (VGG19, InceptionV3, MobileNetV3Large) get a new head:
GlobalAveragePooling -> Dense(256, ReLU, L2) -> Dropout(0.5) -> Dense(1, sigmoid).
The sigmoid output is P(non-accident) because accident is label 0; detect.py turns it
into P(accident) = 1 - p.

The custom CNN has 4 convolution blocks (Conv -> BatchNorm -> MaxPool), as in Fig. 1.
"""
from __future__ import annotations


def _keras():
    import tensorflow as tf  # imported lazily so tests and the data tools run without TF
    return tf.keras


BACKBONES = {
    "vgg19": ("VGG19", "vgg19"),
    "inceptionv3": ("InceptionV3", "inception_v3"),
    "mobilenetv3large": ("MobileNetV3Large", "mobilenet_v3"),
}


def preprocess_fn(name: str):
    """Per-model input scaling. The paper normalises to [0, 1]; Keras backbones
    expect their own scaling, so the [0, 1] image is mapped back for them."""
    keras = _keras()
    if name == "custom_cnn":
        return lambda x: x
    module = getattr(keras.applications, BACKBONES[name][1])
    return lambda x: module.preprocess_input(x * 255.0)


def build(name: str, input_size: int, l2: float, weights: str | None = "imagenet"):
    keras = _keras()
    L, reg = keras.layers, keras.regularizers.l2(l2)
    inp = keras.Input((input_size, input_size, 3))
    if name == "custom_cnn":
        x = inp
        for filters in (32, 64, 128, 256):
            x = L.Conv2D(filters, 3, padding="same", activation="relu", kernel_regularizer=reg)(x)
            x = L.BatchNormalization(momentum=0.9)(x)
            x = L.MaxPooling2D()(x)
        x = L.Flatten()(x)
        x = L.Dense(256, activation="relu", kernel_regularizer=reg)(x)
        x = L.Dropout(0.5)(x)
        base = None
    else:
        cls = getattr(keras.applications, BACKBONES[name][0])
        kwargs = {"include_top": False, "weights": weights, "input_shape": (input_size, input_size, 3)}
        if name == "mobilenetv3large":
            kwargs["include_preprocessing"] = False
        base = cls(**kwargs)
        base.trainable = False
        x = L.Lambda(preprocess_fn(name), name="preprocess")(inp)
        x = base(x, training=False)
        x = L.GlobalAveragePooling2D()(x)
        x = L.Dense(256, activation="relu", kernel_regularizer=reg)(x)
        x = L.Dropout(0.5)(x)
    out = L.Dense(1, activation="sigmoid", kernel_regularizer=reg)(x)
    model = keras.Model(inp, out, name=name)
    return model, base


def unfreeze_top(base, n_layers: int):
    """Phase 2: unfreeze the last n layers of the backbone (BatchNorm stays frozen)."""
    keras = _keras()
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
