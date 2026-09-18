from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


SEED = 8401


def set_reproducibility(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    phase: str
    hidden_units: tuple[int, ...]
    activation: str = "relu"
    initializer: str = "he_normal"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 12
    dropout: float = 0.0
    l2: float = 0.0


def make_optimizer(name: str, learning_rate: float) -> keras.optimizers.Optimizer:
    key = name.lower()
    if key == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if key == "sgd_momentum":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    if key == "rmsprop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")


def build_mlp(config: ExperimentConfig) -> keras.Model:
    regularizer = keras.regularizers.l2(config.l2) if config.l2 else None
    inputs = keras.Input(shape=(784,), name="pixels")
    x = inputs
    for index, units in enumerate(config.hidden_units, start=1):
        x = keras.layers.Dense(
            units,
            activation=config.activation,
            kernel_initializer=config.initializer,
            kernel_regularizer=regularizer,
            name=f"hidden_{index}",
        )(x)
        if config.dropout:
            x = keras.layers.Dropout(config.dropout, name=f"dropout_{index}")(x)
    outputs = keras.layers.Dense(10, activation="softmax", name="digit")(x)
    model = keras.Model(inputs, outputs, name=config.name)
    model.compile(
        optimizer=make_optimizer(config.optimizer, config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(
    learning_rate: float = 1e-3,
    optimizer_name: str = "adam",
) -> keras.Model:
    inputs = keras.Input(shape=(28, 28, 1), name="image")
    x = keras.layers.Conv2D(32, 5, activation="relu", padding="same", name="conv_1")(inputs)
    x = keras.layers.MaxPooling2D(2, name="pool_1")(x)
    x = keras.layers.Conv2D(64, 5, activation="relu", padding="valid", name="conv_2")(x)
    x = keras.layers.MaxPooling2D(2, name="pool_2")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(120, activation="relu", name="dense_1")(x)
    x = keras.layers.Dense(84, activation="relu", name="dense_2")(x)
    outputs = keras.layers.Dense(10, activation="softmax", name="digit")(x)
    model = keras.Model(inputs, outputs, name="classical_cnn")
    model.compile(
        optimizer=make_optimizer(optimizer_name, learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class GradientNormLogger(keras.callbacks.Callback):
    """Record per-layer gradient magnitudes for the deep sigmoid diagnostic.

    Timing is the whole point. Epoch 0 is recorded before any weight update,
    because that is where a saturating stack attenuates most. Glorot
    initialization gives a six-layer sigmoid network a backward gain near
    ``sqrt(128 * 2/256) * 0.25 = 0.25`` per layer, so the signal reaching the
    first hidden layer starts around ``0.25 ** 5`` of the signal at the last
    one. Training removes this quickly: the network learns larger weights that
    push signal through, so by the end of the first epoch the per-layer norms
    are almost flat. Measuring only at epoch end therefore shows nothing, which
    is what an earlier version of this callback did.

    Two quantities are recorded per hidden layer. ``activation_grad_norm`` is
    the gradient with respect to the layer's output activations, which is what
    backpropagation multiplies by a saturating derivative at every step.
    ``kernel_grad_norm`` is the gradient with respect to the weight matrix; it
    also carries the layer's fan-in and the scale of its incoming activations,
    so comparing it across layers of different widths is misleading.
    ``weights`` is recorded so a per-weight figure can be derived rather than
    assumed.
    """

    def __init__(self, x_sample: np.ndarray, y_sample: np.ndarray):
        super().__init__()
        self.x_sample = tf.convert_to_tensor(x_sample)
        self.y_sample = tf.convert_to_tensor(y_sample)
        self.rows: list[dict[str, Any]] = []

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        self._record(0)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self._record(epoch + 1)

    def _record(self, epoch: int) -> None:
        hidden_layers = [
            layer for layer in self.model.layers if layer.name.startswith("hidden_")
        ]
        output_layer = self.model.get_layer("digit")

        # Step through the Dense layers by hand so each activation can be watched
        # the moment it exists. A single opaque model call gives no handle on the
        # intermediate tensors, so their gradients would come back as None.
        # Dropout is skipped, which is what inference does anyway.
        with tf.GradientTape(persistent=True) as tape:
            activation = self.x_sample
            activations = []
            for layer in hidden_layers:
                activation = layer(activation)
                tape.watch(activation)
                activations.append(activation)
            predictions = output_layer(activation)
            loss = tf.reduce_mean(
                keras.losses.sparse_categorical_crossentropy(
                    self.y_sample, predictions
                )
            )

        activation_gradients = tape.gradient(loss, activations)
        kernels = [layer.kernel for layer in hidden_layers] + [output_layer.kernel]
        kernel_gradients = tape.gradient(loss, kernels)
        del tape

        def norm(gradient: Any) -> float:
            return np.nan if gradient is None else float(tf.norm(gradient).numpy())

        layers = hidden_layers + [output_layer]
        # The output layer has no activation gradient of its own to report.
        activation_gradients = list(activation_gradients) + [None]
        for layer, activation_gradient, kernel_gradient in zip(
            layers, activation_gradients, kernel_gradients
        ):
            self.rows.append(
                {
                    "epoch": epoch,
                    "layer": layer.name,
                    "activation_grad_norm": norm(activation_gradient),
                    "kernel_grad_norm": norm(kernel_gradient),
                    "gradient_norm": norm(kernel_gradient),
                    "weights": int(np.prod(layer.kernel.shape)),
                }
            )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def run_experiment(
    config: ExperimentConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    callbacks: Iterable[keras.callbacks.Callback] = (),
    verbose: int = 2,
    show_summary: bool = True,
) -> tuple[keras.Model, pd.DataFrame, dict[str, Any]]:
    set_reproducibility()
    model = build_mlp(config)
    if show_summary:
        model.summary()
    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=list(callbacks),
        verbose=verbose,
    )
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(history.history)
    best_epoch = int(frame["val_accuracy"].idxmax()) + 1
    record = {
        **asdict(config),
        "hidden_units": "-".join(map(str, config.hidden_units)),
        "parameters": model.count_params(),
        "training_seconds": round(elapsed, 3),
        "best_epoch": best_epoch,
        "best_val_accuracy": float(frame["val_accuracy"].max()),
        "best_val_loss": float(frame["val_loss"].min()),
        "final_train_accuracy": float(frame["accuracy"].iloc[-1]),
        "final_val_accuracy": float(frame["val_accuracy"].iloc[-1]),
        "generalization_gap": float(
            frame["accuracy"].iloc[-1] - frame["val_accuracy"].iloc[-1]
        ),
    }
    return model, frame, record


def train_cnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    verbose: int = 2,
) -> tuple[keras.Model, pd.DataFrame, dict[str, Any]]:
    set_reproducibility()
    model = build_cnn(learning_rate, optimizer_name)
    model.summary()
    started = time.perf_counter()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(history.history)
    record = {
        "name": "classical_cnn",
        "phase": "cnn_benchmark",
        "hidden_units": "Conv32-Pool-Conv64-Pool-120-84",
        "activation": "relu",
        "initializer": "glorot_uniform",
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "dropout": 0.0,
        "l2": 0.0,
        "parameters": model.count_params(),
        "training_seconds": round(elapsed, 3),
        "best_epoch": int(frame["val_accuracy"].idxmax()) + 1,
        "best_val_accuracy": float(frame["val_accuracy"].max()),
        "best_val_loss": float(frame["val_loss"].min()),
        "final_train_accuracy": float(frame["accuracy"].iloc[-1]),
        "final_val_accuracy": float(frame["val_accuracy"].iloc[-1]),
        "generalization_gap": float(
            frame["accuracy"].iloc[-1] - frame["val_accuracy"].iloc[-1]
        ),
    }
    return model, frame, record


def save_metadata(path: str | Path, **values: Any) -> None:
    Path(path).write_text(json.dumps(values, indent=2), encoding="utf-8")
