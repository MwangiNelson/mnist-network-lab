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
    def __init__(self, x_sample: np.ndarray, y_sample: np.ndarray):
        super().__init__()
        self.x_sample = tf.convert_to_tensor(x_sample)
        self.y_sample = tf.convert_to_tensor(y_sample)
        self.rows: list[dict[str, Any]] = []

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        with tf.GradientTape() as tape:
            predictions = self.model(self.x_sample, training=False)
            loss = keras.losses.sparse_categorical_crossentropy(
                self.y_sample, predictions
            )
            loss = tf.reduce_mean(loss)
        kernels = [
            variable
            for variable in self.model.trainable_variables
            if "kernel" in variable.name
        ]
        gradients = tape.gradient(loss, kernels)
        for variable, gradient in zip(kernels, gradients):
            norm = np.nan if gradient is None else float(tf.norm(gradient).numpy())
            self.rows.append(
                {
                    "epoch": epoch + 1,
                    "layer": variable.name.split("/")[0],
                    "gradient_norm": norm,
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
