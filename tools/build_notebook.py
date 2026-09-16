from __future__ import annotations

import json
from pathlib import Path


def lines(text: str) -> list[str]:
    return [line + "\n" for line in text.strip("\n").split("\n")]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(
        """
# Training neural networks on MNIST

**Course:** DSA 8401 Applied Machine Learning<br>
**Programme:** Master in Data Science and Analytics<br>
**Student:** Nelson Mwangi<br>
**Random seed:** 8401

This notebook designs and evaluates fully connected networks before introducing
a classical CNN benchmark. Model selection uses the validation set. The test set
remains untouched until the final fully connected architecture is fixed.

Run this notebook in Google Colab. Start with smoke mode, then restart the
runtime and use full mode for the submitted results.
"""
    ),
    code(
        """
# Set this to True for the graded run.
RUN_FULL = False
SEED = 8401

%pip install -q "tensorflow>=2.16,<2.20" "seaborn>=0.13,<1" "scikit-learn>=1.4,<2"
"""
    ),
    code(
        """
from pathlib import Path
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

REPO_DIR = Path("/content/mnist-network-lab")
if not (REPO_DIR / "training" / "experiment_utils.py").exists():
    subprocess.run(
        ["git", "clone", "-q", "https://github.com/MwangiNelson/mnist-network-lab.git", str(REPO_DIR)],
        check=True,
    )
sys.path.insert(0, str(REPO_DIR))

from training.experiment_utils import (
    ExperimentConfig,
    GradientNormLogger,
    build_mlp,
    run_experiment,
    save_metadata,
    set_reproducibility,
    train_cnn,
)

set_reproducibility(SEED)
sns.set_theme(style="whitegrid", context="notebook")
ARTIFACT_DIR = Path("/content/artifacts")
PLOT_DIR = ARTIFACT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("Python:", platform.python_version())
print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("Hardware:", tf.config.list_physical_devices())
print("Mode:", "FULL GRADED RUN" if RUN_FULL else "SMOKE TEST ONLY")
"""
    ),
    markdown(
        """
## 1. Data preparation

Pixel scaling keeps inputs in a small, consistent range. This prevents large
input values from producing poorly scaled updates and usually makes gradient
descent more stable. Fully connected models receive a 784-value vector. The CNN
keeps the 28 by 28 spatial grid and adds one channel.
"""
    ),
    code(
        """
(x_pool_raw, y_pool), (x_test_raw, y_test) = keras.datasets.mnist.load_data()
print("Original training pool:", x_pool_raw.shape, y_pool.shape)
print("Held-out test set:", x_test_raw.shape, y_test.shape)
assert x_pool_raw.shape == (60_000, 28, 28)
assert x_test_raw.shape == (10_000, 28, 28)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for axis, image, label in zip(axes.flat, x_pool_raw[:10], y_pool[:10]):
    axis.imshow(image, cmap="gray_r")
    axis.set_title(f"Label {label}")
    axis.axis("off")
fig.suptitle("Sample MNIST training images")
fig.tight_layout()
fig.savefig(PLOT_DIR / "sample_digits.png", dpi=160, bbox_inches="tight")
plt.show()

class_counts = pd.Series(y_pool).value_counts().sort_index()
ax = class_counts.plot.bar(figsize=(9, 4), color="#1463ff", width=0.78)
ax.set(title="Class distribution in the 60,000-image training pool", xlabel="Digit", ylabel="Images")
ax.bar_label(ax.containers[0], fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR / "class_distribution.png", dpi=160, bbox_inches="tight")
plt.show()
display(class_counts.rename("count").to_frame())
"""
    ),
    code(
        """
x_train_raw, x_val_raw, y_train_all, y_val_all = train_test_split(
    x_pool_raw,
    y_pool,
    test_size=5_000,
    random_state=SEED,
    stratify=y_pool,
)
assert len(x_train_raw) == 55_000 and len(x_val_raw) == 5_000

x_train_fc_all = x_train_raw.astype("float32").reshape(-1, 784) / 255.0
x_val_fc_all = x_val_raw.astype("float32").reshape(-1, 784) / 255.0
x_test_fc = x_test_raw.astype("float32").reshape(-1, 784) / 255.0

x_train_cnn_all = x_train_raw.astype("float32")[..., None] / 255.0
x_val_cnn_all = x_val_raw.astype("float32")[..., None] / 255.0
x_test_cnn = x_test_raw.astype("float32")[..., None] / 255.0

if RUN_FULL:
    x_train_fc, y_train = x_train_fc_all, y_train_all
    x_val_fc, y_val = x_val_fc_all, y_val_all
    x_train_cnn, x_val_cnn = x_train_cnn_all, x_val_cnn_all
    EXPERIMENT_EPOCHS, FINAL_EPOCHS = 10, 18
else:
    x_train_fc, y_train = x_train_fc_all[:12_000], y_train_all[:12_000]
    x_val_fc, y_val = x_val_fc_all[:2_000], y_val_all[:2_000]
    x_train_cnn, x_val_cnn = x_train_cnn_all[:12_000], x_val_cnn_all[:2_000]
    EXPERIMENT_EPOCHS, FINAL_EPOCHS = 2, 3

print("Official split:", len(x_train_raw), len(x_val_raw), len(x_test_raw))
print("Run subset:", len(x_train_fc), len(x_val_fc))
print("FC shapes:", x_train_fc_all.shape, x_val_fc_all.shape, x_test_fc.shape)
print("CNN shapes:", x_train_cnn_all.shape, x_val_cnn_all.shape, x_test_cnn.shape)
print("Scaled range:", float(x_train_fc_all.min()), float(x_train_fc_all.max()))
"""
    ),
    markdown(
        """
## Experiment record

Every run contributes one row to the same table. The four questions below guide
the written analysis after each experiment.

1. What did I change?
2. What happened?
3. Why did it happen?
4. What did I learn?
"""
    ),
    code(
        """
records = []
histories = {}
models = {}

def execute(config, callbacks=()):
    model, history, record = run_experiment(
        config,
        x_train_fc,
        y_train,
        x_val_fc,
        y_val,
        callbacks=callbacks,
    )
    records.append(record)
    histories[config.name] = history
    models[config.name] = model
    return model, history, record

def phase_table(phase):
    columns = [
        "name", "hidden_units", "parameters", "activation", "initializer",
        "optimizer", "learning_rate", "batch_size", "dropout", "l2",
        "training_seconds", "best_val_accuracy", "generalization_gap"
    ]
    return pd.DataFrame(records).query("phase == @phase")[columns].sort_values(
        "best_val_accuracy", ascending=False
    )

def plot_histories(names, metric="accuracy", title=None, filename=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in names:
        frame = histories[name]
        axes[0].plot(frame[metric], label=name)
        axes[1].plot(frame[f"val_{metric}"], label=name)
    axes[0].set(title=f"Training {metric}", xlabel="Epoch", ylabel=metric)
    axes[1].set(title=f"Validation {metric}", xlabel="Epoch", ylabel=metric)
    axes[1].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if filename:
        fig.savefig(PLOT_DIR / filename, dpi=160, bbox_inches="tight")
    plt.show()
"""
    ),
    markdown(
        """
## 2. My fully connected architectures

The variants cover one to five hidden layers. `shallow_wide` and `deep_narrow`
have similar parameter counts, which makes their depth versus width comparison
more meaningful than comparing arbitrarily sized networks.
"""
    ),
    code(
        """
architecture_specs = {
    "one_hidden": (256,),
    "two_hidden": (256, 128),
    "three_hidden": (256, 128, 64),
    "four_hidden": (256, 128, 64, 32),
    "five_hidden": (256, 128, 64, 32, 16),
    "deep_narrow": (224, 96, 48, 24),
}
architecture_configs = [
    ExperimentConfig(
        name=name,
        phase="architecture",
        hidden_units=units,
        epochs=EXPERIMENT_EPOCHS,
    )
    for name, units in architecture_specs.items()
]
for config in architecture_configs:
    execute(config)

arch_results = phase_table("architecture")
display(arch_results)
plot_histories(
    [config.name for config in architecture_configs],
    title="Architecture comparison",
    filename="architecture_accuracy.png",
)
best_arch_name = arch_results.iloc[0]["name"]
best_units = architecture_specs[best_arch_name]
print("Validation-selected architecture:", best_arch_name, best_units)
"""
    ),
    markdown(
        """
### Architecture analysis

`[WRITE AFTER RUN]` Compare the one-layer network with the deepest network and
compare `one_hidden` with `deep_narrow`. Quote their parameter counts, training
times, validation accuracies, and generalization gaps. Explain whether added
depth improved the representation or mainly added optimization difficulty.
"""
    ),
    markdown(
        """
## 3. Activation functions and gradients

The activation comparison holds architecture, initialization, optimizer,
learning rate, batch size, and epochs fixed. The separate six-layer sigmoid
network records kernel-gradient norms at the end of every epoch.
"""
    ),
    code(
        """
activation_configs = [
    ExperimentConfig(
        name=f"activation_{activation}",
        phase="activation",
        hidden_units=best_units,
        activation=activation,
        initializer="he_normal" if activation == "relu" else "glorot_uniform",
        epochs=EXPERIMENT_EPOCHS,
    )
    for activation in ("relu", "tanh", "sigmoid")
]
for config in activation_configs:
    execute(config)

activation_results = phase_table("activation")
display(activation_results)
plot_histories(
    [config.name for config in activation_configs],
    metric="loss",
    title="Activation loss comparison",
    filename="activation_loss.png",
)
best_activation = activation_results.iloc[0]["activation"]
print("Validation-selected activation:", best_activation)
"""
    ),
    code(
        """
gradient_logger = GradientNormLogger(x_train_fc[:256], y_train[:256])
deep_sigmoid = ExperimentConfig(
    name="deep_sigmoid_gradient_test",
    phase="gradient_diagnostic",
    hidden_units=(128, 128, 128, 128, 128, 128),
    activation="sigmoid",
    initializer="glorot_uniform",
    epochs=EXPERIMENT_EPOCHS,
)
execute(deep_sigmoid, callbacks=[gradient_logger])
gradient_frame = gradient_logger.to_frame()
display(gradient_frame.head())

plt.figure(figsize=(10, 5))
sns.lineplot(data=gradient_frame, x="epoch", y="gradient_norm", hue="layer", marker="o")
plt.yscale("log")
plt.title("Deep sigmoid network: per-layer kernel gradient norms")
plt.ylabel("L2 gradient norm on log scale")
plt.tight_layout()
plt.savefig(PLOT_DIR / "sigmoid_gradient_norms.png", dpi=160, bbox_inches="tight")
plt.show()
"""
    ),
    markdown(
        """
### Activation and gradient analysis

`[WRITE AFTER RUN]` State which activation converged fastest and quote the final
validation losses. Use the gradient-norm plot to compare the first and last
hidden layers. Sigmoid derivatives are small in saturated regions; multiplying
many such derivatives can shrink the signal sent to early layers. ReLU keeps a
unit derivative for positive inputs, though inactive units can still receive a
zero derivative. Exploding gradients are the opposite failure: repeated large
derivatives make updates unstable or non-finite.
"""
    ),
    markdown("## 4. Weight initialization"),
    code(
        """
initializers = ("zeros", "glorot_uniform", "he_normal")
initialization_configs = [
    ExperimentConfig(
        name=f"initializer_{initializer}",
        phase="initialization",
        hidden_units=best_units,
        activation=best_activation,
        initializer=initializer,
        epochs=EXPERIMENT_EPOCHS,
    )
    for initializer in initializers
]
for config in initialization_configs:
    execute(config)

initialization_results = phase_table("initialization")
display(initialization_results)
plot_histories(
    [config.name for config in initialization_configs],
    metric="loss",
    title="Initialization loss comparison",
    filename="initialization_loss.png",
)
eligible_initializers = initialization_results.query("initializer != 'zeros'")
best_initializer = eligible_initializers.iloc[0]["initializer"]
print("Validation-selected non-zero initializer:", best_initializer)
"""
    ),
    markdown(
        """
### Initialization analysis

`[WRITE AFTER RUN]` Quote the zero, Glorot, and He results. All-zero weights make
neurons in the same layer start identically and receive identical updates. The
network cannot break that symmetry, so increasing the number of hidden neurons
does not create distinct learned features. Relate the winning initializer to
the selected activation and its intended variance scaling.
"""
    ),
    markdown("## 5. Optimization, learning rate, and batch size"),
    code(
        """
optimizer_configs = [
    ExperimentConfig(
        name=f"optimizer_{optimizer}",
        phase="optimizer",
        hidden_units=best_units,
        activation=best_activation,
        initializer=best_initializer,
        optimizer=optimizer,
        learning_rate=0.001,
        epochs=EXPERIMENT_EPOCHS,
    )
    for optimizer in ("adam", "sgd_momentum", "rmsprop")
]
for config in optimizer_configs:
    execute(config)
optimizer_results = phase_table("optimizer")
display(optimizer_results)
best_optimizer = optimizer_results.iloc[0]["optimizer"]

learning_rate_configs = [
    ExperimentConfig(
        name=f"lr_{str(rate).replace('.', '_')}",
        phase="learning_rate",
        hidden_units=best_units,
        activation=best_activation,
        initializer=best_initializer,
        optimizer=best_optimizer,
        learning_rate=rate,
        epochs=EXPERIMENT_EPOCHS,
    )
    for rate in (0.1, 0.01, 0.001)
]
for config in learning_rate_configs:
    execute(config)
learning_rate_results = phase_table("learning_rate")
display(learning_rate_results)
best_learning_rate = float(learning_rate_results.iloc[0]["learning_rate"])

batch_configs = [
    ExperimentConfig(
        name=f"batch_{batch_size}",
        phase="batch_size",
        hidden_units=best_units,
        activation=best_activation,
        initializer=best_initializer,
        optimizer=best_optimizer,
        learning_rate=best_learning_rate,
        batch_size=batch_size,
        epochs=EXPERIMENT_EPOCHS,
    )
    for batch_size in (32, 128, 512)
]
for config in batch_configs:
    execute(config)
batch_results = phase_table("batch_size")
display(batch_results)
best_batch_size = int(batch_results.iloc[0]["batch_size"])

plot_histories(
    [config.name for config in optimizer_configs],
    metric="loss",
    title="Optimizer loss comparison",
    filename="optimizer_loss.png",
)
plot_histories(
    [config.name for config in learning_rate_configs],
    metric="loss",
    title="Learning-rate loss comparison",
    filename="learning_rate_loss.png",
)
plot_histories(
    [config.name for config in batch_configs],
    metric="loss",
    title="Batch-size loss comparison",
    filename="batch_size_loss.png",
)
print("Selected:", best_optimizer, best_learning_rate, best_batch_size)
"""
    ),
    markdown(
        """
### Optimization analysis

`[WRITE AFTER RUN]` Compare convergence in the first few epochs and the best
validation result for all three optimizers. Explain any instability at learning
rate 0.1 using the observed loss, not a generic claim. Compare batch sizes using
both training time and validation accuracy. Smaller batches produce noisier,
more frequent updates; larger batches make fewer updates per epoch.
"""
    ),
    markdown("## 6. Regularization"),
    code(
        """
regularization_specs = [
    ("no_regularization", 0.0, 0.0),
    ("dropout_020", 0.20, 0.0),
    ("l2_0001", 0.0, 1e-4),
    ("dropout_l2", 0.20, 1e-4),
]
regularization_configs = [
    ExperimentConfig(
        name=name,
        phase="regularization",
        hidden_units=best_units,
        activation=best_activation,
        initializer=best_initializer,
        optimizer=best_optimizer,
        learning_rate=best_learning_rate,
        batch_size=best_batch_size,
        epochs=EXPERIMENT_EPOCHS,
        dropout=dropout,
        l2=l2,
    )
    for name, dropout, l2 in regularization_specs
]
for config in regularization_configs:
    execute(config)

regularization_results = phase_table("regularization")
display(regularization_results)
plot_histories(
    [config.name for config in regularization_configs],
    title="Regularization accuracy comparison",
    filename="regularization_accuracy.png",
)
best_regularization_name = regularization_results.iloc[0]["name"]
selected_config = next(c for c in regularization_configs if c.name == best_regularization_name)
print("Validation-selected final configuration:", selected_config)
"""
    ),
    markdown(
        """
### Regularization analysis

`[WRITE AFTER RUN]` Compare the training-validation gap with and without
regularization. State whether the unregularized model overfit enough for dropout
or L2 to help. MNIST is clean, balanced, and large relative to these networks,
so a well-sized model may show only a small gap. Strong dropout can then reduce
both training and validation performance.
"""
    ),
    markdown(
        """
## 7. Final fully connected model

The choices above are now frozen. The next cell trains once with early stopping,
evaluates the untouched test set, saves the model, reloads it, and verifies that
the restored predictions match.
"""
    ),
    code(
        """
final_config = ExperimentConfig(
    name="final_fully_connected",
    phase="final_fc",
    hidden_units=selected_config.hidden_units,
    activation=selected_config.activation,
    initializer=selected_config.initializer,
    optimizer=selected_config.optimizer,
    learning_rate=selected_config.learning_rate,
    batch_size=selected_config.batch_size,
    epochs=FINAL_EPOCHS,
    dropout=selected_config.dropout,
    l2=selected_config.l2,
)
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
final_fc, final_history, final_record = run_experiment(
    final_config,
    x_train_fc_all if RUN_FULL else x_train_fc,
    y_train_all if RUN_FULL else y_train,
    x_val_fc_all if RUN_FULL else x_val_fc,
    y_val_all if RUN_FULL else y_val,
    callbacks=[early_stopping],
)
records.append(final_record)
histories[final_config.name] = final_history

fc_test_loss, fc_test_accuracy = final_fc.evaluate(x_test_fc, y_test, verbose=0)
fc_probabilities = final_fc.predict(x_test_fc, verbose=0)
fc_predictions = fc_probabilities.argmax(axis=1)
print(f"FC test loss: {fc_test_loss:.4f}")
print(f"FC test accuracy: {fc_test_accuracy:.4f}")
print(classification_report(y_test, fc_predictions, digits=4))

ConfusionMatrixDisplay.from_predictions(
    y_test, fc_predictions, cmap="Blues", colorbar=False, values_format="d"
)
plt.title("Final fully connected model: test confusion matrix")
plt.tight_layout()
plt.savefig(PLOT_DIR / "fc_confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.show()

error_indices = np.flatnonzero(fc_predictions != y_test)
rng = np.random.default_rng(SEED)
shown = rng.choice(error_indices, size=min(15, len(error_indices)), replace=False)
fig, axes = plt.subplots(3, 5, figsize=(11, 7))
for axis in axes.flat:
    axis.axis("off")
for axis, index in zip(axes.flat, shown):
    axis.imshow(x_test_raw[index], cmap="gray_r")
    axis.set_title(
        f"True {y_test[index]} | Pred {fc_predictions[index]}\n"
        f"p={fc_probabilities[index, fc_predictions[index]]:.2f}"
    )
    axis.axis("off")
fig.suptitle("Misclassified test digits")
fig.tight_layout()
fig.savefig(PLOT_DIR / "fc_misclassifications.png", dpi=160, bbox_inches="tight")
plt.show()
"""
    ),
    code(
        """
model_path = ARTIFACT_DIR / "final_fc.keras"
final_fc.save(model_path)
restored_fc = keras.models.load_model(model_path)
before = final_fc.predict(x_test_fc[:32], verbose=0)
after = restored_fc.predict(x_test_fc[:32], verbose=0)
np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-6)
print("Restored predictions match the original model.")

save_metadata(
    ARTIFACT_DIR / "model_metadata.json",
    model="final_fully_connected",
    input_shape=[784],
    classes=list(range(10)),
    scaling="pixel / 255.0",
    seed=SEED,
    test_accuracy=float(fc_test_accuracy),
    test_loss=float(fc_test_loss),
    config={
        "hidden_units": list(final_config.hidden_units),
        "activation": final_config.activation,
        "initializer": final_config.initializer,
        "optimizer": final_config.optimizer,
        "learning_rate": final_config.learning_rate,
        "batch_size": final_config.batch_size,
        "dropout": final_config.dropout,
        "l2": final_config.l2,
    },
)
"""
    ),
    markdown(
        """
### Final fully connected model analysis

`[WRITE AFTER RUN]` Report the selected configuration, best validation epoch,
test accuracy, most common confusion pairs, and two visually ambiguous errors.
Explain whether the held-out result agrees with the validation estimate.
"""
    ),
    markdown(
        """
## 8. Classical CNN benchmark

The fully connected work is complete. This convolution-pooling-convolution-
pooling stack now receives the image grid and uses the same optimizer, learning
rate, batch size, and maximum epoch budget where reasonable.
"""
    ),
    code(
        """
cnn, cnn_history, cnn_record = train_cnn(
    x_train_cnn_all if RUN_FULL else x_train_cnn,
    y_train_all if RUN_FULL else y_train,
    x_val_cnn_all if RUN_FULL else x_val_cnn,
    y_val_all if RUN_FULL else y_val,
    epochs=FINAL_EPOCHS,
    batch_size=final_config.batch_size,
    learning_rate=final_config.learning_rate,
    optimizer_name=final_config.optimizer,
)
records.append(cnn_record)
histories["classical_cnn"] = cnn_history

cnn_test_loss, cnn_test_accuracy = cnn.evaluate(x_test_cnn, y_test, verbose=0)
cnn_probabilities = cnn.predict(x_test_cnn, verbose=0)
cnn_predictions = cnn_probabilities.argmax(axis=1)
print(f"CNN test loss: {cnn_test_loss:.4f}")
print(f"CNN test accuracy: {cnn_test_accuracy:.4f}")

ConfusionMatrixDisplay.from_predictions(
    y_test, cnn_predictions, cmap="Blues", colorbar=False, values_format="d"
)
plt.title("Classical CNN: test confusion matrix")
plt.tight_layout()
plt.savefig(PLOT_DIR / "cnn_confusion_matrix.png", dpi=160, bbox_inches="tight")
plt.show()

comparison = pd.DataFrame([
    {
        "model": "Student-designed fully connected",
        "parameters": final_fc.count_params(),
        "training_seconds": final_record["training_seconds"],
        "test_accuracy": fc_test_accuracy,
        "test_errors": int((fc_predictions != y_test).sum()),
    },
    {
        "model": "Classical CNN benchmark",
        "parameters": cnn.count_params(),
        "training_seconds": cnn_record["training_seconds"],
        "test_accuracy": cnn_test_accuracy,
        "test_errors": int((cnn_predictions != y_test).sum()),
    },
])
display(comparison)
"""
    ),
    markdown(
        """
### Fully connected model versus CNN

`[WRITE AFTER RUN]` State what the student-designed model achieved before the
CNN was introduced, then give the accuracy gap in percentage points. Convolution
uses local receptive fields and shares a filter across positions. Pooling makes
small translations less disruptive. Flattening first discards the explicit row
and column relationships, so a fully connected layer must learn similar stroke
detectors separately at different positions. Compare the models' largest
confusion pairs to see whether the CNN reduced the same errors or different ones.
"""
    ),
    markdown("## Complete experiment table and conclusions"),
    code(
        """
results = pd.DataFrame(records)
results.to_csv(ARTIFACT_DIR / "experiment_results.csv", index=False)
display(results.sort_values(["phase", "best_val_accuracy"], ascending=[True, False]))

summary_columns = [
    "phase", "name", "hidden_units", "parameters", "training_seconds",
    "best_val_accuracy", "generalization_gap"
]
display(results[summary_columns])
"""
    ),
    markdown(
        """
## Personal conclusion

`[WRITE AFTER RUN]` Write this in your own voice. Use exact numbers to answer:

- Which depth and width worked best, and what trade-off did you observe?
- What evidence showed weak gradient flow in the deep sigmoid model?
- How did zero, Glorot, and He initialization differ?
- Which optimizer, learning rate, and batch size produced the best validation result?
- Did regularization fix measurable overfitting or add unnecessary bias?
- What did the fully connected model achieve before the CNN comparison?
- How large was the CNN gap, and which image-specific assumptions explain it?

End with one limitation. A single seed and one train-validation split do not
measure run-to-run uncertainty; repeated seeds would strengthen the comparison.
"""
    ),
    code(
        """
archive_base = "/content/mnist_assignment_artifacts"
archive_path = shutil.make_archive(archive_base, "zip", ARTIFACT_DIR)
print("Download:", archive_path)
print("Also download this executed notebook from Colab's File menu.")
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "mnist_neural_networks.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).resolve().parents[1] / "notebooks" / "mnist_neural_networks.ipynb"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(output)
