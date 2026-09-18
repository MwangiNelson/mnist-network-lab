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

%pip install -q "tensorflow>=2.20,<2.22" "seaborn>=0.13,<1" "scikit-learn>=1.4,<2"
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
if (REPO_DIR / ".git").exists():
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "pull", "-q", "--ff-only"],
        check=True,
    )
else:
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

**What I changed.** Hidden layers from one to five, all ReLU with He
initialization, Adam at 0.001 and batch 128, ten epochs each. The sixth network,
`deep_narrow` at 224-96-48-24, exists to hold the parameter count fixed: its
203,522 weights sit within eight of `one_hidden`'s 203,530.

**What happened.** The single 256-unit layer won at 98.04% validation accuracy.
Depth did not help once: 97.66%, 97.68%, 97.94% and 97.58% for two through five
layers, and the largest network at 244,890 parameters finished last of the
stack. The matched-parameter comparison is the cleaner one, and width took it,
98.04% against 97.46%, a gap of 0.58 points with the same weight budget.

Generalization gaps stayed between 1.60 and 2.10 points everywhere, with no
trend against depth. The deepest network did not overfit more than the
shallowest.

**Why.** MNIST digits are 784 inputs that a single wide layer can already
separate. A deeper stack buys hierarchical features that this problem does not
need, and pays for them with a longer path for gradients to travel. The
accuracy differences here are small enough that most of the ordering between
two, three, four and five layers is run-to-run noise, and I do not claim a
mechanism for it. What the matched-parameter pair does support is that spending
203,530 weights on one wide layer beats spreading the same weights across four
narrow ones.

**One caveat about the time column.** `one_hidden` reports 41.5 seconds, but
this exact configuration is retrained six more times later in the notebook as
the baseline for the activation, initialization, optimizer, learning-rate,
batch-size and regularization phases. Those six report 23.0, 22.6, 22.3, 21.3,
22.0 and 22.2 seconds, and all seven reach 98.04%. The extra 19 seconds in the
first row is GPU warm-up and graph compilation charged to whichever experiment
runs first. Its real training cost is about 22 seconds, and I read the
architecture time column with that in mind.

**What I learned.** Depth is not free and is not automatically better. Matching
parameter counts is what turns a depth-versus-width question into a fair one,
and timing the first fit in a session measures the session, not the model.
"""
    ),
    markdown(
        """
## 3. Activation functions and gradients

The activation comparison holds architecture, initialization, optimizer,
learning rate, batch size, and epochs fixed. The separate six-layer sigmoid
network records two gradient norms per layer, once before any weight update
and again at the end of every epoch: the gradient with respect to the layer's
output activations, and the gradient with respect to its weight matrix.

The measurement before training is the one that matters. Glorot initialization
gives this stack a backward gain near `sqrt(128 * 2/256) * 0.25 = 0.25` per
layer, so at epoch 0 the signal reaching the first hidden layer is a small
fraction of the signal at the last. Training removes that within one epoch,
because the network learns larger weights that push signal through and the
per-layer norms flatten out. Recording only at epoch end gives an almost flat
profile and proves nothing.

Of the two quantities, only the activation gradient answers the
vanishing-gradient question. Backpropagation multiplies it by a saturating
derivative at every layer it passes through, so attenuation shows up there. A
weight gradient also carries the layer's fan-in and the scale of its incoming
activations, which makes it useless for comparing layers of different widths:
`hidden_1` holds 784x128 weights against 128x128 for the rest, so its norm is
inflated by element count alone. Both are recorded, along with the weight
count, so the analysis can divide by it rather than assume it.
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
gradient_frame.to_csv(ARTIFACT_DIR / "gradient_norms.csv", index=False)
display(gradient_frame.head())

hidden_gradients = gradient_frame[gradient_frame["layer"].str.startswith("hidden_")]
hidden_gradients = hidden_gradients.assign(
    depth=hidden_gradients["layer"].str.removeprefix("hidden_").astype(int)
)


def attenuation_at(epoch):
    stage = hidden_gradients[hidden_gradients["epoch"] == epoch].set_index("layer")
    return (
        stage.loc["hidden_6", "activation_grad_norm"]
        / stage.loc["hidden_1", "activation_grad_norm"]
    )


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Depth on the x axis is what makes a per-layer multiplicative decay read as a
# straight line on a log scale.
for epoch, style in ((0, "o-"), (1, "s--"), (EXPERIMENT_EPOCHS, "^:")):
    stage = hidden_gradients[hidden_gradients["epoch"] == epoch].sort_values("depth")
    if stage.empty:
        continue
    label = f"epoch {epoch}"
    if epoch == 0:
        label += " (before training)"
    axes[0].plot(stage["depth"], stage["activation_grad_norm"], style, label=label)
axes[0].set_yscale("log")
axes[0].set_xlabel("hidden layer, 1 is nearest the input")
axes[0].set_ylabel("activation gradient L2 norm, log scale")
axes[0].set_title("Attenuation at initialization, trained away within one epoch")
axes[0].legend()

sns.lineplot(
    data=hidden_gradients,
    x="epoch",
    y="activation_grad_norm",
    hue="layer",
    marker="o",
    ax=axes[1],
)
axes[1].set_yscale("log")
axes[1].set_title("Per-layer activation gradient over training")
axes[1].set_ylabel("L2 norm on log scale")
axes[1].legend(fontsize=7)

fig.suptitle("Deep sigmoid network: where the gradient signal goes")
fig.tight_layout()
fig.savefig(PLOT_DIR / "sigmoid_gradient_norms.png", dpi=160, bbox_inches="tight")
plt.show()

print(f"Attenuation hidden_6 to hidden_1 at initialization: {attenuation_at(0):.1f}x")
print(f"Attenuation hidden_6 to hidden_1 after one epoch:   {attenuation_at(1):.1f}x")
initial = hidden_gradients[hidden_gradients["epoch"] == 0].set_index("layer")
display(initial[["activation_grad_norm", "kernel_grad_norm", "weights"]])
"""
    ),
    markdown(
        """
### Activation and gradient analysis

**What I changed.** The activation on the selected 256-unit network, across
ReLU, tanh and sigmoid, with He initialization for ReLU and Glorot for the two
saturating ones. Then a separate six-layer sigmoid network at 128 units per
layer, purely as a gradient diagnostic.

**What happened at one hidden layer.** ReLU reached 98.04% with a validation
loss of 0.0770, tanh 97.82% at 0.0782, and sigmoid 96.98% at 0.0996. ReLU and
tanh are close enough that I would not read much into the 0.22 point difference.
Sigmoid is the outlier, and the interesting part is how it failed: its training
accuracy finished at 98.02% against 99.74% for ReLU, and it had the smallest
generalization gap of the three at 1.04 points. It underfitted. It did not
overfit and it did not diverge, it simply learned more slowly within the same
ten epochs.

**What happened with depth.** The six-layer sigmoid network reached 94.30% with
a validation loss of 0.2026, which is 2.6 times the ReLU loss. Putting the two
comparisons together gives the cleanest result in this section. Going from ReLU
to sigmoid at one layer costs 1.06 points. Adding depth to ReLU costs about 0.5
points, from 98.04% at one layer to 97.58% at five. Adding depth to sigmoid
costs a further 2.68 points, from 96.98% to 94.30%. Depth punishes sigmoid about
five times harder than it punishes ReLU, and that interaction needs no caveats
about how the gradient was measured.

**The gradient evidence, and when to look for it.** At initialization the
activation gradient falls geometrically from the layer nearest the loss to the
layer nearest the input: 8.451e-02 at `hidden_6`, then 2.003e-02, 4.894e-03,
1.207e-03, 2.800e-04, and 6.614e-05 at `hidden_1`. Each step multiplies by about
0.24, and across five steps that is a total attenuation of 1277.8x. On the
log-scale plot against depth it is a straight line, which is what a constant
per-layer factor looks like.

That factor is predictable in advance. Glorot uniform gives these 128 by 128
layers a weight variance of `2/256`, so the backward pass through one layer
scales the gradient by about `sqrt(128 * 2/256) = 1` from the weights and by the
sigmoid derivative from the activation. A sigmoid sitting near zero has a
derivative of 0.25, giving a predicted gain of 0.25 per layer against the 0.238
I measured.

Then it disappears. By the end of epoch 1 the same ratio is 0.9x, and by epoch
10 it is 0.1x, meaning the first hidden layer now carries the largest gradient
of the six. The network learns weights big enough to push signal back through
the saturating layers, and it does so within a single epoch.

**Why the timing decides the experiment.** My first attempt at this diagnostic
recorded only at epoch end, and measured no attenuation whatsoever: 0.0721 at
`hidden_1` against 0.0656 at `hidden_6`, a ratio of 0.9x. I nearly concluded that
this network shows no vanishing gradient. The effect was there all along, three
orders of magnitude in size, and had been trained away 430 optimizer steps
before the first measurement was taken.

I also recorded the kernel gradient for contrast, and it is the wrong instrument
here for a second reason. A weight gradient carries the layer's fan-in and the
scale of its incoming activations, so `hidden_1` at 784 by 128 cannot be compared
with five 128 by 128 layers on raw norm: its 100,352 weights inflate the
Frobenius norm by about 2.5 times on element count alone. That is why the weight
count is recorded alongside it.

**How activations shape this.** Sigmoid derivatives are at most 0.25 and fall
towards zero once a unit saturates, so a stack of them multiplies the
backpropagated signal down at every step. ReLU keeps a derivative of exactly one
wherever its input is positive, which is why the same depth costs it so much
less, though a unit stuck at a negative input passes back zero and stops
learning entirely. Exploding gradients are this mechanism with the factor above
one: had the initial scaling put the per-layer gain at 4 rather than 0.24, these
five layers would amplify by roughly 1000x instead of attenuating, and updates
would oscillate or overflow. The learning rate 0.1 run in the next section is the
practical version of that failure.

**What I learned.** Vanishing gradients are a property of a network at
initialization, not a permanent condition, and an instrument that samples at the
wrong moment reports their absence just as confidently as their presence. The
quantity has to be the gradient with respect to activations, the moment has to be
before the first update, and depth belongs on the x axis so that a constant
per-layer factor shows up as a straight line.
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

**What I changed.** The initializer on the selected 256-unit ReLU network,
across all zeros, Glorot uniform and He normal. Nothing else moved.

**What happened.** Zeros reached 11.24% and stopped there. Its best epoch was 1,
its training accuracy was also 11.24%, and its generalization gap was 0.00
points. Its validation loss sat at 2.3012, which is essentially `ln(10)` at
2.3026, the loss of a model assigning equal probability to all ten classes.
Glorot reached 97.76% and He reached 98.04%.

**Why.** With every weight at zero, all 256 units in the layer compute the same
output, receive the same gradient and take the same step. They stay identical
for the whole run, so a 256-unit layer has the effective capacity of a single
unit, and the best it can do is predict the class prior. The width of the layer
is irrelevant, which is the point: the failure is symmetry, not size. The
accuracy of 11.24% is the share of the most common digit in my 5,000-image
validation split, not a tenth.

He beat Glorot by 0.28 points, which is the ordering I expected from the
activation. He scales the initial variance by `2/fan_in`, and the factor of two
compensates for ReLU zeroing roughly half its inputs. Glorot uses
`2/(fan_in+fan_out)` and assumes an activation that is symmetric about zero,
which ReLU is not. The margin is small because one layer gives the mismatch
little room to compound.

**What I learned.** Initialization is not a detail to accept from the defaults.
Zero weights do not train slowly, they do not train at all, and the loss value
tells you the model has collapsed to the class prior rather than merely
underperforming.
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

**What I changed.** Three optimizers at a fixed learning rate of 0.001, then
three learning rates on Adam, then three batch sizes on Adam.

**What happened.** Adam reached 98.04%, RMSprop 97.92% and SGD with momentum
91.52%. Learning rate 0.001 gave 98.04%, 0.01 gave 97.00% and 0.1 gave 89.56%.
Batch 128 gave 98.04% in 22.0 seconds, batch 32 gave 97.74% in 79.1 seconds and
batch 512 gave 97.40% in 7.6 seconds.

**Why.** The SGD result is not the optimizer being worse, and I want to be
careful here. Its training accuracy finished at 91.85% against a validation
accuracy of 91.52%, a gap of 0.33 points, and its best epoch was the last one.
That is a model still climbing when the budget ran out, not one that converged
somewhere poor. A step size of 0.001 is small for plain SGD while Adam's
per-parameter scaling makes the same nominal rate an effective one. Comparing
optimizers at a single shared learning rate measures the pairing, not the
optimizer, and a fair comparison would tune the rate separately for each.

Learning rate 0.1 failed in a specific way worth naming. Its best validation
accuracy, 89.56%, came at epoch 1 and the run never beat it again, finishing at
84.12% validation against 83.06% training. Training accuracy below validation
accuracy, a gap of minus 1.06 points, is the signature of steps large enough to
keep throwing the weights back out of whatever basin they just found. Training
accuracy is averaged across the epoch while validation accuracy is measured once
at the end, so a run that thrashes throughout scores worse on the average than
on the snapshot. A learning rate that peaks in the first epoch and decays
afterwards is diverging, not converging slowly.

Batch size traded time against accuracy almost linearly in the direction I
expected. Batch 32 makes 1,719 updates per epoch and batch 512 makes 108, so
the small batch spent 3.6 times the wall clock of batch 128 for 0.30 points
less accuracy. Batch 512 is the interesting one: it ran in under a tenth of
batch 32's time and gave up only 0.34 points, because fewer, less noisy updates
cover less ground per epoch at this budget.

**What I learned.** Adam at 0.001 with batch 128 is the configuration I carry
forward, but the batch 512 result is the one I would reach for if I had to run
this sweep twenty times. The 0.1 run is also a reminder that a rising then
falling validation curve is diagnostic on its own, before any loss value is
quoted.
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

**What I changed.** Dropout at 0.2, L2 at 0.0001, and both together, on the
selected 256-unit ReLU network.

**What happened.** Nothing improved validation accuracy. The unregularized
network reached 98.04%, dropout 97.98%, L2 97.96%, and both together 97.86%.
The generalization gap did fall in step with how much regularization I added:
1.70 points with none, 1.43 with L2, 1.15 with dropout, and 0.79 with both.

One number moved the other way. Dropout gave the lowest validation loss of the
four, 0.0702 against 0.0770 unregularized, while scoring 0.06 points lower on
accuracy. It produced better calibrated probabilities on roughly the same set of
correct answers.

**Why.** There was no overfitting here worth fixing. A 1.70 point gap on 55,000
training images with 203,530 parameters is close to the floor for this problem.
MNIST is clean, balanced across the ten digits, and large relative to a network
this size. Regularization works by trading variance for bias, and with almost no
variance to reclaim, every configuration paid the bias and got nothing back. The
gap declining monotonically alongside accuracy declining monotonically is what
that trade looks like when the trade is not worth making.

**What I learned.** A shrinking train-validation gap is not evidence that
regularization helped. It only shows the regularizer is doing something. I kept
the unregularized configuration for the final model and would revisit that on a
smaller or dirtier dataset, where the gap would give dropout something to work
with. Early stopping still earns its place in the final run for a different
reason: it picks the best epoch rather than the last one.
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
fc_confusion = confusion_matrix(y_test, fc_predictions)
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
pd.DataFrame({
    "test_index": error_indices,
    "true_label": y_test[error_indices],
    "predicted_label": fc_predictions[error_indices],
    "predicted_probability": fc_probabilities[
        error_indices, fc_predictions[error_indices]
    ],
}).to_csv(ARTIFACT_DIR / "fc_misclassifications.csv", index=False)
pd.DataFrame(fc_confusion).to_csv(ARTIFACT_DIR / "fc_confusion_matrix.csv", index=False)
rng = np.random.default_rng(SEED)
shown = rng.choice(error_indices, size=min(15, len(error_indices)), replace=False)
fig, axes = plt.subplots(3, 5, figsize=(11, 7))
for axis in axes.flat:
    axis.axis("off")
for axis, index in zip(axes.flat, shown):
    axis.imshow(x_test_raw[index], cmap="gray_r")
    axis.set_title(
        f"True {y_test[index]} | Pred {fc_predictions[index]}\\n"
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

**The selected configuration.** One hidden layer of 256 ReLU units, He normal
initialization, Adam at a learning rate of 0.001, batch size 128, no dropout and
no L2, with early stopping on validation loss at a patience of three. That is
203,530 parameters. Every choice in that list came from a validation comparison
in the sections above, and the test set was not touched until this cell.

**Results.** Early stopping selected epoch 11, at 98.06% validation accuracy. On
the 10,000 held-out test images the model reached 97.89% accuracy with a loss of
0.0753, making 211 errors.

Validation said 98.06% and the test set said 97.89%, a difference of 0.17
points. On 10,000 images one standard error is roughly 0.14 points, so the two
agree to within the noise. The validation set never fed a gradient update, but it
did choose between about twenty-five candidates and pick the stopping epoch, and
selecting that many times against 5,000 images is enough to bias the estimate
optimistically by a fraction of a point. The 0.17-point drop is what that looks
like.

**Where the errors fall.** Digit 8 is the weakest class with 35 errors, then 7
with 33, 9 with 26 and 5 with 25. The most frequent single confusions are 7
predicted as 9 (12 times), 5 predicted as 3 (11), 7 predicted as 2 (10), then 2
as 8 and 6 as 0 at 7 each.

**On confidence.** The error set is not mostly near-misses. Of the 211 errors, 74
were made with more than 90% probability on the wrong class. Test image 3520 is a
6 called a 4 at 100.00% confidence, and image 9587 is a 9 called a 4 at 99.99%.
At the other end, image 3060 is a 9 called a 7 at 28.10% and image 1941 is a 7
called a 9 at 28.15%, where the model is visibly unsure and the digits really are
ambiguous. A softmax probability from this network is not a reliable measure of
how likely it is to be right, which matters directly for the demo application
that shows these numbers to a user.

**Save and restore.** The model is written to `final_fc.keras`, reloaded, and the
reloaded copy's predictions are compared against the original with
`np.testing.assert_allclose` at a relative tolerance of 1e-5. The cell fails if
they diverge. The comparison covers the first 32 test images rather than all
10,000, which is enough to catch a broken round trip, since serialization
failures affect every prediction rather than a rare few.
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
cnn_confusion = confusion_matrix(y_test, cnn_predictions)
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
comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
cnn_error_indices = np.flatnonzero(cnn_predictions != y_test)
pd.DataFrame({
    "test_index": cnn_error_indices,
    "true_label": y_test[cnn_error_indices],
    "predicted_label": cnn_predictions[cnn_error_indices],
    "predicted_probability": cnn_probabilities[
        cnn_error_indices, cnn_predictions[cnn_error_indices]
    ],
}).to_csv(ARTIFACT_DIR / "cnn_misclassifications.csv", index=False)
pd.DataFrame(cnn_confusion).to_csv(ARTIFACT_DIR / "cnn_confusion_matrix.csv", index=False)
display(comparison)
"""
    ),
    markdown(
        """
### Fully connected model versus CNN

**Where my own architecture finished.** Before any convolution appeared in this
notebook, my fully connected model reached 97.89% test accuracy with 211 errors
on 203,530 parameters, trained in about 29 seconds.

**The benchmark.** The Conv-Pool-Conv-Pool-Dense stack reached 99.02% test
accuracy with 98 errors on 255,230 parameters, trained in 65.0 seconds under the
same optimizer and the same epoch budget. That is a gap of 1.13 percentage
points, and it more than halves the error count, a 54% reduction, for 25% more
parameters and 2.3 times the training time.

**Why convolution suits this data.** A 28 by 28 image flattened into 784
independent inputs keeps every pixel value and discards every statement about
which pixels are adjacent. The dense layer has to rediscover from data that input
100 and input 128 are vertical neighbours, and it has to learn a stroke detector
separately for every position that stroke might occupy. A convolution is handed
that structure. Each filter sees a local patch, and the same weights slide across
all positions, so one learned edge detector applies everywhere and costs one set
of weights rather than 784. Pooling then discards precise position within each
window, which is why a digit shifted by a pixel or two produces nearly the same
features.

**Did it fix the same errors?** Mostly, and this is the part I found most
interesting. Of my 211 errors the CNN fixed 165, but it introduced 52 of its own,
so only 46 of its 98 mistakes are ones I also made. It is not uniformly better
image by image.

The per-class picture is sharper. My worst class was 8 with 35 errors, which the
CNN cut to 8. But digit 7 came out at exactly 33 errors for both models, with
both getting 995 of them right. The CNN did not improve on 7s at all. It
relocated the mistakes: my 7s scattered into 9, 2 and 1, while the CNN put 22 of
its 33 into class 2 alone, against my 10. A 7 with a flat top and a 2 with a
small loop share a lot of local stroke structure, and pooling away exact position
makes that pair harder to separate rather than easier.

**What I learned.** The gap closes by about half the remaining error, not by an
order of magnitude, and what buys it is an architectural assumption rather than
capacity, since the CNN carries only 25% more parameters. Convolution and pooling
are the right prior for images. Pooling's translation tolerance is also not free:
it costs precision on the one pair where exact position was the distinguishing
feature.
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

**Depth and width.** One hidden layer of 256 units won, at 98.04% validation
accuracy, and it was also the smallest network I tried at 203,530 parameters.
Two through five layers gave 97.66%, 97.68%, 97.94% and 97.58%, so depth never
paid. The comparison I trust is the matched-parameter one, because it removes
capacity as an explanation: `deep_narrow` at 224-96-48-24 holds 203,522 weights,
within eight of the winner, and reached 97.46%. Width took that by 0.58 points.
The trade-off is not accuracy against capacity, since both models have the same
budget. It is that MNIST does not need a feature hierarchy, so extra depth only
lengthens the path a gradient has to travel.

**Gradient flow in the deep sigmoid model.** At initialization the activation
gradient shrank by a factor of about 0.24 per layer, from 8.451e-02 at the last
hidden layer to 6.614e-05 at the first, a total of 1277.8x across five layers.
That factor is close to the 0.25 predicted from Glorot variance and a sigmoid
derivative near zero, and the decay is a straight line on a log plot against
depth. The behavioural consequence is a network that reached only 94.30% against
98.04% for the shallow ReLU model on the same budget, at a validation loss 2.6
times higher.

The thing I did not expect is that the attenuation is gone after one epoch, at
0.9x, and inverted by epoch 10, at 0.1x. I only found this because a first
version of the diagnostic measured at epoch end and reported no attenuation at
all. Had I trusted that, I would have written the opposite conclusion from the
same network.

**Initialization.** Zeros reached 11.24% and never moved, at a validation loss of
2.3012 against `ln(10)` at 2.3026. Every unit in the layer starts identical and
receives an identical gradient, so 256 units behave as one and the model can only
output the class prior. Glorot reached 97.76% and He reached 98.04%. He is the
right pairing for ReLU because its `2/fan_in` scaling compensates for ReLU
zeroing about half its inputs, though at one layer the mismatch has little room
to compound and the margin is only 0.28 points.

**Optimizer, learning rate, batch size.** Adam at 0.001 with batch 128, which
gave 98.04%. RMSprop was within noise at 97.92%. SGD with momentum reached
91.52%, but I will not call it the worse optimizer: its training accuracy
finished at 91.85% with its best epoch last, which is a model still improving
when the budget ended rather than one that converged badly. Comparing optimizers
at one shared learning rate measures the pairing. Learning rate 0.1 peaked at
89.56% in epoch 1 and decayed after, ending with training accuracy below
validation accuracy, which is what thrashing looks like. Batch 512 deserves a
mention: 7.6 seconds against 22.0 for batch 128 and 79.1 for batch 32, for only
0.64 points less accuracy than the winner.

**Regularization.** It fixed nothing, because there was nothing to fix. The
unregularized gap was 1.70 points, and dropout, L2 and both together pushed it to
1.15, 1.43 and 0.79 while pushing accuracy down to 97.98%, 97.96% and 97.86%.
That is bias bought at full price with no variance to sell. Dropout did give the
best validation loss of the four at 0.0702, so it improved calibration without
improving accuracy. I kept the unregularized configuration.

**Before the CNN.** 97.89% test accuracy, 211 errors on 10,000 images, 203,530
parameters, about 29 seconds of training. Validation had predicted 98.06%, and
the 0.17 point drop is within the roughly 0.14 point standard error on a
10,000-image test set.

**The CNN gap.** 99.02% against 97.89%, so 1.13 percentage points, which removes
54% of the remaining errors for 25% more parameters and 2.3 times the training
time. The assumptions that buy it are local receptive fields, one filter shared
across every position, and pooling that tolerates small translations. Flattening
to 784 inputs throws away which pixels are adjacent, so my dense layer had to
learn each stroke detector separately for each place a stroke might appear.

Two details stopped me treating the CNN as simply better. It fixed 165 of my 211
errors but introduced 52 new ones, so only 46 of its 98 mistakes overlap with
mine. And on digit 7 both models made exactly 33 errors: the CNN did not improve
that class at all, it concentrated the failures, putting 22 of them into class 2
where I had put 10. Pooling away exact position is what makes a flat-topped 7 and
a small-looped 2 harder to tell apart, so the mechanism that wins overall costs
precision on the one pair where position was the distinguishing feature.

**Limitations.** Every number here comes from one seed, 8401, and one
55,000/5,000 split. Differences under about half a point, which covers most of
the architecture table and the whole regularization table, are not separable from
run-to-run variation on this evidence, and repeated seeds with error bars would
be needed to claim otherwise. The optimizer comparison shares a single learning
rate and so measures pairings rather than optimizers. The deep sigmoid diagnostic
uses one width and one initializer, so the 0.24 per-layer factor is specific to
Glorot at 128 units and is not a general constant.
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
