"""Check that GradientNormLogger catches attenuation at initialization.

Needs TensorFlow, so run it in Colab or any environment that has it:

    python -m training.test_gradient_logger

Two properties are asserted, and the second one is why the callback records
epoch 0 at all. At initialization a Glorot sigmoid stack attenuates the
backpropagated signal by roughly 0.25 per layer, so the gradient reaching the
first hidden layer is far smaller than the one at the last. After a single epoch
the network has learned weights that push signal through and the per-layer norms
flatten out. An earlier version measured only at epoch end and so recorded no
attenuation at all, which left task 3 with no evidence.
"""

from __future__ import annotations

import numpy as np

from training.experiment_utils import (
    ExperimentConfig,
    GradientNormLogger,
    build_mlp,
    set_reproducibility,
)


def _activation_norms(frame, epoch: int) -> np.ndarray:
    rows = frame[(frame["epoch"] == epoch) & frame["layer"].str.startswith("hidden_")]
    return rows.sort_values("layer")["activation_grad_norm"].to_numpy()


def test_attenuation_is_visible_at_initialization() -> None:
    set_reproducibility()
    rng = np.random.default_rng(0)
    x = rng.random((256, 784)).astype("float32")
    y = rng.integers(0, 10, size=256)

    config = ExperimentConfig(
        name="gradient_logger_check",
        phase="test",
        hidden_units=(128,) * 6,
        activation="sigmoid",
        initializer="glorot_uniform",
        epochs=1,
    )
    model = build_mlp(config)
    logger = GradientNormLogger(x, y)
    model.fit(x, y, epochs=1, batch_size=64, verbose=0, callbacks=[logger])

    frame = logger.to_frame()
    initial = _activation_norms(frame, 0)
    trained = _activation_norms(frame, 1)

    assert len(initial) == 6, f"expected 6 hidden layers at epoch 0, got {len(initial)}"
    assert np.isfinite(initial).all(), f"non-finite gradients at epoch 0: {initial}"

    # hidden_1 is nearest the input, hidden_6 nearest the loss.
    initial_ratio = initial[-1] / initial[0]
    assert initial_ratio > 10, (
        "a six-layer sigmoid stack should attenuate by more than 10x at "
        f"initialization; got {initial_ratio:.1f}x from {initial}"
    )

    # Training flattens it, which is why epoch 0 is the measurement that matters.
    trained_ratio = trained[-1] / trained[0]
    assert trained_ratio < initial_ratio, (
        "attenuation should shrink after an epoch of training; got "
        f"{initial_ratio:.1f}x at epoch 0 and {trained_ratio:.1f}x at epoch 1"
    )

    # Recorded so the analysis can divide by it instead of assuming a width.
    weights = frame[frame["layer"] == "hidden_2"]["weights"].iloc[0]
    assert weights == 128 * 128, f"expected 16384 weights in hidden_2, got {weights}"


if __name__ == "__main__":
    test_attenuation_is_visible_at_initialization()
    print("gradient logger check passed")
