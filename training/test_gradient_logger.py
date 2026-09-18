"""Check that GradientNormLogger measures decay toward the input.

Needs TensorFlow, so run it in Colab or any environment that has it:

    python -m training.test_gradient_logger

The point of the check is the direction. A sigmoid stack attenuates the
backpropagated signal at every layer, so the gradient with respect to the
activations must be smallest at the layer nearest the input. An earlier version
of this logger recorded only kernel gradients, which mix in fan-in and
activation scale and showed no such ordering.
"""

from __future__ import annotations

import numpy as np

from training.experiment_utils import (
    ExperimentConfig,
    GradientNormLogger,
    build_mlp,
    set_reproducibility,
)


def test_activation_gradients_shrink_toward_the_input() -> None:
    set_reproducibility()
    rng = np.random.default_rng(0)
    x = rng.random((64, 784)).astype("float32")
    y = rng.integers(0, 10, size=64)

    config = ExperimentConfig(
        name="gradient_logger_check",
        phase="test",
        hidden_units=(32,) * 6,
        activation="sigmoid",
        initializer="glorot_uniform",
        epochs=1,
    )
    model = build_mlp(config)
    logger = GradientNormLogger(x, y)
    model.fit(x, y, epochs=1, batch_size=32, verbose=0, callbacks=[logger])

    frame = logger.to_frame()
    hidden = frame[frame["layer"].str.startswith("hidden_")].sort_values("layer")
    norms = hidden["activation_grad_norm"].to_numpy()

    assert len(norms) == 6, f"expected 6 hidden layers, got {len(norms)}"
    assert np.isfinite(norms).all(), f"non-finite activation gradients: {norms}"
    assert norms[0] < norms[-1], (
        "activation gradient at the input side should be the smaller one; "
        f"got hidden_1={norms[0]:.3e} and hidden_6={norms[-1]:.3e}"
    )
    assert norms[-1] / norms[0] > 2, (
        "sigmoid stack should attenuate the signal by more than 2x across six "
        f"layers; got {norms[-1] / norms[0]:.2f}x from {norms}"
    )

    # Recorded so the analysis can divide by it instead of assuming a width.
    weights = frame[frame["layer"] == "hidden_2"]["weights"].iloc[0]
    assert weights == 32 * 32, f"expected 1024 weights in hidden_2, got {weights}"


if __name__ == "__main__":
    test_activation_gradients_shrink_toward_the_input()
    print("gradient logger check passed")
