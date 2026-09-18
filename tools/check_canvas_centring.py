"""Compare the app's old and new canvas preprocessing on real MNIST digits.

Needs TensorFlow. Run it in the API container, where the model already lives:

    docker cp tools/check_canvas_centring.py deploy-api-1:/tmp/
    docker exec deploy-api-1 python /tmp/check_canvas_centring.py

or anywhere else with MODEL_PATH pointing at final_fc.keras.

Each test digit is upscaled to the 280 x 280 drawing canvas, then pushed through
a Python replica of web/src/App.tsx preparePixels: threshold at 20 to find the
stroke bounds, fit the longest side to 20 pixels, and place it on a 28 x 28
black field. The only difference between the two pipelines is placement:

  old  centres the bounding box
  new  puts the centre of mass at (13.5, 13.5), which is what MNIST itself does

This is a replica, not the shipped JavaScript. Canvas smoothing is approximated
with bilinear resizing and placement is rounded to whole pixels, so absolute
numbers will not match the browser exactly. The comparison between the two
placements is what it measures.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

N = 1000

model = keras.models.load_model(os.getenv("MODEL_PATH", "/models/final_fc.keras"))
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
x_test, y_test = x_test[:N], y_test[:N]


def prepare(digit28: np.ndarray, centre_of_mass: bool) -> np.ndarray:
    canvas = tf.image.resize(
        digit28[..., None].astype("float32"), (280, 280), method="bilinear"
    ).numpy()[..., 0]
    ys, xs = np.nonzero(canvas > 20)
    crop = canvas[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    height, width = crop.shape
    scale = 20 / max(width, height)
    draw_w = max(1, round(width * scale))
    draw_h = max(1, round(height * scale))
    small = tf.image.resize(
        crop[..., None], (draw_h, draw_w), method="bilinear", antialias=True
    ).numpy()[..., 0]

    if centre_of_mass:
        mass = small.sum()
        cy = (small.sum(axis=1) * np.arange(draw_h)).sum() / mass
        cx = (small.sum(axis=0) * np.arange(draw_w)).sum() / mass
        off_x, off_y = round(13.5 - cx), round(13.5 - cy)
    else:
        off_x, off_y = (28 - draw_w) // 2, (28 - draw_h) // 2

    off_x = min(max(off_x, 0), 28 - draw_w)
    off_y = min(max(off_y, 0), 28 - draw_h)
    field = np.zeros((28, 28), dtype="float32")
    field[off_y : off_y + draw_h, off_x : off_x + draw_w] = small
    return np.clip(field / 255.0, 0.0, 1.0).reshape(784)


raw = (x_test.reshape(-1, 784) / 255.0).astype("float32")
old = np.stack([prepare(d, centre_of_mass=False) for d in x_test])
new = np.stack([prepare(d, centre_of_mass=True) for d in x_test])

for label, batch in (("raw MNIST, no canvas", raw), ("old: bounding box", old), ("new: centre of mass", new)):
    probs = model.predict(batch, verbose=0)
    pred = probs.argmax(axis=1)
    acc = (pred == y_test).mean()
    conf = probs[np.arange(N), y_test].mean()
    sevens = y_test == 7
    acc7 = (pred[sevens] == 7).mean()
    print(
        f"{label:<22} accuracy {acc * 100:6.2f}%   "
        f"mean prob on true class {conf * 100:6.2f}%   digit 7 accuracy {acc7 * 100:6.2f}%"
    )

p_old = model.predict(old[:1], verbose=0)[0]
p_new = model.predict(new[:1], verbose=0)[0]
print()
print(f"test image 0 (a 7): old reads {p_old.argmax()} at {p_old.max() * 100:.1f}%, "
      f"new reads {p_new.argmax()} at {p_new.max() * 100:.1f}%")
