# Hand-in checklist

Submit the executed notebook, not a clean notebook with empty outputs.

Ticked items were verified against the committed run. The run is
`notebooks/mnist_neural_networks.ipynb`, executed on a Colab T4 from a fresh
runtime with `RUN_FULL = True`, and its artifacts are in `artifacts/`.

## Required submission

- [x] `mnist_neural_networks.ipynb` opens without errors. Execution counts run 1
      to 16 with no gaps, so the cells ran top to bottom in one pass.
- [x] The notebook records the package versions and runtime: Python 3.13.15,
      TensorFlow 2.20.0, NumPy 2.1.3, pandas 2.2.3, scikit-learn 1.6.1, on a GPU
      device. The seed appears in the title cell, in `SEED = 8401`, and in
      `artifacts/model_metadata.json`.
- [x] MNIST shapes show 60,000 training and 10,000 test images, asserted in code.
- [x] Sample digits and class-distribution plots are visible.
- [x] The split is 55,000 train, 5,000 validation, and 10,000 untouched test.
- [x] Fully connected inputs have 784 values; CNN inputs have shape 28 by 28 by 1.
- [x] Each discussed model has a visible `model.summary()`. There are 28 of them.
- [x] The architecture table includes 1 to 5 hidden layers.
- [x] Shallow-wide and deep-narrow models have roughly comparable parameters:
      203,530 against 203,522, a difference of eight weights.
- [x] ReLU and sigmoid or tanh use the same non-activation settings.
- [x] The deep sigmoid run includes per-layer gradient norms and a loss curve.
      Gradients are recorded before the first weight update as well as at every
      epoch end, because the attenuation is trained away within one epoch.
- [x] Initialization runs include He, Glorot and all-zero weights.
- [x] Optimizer runs include Adam, SGD with momentum, and RMSprop.
- [x] Learning-rate and batch-size comparisons are present.
- [x] Regularization results include dropout and L2 and discuss overfitting.
- [x] The final fully connected model is chosen before any CNN result is used.
      The CNN appears only in section 8, after the final model is trained, saved,
      reloaded and tested.
- [x] The held-out test set is evaluated only for the two final models.
- [x] Final evaluation includes accuracy, a confusion matrix, and errors.
- [x] The saved fully connected model is reloaded and its predictions are
      checked against the original.
- [x] The CNN uses a comparable epoch budget: the same optimizer, the same
      learning rate, and the same 18-epoch cap.
- [x] The final table compares accuracy, parameters, time, and error patterns.
- [x] Every major experiment answers: what changed, what happened, why, and what
      was learned.
- [x] The conclusion uses the actual run numbers and does not claim unsupported
      causes. Differences smaller than about half a point are named as
      inseparable from run-to-run noise on a single seed.

## Companion deliverables

- [x] `artifacts/final_fc.keras`
- [x] `artifacts/model_metadata.json`
- [x] `artifacts/experiment_results.csv`, 28 rows across 10 phases
- [x] Public GitHub repository: https://github.com/MwangiNelson/mnist-network-lab
- [x] Public web application: https://mnist-network-lab.vercel.app
- [x] Public API health: https://mnist-api.astralyngroup.com/health

## Final integrity check

- [x] No `[WRITE AFTER RUN]` marker remains.
- [x] No scratch cells remain. The notebook is 35 cells: 16 code, 19 markdown.
- [x] Every number quoted in the analysis was checked against the artifact CSVs
      and JSON. There are no mismatches.
- [x] No test result influenced model selection. Each stage selects on
      validation accuracy and passes its winner to the next.
- [x] The notebook ran from a fresh Colab runtime.

## Known limitations, stated in the notebook

- One seed, 8401, and one train-validation split, so no run-to-run error bars.
- The optimizer comparison shares one learning rate, so it measures optimizer
  and learning-rate pairings rather than optimizers alone.
- The architecture table's training-time column charges GPU warm-up and graph
  compilation to whichever experiment runs first. The same configuration
  retrained later in the notebook takes about 22 seconds rather than 41.5.
- The restore check covers 32 test images, not all 10,000.
