# Hand-in checklist

Submit the executed notebook, not a clean notebook with empty outputs.

## Required submission

- [ ] `mnist_neural_networks.ipynb` opens without errors.
- [ ] The notebook records the random seed, package versions, and runtime.
- [ ] MNIST shapes show 60,000 training and 10,000 test images.
- [ ] Sample digits and class-distribution plots are visible.
- [ ] The split is 55,000 train, 5,000 validation, and 10,000 untouched test.
- [ ] Fully connected inputs have 784 values; CNN inputs have shape 28 by 28 by 1.
- [ ] Each discussed model has a visible `model.summary()`.
- [ ] The architecture table includes 1 to 5 hidden layers.
- [ ] Shallow-wide and deep-narrow models have roughly comparable parameters.
- [ ] ReLU and sigmoid or tanh use the same non-activation settings.
- [ ] The deep sigmoid run includes per-layer gradient norms and a loss curve.
- [ ] Initialization runs include He or Glorot and all-zero weights.
- [ ] Optimizer runs include Adam, SGD with momentum, and RMSprop.
- [ ] Learning-rate and batch-size comparisons are present.
- [ ] Regularization results include dropout or L2 and discuss overfitting.
- [ ] The final fully connected model is chosen before any CNN result is used.
- [ ] The held-out test set is evaluated only for the two final models.
- [ ] Final evaluation includes accuracy, confusion matrix, and errors.
- [ ] The saved fully connected model is reloaded and its predictions are checked.
- [ ] The CNN uses a comparable epoch budget.
- [ ] The final table compares accuracy, parameters, time, and error patterns.
- [ ] Every major experiment answers: what changed, what happened, why, and what was learned.
- [ ] The conclusion uses the actual run numbers and does not claim unsupported causes.

## Companion deliverables

- [ ] `final_fc.keras`
- [ ] `model_metadata.json`
- [ ] `experiment_results.csv`
- [ ] Public GitHub repository URL
- [ ] Public web application URL
- [ ] Public API health URL

## Final integrity check

- [ ] Replace every `[WRITE AFTER RUN]` marker with personal analysis.
- [ ] Remove failed scratch cells that do not support the discussion.
- [ ] Confirm all table numbers match visible cell output.
- [ ] Confirm no test result influenced model selection.
- [ ] Run the notebook from a fresh Colab runtime before submission.
