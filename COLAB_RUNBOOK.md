# Colab runbook

Use this after the repository is online.

## First pass

1. Open the notebook from GitHub in Colab.
2. Choose **Runtime > Change runtime type** and select a GPU.
3. Keep `RUN_FULL = False`.
4. Run all cells. Fix any environment error before spending time on a full run.

The smoke run uses fewer epochs and a subset. Its numbers are not submission
results.

## Graded run

1. Set `RUN_FULL = True`.
2. Choose **Runtime > Disconnect and delete runtime**.
3. Reconnect and run all cells.
4. Keep the browser tab open until the final artifact cell completes.
5. Read every generated table and plot.
6. Replace each `[WRITE AFTER RUN]` prompt in the analysis cells using the
   adjacent evidence.

Do not tune against the test set. The notebook only exposes test results after
the final fully connected configuration has been selected from validation data.

## Save artifacts

The final cell creates `mnist_assignment_artifacts.zip`. Download that archive
and the executed notebook through **File > Download > Download .ipynb**.

Copy these files from the archive into the repository's `artifacts/` directory:

- `final_fc.keras`
- `model_metadata.json`
- `experiment_results.csv`

Commit the executed notebook and small tables. The `.gitignore` excludes model
weights by default because GitHub rejects large files. The VPS receives the
model through `scp` during deployment.
