# MNIST network lab

This repository contains the DSA 8401 neural-network assignment and a public
digit classifier built around the final student-designed fully connected model.

The notebook covers the required architecture, activation, gradient,
initialization, optimizer, learning-rate, batch-size, and regularization
experiments. It selects a final fully connected model before introducing the
CNN benchmark. The web application sends a normalized 28 by 28 drawing to a
FastAPI service and displays the ten class probabilities.

## Repository contents

| Path | Purpose |
| --- | --- |
| `notebooks/mnist_neural_networks.ipynb` | Colab hand-in notebook |
| `training/experiment_utils.py` | Reusable Keras experiment runner |
| `api/` | Public FastAPI inference service |
| `web/` | React drawing interface for Vercel |
| `deploy/` | VPS Docker Compose and Caddy example |
| `HAND_IN_CHECKLIST.md` | Submission and marking checklist |
| `COLAB_RUNBOOK.md` | Exact full-run and artifact download steps |

## Run the notebook

1. Open `notebooks/mnist_neural_networks.ipynb` in Google Colab.
2. Select a GPU runtime if one is available.
3. Run all cells once with `RUN_FULL = False` to check the pipeline.
4. Set `RUN_FULL = True`, restart the runtime, and run all cells for the hand-in.
5. Download the executed notebook and the generated `artifacts` directory.

The full run is the source of truth. The repository does not contain invented
metrics or a borrowed model.

## Run the application locally

Place the notebook's `final_fc.keras` and `model_metadata.json` files in
`artifacts/`, then start the API:

```bash
docker compose -f deploy/docker-compose.yml up --build
```

In another terminal, start the web client:

```bash
cd web
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:5173`. The API listens on `http://localhost:8010`.

## Reproducibility

The notebook fixes Python, NumPy, and TensorFlow seeds. It prints library and
hardware versions near the top and exports all experiment configurations to
CSV. Exact GPU training times can vary between Colab sessions.

## Public deployment

The API has no authentication, as required for this demo. The deployment limits
request size, validates all 784 pixels, and does not store drawings. Read
`deploy/README.md` before exposing it to the internet.
