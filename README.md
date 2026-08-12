# MLOps Intent Classifier with KServe

A small end-to-end example of serving a text intent-classification model through a REST API and preparing the model for Kubernetes-based inference with **KServe**.

The project is intentionally lightweight, but it demonstrates an important MLOps transition: from a locally trained scikit-learn model to a deployable inference service.

## Project Overview

The model classifies short customer-style messages into four intent categories:

* `greeting`
* `question`
* `complaint`
* `praise`

The training pipeline uses a scikit-learn `Pipeline` combining `CountVectorizer` and `MultinomialNB`, then serializes the trained pipeline with `joblib`.

The model is loaded by a small inference layer and exposed through a Flask API with health and prediction endpoints.

## Architecture

```text
Training data
     │
     ▼
CountVectorizer + MultinomialNB
     │
     ▼
scikit-learn Pipeline
     │
     ▼
joblib artifact
(intent_model.pkl)
     │
     ├──────────────► Flask REST API
     │                    ├── /health
     │                    └── /predict
     │
     └──────────────► KServe / Kubernetes
                           └── InferenceService
```

## Tech Stack

| Component                   | Technology      |
| --------------------------- | --------------- |
| Language                    | Python          |
| ML framework                | scikit-learn    |
| Model persistence           | joblib          |
| API                         | Flask           |
| Production WSGI entry point | Gunicorn / WSGI |
| Model serving               | KServe          |
| Container orchestration     | Kubernetes      |
| Package management          | pip             |

## Repository Structure

```text
.
├── app.py
├── wsgi.py
├── requirements.txt
├── KServe-implementation.md
└── model/
    ├── train.py
    ├── intent_model.py
    └── artifacts/
        └── intent_model.pkl
```

## 1. Run Locally

### Create the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train the model

```bash
python model/train.py
```

The training script creates:

```text
model/artifacts/intent_model.pkl
```

### Start the API

```bash
python app.py
```

The service runs on:

```text
http://127.0.0.1:6000
```

### Health check

```bash
curl http://127.0.0.1:6000/health
```

Expected response:

```json
{"status":"ok"}
```

### Prediction

```bash
curl -X POST http://127.0.0.1:6000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I want to cancel my subscription"}'
```

Expected response:

```json
{"intent":"complaint"}
```

The current inference implementation calculates class probabilities internally but returns only the predicted intent in the API response.

## 2. Run with Gunicorn

The repository includes a WSGI entry point:

```python
from app import app
application = app
```

Run:

```bash
gunicorn wsgi:application --bind 0.0.0.0:6000
```

## 3. Deploy with KServe

The repository also includes a Kubernetes/KServe implementation guide in [`KServe-implementation.md`](KServe-implementation.md).

The deployment uses a KServe `InferenceService` with the scikit-learn model server and exposes the model through KServe's inference protocol.

### KServe prerequisites

The documented setup uses:

* Kubernetes
* cert-manager
* KServe v0.16.0
* Helm

Install the KServe CRDs and controller following the commands in [`KServe-implementation.md`](KServe-implementation.md).

### Important deployment note

The current KServe guide still contains a sample `storageUri` pointing to the public KServe sklearn example model.

That is useful for validating the KServe installation, but it is **not the same artifact trained by this repository**.

For a production-ready version of this project, the next step is to publish `intent_model.pkl` to object storage accessible by Kubernetes, such as Google Cloud Storage or Amazon S3, and update the `InferenceService` to reference that artifact.

## MLOps Concepts Demonstrated

This repository is intentionally small, but it covers several core MLOps building blocks:

1. **Model training** — reproducible model creation from training data.
2. **Model serialization** — storing the fitted pipeline as a versionable artifact.
3. **Inference abstraction** — separating model loading/prediction from the API layer.
4. **Health checks** — exposing a service endpoint for operational checks.
5. **API-based serving** — serving predictions through HTTP.
6. **Kubernetes inference** — preparing the model for KServe-based deployment.
7. **Production WSGI serving** — providing a Gunicorn-compatible entry point.

## Limitations and Future Improvements

This is a demonstration project rather than a production ML system.

Potential improvements include:

* Replace the toy training dataset with a larger, representative dataset.
* Add train/validation/test splits and model evaluation metrics.
* Add automated tests for training and inference.
* Version the model artifact and training data.
* Store the model in object storage for KServe deployment.
* Containerize the application and inference workflow.
* Add CI/CD with automated testing and deployment.
* Add experiment tracking and model registry support, for example with MLflow.
* Add observability for latency, errors and prediction distributions.
* Add data drift and prediction drift monitoring.
* Return calibrated class probabilities and confidence thresholds.

## Why KServe?

KServe provides a Kubernetes-native abstraction for deploying and operating machine learning inference services.

In this project it represents the step from a simple Python/Flask service toward a standardized model-serving platform.

## Related Documentation

* [`KServe-implementation.md`](KServe-implementation.md) — Kubernetes and KServe deployment notes.

## Author

**Anderson Cruz**
Data Scientist | Machine Learning | MLOps

GitHub: [@AnderCruz](https://github.com/AnderCruz)

---

*This repository is part of an MLOps portfolio focused on model deployment, serving, monitoring and production-oriented machine learning workflows.*
