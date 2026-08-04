# FaKe API

This directory contains the API deployed at <https://fph-ml.onrender.com>. The Netlify dashboard and Tampermonkey userscript send Filipino news text to `POST /check-news`.

## Run locally

Use Python 3.11.15, which is pinned in `.python-version`.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 5000
```

`GET /health` reports readiness. `POST /check-news` accepts JSON in the form `{"news_body":"..."}` and returns the compatibility field `status` together with a readable label, model name, and API version.

## Render configuration

The Render web service uses `server` as its root directory. Its build command is `pip install -r requirements.txt`, its start command is `uvicorn app:app --host 0.0.0.0 --port $PORT`, and its health-check path is `/health`.

The serialized deployment model was produced with scikit-learn 1.3.2. PYSEC-2024-110 concerns tokens stored while fitting a `TfidfVectorizer`. This API does not fit or serialize user submissions, and the model's training corpora are public. A scikit-learn upgrade requires a validated serialization migration or model retraining because cross-version pickle loading is unsupported.
