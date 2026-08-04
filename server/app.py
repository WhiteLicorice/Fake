import logging
from contextlib import asynccontextmanager
from pathlib import Path
from pickle import load as ml_load
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


LOGGER = logging.getLogger("fake_api")
MODEL_ID = "LogisticRegression"
VERSION = "1.0.0"
MODEL_PATH = Path(__file__).resolve().parent / "root" / "models" / f"{MODEL_ID}.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the deployment model once before the service accepts requests."""
    with MODEL_PATH.open("rb") as model_file:
        app.state.ml_model = ml_load(model_file)
    LOGGER.info("Loaded model %s from %s", MODEL_ID, MODEL_PATH)
    yield
    app.state.ml_model = None


app = FastAPI(title="FaKe API", version=VERSION, lifespan=lifespan)

# The userscript can run on arbitrary article origins, so the API must accept
# cross-origin requests. Credentials are not used or accepted.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


class News(BaseModel):
    news_body: str = Field(min_length=20, max_length=1_000_000)


class PredictionResponse(BaseModel):
    # `status` is retained for compatibility with the dashboard and userscript.
    status: bool
    label: Literal["Fake", "Real"]
    model: str
    version: str


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "health": "ready",
        "model": MODEL_ID,
        "version": VERSION,
    }


@app.post("/check-news", response_model=PredictionResponse)
def check_news(news: News):
    LOGGER.info("Classifying article with %d characters", len(news.news_body))

    try:
        prediction = app.state.ml_model.predict([news.news_body])
    except Exception as error:
        LOGGER.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from error

    is_fake = bool(prediction[0] == 0)
    return PredictionResponse(
        status=is_fake,
        label="Fake" if is_fake else "Real",
        model=MODEL_ID,
        version=VERSION,
    )
