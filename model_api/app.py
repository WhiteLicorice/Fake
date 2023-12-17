from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from pickle import load as ml_load

from contextlib import asynccontextmanager

from filipino_transformers import TRADExtractor, SYLLExtractor
from bpe_tokenizer import BPETokenizer
import nltk

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    with open(f"root/models/{model_id}.pkl", "rb") as file:
        ml_model = ml_load(file)
    yield
    ml_model.clear()

# Declaring our FastAPI instance
app = FastAPI(lifespan=lifespan)
version = "0.0.0.0.0.0.0.1"
model_id = "LogisticRegression"

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class Payload(BaseModel):
    tokens: str

@app.get('/')
def health_check():
    return {'health': f'Running version {version} of Fake_ML with model {model_id}'}

@app.post("/predict")
async def predict(payload: Payload):
    print (payload.tokens)
    prediction = await model_predict(payload.tokens)
    print(prediction)
    return prediction

async def model_predict(tokens):
    if not isinstance(tokens, list):
        tokens = [tokens]
    # Use the global bpe_tokenizer function
    y_pred = ml_model.predict(tokens)
    if y_pred[0] == 1:
        return False  # Real    
    else:
        return True  # Fake
