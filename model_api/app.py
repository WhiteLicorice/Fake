from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from pickle import load as ml_load

from scipy.sparse import hstack

from contextlib import asynccontextmanager

#   Initialize objects that will live across the lifespan of the app
@asynccontextmanager
async def lifespan(app: FastAPI):
    #   Load machine learning model
    global ml_model
    with open(f"root/models/{model_id}.pkl", "rb") as file:
        ml_model = ml_load(file)
    global tf_idf
    with open(r"root\models\tfidf.pkl", "rb") as file:
        tf_idf = ml_load(file)
    global bag_of_words
    with open(r'root\models\cv.pkl', "rb") as file:
        bag_of_words = ml_load(file)
    yield

#   Declare FastAPI instance
app = FastAPI(lifespan=lifespan)
version = "0.0.0.0.0.0.0.1"
model_id = "LogisticRegression"

#   Configure middleware for application and allow all requests from all sources
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

#   Prototype for valid payloads -> tokens : string
class Payload(BaseModel):
    article: str
    trad: list
    syll: list

@app.get('/')
def health_check():
    return {'health': f'Running version {version} of Fake_ML with model {model_id}'}

@app.post("/predict")
async def predict(payload: Payload):
    #   Log received tokens in the console
    #print (payload.tokens)
    print("Payload received!")
    #   Await model prediction
    prediction = await model_predict(payload)
    #   Log prediction in the console
    print(prediction)
    #   Return prediction as Bool -> True | False
    return prediction

async def model_predict(payload: Payload):
    ngrams = tf_idf.transform([payload.article])
    bow = bag_of_words.transform([payload.article])
    features = hstack([payload.trad, payload.syll, ngrams, bow])
    #   Make predictions on the tokens
    y_pred = ml_model.predict(features)
    if y_pred[0] == 1:
        return False  # Real    
    else:
        return True  # Fake
