from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from pickle import load as ml_load

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan():
	global vectorizer, ml_model
	with open(f"root/{model_id}.pkl", "rb") as model_in:       
		vectorizer, ml_model = ml_load(model_in)
	yield
	vectorizer.clear()
	ml_model.clear()
	
# Declaring our FastAPI instance
app = FastAPI(lifespan=lifespan)
version = "0.0.0.0.0.0.0.1"
model_id = "svm"

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
	prediction = await model_predict(payload.tokens)
	print(prediction)
	return prediction

async def model_predict(tokens):
	X = vectorizer.transform([tokens])
	y_pred = ml_model.predict(X)
	if y_pred[0] == 1:
		return False	#	Real
	else:
		return True		#	Fake