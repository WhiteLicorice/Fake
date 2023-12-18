from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from json import load as js_load

import string as string
import httpx

import root.LM as LM
import root.SYLL as SYLL
import root.TRAD as TRAD

from contextlib import asynccontextmanager

#   Initialize objects that will live across the lifespan of the app
@asynccontextmanager
async def lifespan(app: FastAPI):
	yield

# 	Declare FastAPI instance
app = FastAPI(lifespan=lifespan)
version = "0.0.0.0.0.0.0.1"
model_id = "svm" 
model_api = "http://127.0.0.1:6996"             	#   Localhost endpoint
#model_api = "https://fake-ph-ml.cyclic.app"      	#   Cyclic endpoint

#	Configure middleware to allow all requests from all sources
app.add_middleware(
	CORSMiddleware,
	allow_origins=['*'],
	allow_credentials=True,
	allow_methods=['*'],
	allow_headers=['*'],
)

#	Define prototype for valid news articles -> news_body : string
class News(BaseModel):
	news_body: str

@app.get('/')
async def health_check():
	return {'health': f'Running version {version} of Fake_API with model {model_id}'}

#	Main endpoint for making requests to machine learning microservice
@app.post("/check-news")
async def check_news(news: News):
	#	Log received news article in the console
	print(news.news_body)
	#	Await call to machine learning model
	is_fake_news = await call_model(news.news_body)
	#	Log returned bool -> True | False as string
	print(is_fake_news)
	#	Return json containing prediction of machine learning model
	return {"status": is_fake_news}

#	Function for making asynchronous calls to machine learning model microservice
async def call_model(tokens):
	async with httpx.AsyncClient() as async_client:
		result = await async_client.post(f"{model_api}/predict", json={'tokens': tokens})
	return result.text