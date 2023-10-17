from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import sklearn
import random
import numpy as np
import nltk
import tokenizers
import scipy
import secrets

# Declaring our FastAPI instance
app = FastAPI()
version = "0.0.0.0.0.0.0.1"

# Replace this list with the actual origins you want to allow
origins = [
    "https://www.philstar.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class News(BaseModel):
	news_body: str

@app.get('/')
def health_check():
	return {'health': f'Running version {version} of Fake_API'}

@app.post("/check-news")
def check_news(news: News):
    is_fake_news = str(bool(random.randint(0, 1)))   ##  Scaffold: Insert pipe(ML_model) here
    print(news.news_body)
    print(is_fake_news)
    return {"status": is_fake_news}