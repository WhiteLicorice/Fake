from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from json import load as js_load
from tokenizers import Tokenizer
#from Stemmer import Stemmer as PorterStemmer

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.data.path.append('/nltk_')

import root.tagalog_stemmer as stemmer_tl
import string as string
import httpx

# Declaring our FastAPI instance
app = FastAPI()
version = "0.0.0.0.0.0.0.1"
model_id = "svm" 
#model_api = "http://127.0.0.1:6996"             #   Localhost endpoint
model_api = "https://fake-ph-ml.cyclic.app"      #   Cyclic enpoint

# Allow all origins to make CORS request
origins = [
    "*",
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

##  TODO: PREDOWNLOAD NLTK STUFF
#   Download "punkt" from nltk
@app.on_event("startup")
def nltk_download_punkt():
    try:
        #nltk.data.find("tokenizers/punkt")
        pass
    except LookupError:
        nltk.download("punkt")
        
#   Download "stopwords" from nltk    
@app.on_event("startup")
def nltk_download_stopwords_en():
    try:
        #nltk.data.find("corpora/stopwords")
        pass
    except LookupError:
        nltk.download("stopwords")

@app.on_event("startup")
async def load_processors():
    global tokenizer
    tokenizer = Tokenizer.from_file(f"root/tokenizer.json")
    global stop_words_tl
    with open("root/stopwords-tl.json", "r") as tl:
        stop_words_tl = set(js_load(tl))
    global stop_words_en
    stop_words_en = set(stopwords.words('english'))
    global stemmer_en
    stemmer_en = PorterStemmer()


@app.get('/')
def health_check():
	return {'health': f'Running version {version} of Fake_API with model {model_id}'}

@app.post("/check-news")
async def check_news(news: News):
    print(news.news_body)
    prepocessed_text = await preprocess_text(news.news_body)
    is_fake_news = await call_model(prepocessed_text)
    print(is_fake_news)
    return {"status": is_fake_news}

async def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    words = tokenizer.encode(text)
    words = [word for word in words.tokens if word not in stop_words_tl and stop_words_en]
    words = [stemmer_en.stem(stemmer_tl.stemmer(word)) for word in words]
    text = ' '.join(words)
    return text

async def call_model(tokens):
    async with httpx.AsyncClient() as async_client:
        result = await async_client.post(f"{model_api}/predict", json={'tokens': tokens})
    return result.text