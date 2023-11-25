from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from json import load as js_load
from tokenizers import Tokenizer

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import PorterStemmer

import root.tagalog_stemmer as stemmer_tl
import string as string
import httpx

import root.LM as LM
import root.SYLL as SYLL
import root.TRAD as TRAD

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
	global tokenizer
	tokenizer = Tokenizer.from_file(f"root/tokenizer.json")
	global stop_words_tl
	with open("root/stopwords-tl.json", "r") as tl:
		stop_words_tl = set(js_load(tl))
	global stop_words_en
	with open("root/stopwords-en.txt", 'r') as en:
		stop_words_en = set(en.read().splitlines())
	global stemmer_en
	stemmer_en = PorterStemmer()
	yield
	stop_words_en.clear()
	stop_words_tl.clear()

# Declaring our FastAPI instance
app = FastAPI(lifespan=lifespan)
version = "0.0.0.0.0.0.0.1"
model_id = "svm" 
#model_api = "http://127.0.0.1:6996"             #   Localhost endpoint
model_api = "https://fake-ph-ml.cyclic.app"      #   Cyclic enpoint

app.add_middleware(
	CORSMiddleware,
	allow_origins=['*'],
	allow_credentials=True,
	allow_methods=['*'],
	allow_headers=['*'],
)

class News(BaseModel):
	news_body: str

@app.get('/')
async def health_check():
	feature_extraction_functions = [
	('word_count', TRAD.word_count_per_doc),
	('sentence_count', TRAD.sentence_count_per_doc),
	('polyslyll_count', TRAD.polysyll_count_per_doc),
	('ave_word_length', TRAD.ave_word_length),
	('ave_syllable_count_of_word', TRAD.ave_phrase_count_per_doc),
	('consonant_cluster', SYLL.get_consonant_cluster),
	('v_density', SYLL.get_v),
	('cv_density', SYLL.get_cv),
	('vc_density', SYLL.get_vc),
	('cvc_density', SYLL.get_cvc),
	('vcc_density', SYLL.get_vcc),
	('cvcc_density', SYLL.get_cvcc),
	('ccvcc_density', SYLL.get_ccvcc),
	('ccvccc_density', SYLL.get_ccvccc),
	]
	text3 = """MANILA, Philippines — Umabot na sa apat na Pilipino ang namamatay sa pagpapatuloy ng bakbakan sa pagitan ng mga militanteng Palestino at Israel, ayon sa Department of Foreign Affairs.
Ito ang kinumpirma ni Foreign Affairs Undersecretary Enrique Manalo sa isang paskil sa X (dating Twitter) ngayong Huwebes nang umaga.
"I regret to inform the nation that we have received confirmation from the Israeli government of another Filipino casualty in Israel," ani Manalo kanina.
"Out of respect for the wishes of the family, we shall be withholding details on the identity of the victim. But we have assured the family of the Government’s full support and assistance."
1/2 I regret to inform the nation that we have received confirmation from the Israeli government of another Filipino casualty in Israel
Kinumpirma rin ni Manalo na isang babaeng caregiver ang ikaapat na nasawing Pinay.
"We commiserate with family... Isa siya sa mga missing," pagbabahagi ni Manalo sa panayam ng Philstar.com.
Bago ito, isang babae, caregiver at manggagawang nagtratrabaho sa kibbutz na inatake ng Hamas ang nasawi sa hanay ng mga Pilipino.
Miyerkules lang nang makauwi ng Pilipinas ang 16 na overseas Filipino workers at isang buwang sanggol galing ng Israel, ang pinakaunang batch sa mga Pinoy na humingi ng repatriation sa gitna ng gulo.
Ang lahat ng ito ay kaugnay ng "Operation al-Aqsa Storm" ng ilang Palestino, na produkto diumano ng pag-atakeng sinimulan ng Jewish settlers at bakbakan sa Jenin at Al-Aqsa mosque na ikinamatay ng higit 200 Palestino. Bukod pa ito sa deka-dekadang illegal Israeli occupation.
Pumalo na sa 1,400 katao ang namamatay sa Israel matapos ang opensiba ng mga Palestino mula Gaza.
Una nang nagdeklara ng giyera si Israeli Prime Minister Benjamin Netanyahu kaugnay ng atake ng mga nabanggit, dahilan para gumanti ng tuloy-tuloy na air strikes, bagay na pumatay na sa mas maraming Palestino sa ngayon.
Kahapon lang nang umabot sa 2,750 ang namatay sa Palestine, kabilang na ang nasa 471 kataong napatay sa pambobomba ng Israel sa Ahli Arab hospital sa Gaza Strip, bagay na nakatatanggap ng malawakang batikos.
Morning news briefings from Philstar.com
Philstar.com is one of the most vibrant, opinionated, discerning communities of readers on cyberspace. With your meaningful insights, help shape the stories that can shape the country. Sign up now!"""
	
	final = {}
	for column, func in feature_extraction_functions:
		final[column] = func(text3)
	
	print(final)
	#await preprocess_text(text3)
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