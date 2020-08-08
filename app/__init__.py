from flask import Flask
import spacy
import en_core_web_sm
from flask_compress import Compress



app = Flask(__name__)
Compress(app)


nlp = en_core_web_sm.load()
model = None
VECTOR_SIZE = 150


from app import views
from app import api
