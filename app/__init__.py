from flask import Flask
import spacy

app = Flask(__name__)

nlp = spacy.load('en_core_web_sm')

VECTOR_SIZE = 150

from app import views
from app import api
