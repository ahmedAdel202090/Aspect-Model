import requests
import numpy as np
from random import seed
from random import randint
from app import VECTOR_SIZE
seed(1)

def seq_to_vec(text):
    body = {"text": text}
    url = "https://test-wordtovec.azurewebsites.net/doc2vec"
    response = requests.post(url, body)
    if response.text[0] != '{':
        return np.zeros((1,VECTOR_SIZE),dtype='float32')    
    vector = eval(response.text)
    return np.array(vector['doc_vec'])

def word2vec(word):
    url = "https://test-wordtovec.azurewebsites.net/word2vec/{0}".format(word)
    response = requests.get(url)
    if response.text[0] != '{':
        return np.zeros((1,VECTOR_SIZE),dtype='float32')
    vector = eval(response.text)
    return np.array(vector['vector'])

def get_sentiment(text):
    return randint(0,4)

