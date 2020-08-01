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

def get_sentiment(self,text):
    return randint(0,4)
# class Sentiment(object):
#     def __init__(self):
#         self.sentiment_class = -1
#         self.states = {0:'DEFAULT' , 1:'IN_PROGRESS',2:'FINISHED',-1:'FAILD'}
#         self.state = self.states[0] #Default    
#     def get_sentiment(self,text , callback=''):
#         url = "https://aspect-based-sentiment.herokuapp.com/api/predict/classname"
#         body = {"review": text,"callback_url":callback}
#         response = requests.post(url, body)
#         json_res = eval(response.text)
#         if json_res['msg'] == "Request Created":
#             self.state = self.states[1]   
#         else:
#             self.state = self.states[-1]    
#     def callback(self,sentiment_class):
#         self.sentiment_class = sentiment_class
#         self.state = self.states[2]