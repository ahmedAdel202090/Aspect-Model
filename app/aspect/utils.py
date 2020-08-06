import requests
import numpy as np
from random import seed
from random import randint
from app import VECTOR_SIZE
import nltk
from nltk import word_tokenize
from app import model,nlp
import threading

seed(1)

# def seq_to_vec(text):
#     body = {"text": text}
#     url = "https://test-wordtovec.azurewebsites.net/doc2vec"
#     response = requests.post(url, body)
#     if response.text[0] != '{':
#         return np.zeros((1,VECTOR_SIZE),dtype='float32')    
#     vector = eval(response.text)
#     return np.array(vector['doc_vec'])
THREAD_SIZE = 30
def partition(arr, num_threads):
    parts = []
    r = len(arr) / num_threads # r represent ratio between part size for each thread
    for i in range(num_threads):
        st = int(i*r)
        ed = int((i+1)*r)
        part = arr[st:ed]
        parts.append(part)
    return parts
def get_num_threads(size):
    if size<=THREAD_SIZE:
        return 1
    for num_threads in range(2,size):
        if ((size*1.0)/num_threads) <= THREAD_SIZE:
            return num_threads
    return size
def segmentation(reviews,result_arr):
    for review in reviews:
        doc = nlp(review["text"])
        sents = [str(sent) for sent in doc.sents]
        review["opinionUnits"] = sents
        result_arr.append(review)
def sentence_segmenter(reviews):
    result_arr = []
    num_threads = len(reviews)
    parts = partition(reviews,num_threads)
    threads = []
    for part in parts:
        thread = threading.Thread(target=segmentation, args=(part, result_arr))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return result_arr    
def seq_to_vec(model,text):
    result = np.zeros((1,VECTOR_SIZE),dtype='float32')
    count = 0
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    reco = ['NN', 'NNP','NNS']
    for tag in tags:
        try:
            if tag[1] in reco:
              wv = model[str(tag[0])]
              wv = wv.reshape(1,wv.shape[0])
              result+=np.nan_to_num(wv)
              count+=1
        except:
            continue
    if count > 0:
        result = result/count
    return result

def word2vec(word):
    url = "https://test-wordtovec.azurewebsites.net/word2vec/{0}".format(word)
    response = requests.get(url)
    if response.text[0] != '{':
        return np.zeros((1,VECTOR_SIZE),dtype='float32')
    vector = eval(response.text)
    return np.array(vector['vector'])


def get_sentiment(text):
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