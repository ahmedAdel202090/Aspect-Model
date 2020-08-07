from app import app
from flask import request, jsonify
import threading
import requests
from app.aspect.aspect_extraction import AspectExtraction
from app.aspect.aspect_knowlege_base import AspectKnowledgeBase
from app.aspect.utils import seq_to_vec, sentence_segmenter
import threading
import os
import time
import gensim
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
MAX_REQUEST_RETRIES = 5
is_loaded = False
MODEL_URL = 'https://storagerat.blob.core.windows.net/telda/word2vec-150-400k.bin?sp=r&st=2020-07-29T22:12:06Z&se=2020-08-30T06:12:06Z&spr=https&sv=2019-12-12&sr=b&sig=l7lQ8gSusC81xeobOP6rzTbARjvq89pSIYCj2uuvYfo%3D'
model = None


def callback(target_fun, args, callback_url, call_name):
    print("[{0}] Request is in progress and will be delivered to {1}".format(
        call_name, callback_url))
    result = target_fun(*args)
    resp = {
        "data": result,
        "msg": "Succeed",
        "errmsg": "NULL"
    }
    webhook_request = requests.post(url=callback_url, json=resp)
    print("[{0}]:Request Delivered to <{1}>.".format(call_name, callback_url))
    del webhook_request
    # for _ in range(MAX_REQUEST_RETRIES):
    #     webhook_request = requests.post(url=callback_url, json=resp)
    #     if webhook_request.status_code == 200:
    #         print("[{0}]:Request Delivered to <{1}>.".format(call_name, callback_url))
    #         break


def load_model():
    global model
    global is_loaded
    if model == None and not is_loaded:
        st = time.time()
        print('Loading Model .....')
        is_loaded = True
        model = KeyedVectors.load_word2vec_format(MODEL_URL, binary=True)
        ed = time.time() - st
        print('Model Loaded in : {0} s'.format(ed))


@app.route('/load_model')
def load_word2vec():
    global model
    global is_loaded
    if model != None and is_loaded:
        return jsonify({'loaded': True, 'msg': 'model is loaded...'})
    else:
        threading.Thread(target=load_model,
                         name="Model-Loading-Thread").start()
        return jsonify({'loaded': False, 'msg': 'model not loaded'})


@app.route('/sentence-seg', methods=['POST'])
def sentence_segmentation():
    reviews = request.json["reviews"]
    callback_url = request.json["callback_url"]
    resp = None
    if reviews is None or callback_url is None:
        resp = {
            "data": "NULL",
            "msg": "Sentence-Segmentation Failed",
            "errMsg": "One of the required parameters is null"
        }
    else:
        threading.Thread(target=callback, args=(
            sentence_segmenter, (reviews,), callback_url, "Sentence-Segmentation")).start()
        resp = {
            "msg": "Sentence-Segmentation starting",
            "errmsg": "NULL"
        }
    return jsonify(resp)


@app.route('/api/aspect_extraction', methods=["POST"])
def aspect_extraction():
    global model
    global is_loaded
    if model != None and is_loaded:
        review = request.json["review"]
        n_grams = request.json['n_grams']
        domains = request.json["tags"]
        callback_url = request.json["callback_url"]
        resp = None
        if callback_url is None or domains is None or n_grams is None:
            resp = {
                "data": "NULL",
                "msg": "Aspect-Extraction Failed",
                "errMsg": "One of the required parameters is null"
            }
        else:
            extractor = AspectExtraction()
            doc_vec = seq_to_vec(model, review)
            threading.Thread(target=callback, args=(extractor, (model, doc_vec, review, int(
                n_grams), domains), callback_url, "Aspect-Extraction")).start()
            resp = {
                "msg": "Aspect-Extraction starting",
                "errmsg": "NULL"
            }
        return jsonify(resp)
    else:
        return jsonify({'loaded': False, 'msg': 'model not loaded'})


@app.route('/api/create_knowledge_base', methods=["POST"])
def create_knowledge_base():
    global model
    global is_loaded
    if model != None and is_loaded:
        reviews = request.json["reviews"]
        n_grams = request.json['n_grams']
        domains = request.json["tags"]
        callback_url = request.json["callback_url"]
        resp = None
        if callback_url is None or domains is None or n_grams is None:
            resp = {
                "data": "NULL",
                "msg": "Aspect-KnowledgeBase Failed",
                "errMsg": "One of the required parameters is null"
            }
        else:
            aspect_knowledge_base = AspectKnowledgeBase()
            threading.Thread(target=callback, args=(aspect_knowledge_base, (model, reviews,int(n_grams), domains), callback_url, "Aspect-KnowledgeBase")).start()
            resp = {
                "msg": "Aspect-KnowledgeBase Start Generating...",
                "errmsg": "NULL"
            }
        return jsonify(resp)
    else:
        return jsonify({'loaded': False, 'msg': 'model not loaded'})
