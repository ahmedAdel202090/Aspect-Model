from app import app
from flask import request, jsonify
import threading
import requests
from app.aspect.aspect_extraction import AspectExtraction
from app.aspect.aspect_knowlege_base import AspectKnowledgeBase
from app.aspect.utils import seq_to_vec


@app.route('/api/aspect_extraction',methods=["POST"])
def aspect_extraction():
    review = request.json["review"]
    n_grams = request.json['n_grams']
    domains = request.json["tags"]
    extractor = AspectExtraction()
    doc_vec = seq_to_vec(review)
    result = extractor(doc_vec,review,int(n_grams),domains)
    return jsonify(result)


@app.route('/api/create_knowledge_base',methods=["POST"])
def create_knowledge_base():
    reviews = request.json["reviews"]
    n_grams = request.json['n_grams']
    domains = request.json["tags"]
    aspect_knowledge_base = AspectKnowledgeBase()
    result = aspect_knowledge_base(reviews,n_grams,domains)
    return jsonify(result)


