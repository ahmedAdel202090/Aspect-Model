from app.aspect.utils import seq_to_vec
from app.aspect.aspect_extraction import AspectExtraction


class AspectKnowledgeBase(object):
  def __init__(self):
    self.texts = []
    self.knowledge_set = {}
    self.aspect_extraction = AspectExtraction()
    self.num_keywords=0
  def __call__(self,model,texts,ngram,skip_keywords):
    self.texts = texts
    text_value = list(map(lambda x:x["text"],texts))
    text = ' '.join(text_value)
    doc_vec = seq_to_vec(model,text)
    for text in texts:
      aspects = self.aspect_extraction(model,doc_vec,text,ngram,skip_keywords)
      for aspect_obj in aspects:
        self.num_keywords +=1
        if not aspect_obj['aspect'] in self.knowledge_set.keys():
          self.knowledge_set[aspect_obj['aspect']] = {'sentiment':[0,0,0,0,0] , 'score':aspect_obj['score']}
        sentiment = aspect_obj['sentiment']  
        self.knowledge_set[aspect_obj['aspect']]['sentiment'][sentiment] +=1
    self.rank_normalize()
    #self.knowledge_set = dict(sorted(self.knowledge_set.items(), key=lambda item: item[1]['score'],reverse=True))
    self.knowledge_set = self.transform_aspect_to_obj(self.knowledge_set)
    return self.knowledge_set    
  def rank_normalize(self):
    for keyword in self.knowledge_set.keys():
      keyword_freq = sum(self.knowledge_set[keyword]['sentiment'])
      self.knowledge_set[keyword]['score'] = self.knowledge_set[keyword]['score'] * (keyword_freq / self.num_keywords)  
  def transform_aspect_to_obj(self,knowledge_set):
    temp_knowledge_set = []
    for aspect,value in knowledge_set.items():
          value['aspect'] = aspect
          temp_knowledge_set.append(value)
    return temp_knowledge_set      

