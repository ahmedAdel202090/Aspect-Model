from app.aspect.utils import seq_to_vec
from app.aspect.aspect_extraction import AspectExtraction
from threading import Thread

MAX_SIZE_THREAD_GET = 5
def partition(arr, num_threads):
    parts = []
    r = len(arr) / num_threads # r represent ratio between part size for each thread
    for i in range(num_threads):
        st = int(i*r)
        ed = int((i+1)*r)
        part = arr[st:ed]
        parts.append(part)
    return parts
def task(extractor,doc_vec,texts,ngram,skip_keywords,result_aspects):
      for text in texts:
          aspects = extractor(doc_vec,text,ngram,skip_keywords)
          result_aspects.extend(aspects)
def assign_threads(parts,results,extractor,doc_vec,ngram,skip_keywords):
  threads = []    
  for part in parts:
    thread = Thread(target=task, args=(extractor,doc_vec,part,ngram,skip_keywords,results))
    thread.start()
    threads.append(thread)
  for thread in threads:
    thread.join()  
def get_num_threads(size):
  if size<2:
    return 1  
  for num_threads in range(2,size):
    if ((size*1.0)/num_threads) < MAX_SIZE_THREAD_GET:
      return num_threads
  return size         
class AspectKnowledgeBase(object):
  def __init__(self):
    self.texts = []
    self.knowledge_set = {}
    self.aspect_extraction = AspectExtraction()
    self.num_keywords=0
  def __call__(self,texts,ngram,skip_keywords):
    self.texts = texts
    text = ' '.join(texts)
    doc_vec = seq_to_vec(text)
    num_threads = get_num_threads(len(texts))
    print('num_threads............................',num_threads)
    texts_partitions = partition(texts,num_threads)
    aspects = []
    assign_threads(texts_partitions,aspects, self.aspect_extraction,doc_vec,ngram,skip_keywords)
    for aspect_obj in aspects:
      self.num_keywords +=1
      if not aspect_obj['aspect'] in self.knowledge_set.keys():
        self.knowledge_set[aspect_obj['aspect']] = {'sentiment':[0,0,0,0,0] , 'score':aspect_obj['score']}
      sentiment = aspect_obj['sentiment']  
      self.knowledge_set[aspect_obj['aspect']]['sentiment'][sentiment] +=1
    self.rank_normalize()
    self.knowledge_set = dict(sorted(self.knowledge_set.items(), key=lambda item: item[1]['score'],reverse=True))
    return self.knowledge_set    
  def rank_normalize(self):
    for keyword in self.knowledge_set.keys():
      keyword_freq = sum(self.knowledge_set[keyword]['sentiment'])
      self.knowledge_set[keyword]['score'] = self.knowledge_set[keyword]['score'] * (keyword_freq / self.num_keywords)
