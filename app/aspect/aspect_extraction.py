from app import nlp, VECTOR_SIZE
from app.aspect.utils import seq_to_vec, get_sentiment
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk import word_tokenize
import numpy as np
from rake_nltk import Rake
from itertools import chain

W1 = 0.3
W2 = 0.7
IGNORED_ENTITIES  = ['GPE','LOC','PRODUCT','WORK_OF_ART','LANGUAGE','DATE','TIME','ORDINAL','CARDINAL']


class AspectExtraction(object):
    def __init__(self):
        self.sent_keywords = []  # candidates
        self.aspects = []
        self.sents = []
        self.sent_index = -1
        self.ignores = []
        self.doc_vector = np.zeros(VECTOR_SIZE, dtype='float32')

    def __call__(self,model ,doc_vector, text, ngrams, topics=[]):
        self.__init__()
        self.ignores = topics
        self.topics = topics
        self.doc_vector = doc_vector
        self.model = model
        self.text = text['text']
        self.sentiment = text['sentiment']
        doc = nlp(self.text)
        # self.sents = [(str(sent), get_sentiment(str(sent)))
        #               for sent in doc.sents]
        self.ignores.extend([str(ent.lemma_) for ent in doc.ents if ent.label_ in IGNORED_ENTITIES])
        temp_ignores = []
        for s in self.ignores:
            temp_ignores.extend(word_tokenize(s))
        self.ignores = temp_ignores  
        # for i in range(0, len(self.sents)):
        #     self.index = i
        self.text_filter()
        self.run_rake()
        self.getNgrams(ngrams)
        self.make_keywords_to_aspects()  # pass sentence index to add its sentiment to keyword
        self.filter_keywords()
        self.calculate_similarity()
        return self.aspects

    def calculate_similarity(self):
        if len(self.aspects) <= 0:
            return
        aspects_vector_space = np.array([seq_to_vec(self.model ,str(aspect_obj['aspect'])).tolist()[0] for aspect_obj in self.aspects])
        doc_similarities = cosine_similarity(aspects_vector_space, self.doc_vector)
        topic_similarities = cosine_similarity(aspects_vector_space, seq_to_vec(self.model,' '.join(self.topics)))
        for i, aspect_obj in enumerate(self.aspects):
            score = 0.0
            if len(self.topics) > 0:
                score = W2 * topic_similarities[i][0] + W1 * doc_similarities[i][0]
            else:
                score = (W1+W2) * doc_similarities[i][0]
            #_, aspect, sentiment = self.aspects[i]
            self.aspects[i]['score'] = score

    def text_filter(self):
        result = []
        text_doc = nlp(self.text, disable=['ner'])
        is_prev_noun = False
        is_added_of = False
        last_of_index = -1
        for token in text_doc:
            if token.pos_ in ['NOUN']:
                result.append(str(token.lemma_).lower())
                if is_prev_noun and is_added_of:
                    is_prev_noun = False
                    is_added_of = False
                    last_of_index = -1
                else:
                    is_prev_noun = True
            elif token.lemma_ == 'of':
                is_added_of = True
                last_of_index = len(result)
                result.append(str(token.lemma_).lower())
            elif last_of_index > -1 and not is_prev_noun:
                result.pop(last_of_index)
                is_prev_noun = False
                is_added_of = False
                last_of_index = -1
            else:
                is_prev_noun = False
        self.updated_sent = ' '.join(result)
        return self

    def getNgrams(self, n):
        for i in range(0, len(self.sent_keywords)):
            tokens = nltk.word_tokenize(self.sent_keywords[i][1])
            grams = n
            tags = nltk.pos_tag(tokens)
            reco = ['NN', 'NNP', 'NNS']
            result = []
            for tag in tags:
                if grams <= 0:
                    break
                if tag[1] in reco:
                    result.append(tag[0])
                    grams -= 1
            result = ' '.join(result)
            if result != '':
                self.sent_keywords[i] = (self.sent_keywords[i][0], result)
            return self

    def filter_keywords(self):
        for i, aspect_obj in enumerate(self.aspects):
            aspect_tokens = word_tokenize(aspect_obj['aspect'])
            for vocab in self.ignores:
                vocab_tokens = set(word_tokenize(vocab))
                if len(vocab_tokens.intersection(aspect_tokens)) > 0:
                    self.aspects.pop(i)
                    break

    def run_rake(self):
        r = Rake()
        r.stopwords = list(set(r.stopwords) - {'of'})
        r.to_ignore = set(chain(r.stopwords, r.punctuations))
        r.extract_keywords_from_text(self.updated_sent)
        self.sent_keywords = r.get_ranked_phrases_with_scores()
        return self

    def make_keywords_to_aspects(self):
        #sum_ranks = sum([tag[0] for tag in self.sent_keywords])
        if len(self.sent_keywords) > 0:
            # rand changed to predict sentiment
            self.aspects.append({
                'aspect': self.sent_keywords[0][1],
                'sentiment': self.sentiment
            })
        return self
