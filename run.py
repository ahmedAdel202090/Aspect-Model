from app import app
import nltk

nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
if __name__ == "__main__":
    app.debug = True
    app.run()
