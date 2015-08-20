__author__ = 'Pelumi'


import nltk.stem
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer

english_stemmer = nltk.stem.SnowballStemmer('english')
class CustomVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))