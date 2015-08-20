__author__ = 'Pelumi'
from os import listdir
from os.path import isfile, join
from sentanal.preprocessing.const import const
from datetime import datetime
import nltk
from nltk.tree import *
from sentanal.preprocessing.christ_tokenizer import Tokenizer
from nltk.draw import tree

from nltk.corpus import stopwords




def timeLabs():
    date_str = 'Tue May 08 15:14:45 +0800 2012'
    time_str = '2011.03.02 11:25:14' #len is 19
    time_str2 = '2010.09.18 12:45' #len is 16
    time_str3 = '5/08/2012 8:23:20 PM' #len is 20-22
    time_str4 = '2011.05.31 0:00'
    # {'date': "datestr", 'count': 7, 'texts': [sms,sms,sms,sms]}

    date = datetime.strptime(date_str, '%a %B %d %H:%M:%S +0800 %Y')
    timeDate = datetime.strptime(time_str, '%Y.%m.%d %H:%M:%S')
    timeDate2 = datetime.strptime(time_str2, '%Y.%m.%d %H:%M')
    timeDate3 = datetime.strptime(time_str3, '%m/%d/%Y %I:%M:%S %p')

    timeDate4 = datetime.strptime(time_str3, '%m/%d/%Y %I:%M:%S %p')

    if date.date() == timeDate3.date():
        print 'same day'
    else:
        print 'diff days'
    print date
    print timeDate
    print timeDate2
    print timeDate3

    print 'the dude '.strip()
#datetime.datetime(2012, 5, 8, 15, 14, 45) http://www.monlp.com/2011/12/16/nltk-trees/

def tokeniseText(text):
    tok = Tokenizer(preserve_case=True)
    tokenized = tok.tokenize(text)
    return tokenized

def loadSuspects():
    #todo load suspects from directory
    onlyfiles = [ f for f in listdir(const.classified_sms_dir) if isfile(join(const.classified_sms_dir,f)) ]
    print(onlyfiles)

def posRendering():
    sentence = """Fly Jon to London."""
    #sentence = """you do all in 1 night? lol."""
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)
    print tagged
  #  sentence = Tree('s', tagged)
   # sentence.draw()


def drawTree():
    tree = nltk.Tree.parse('(Tweet (Adj old) (NP (N men) (Conj and) (N women)))')

    cmutree = nltk.Tree.parse('(Tweet (Adj old) (NP (N men) (Conj and) (N women)))')
    tree.draw()

def tokenize():
    tweet = 'Tom, Expect fair weather tomorrow!:)'
    tokens = nltk.word_tokenize(tweet)

    toks = nltk.wordpunct_tokenize(tweet)

    tok = tokeniseText(tweet)

    print tokens
    print toks
    print tok


#print stopwords.words('english')
drawTree()

posRendering()

#http://en.wikipedia.org/wiki/Support_vector_machine#mediaviewer/File:Svm_max_sep_hyperplane_with_margin.png

