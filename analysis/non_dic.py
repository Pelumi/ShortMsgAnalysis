__author__ = 'Pelumi'

import nltk
from nltk.tokenize import RegexpTokenizer

from sms import *
from util.const import *
import enchant
from enchant.checker import *
from enchant.tokenize import URLFilter, WikiWordFilter, EmailFilter
from enchant.utils import *


#imports for language model
from nltk.model import NgramModel
from nltk.probability import LidstoneProbDist


class normalization:
    norm_list = []
    norm_lexi_list = []
    norm_dict = {}
    def __init__(self):

        print "Normalisation started..."

    def loadFile(self):
        lines = [line.strip() for line in open(const.norm_dataset)]

        print "Total list entries are ", len(lines)

        for i in range(0,6000,3):
            #print "The default is: ", lines[i]
            #print "Cleaned is: ", lines[i+1]

            self.norm_list.append(sms(lines[i], lines[i+1]))


    #function to tokenize twitter normalisation lexicon
    def loadNormLexicon(self):
        lines = [line.strip() for line in open(const.norm_lexicon_data)]
        print "Total list entries are ", len(lines)

        count = len(lines)
        cou = 0
        for j in range(0,count,1):
            line = lines[j].translate(None, ',.')
            tokens = nltk.word_tokenize(line)
            if(len(tokens) > 2 and tokens[1]=="OOV"):
                if tokens[0]!=tokens[2]:
                    cou = cou + 1
                    print "",cou, " - ", tokens[0], ">>>", tokens[2]
          #  else:
              #  print "invalid entry.."



    def createOOVMap(self):
        #tokenize removing punctuations
        norm = normalization()
        for j in range(0,2000,1):
            tokenizer = RegexpTokenizer(r'\w+')
            text_tokens = tokenizer.tokenize(norm.norm_list[j].text)
            #print "total tokens are ", len(text_tokens)
            clean_text_tokens = tokenizer.tokenize(norm.norm_list[j].clean_text)

            if len(text_tokens) == len(clean_text_tokens):
                tok_count = len(clean_text_tokens)
              #  print "Same number of tokens therefore direct mapping can be done..."
                for token in range(0,tok_count,1):
                    if text_tokens[token].lower() != clean_text_tokens[token].lower():
                        print text_tokens[token].lower(), " >>> ", clean_text_tokens[token].lower()
                        norm.norm_dict[text_tokens[token].lower()] = clean_text_tokens[token]

            else:
               # print "token count not equal"
                with open("../data/rem.dat", "a") as rem:
                    rem.write(norm.norm_list[j].text.join("\n"))
                    rem.write(norm.norm_list[j].clean_text.join("\n"))
        #print norm.norm_dict

    def calcWordProb(self):

        word_seq = ['foo', 'foo', 'foo', 'foo', 'bar', 'baz']

        text = "They become more expensive already. Mine is like 25. So horrible and they did less things than I did last time."
        text = nltk.word_tokenize(text.translate(None, ',.'))

        print text
        #estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
        #lm = NgramModel(2, word_seq, estimator)

        est = lambda freqdist, bins: LidstoneProbDist(freqdist, 0.2, bins)
        model = NgramModel(3, text, True, True, est, 21)

        print model.prob("more", text)


    def detectOOV(self, word):
        d = enchant.Dict("en_US")
        chkr = SpellChecker("en_US", word, filters=[EmailFilter,URLFilter,WikiWordFilter])

        for err in chkr:
            #print d.suggest(chkr.word)
            print trim_suggestions(chkr.word, d.suggest(chkr.word)[:10], 10)
            #err.replace("SPAM")
            print chkr.word

        print word

        print chkr.get_text()

        # if d.check(word):
        #     print "Its a dic word"
        # else:
        #     print "OOV word detected, suggestions are:\n"
        #     print d.suggest(word)




norm = normalization()
#norm.loadFile();
#norm.loadNormLexicon()
#norm.createOOVMap()
#norm.calcWordProb()

norm.detectOOV("Weekend pressure at TPC Sawgrass? Look no further than Harris English 4get it nyt gurl - 12 shot swing in a day - goes from 6");