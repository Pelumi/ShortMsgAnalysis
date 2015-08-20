from multiprocessing import Pool
import random
import string

__author__ = 'Pelumi'

import numpy as numpy
from sklearn.base import BaseEstimator
from util.FreqCounters import *
from util.utils import tokeniseText
from util.utils import ManageLexicon
import time
import subprocess
from nltk import bigrams
from util import  const as constants
from util.utils import normalise_tweets


class SemanticVectorizer(BaseEstimator):

    #loading persisted dictionaries
    lexicon = ManageLexicon()
    NRCManualLexiconPos = lexicon.loadData(constants.const.NRCPosPickle)
    NRCManualLexiconNeg = lexicon.loadData(constants.const.NRCNegPickle)

    MPQAManualLexiconPos = lexicon.loadData(constants.const.MPQAPosPickle)
    MPQAManualLexiconNeg = lexicon.loadData(constants.const.MPQANegPickle)

    BingLiuManualLexiconPos = lexicon.loadData(constants.const.BingLiuPosPickle)
    BingLiuManualLexiconNeg = lexicon.loadData(constants.const.BingLiuNegPickle)

    NRCHashTagAutoLexiconUni = lexicon.loadData(constants.const.NRCUniPickle)
    NRCHashTagAutoLexiconBi = lexicon.loadData(constants.const.NRCBiPickle)
    NRCHashTagAutoLexiconPairs = {}

    Sent140AutoLexiconUni = lexicon.loadData(constants.const.Sent140BiPickle)
    Sent140AutoLexiconBi = lexicon.loadData(constants.const.Sent140UniPickle)

    #saved POS tags data for each sms
    POSTagCountData = lexicon.loadData(constants.const.PersistedPOSTag)
    TestPOSTagCountData = lexicon.loadData(constants.const.TestPersistedPOSTag)
    Sent140AutoLexiconPairs = {}

    def get_feature_names(self):

        return numpy.array(['ucaseCount', 'hashTagCount', 'mentionCount', 'negEmoCount', 'posEmoCount', 'lastTermIsEmo', 'lastTermIsPunc', 'puncSeqCount',
                            'tokenCount', 'charSeqCount', 'negationCount', 'tokCountNRCEmo', 'tokCountMPQA', 'tokCountBingLiu', 'maxScoreNRCEmo', 'maxScoreMPQA',
                            'maxScoreBingLiu', 'totScoreNRCEmo', 'totScoreMPQA', 'totScoreBingLiu', 'lastPosScNRC', 'lastPosScoreMPQA', 'lastPosScoreBingLiu',
                            'tokCountNRCHashUni', 'maxScoreNRCHashUni', 'lastPosScoreNRCHashUni', 'totScoreNRCHashUni', 'tokCountSent140Uni', 'maxScoreSent140Uni',
                            'lastPosScoreSent140Uni', 'totScoreSent140Uni', 'tokCountNRCHashBi', 'maxScoreNRCHashBi', 'lastPosScoreNRCHashBi', 'totScoreNRCHashBi',
                            'tokCountSent140Bi', 'maxScoreSent140Bi', 'lastPosScoreSent140Bi', 'totScoreSent140Bi',

                            'N',  'O',  'S',  'B',  'Z',  'L',  'M', 'V', 'A', 'R', 'C', 'D', 'P', 'F', 'T', 'X','Y', 'H',  'I',  'J',  'U', 'E', 'K',  'L', 'G'])


    def getManualLexiCounts(self, tokens):
        countNRC, countMPQA, countBingLiu, maxLastBingLiu, maxLastNRC, maxLastMPQA = 0,0,0,0,0,0
        for word in tokens:
            if word in self.NRCManualLexiconPos:
                countNRC+=1
            if word in self.MPQAManualLexiconPos:
                countMPQA+=1
            if word in self.BingLiuManualLexiconPos:
                countBingLiu+=1

        if countNRC > 0:
            maxLastNRC =1
        if countBingLiu> 0:
            maxLastBingLiu =1
        if countMPQA > 0:
            maxLastMPQA = 1

        return countNRC, countMPQA, countBingLiu, maxLastNRC, maxLastMPQA, maxLastBingLiu

    def getAutoLexiconCountsUni(self,tokens):
        countNRCPos, maxNRCScorePos, lastNRCPosTokenScore, totNRCScore = 0,0,0,0
        countSent140Pos, maxSent140ScorePos, lastSent140PosTokenScore, totSent140Score = 0,0,0,0

        for word in tokens:

            if word in SemanticVectorizer.NRCHashTagAutoLexiconUni:
                wordScoreNRC = float(SemanticVectorizer.NRCHashTagAutoLexiconUni[word])
                if wordScoreNRC > 1:
                    countNRCPos+=1
                    lastNRCPosTokenScore = wordScoreNRC
                    if wordScoreNRC > maxNRCScorePos:
                        maxNRCScorePos = wordScoreNRC
                totNRCScore += wordScoreNRC

            #sentiment 140 params retrieval
            if word in SemanticVectorizer.Sent140AutoLexiconUni:
                wordScoreSent140 = float(SemanticVectorizer.Sent140AutoLexiconUni[word])
                if wordScoreSent140 > 1:
                    countSent140Pos+=1
                    lastSent140PosTokenScore = wordScoreSent140
                    if wordScoreSent140 > maxSent140ScorePos:
                        maxSent140ScorePos = wordScoreSent140
                totSent140Score += wordScoreSent140

        return countNRCPos, maxNRCScorePos, lastNRCPosTokenScore, totNRCScore, countSent140Pos, maxSent140ScorePos, \
               lastSent140PosTokenScore, totSent140Score

    def getAutoLexiconCountsBi(self,text):
        countNRCPos, maxNRCScorePos, lastNRCPosTokenScore, totNRCScore = 0,0,0,0
        countSent140Pos, maxSent140ScorePos, lastSent140PosTokenScore, totSent140Score = 0,0,0,0

        tokens = bigrams(text)

        for word in tokens:

            if word in SemanticVectorizer.NRCHashTagAutoLexiconBi:
                wordScoreNRC = float(SemanticVectorizer.NRCHashTagAutoLexiconBi[word])
                if wordScoreNRC > 1:
                    countNRCPos+=1
                    lastNRCPosTokenScore = wordScoreNRC
                    if wordScoreNRC > maxNRCScorePos:
                        maxNRCScorePos = wordScoreNRC
                totNRCScore += wordScoreNRC

            #sentiment 140 params retrieval
            if word in SemanticVectorizer.Sent140AutoLexiconBi:
                wordScoreSent140 = float(SemanticVectorizer.Sent140AutoLexiconBi[word])
                if wordScoreSent140 > 1:
                    countSent140Pos+=1
                    lastSent140PosTokenScore = wordScoreSent140
                    if wordScoreSent140 > maxSent140ScorePos:
                        maxSent140ScorePos = wordScoreSent140
                totSent140Score += wordScoreSent140

        return countNRCPos, maxNRCScorePos, lastNRCPosTokenScore, totNRCScore, countSent140Pos, maxSent140ScorePos, \
               lastSent140PosTokenScore, totSent140Score



    def getSentFeats(self, sms):
        #normalise the texts
        normalised_sms = normalise_tweets(sms)
        charSeqCount, norm_text = sequenceCounter(normalised_sms)
        tokens = tokeniseText(norm_text)
        tokenCount = len(tokens)

        ucaseCount, hashTagCount, mentionCount = getCounts(tokens)
        posEmoCount, negEmoCount, lastTermIsEmo = emoticonCount(tokens)
        lastTermIsPunc, puncSeqCount = punctuationCount(tokens)
        negatedTokens, negationCount = getNegations(tokens)

        negatedTokens = tokens

        #get manual lexicon features
        tokCountNRCEmo, tokCountMPQA, tokCountBingLiu, maxScoreNRCEmo, \
        maxScoreMPQA, maxScoreBingLiu = self.getManualLexiCounts(negatedTokens)

        totScoreNRCEmo = tokCountNRCEmo
        totScoreMPQA = tokCountMPQA
        totScoreBingLiu = tokCountBingLiu

        lastPosScNRC = maxScoreNRCEmo
        lastPosScoreMPQA = maxScoreMPQA
        lastPosScoreBingLiu = maxScoreBingLiu
        #end of manual lexicon features

        # #get auto lexicon feature
        # #unigrams
        tokCountNRCHashUni, maxScoreNRCHashUni, lastPosScoreNRCHashUni, \
        totScoreNRCHashUni, tokCountSent140Uni, maxScoreSent140Uni, \
        lastPosScoreSent140Uni, totScoreSent140Uni = self.getAutoLexiconCountsUni(negatedTokens)
        #
        # #bigrams
        tokCountNRCHashBi, maxScoreNRCHashBi, lastPosScoreNRCHashBi, \
        totScoreNRCHashBi, tokCountSent140Bi, maxScoreSent140Bi, \
        lastPosScoreSent140Bi, totScoreSent140Bi = self.getAutoLexiconCountsBi(negatedTokens)

        #apply tweet specific POS tagging to the unnormalised messages
        if sms in SemanticVectorizer.POSTagCountData:
            #print 'found in generic dict, sms is: ', sms, ' value is: ', SemanticVectorizer.POSTagCountData
            currentPOSTags = SemanticVectorizer.POSTagCountData[sms]
        elif sms in SemanticVectorizer.TestPOSTagCountData:
            #print 'found in test dict, sms is: ', sms, ' value is: ', SemanticVectorizer.TestPOSTagCountData
            currentPOSTags = SemanticVectorizer.TestPOSTagCountData[sms]
        else:
            currentPOSTags = getPosTag(sms)
            print 'not found anywhere, sms is: ', sms
            SemanticVectorizer.POSTagCountData[sms] = currentPOSTags

        return [ucaseCount, hashTagCount, mentionCount, negEmoCount, posEmoCount, lastTermIsEmo, lastTermIsPunc, puncSeqCount,
                tokenCount, charSeqCount, negationCount, tokCountNRCEmo, tokCountMPQA, tokCountBingLiu, maxScoreNRCEmo, maxScoreMPQA,
                maxScoreBingLiu, lastPosScNRC, lastPosScoreMPQA, lastPosScoreBingLiu,
                tokCountNRCHashUni, maxScoreNRCHashUni, lastPosScoreNRCHashUni, tokCountSent140Uni, maxScoreSent140Uni,
                lastPosScoreSent140Uni, tokCountNRCHashBi, maxScoreNRCHashBi, lastPosScoreNRCHashBi,
                tokCountSent140Bi, maxScoreSent140Bi, lastPosScoreSent140Bi,
                totScoreNRCEmo, totScoreMPQA, totScoreBingLiu] + currentPOSTags
                # SemanticVectorizer.POSTagCountData[sms]

        # , totScoreSent140Bi, totScoreNRCHashBi, totScoreSent140Uni, totScoreNRCHashUni,

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        print 'Performing Semantic vectorisation...'
        ucaseCount, hashTagCount, mentionCount, negEmoCount, posEmoCount, lastTermIsEmo, lastTermIsPunc, puncSeqCount, \
        tokenCount, charSeqCount, negationCount, tokCountNRCEmo, tokCountMPQA, tokCountBingLiu, maxScoreNRCEmo, maxScoreMPQA, \
        maxScoreBingLiu,  lastPosScNRC, lastPosScoreMPQA, lastPosScoreBingLiu, \
        tokCountNRCHashUni, maxScoreNRCHashUni, lastPosScoreNRCHashUni, tokCountSent140Uni, maxScoreSent140Uni, \
        lastPosScoreSent140Uni, tokCountNRCHashBi, maxScoreNRCHashBi, lastPosScoreNRCHashBi, \
        tokCountSent140Bi, maxScoreSent140Bi, lastPosScoreSent140Bi, \
        totScoreNRCEmo, totScoreMPQA, totScoreBingLiu,   N,  O,  S,  B,  Z,  L,  M, V, A, R, C, D, P, F, T, X,Y, H,  I,  J,  U, E, K,  L, G = numpy.array([self.getSentFeats(d) for d in documents]).T

        #, totScoreSent140Bi totScoreSent140Uni,  totScoreNRCHashBi, totScoreNRCHashUni, negative terms removed for nvb

        otherFeatures = numpy.array([ucaseCount, hashTagCount, mentionCount, negEmoCount, posEmoCount, lastTermIsEmo, lastTermIsPunc, puncSeqCount,
                                     tokenCount, charSeqCount, negationCount, tokCountNRCEmo, tokCountMPQA, tokCountBingLiu, maxScoreNRCEmo, maxScoreMPQA,
                                     maxScoreBingLiu, lastPosScNRC, lastPosScoreMPQA, lastPosScoreBingLiu,
                                     tokCountNRCHashUni, maxScoreNRCHashUni, lastPosScoreNRCHashUni, tokCountSent140Uni, maxScoreSent140Uni,
                                     lastPosScoreSent140Uni,  tokCountNRCHashBi, maxScoreNRCHashBi, lastPosScoreNRCHashBi,
                                     tokCountSent140Bi, maxScoreSent140Bi, lastPosScoreSent140Bi,
                                     totScoreNRCEmo, totScoreMPQA, totScoreBingLiu,
                                     N,  O,  S,  B,  Z,  L,  M, V, A, R, C, D, P, F, T, X,Y, H,  I,  J,  U, E, K,  L, G]).T

        #, totScoreSent140Bi totScoreSent140Uni,  totScoreNRCHashBi, totScoreNRCHashUni, negative terms removed for nvb

        return otherFeatures

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

#used to get post tags into a variable
def getPosTag(text):
    filename = const.dump_dir + id_generator(12)
    fo = open(filename, "w+")
    #fo = open(const.poslabs, "wb")

    fo.write(text)
    fo.close()
    output = subprocess.check_output(["../data/CMUPOSTool/runTagger.sh", "--output-format", "conll", "--no-confidence", filename])
    #output = subprocess.check_output(["../data/CMUPOSTool/runTagger.sh", "--output-format", "conll", "--no-confidence", const.poslabs])

    #print 'output from py is: ', output

    smsTags = {  "N" : 0,  "O" : 0,  "S" : 0,  "^" : 0,  "Z" : 0,  "L" : 0,  "M" : 0, "V" : 0, "A" : 0, "R" : 0, "!" : 0, "D" : 0, "P" : 0, "&" : 0,
         "T" : 0, "X" : 0,"Y" : 0, "#" : 0,  "@" : 0,  "~" : 0,  "U" : 0, "E" : 0, "$" : 0,  "," : 0,  "G" : 0}

    tokens = output.strip().split("\n")
    posTags = []
    #print smsTags
    for token in tokens:
        toks = token.split('\t', 2)
        smsTags[toks[1]] += 1
    # print smsTags
    posTags = list(smsTags.values())
    #print posTags
    return posTags

def persist_test_POS(test_set):
    pool = Pool(processes=8)
    pages = pool.map(getPosTag, test_set)

    lexicon = ManageLexicon()
    POSDict = dict(zip(test_set, pages))
    lexicon.savedata(POSDict, const.TestPersistedPOSTag)
    print 'saved as ', const.TestPersistedPOSTag
    #print pages
    #print POSDict

#lexicon = ManageLexicon()

#TestPOSTagCountData = lexicon.loadData(const.ImprovedPersistedPOSTag)
#print TestPOSTagCountData

# #print getPosTag("USER_PHOLDA get USER_PHOLDA on february for valentine's day, the suave fuckers")
# print len(SemanticVectorizer.POSTagCountData)
# for sms in SemanticVectorizer.POSTagCountData:
#     print sms
#     print SemanticVectorizer.POSTagCountData[sms]
#       #      currentPOSTags = SemanticVectorizer.POSTagCountData[sms]
#        # else:

#etPosTag('Yeah he got in at 2 and was v apologetic. n had fallen out and she was actin like spoilt child and he got caught up in that. Till 2! But we won\'t go there! Not doing too badly cheers. You?')