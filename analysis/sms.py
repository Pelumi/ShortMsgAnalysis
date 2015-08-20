__author__ = 'Pelumi'
from util.const import *

class sms:
    text = 'default sms'
    clean_text = ''

    def __init__(self, text, label=const.NEUTRAl, clean_text=text, preprocessed_text=text, sender="NA", recipient="NA",
                 tokens=[], negTokens =[], uCaseCount = 0, charSeqCount="", hashTagCount= 0, mentionCount = 0, posEmoCount = 0, negEmoCount = 0,
                 lastTermIsEmo=0, lastTermisPunct=0,puncSeqCount=0, negationCount=0):
        self.text = text
        self.clean_text = clean_text
        self.preprocessed_text = preprocessed_text
        self.sender = sender
        self.recipient = recipient
        self.label = label
        self.tokens = tokens
        self.tokenCount = 0
        self.negTokens = negTokens
        self.uCaseCount = uCaseCount
        self.charSeqCount = charSeqCount
        self.hashTagCount = hashTagCount
        self.mentionCount = mentionCount

        self.posEmoCount = posEmoCount
        self.negEmoCount = negEmoCount
        self.lastTermIsEmo = lastTermIsEmo
        self.lastTermisPunct = lastTermisPunct
        self.puncSeqCount = puncSeqCount
        self.negationCount = negationCount

        #positive polarity params from lexicon
        self.tokCountNRCEmo = 0
        self.tokCountBingLiu = 0
        self.tokCountMPQA = 0
        self.tokCountNRCHashUni = 0
        self.tokCountNRCHashBi = 0
        self.tokCountSent140Uni = 0
        self.tokCountSent140Bi = 0


        self.totScoreNRCEmo = 0
        self.totScoreBingLiu = 0
        self.totScoreMPQA = 0
        self.totScoreNRCHashUni = 0
        self.totScoreNRCHashBi = 0
        self.totScoreSent140Uni = 0
        self.totScoreSent140Bi = 0

        self.maxScoreNRCEmo = 0
        self.maxScoreBingLiu = 0
        self.maxScoreMPQA = 0
        self.maxScoreNRCHashUni = 0
        self.maxScoreNRCHashBi = 0
        self.maxScoreSent140Uni = 0
        self.maxScoreSent140Bi = 0


        self.scoreNRCEmo = 0
        self.scoreBingLiu = 0
        self.scoreMPQA = 0
        self.scoreNRCHashUni = 0
        self.scoreNRCHashBi = 0
        self.scoreSent140Uni = 0
        self.scoreSent140Bi = 0

        self.lastPosScoreNRCEmo = 0
        self.lastPosScoreBingLiu = 0
        self.lastPosScoreMPQA = 0
        self.lastPosScoreNRCHashUni = 0
        self.lastPosScoreNRCHashBi = 0
        self.lastPosScoreSent140Uni = 0
        self.lastPosScoreSent140Bi = 0

        self.POSTags = []

samplesms = sms("", "")
print samplesms.text

class dataholder:
    def __init__(self, smsList, labelList, tweetModel, features):
        self.smsList = smsList
        self.labelList = labelList
        self.tweetModel = tweetModel
        self.features = features