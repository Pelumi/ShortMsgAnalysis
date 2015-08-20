__author__ = 'Pelumi'

import pickle
from time import time
import os
import collections
import csv
from os import listdir
from os.path import isfile, join
import re

import numpy as np
from matplotlib import pylab
from nltk.stem import PorterStemmer

from util.const import *
from christ_tokenizer import Tokenizer


class ManageLexicon:
    def savedata(self, dict, fileName):
        pickle.dump( dict, open( fileName, "wb" ) )

    def loadData(self, fileName):
        persistedPickle = pickle.load( open( fileName, "rb" ) )
        return persistedPickle

class tools:
       def tabSpaceFile(self):
        lines = [line.strip() for line in open(const.trimmed_clean_dataset_B_Final)]
        f = open(const.raw_data, 'a')
        data_count = len(lines)
        for i in range(0,data_count,1):
                temp = lines[i].replace('\t', ' ')
                #split by tabs, only the first two
                tokens = temp.split(' ', 1)
#                print tokens[1]
                f.write(tokens[0].strip())
                f.write("\t")
                f.write(tokens[1].strip())
                f.write("\n")
            #self.norm_list.append(sms(lines[i], lines[i+1]))

        def filterInvalidData(self):
            lines = [line.strip() for line in open(const.sent_cleaned_dataset)]
            f = open(const.trimmed_clean_dataset_B, 'a')
            data_count = len(lines)

            for i in range(0,data_count,1):
                #print "The default is: ", lines[i]

                if "Not Available" in lines[i]: continue

                #print lines[i]
                #split by tabs, only the first two
                tokens = lines[i].split('\t', 2)
                toks = tokens[2].split('\t')
                count = len(toks)

                #print count

                #print tokens[2]
                f.write(toks[0].strip())
                f.write("\t")
                f.write(toks[1].strip())
                f.write("\n")
                #self.norm_list.append(sms(lines[i], lines[i+1]))

        def printStopwordList(self):
            #read the stopwords file and build a list
            stop_dict = {}
            lines = [line.strip() for line in open(const.stop_word_list)]
            data_count = len(lines)

            for i in range(0,data_count,1):
                #print lines[i]
                stop_dict[lines[i]] = 0
                print '\'{}\' : \'0\','.format(lines[i])
                #print stop_dict

            return stop_dict
            #end

class PersistLexicons:

    persist = ManageLexicon()
    def __init__(self):
        t0 = time()
        self.loadAutoLexicon(const.NRCHashAutoLexiconUnigram, const.NRCUniPickle)
        self.loadAutoLexicon(const.NRCHashAutoLexiconBigram, const.NRCBiPickle)

        self.loadAutoLexicon(const.sent140AutoLexiconBigram, const.Sent140BiPickle)
        self.loadAutoLexicon(const.sent140AutoLexiconUnigram, const.Sent140UniPickle)

        self.getScoreBingLiu()
        self.getScoreMPQA()
        self.getScoreNRC()

        print "All lexicon dicts created and persisted..."
        print("done in %fs" % (time() - t0))

    #todo persist lexicon data so as not to reload each time
    #use NRC manual lexicon
    def getScoreNRC(self):
        lines = [line.strip() for line in open(const.NRCManualLexicon)]
        data_count = len(lines)
        nrclex_NEG = {}
        nrclex_POS = {}
        for i in range(0,data_count,1):
            tokens = lines[i].split('\t')
            if tokens[1] == 'positive':
                if tokens[2] == '1':
                    nrclex_POS[tokens[0]] = tokens[2]
            elif tokens[1] == 'negative':
                if tokens[2] == '1':
                    nrclex_NEG[tokens[0]] = tokens[2]
            else:
                continue

        PersistLexicons.persist.savedata(nrclex_NEG, const.NRCNegPickle)
        PersistLexicons.persist.savedata(nrclex_POS, const.NRCPosPickle)

    #load MPQA manual lexicon
    def getScoreMPQA(self, loadWeakSubj=True):
        lines = [line.strip() for line in open(const.MPQAManualLexicon)]
        data_count = len(lines)
        mpqalex_NEG = {}
        mpqalex_POS = {}
        for i in range(0,data_count,1):
            tokens = lines[i].split()
            if tokens[5] == 'priorpolarity=positive':
                if loadWeakSubj:
                    mpqalex_POS[tokens[2][6:]] = 1
                else:
                    if tokens[0] == 'type=strongsubj':
                        mpqalex_POS[tokens[2][6:]] = 1
            elif tokens[5] == 'priorpolarity=negative':
                if loadWeakSubj:
                    mpqalex_NEG[tokens[2][6:]] = 1
                else:
                    if tokens[0] == 'type=strongsubj':
                        mpqalex_NEG[tokens[0][6:]] = 1
            else:
                continue

        PersistLexicons.persist.savedata(mpqalex_NEG, const.MPQANegPickle)
        PersistLexicons.persist.savedata(mpqalex_POS, const.MPQAPosPickle)

    #load Bing Liu manual lexicon
    def getScoreBingLiu(self):
        neglines = [line.strip() for line in open(const.BingLiuNegLexicon)]
        poslines = [line.strip() for line in open(const.BingLiuPosLexicon)]
        pos_count = len(poslines)
        neg_count = len(neglines)
        bingLiuLex_NEG = {}
        bingLiuLex_POS = {}
        for i in range(0,neg_count,1):
            bingLiuLex_NEG[neglines[i]] = 1

        for i in range(0,pos_count,1):
            bingLiuLex_POS[poslines[i]] = 1

        PersistLexicons.persist.savedata(bingLiuLex_POS, const.BingLiuPosPickle)
        PersistLexicons.persist. savedata(bingLiuLex_NEG, const.BingLiuNegPickle)

    #load  auto lexicons sentiment140/NRCHashtag all
    def loadAutoLexicon(self, lexiconFile, outputFile):
        lines = [line.strip() for line in open(lexiconFile)]

        data_count = len(lines)
        lexiconDict = {}
        for i in range(0,data_count,1):
            tokens = lines[i].split('\t')
            lexiconDict[tokens[0]] = tokens[1]
        PersistLexicons.persist.savedata(lexiconDict, outputFile)


#lex = Lexicons()

def tweak_labels(Y, pos_sent_list):
    pos = Y == pos_sent_list[0]
    for sent_label in pos_sent_list[1:]:
        pos |= Y == sent_label

    Y = np.zeros(Y.shape[0])
    Y[pos] = 1
    Y = Y.astype(int)

    return Y




    #tokenise text using custom christopher's tokeniser
    #todo explore if CMU tokeniser will be better
def tokeniseText(text):
    tok = Tokenizer(preserve_case=True)
    tokenized = tok.tokenize(text)
    return tokenized

def plot_pr(auc_score, name, phase, precision, recall, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.plot(recall, precision, lw=1)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('AUC=%0.2f | %s' % (auc_score, label))
    #pylab.title('P/R curve (AUC=%0.2f) / %s' % (auc_score, label))
    filename = name.replace(" ", "_")
    pylab.savefig(os.path.join(const.CHART_DIR, "pr_%s_%s.png"%(filename, phase)), bbox_inches="tight")


class LoadDataSet:

    def load_semeval_data(self):
        dt = LoadDataSet()

        labels = []
        tweets = []

        lines = [line.strip() for line in open(const.trimmed_clean_dataset_B_Final)]

        for line in lines:
            tokens = line.split('\t', 1)
            if(len(tokens) == 2):
                label, tweet = tokens[0].strip(), tokens[1].strip()
                labels.append(label)
                tweets.append((tweet))

        labels = np.asarray(labels)
        tweets = np.asarray(dt.processTweet(tweet, stem=True))

        # return tweets, labels
        return tweets, labels

    def processTweet(self, tweet,stem= True,remove_stopwords=True):

        #Convert to lower case
        #tweet = tweet.lower()

        #replace emos with placeholders
        for j in emoticons.iterkeys():
            tweet = tweet.replace(j , emoticons[j])

        for k in emo_repl_order:
            tweet = tweet.replace(k, emo_repl[k])

        #Convert www.* or https?://* to URL
       # tweet = re.sub('((www\.[\s])|(https?://[^\s]))','',tweet)

        tweet = re.sub('((www\.[\s])|(https?://[^\s]))','URL_PHOLDA',tweet)
        #Convert @username to AT_USER
        tweet = re.sub('@[^\s]','@USER_PHOLDA',tweet)

        tweet = tweet.replace('RT', '')

        #tweet = re.sub('@[^\s]','',tweet)

        return_tweet = tweet
        # if remove_stopwords:
        #     ntweet = []
        #     #remove stop words
        #     for word in return_tweet.split(): # iterate over word_list
        #         if const.stopwords.has_key(word):
        #             continue
        #         else:
        #             ntweet.append(word)
        #     return_tweet = " ".join(ntweet)

        for word in return_tweet.split(): # iterate over word_list
            if word in NormalizationDict:
                return_tweet.replace(word, NormalizationDict[word])


        #stem tweet
        if stem:
            st_tweet = []
            porterStemmer = PorterStemmer()
            for word in return_tweet.split(): # iterate over word_list
                #snowball stemmer resulted in better performance, wordnet was poor
                st_tweet.append(porterStemmer.stem(word))
                #st_tweet.append(SnowballStemmer("english").stem(word))
            #st_tweet,dumpshit = getNegations(st_tweet)
            return_tweet = " ".join(st_tweet)

        return return_tweet

    def load_norm_lexicon(self):
        norm_dict = {}

        with open(const.norm_lexicon, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            for line in reader:
                #print line
                norm_dict[line[0]] = line[1]
        return norm_dict

    def load_sent_word_net(self):

        sent_scores = collections.defaultdict(list)

        with open(const.SentiwordNet, "r") as csvfile:
            reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            for line in reader:
                if line[0].startswith("#"):
                    continue
                if len(line)==1:
                    continue

                POS,ID,PosScore,NegScore,SynsetTerms,Gloss = line
                if len(POS)==0 or len(ID)==0:
                    continue
                #print POS,PosScore,NegScore,SynsetTerms
                for term in SynsetTerms.split(" "):
                    term = term.split("#")[0] # drop #number at the end of every term
                    term = term.replace("-", " ").replace("_", " ")
                    key = "%s/%s"%(POS,term.split("#")[0])
                    sent_scores[key].append((float(PosScore), float(NegScore)))
        for key, value in sent_scores.iteritems():
            sent_scores[key] = np.mean(value, axis=0)

        return sent_scores

    def loadSuspects(self):
        #todo load suspects from directory
        onlyfiles = [ f for f in listdir(const.classified_sms_dir) if isfile(join(const.classified_sms_dir,f)) ]
        #print(onlyfiles)
        #delete wierd mac .DS_Store file
        #print onlyfiles[1:]
        #onlyfiles = onlyfiles.dremove('.DS_Store')

        return onlyfiles[1:]

#loadDataSemeval = LoadDataSet()
#ddd = LoadDataSet()
#ddd.load_norm_lexicon()
def load_norm_lexicon():
    norm_dict = {}

    with open(const.norm_lexicon, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for line in reader:
                #print line
            norm_dict[line[0]] = line[1]
    return norm_dict

NormalizationDict = load_norm_lexicon()

def normalise_tweets(tweet):
    return_tweet = tweet
    # print "norm method called"
    for word in return_tweet.split():
        if word in NormalizationDict:
            #print 'Normalizing word: ', word , ' to ', normalization_dict[word]
            #persist_norm_terms(word + "->" + normalization_dict[word] + "\n") #used to persist normalised terms to file
            return_tweet.replace(word, NormalizationDict[word])
    return return_tweet