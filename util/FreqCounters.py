__author__ = 'Pelumi'
import re

from util.const import *


#get number of hashtags, mentions and upper case
def getCounts(text_tokens):
    upperCount = 0;
    hashCount = 0;
    mentionCount = 0
    for word in text_tokens:
        if(word.isupper()):
            upperCount+=1
        if(word.startswith("#")):
            hashCount+=1
        if(word.startswith("@")):
            mentionCount+=1
    return upperCount, hashCount, mentionCount

#determine number of positive or negative emoticons used and if last term is an emoticon
def emoticonCount(textTokens):
    positiveEmoCount = 0
    negativeEmoCount = 0
    lastTermIsEmo = 0
    #print "index is ", len(textTokens) - 1
    #print textTokens
    if textTokens[len(textTokens) - 1] in emoticonsPolarity:
        lastTermIsEmo = 1
    for word in textTokens:
        if word in emoticonsPolarity:
            if emoticonsPolarity[word]==1:
                positiveEmoCount+=1
            else:
                negativeEmoCount+=1

    return positiveEmoCount, negativeEmoCount, lastTermIsEmo

#determine number of repeated punctuation sequence and if punctuation is last term
def punctuationCount(textTokens):
    punctSeqCount = 0
    punctseqcount = 0
    lastTermIsPunc = 0
    count = 0

    if textTokens[len(textTokens) - 1] in punctuations:
        lastTermIsPunc = 1
    for word in textTokens:
        count += 1
        if word in punctuations:
            punctseqcount += 1
            if count==len(textTokens) and punctseqcount >1:
                punctSeqCount += 1
            continue
        else:
            if punctseqcount > 1:
                punctSeqCount += 1
            punctseqcount =0
    return lastTermIsPunc, punctSeqCount

#count number of elongated terms, normalise to a max of len 3 then normalise urls and handles
def sequenceCounter(text):
    #find all elongated strings, count and reduce max length to 3
    reps = re.findall(r'((\w)\2{2,})', text)
    #ensure characters are not repeated more than 3 times
    for i in range(0, len(reps), 1):
        text = text.replace(reps[i][0], reps[i][1]+reps[i][1]+reps[i][1] , 1);

    #Convert www.* or https?://* to URL
    text = re.sub('((www\.[\s]+)|(https?://[^\s]+))','http://URL_PHOLDA',text)
    #Convert @username to AT_USER
    text = re.sub('@[^\s]+','@UserR_PHOLDA',text)

    return len(reps), text


#identify negated contexts, count and mark affected words with _NEG
def getNegations(textTokens):
    newTextToxens = []
    negRegEx = '(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n\'t'
    puncRegEx = '^[.:;!?]$'
    pRE = re.compile(puncRegEx)
    nRE = re.compile(negRegEx)
    negSequenceActive = False
    negContext =0
    for word in textTokens:
        newWord = word
        if negSequenceActive:
            if pRE.match(word):
                negSequenceActive = False
                newWord = word
            else:
                newWord = word + "_NEG"

        if nRE.search(word):
             negContext+=1
             negSequenceActive = True
             newWord = word
        newTextToxens.append(newWord)
    return newTextToxens, negContext