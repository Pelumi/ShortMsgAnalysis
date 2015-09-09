__author__ = 'Pelumi'

import os
import sys

from nltk.stem.snowball import SnowballStemmer
from sklearn import svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.learning_curve import learning_curve
from vectorizers import SemanticVectorizer as vectorizers

import csv
from textblob import TextBlob
import pandas
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from util.const import emoticons
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.grid_search import GridSearchCV
from util.const import normalization_lexicon
from util.const import stops
from util.const import negative_test_set, positive_test_set, neutral_test_set
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import unicodedata

TRAINING_DF_PICKLED = "../data/dataframes/dataframe.pkl"
TEST_DF_PICKLED = "../data/dataframes/test_dataframe.pkl"
RAW_TRAINING_DATA = "../data/cleanedBFinal.txt"
CHART_DIR = os.path.join("..", "charts")


LABEL_NAME = "label"
INSTANCES = "tweet"
#porterStemmer = PorterStemmer()
snowballStemmer = SnowballStemmer("english")

def load_norm_lexicon():
    norm_dict = {}
    print 'Loading Normalization dictionary...'
    with open(normalization_lexicon, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for line in reader:
            norm_dict[line[0]] = line[1]
    print 'Normalization dictionary loaded...'
    return norm_dict

def remove_accents(input_str):
    input_str = input_str.decode('utf-8', 'ignore')
    nkfd_form = unicodedata.normalize('NFKD', input_str)
    only_ascii = nkfd_form.encode('ASCII', 'ignore')
    return only_ascii

def describe_data(messages, label_describe=False, length_describe=False):
    if label_describe:
        print '============Messages grouped by label============='
        print messages.groupby('label').describe()

    if length_describe:
        # add length coloumn to messages
        messages['length'] = messages['tweet'].map(lambda text: len(text))
        #print the head of messages with the length coloumn added
        #print '============Description of the length of messages============='
        #print messages.length.describe()

    # print messages.head()
    print messages[10:30]
    # print messages.hist(column='length', by='label', bins=50)
    #messages.length.plot(bins=20, kind='hist')

def load_training_data(load=True):
    if load:
        df = pandas.read_pickle(TRAINING_DF_PICKLED)
        print 'Training data frame loaded from disk'
    else:
        df = pandas.read_csv(RAW_TRAINING_DATA, skipinitialspace=True, sep="\t", quoting=csv.QUOTE_NONE, names=["label", "tweet"])
        df = df.drop_duplicates()
        df.ix[df.label == 'neutral ', 'label'] = 'neutral'
        df.save(TRAINING_DF_PICKLED)
        print "Data frame created from csv and saved"
    return df

def preprocess():
    df = load_training_data(True)
    # print df.groupby('label').describe()
    df_tweets = df.tweet.apply(preprcessing_task)
    # df_tweets = df_tweets['tweet'].tail().apply(split_into_lemmas)
    #  df_tweets = df_tweets.tail().apply(split_into_tokens)
    print df_tweets.describe()
    #print df_tweets
    tweet_vectors = vectorise(df_tweets)

    classifier(tweet_vectors, df.label)
    experiemnt(df_tweets, df.label)
    #tune_params(df_tweets, df.label)


def classifier(vectors, labels):
    sentiment_detector = MultinomialNB().fit(vectors, labels)
    print 'predicted:', sentiment_detector.predict(vectors[0])
    print 'expected:', labels[0]


def persist_norm_terms(norm_term):
    with open("../data/normalised_terms.txt", "a") as normFile:
        normFile.write(norm_term)


def preprcessing_task(text):
    # print 'preprocessing task called'
    text = replace_emo_links_mentions(text)
    text = normalise_tweets(text)
    text = remove_accents(text)
    text = stem_tweets(text)

    #text = split_into_lemmas(text)
    #text = split_into_tokens(text)
    return text

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    #to have a figure object, this can be done figure = plt.figure() then the figure object can be referenced subsequently
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='f1_weighted')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.get_current_fig_manager().window.raise_()

    plt.show()
    return plt

def confusion_matrix(labels, predictions):
    plt.matshow(confusion_matrix(labels, predictions), cmap=plt.cm.binary, interpolation='nearest')
    plt.title('confusion matrix')
    plt.colorbar()
    plt.ylabel('expected label')
    plt.xlabel('predicted label')

    print classification_report(labels, predictions)

def normalise_tweets(tweet):
    return_tweet = tweet
    # print "norm method called"
    for word in return_tweet.split():
        if word in normalization_dict:
            #print 'Normalizing word: ', word , ' to ', normalization_dict[word]
            #persist_norm_terms(word + "->" + normalization_dict[word] + "\n") #used to persist normalised terms to file
            return_tweet.replace(word, normalization_dict[word])
    return return_tweet

def stem_tweets(tweet):
    tweetBlob = TextBlob(tweet)
    #tweetBlob = tweetBlob.correct()
    words = tweetBlob.words
    words = [snowballStemmer.stem(word) for word in words] #snowball stemmer was better than porter
    return " ".join(words)

def split_into_tokens(text):
    #print 'split into tokens called'
    message = unicode(text, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words


def split_into_lemmas(message):
    # print 'split into lemmas called'
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]


def experiemnt(tweets, labels):
    tweet_train, tweet_test, label_train, label_test = train_test_split(tweets, labels, test_size=0.2)
    clf = MultinomialNB()
    pipeline = configure_pipeline(clf)

    scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                             tweet_train,  # training data
                             label_train,  # training labels
                             cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                             scoring='f1',  # which scoring metric?
                             n_jobs=-1,)  # -1 = use all cores = faster
    print scores
    print scores.mean(), scores.std()


def tune_params(tweets, labels, output_file="tuned_params.txt"):
    # Vectoriser parameters
    params = {
            'features__vectorizer__counts__ngram_range': [(1,3), (1,2)],  # n-grams are subsequences of "tokens"
            'features__vectorizer__counts__analyzer': ['word', split_into_lemmas, split_into_tokens],  # words are our tokens
            'features__vectorizer__counts__min_df': [1, 2, 3],  # n-grams need to appear in at least this many documents in the dataset 'features__vectorizer__counts__min_df': [8, 9, 10],
             'features__vectorizer__counts__max_df': [2900, 2950, 3000, 2950],
             'features__vectorizer__counts__lowercase': [False],
             'features__vectorizer__counts__preprocessor': [preprcessing_task],
             'features__vectorizer__counts__stop_words': [stops],

             #tfidf trans
             'features__vectorizer__tf_idf__norm': ['l1'],
             'features__vectorizer__tf_idf__use_idf': [False],
             'features__vectorizer__tf_idf__smooth_idf': [False],
            # Classifier parameters
            'classifier__C': [1.44, 1.46, 1.48, 1.458],  # See Support Vector Machines information
            'classifier__penalty': ['l1', 'l2'],
            'classifier__tol': [0.6e-3, 0.5e-4, 0.7e-3, 0.8e-3],
            'classifier__class_weight': ['auto'],

           }
    tweet_train, tweet_test, label_train, label_test = train_test_split(tweets, labels, test_size=0.2)
    pipeline = configure_pipeline(max_ent)

    grid = GridSearchCV(pipeline,  # pipeline from above
                        params,  # parameters to tune via cross validation
                        refit=True,  # fit using all available data at the end, on the best found param combination
                        n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
                        scoring='f1_weighted',  # what score are we optimizing?
                        cv=StratifiedKFold(label_train, n_folds=5), )
    sentiment_classifier = grid.fit(tweets, labels)
    print sentiment_classifier.grid_scores_
    with open(output_file, "w") as scores_grid:
        scores_grid.write(sentiment_classifier.grid_scores_)


# vectorise tweets with tfidf transformer and vectorise using a bag of words approach
def vectorise(tweets):
    bow_transformer = CountVectorizer(analyzer=split_into_lemmas).fit(tweets)
    messages_bow = bow_transformer.transform(tweets)
    tfidf_transformer = TfidfTransformer().fit(messages_bow)
    messages_tfidf = tfidf_transformer.transform(messages_bow)
    print "TFIDF Vectoriser shape: ", messages_tfidf.shape
    return messages_tfidf

def configure_pipeline(clf):
    pipeline = Pipeline([
    ('features', FeatureUnion([
                                  ('vectorizer', Pipeline([
                                      ('counts', CountVectorizer(binary=True, strip_accents=True, preprocessor=preprcessing_task, min_df=0.0004, ngram_range=(1,2), max_df=0.69, lowercase=False, stop_words='english')), #2950maxdf, noLowercase, stopwords, minDF=9
                                      #min_df=0.0004 is the best, ('counts', CountVectorizer(preprocessor=preprcessing_task, ngram_range=(1, 2), min_df=9, max_df=2950, lowercase=False, stop_words=stops, decode_error='replace')),
                                      #mindf > 12 was poor, 9 was the best, 2950 was the best for max features
                                      ('tf_idf', TfidfTransformer())  #its good to use idf, always set to true
                                  ])),
                                  ('semantic_feats', vectorizers.SemanticVectorizer()),
                              #    ('book_feats', PyBVectorizer())
                              ],)),
    ('classifier', clf)
])
    return pipeline

def replace_emo_links_mentions(text):
    #replace emos with placeholders
    for j in emoticons.iterkeys():
        text = text.replace(j, emoticons[j])
    text = re.sub('((www\.[\s])|(https?://[^\s]))', ' ', text)
    #text = re.sub('((www\.[\s])|(https?://[^\s]))','URL_PHOLDA',text)
    #Convert @username to AT_USER
    #replace mentions
    text = re.sub('@([A-Za-z0-9_]+)', ' ', text)
    text = text.replace('RT', ' ')
    text = text.replace('RT,', ' ')
    text = text.replace('&', ' and ')
    text = text.replace('A14', ' you ')
    return text.strip()

def plot_decision_boundary(logreg, X, Y):
    #logreg.fit(X, Y)
    h = .02  # step size in the mesh

    # Plot the decision boundary. For that, we will asign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    #plt.savefig(os.path.join(CHART_DIR, "pr_%s_%s.png"%("filename", "phase")), bbox_inches="tight")
    plt.show()

def read_file_as_list(filename):
    with open(filename) as f:
        content_as_list = f.readlines()
    return content_as_list

def load_test_set(load=True):
    if load:
        test_dataframe = pandas.read_pickle(TEST_DF_PICKLED)
        print 'Test data frame loaded from disk'
    else:
        positive_test_instances = read_file_as_list(positive_test_set)
        negative_test_instances = read_file_as_list(negative_test_set)
        neutral_test_instances = read_file_as_list(neutral_test_set)

        #add positive instances to data frame
        test_data = {'tweet': positive_test_instances}
        test_dataframe = pandas.DataFrame(test_data, columns=['tweet'])
        test_dataframe['label'] = 'positive'
        test_dataframe = test_dataframe.drop_duplicates()

        #add negative instances to data frame
        test_data_neg = {'tweet': negative_test_instances}
        test_dataframe_neg = pandas.DataFrame(test_data_neg, columns=['tweet'])
        test_dataframe_neg['label'] = 'negative'
        test_dataframe_neg = test_dataframe_neg.drop_duplicates()

        #add neutral instances to data frame
        test_data_neut = {'tweet': neutral_test_instances}
        test_dataframe_neut = pandas.DataFrame(test_data_neut, columns=['tweet'])
        test_dataframe_neut['label'] = 'neutral'
        test_dataframe_neut = test_dataframe_neut.drop_duplicates()

        #append negative and neutral to positive df
        test_dataframe = test_dataframe.append(test_dataframe_neg)
        test_dataframe = test_dataframe.append(test_dataframe_neut)
        test_dataframe.to_pickle(TEST_DF_PICKLED)

    return test_dataframe

#load dataframe and normalization lexicon
training_data = load_training_data(load=True)
normalization_dict = load_norm_lexicon()
test_data = load_test_set(load=False)
#print test_data.describe
test_data[INSTANCES] = test_data[INSTANCES].str.strip()
test_data[INSTANCES] = test_data[INSTANCES].apply(remove_accents)
#test_data[INSTANCES] = test_data[INSTANCES]

#persist_test_POS(test_data[INSTANCES].as_matrix())
#sys.exit()
#describe_data(test_data, label_describe=True, length_describe=True)

clf = MultinomialNB(alpha=0.01)
max_ent = LogisticRegression(fit_intercept=False, class_weight='auto', penalty='l1', tol=0.6e-3, C=1.5, )  # auto was great, penalty l1 limped to 69 e-3 was better with 0.6, c=1.46 was awesome
svmClassifier = svm.SVC(kernel='linear', C=1000, gamma=0.001)

pipeline = configure_pipeline(max_ent)

# #testing the classifier with SMS dataset
sentiment_clf = pipeline.fit(training_data[INSTANCES], training_data[LABEL_NAME])
# #print sentiment_clf.get_params()
predicted = pipeline.predict(test_data[INSTANCES])
print np.mean(predicted == test_data[LABEL_NAME])
print(metrics.classification_report(test_data[LABEL_NAME], predicted, digits=6))
sys.exit(0)
#test_pipeline = vectorization_pipeline()
#plot_decision_boundary(pipeline, training_data[INSTANCES][:100], training_data[LABEL_NAME][:100])
#plot_learning_curve(pipeline, "accuracy vs. training set size", training_data[INSTANCES][:100], training_data[LABEL_NAME][:100], cv=5)

#tune_params(training_data[INSTANCES][:100], training_data[LABEL_NAME][:100], "100_instaces.txt")

#sentiment_clf = pipeline.fit(training_data[INSTANCES], training_data[LABEL_NAME])

#ten fold cross validation
#scores = cross_val_score(pipeline, training_data[INSTANCES], training_data[LABEL_NAME], cv=10, scoring='f1_weighted', n_jobs=-1, )
scores = cross_val_score(pipeline, training_data[INSTANCES], training_data[LABEL_NAME], cv=10, n_jobs=-1, )
print scores
print scores.mean(), scores.std()
