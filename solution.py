from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
import time
import re

from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import io
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords as sw
from scipy.sparse import *

TOTREVIEWS = 41077
DEVREVIEWS = 28754
EVALREVIEWS = 12323


"""  |   =======================================================================================   |
     |   =======================================================================================   |
     |   =======================================================================================   |
     |   -------------------------------------- FUNCTIONS --------------------------------------   |
     |   =======================================================================================   |
     |   =======================================================================================   |
     V   =======================================================================================   V
"""


"""  |   =======================================================================================   |
     |   -------------------------------- ALWAYS USED FUNCTIONS --------------------------------   |
     V   =======================================================================================   V
"""


# prints how many positive and how many negative reviews are in the development dataset
def get(labels):
    p = 0
    n = 0
    for l in labels:
        if l == 'pos':
            p = p + 1
        else:
            n = n + 1
    print(p)
    print(n)


# reads development file and return list of reviews
def readDevfile(fname="../development.csv"):

    reviews = []
    labels = []
    counter = 0
    tmpReview = ""
    header = True

    pos = re.compile(".*,pos\n$")
    neg = re.compile(".*,neg\n$")

    with io.open(fname, "r", encoding="utf8") as opened_file:  # open (use utf-8 due to Emoji
        for line in opened_file:
            if header:
                header = False
                continue
            if re.match(pos, line) is not None:
                tmpReview = tmpReview + line[:-5]
                reviews.append(tmpReview)
                labels.append("pos")
                counter = counter + 1
                tmpReview = ""
            elif re.match(neg, line) is not None:
                tmpReview = tmpReview + line[:-5]
                reviews.append(tmpReview)
                labels.append("neg")
                counter = counter + 1
                tmpReview = ""
            else:
                tmpReview = tmpReview + line

    if counter == len(reviews) == len(labels) == DEVREVIEWS:
        print(f"Successfully read {counter} reviews")
        return reviews, labels
    else:
        return None, None


# read evaluation file and return list of reviews
def readEvalFile(fname="../evaluation.csv"):
    header = True
    counter = 0

    start1 = re.compile("^\"[^\"].*")  # a review can start with " but not followed by another "
    start2 = "\"\n"  # a review start line can simply be a "\n
    start3 = re.compile("^\"\"\"[^\"].*")  # a review start line can start with """ but not followed by another "

    end1 = re.compile(".*[^\"]\"\n$")  # a line can end with "\n but not preceded by another "
    end2 = "\"\n"  # or it can be: "\n     only that on a line
    end3 = re.compile(".*[^\"]\"\"\"\n$")  # a line can end with """\n but not preceded by another "

    status = 0  # --->          # 0. look for a start of a review with start1, start2 or start3
    tmpReview = ""  # 1. look for an end of a review with a end1, end2, end3

    targets = []

    with io.open(fname, "r", encoding="utf8") as opened_file:  # open (use utf-8 due to Emoji)
        for line in opened_file:

            counter = counter + 1
            # print(f"---line {counter}")
            # print("---"+line)

            if header:
                header = False
                continue

            if status == 0:
                if re.match(start1, line) is not None or re.match(start3, line) is not None or line == start2:
                    # found review starting regularly with some quotes

                    # check if it also ends on same line
                    if re.match(end1, line) is not None or re.match(end3, line):
                        # entire review is on a line
                        tmpReview = line
                        targets.append(tmpReview)
                        tmpReview = ""
                        # print("0A: " + str(status) + " staying 0")
                        continue
                        # status doesn't change because we want another ordinary review
                    else:
                        # start on this line, but ends later
                        tmpReview = tmpReview + line
                        # print("0B: " + str(status) + " going to 1")
                        status = 1
                        continue

                else:
                    # we were looking for an ordinary review, but it's not
                    tmpReview = tmpReview + line
                    targets.append(tmpReview)
                    tmpReview = ""
                    # print("0C: " + str(status) + " going to 0")
                    # status = 2
                    status = 0
                    continue

            elif status == 1:
                # we found the start of an ordinary review and we want its ordinary end
                if re.match(end1, line) is not None or re.match(end3, line) is not None or line == end2:
                    # ordinary end found
                    tmpReview = tmpReview + line
                    targets.append(tmpReview)
                    tmpReview = ""
                    # print("1A: " + str(status) + " going to 0")
                    status = 0
                    continue
                else:
                    # ordinary end not found yet
                    tmpReview = tmpReview + line
                    # print("1B: " + str(status) + " staying on 1")
                    continue

    if TOTREVIEWS - DEVREVIEWS == len(targets):
        print(f"Successfully read {EVALREVIEWS} reviews")
        return targets
    else:
        return None


# each review is tokenized                     --> tokens
# each token is split according to a RegEx     --> words
def tokenizing(reviews):

    size = len(reviews)

    tokenizedReviews = []
    counter = 0

    for rev in reviews:
        tokenized_rev = word_tokenize(rev, language="italian")
        new_rev = []
        for token in tokenized_rev:
            s = re.compile(",|\.|'|!|`|\(|\)|/|:|;|&|%|\"|=|\?|@|\^|“|”|\\|_|…").split(token)
            for word in s:
                new_rev.append(word.strip().lower())

        tokenizedReviews.append(new_rev)
        counter = counter + 1
        print("\rComputing... " + str(int((counter / size) * 100)) + "%", end="", flush=True)

    if DEVREVIEWS == len(tokenizedReviews) or EVALREVIEWS == len(tokenizedReviews):
        print()
        print(f"Successfully tokenized {len(tokenizedReviews)} reviews")
        print()
        return tokenizedReviews
    else:
        return None


# starting from the reviews in the shape of list of words
# --> remove all the words that are strictly shorter than given threshold
def clean(tokenizedReviews, threshold=2):
    # remove the words with len<=2 (default)

    size = len(tokenizedReviews)

    counter = 0
    wordcount = 0
    remcount = 0

    for rev in tokenizedReviews:
        for word in rev:
            wordcount = wordcount + 1
            if len(word) <= threshold:
                remcount = remcount + 1
                rev.remove(word)
        counter = counter + 1
        print("\rComputing... " + str(int((counter / size) * 100)) + "%", end="", flush=True)

    print()
    print(f"Scanned {wordcount} words, removed {remcount}, remaining: {wordcount-remcount}")
    print()
    return tokenizedReviews


# each word is stemmed and a new version of the review is created   --> list of stemmed words
def stemming(reviews):

    size = len(reviews)

    stemmedReviews = []

    stemmer = SnowballStemmer("italian")
    counter = 0

    for rev in reviews:
        new_rev = []
        for word in rev:
            word = stemmer.stem(word)
            new_rev.append(word)

        stemmedReviews.append(new_rev)
        counter = counter + 1
        print("\rComputing... " + str(int((counter / size) * 100)) + "%", end="", flush=True)

    if DEVREVIEWS == len(stemmedReviews) or EVALREVIEWS == len(stemmedReviews):
        print()
        print(f"Successfully stemmed {len(stemmedReviews)} reviews")
        print()
        return stemmedReviews
    else:
        return None


# generate the list of stopwords and stem them:
#   - most frequent words from frequency document (with freq higher than threshold)
#   - nltk italian stopwords
#   - some symbols (that should not be among the words)
#   - from the first two lists remove some meaningful words
def getStopwordlist(filename="stopwordsStemImproved.txt", threshold=15000):

    stpw = sw.words("italian")

    toKeep = ['quanta', 'avrò', 'aveste', 'avessero', 'sareste', 'fui', 'fareste', 'farebbero', 'feci', 'facemmo',
              'facessero', 'stiano', 'staremmo', 'stessero', 'con', 'contro', 'perché', 'non', 'più', 'piu']
    toKeep = ['quanta', 'avrò', 'avessero', 'farebbero', 'feci', 'facessero', 'stiano', 'stessero', 'non']  # best
    for w in toKeep:
        if w in stpw:
            stpw.remove(w)

    symb = [' ', ',', '.', "'", "!", "`", "(", ")", "/", ":", ";", "&", "%", '"', "=", "?", "@", "^", "“", "”", "\\",
            "_", "…", "-", "[", "]", "{", "}"]
    for w in symb:
        stpw.append(w)

    with io.open(filename, "r", encoding="utf8") as opened_file:       # open (use utf-8 due to Emoji
        # skipped = 0
        for line in opened_file:
            pcs = line.split()
            if len(pcs) != 2:            # skip lines not well formatted
                # skipped = skipped + 1
                continue
            if int(pcs[1]) > threshold:  # not including here the ones with frequency=1, removed by tfidf vectorizer
                if len(pcs[0]) > 2:      # with length 2 or less are already removed in main()
                    stpw.append(pcs[0])
        # print(f"Skipped {skipped} lines")

    toKeep = ['non', 'ottim', 'buon', 'posizion', 'pul', 'serviz', 'dispon', 'bell', 'ben']  # fill it! (11000)
    for w in toKeep:
        if w in stpw:
            stpw.remove(w)

    return stemStopwords(stpw)


# stem the stop words in order to have them matching the stems in the reviews (that are stemmed in the same way)
def stemStopwords(stopwords):

    size = len(stopwords)

    stemmed = []
    stemmer = SnowballStemmer("italian")
    for w in stopwords:
        stemmed.append(stemmer.stem(w))
    print("lost stopwords: " + str(size-len(stemmed)))

    return stemmed


# rebuild the reviews
# --> from a list of stemmed words to a single string (for each review)
def assemble(reviews):

    size = len(reviews)

    # rebuild reviews
    processedReviews = []
    counter = 0
    for rev in reviews:
        tmpReview = ""
        for word in rev:
            if word != '':
                tmpReview = tmpReview + word + " "
        processedReviews.append(tmpReview)
        counter = counter + 1
        print("\rComputing... " + str(int((counter / size) * 100)) + "%", end="", flush=True)

    print()
    print(f"Re-assembled {counter} reviwes ({size-counter} missing)")
    print()
    return processedReviews


# explore a range of C values
# get the F1_weighted score from cross validation with 20 folds
def tuneC(tfidf_X, labels):
    print("==========CROSS-VALIDATION START==========")

    print("--- Cross validating for C parameter:")

    start =0.2       # 0.1      # 0.150
    stop = 1         # 10       # 0.6
    step = 0.001     # 1        # 0.001
    cRange = np.arange(start, stop, step)
    minim = np.zeros(np.shape(cRange))
    maxim = np.zeros(np.shape(cRange))
    avg = np.zeros(np.shape(cRange))
    cnt = 0
    for c in cRange:

        svcTest = LinearSVC(dual=False, max_iter=10000, C=c)

        # cross val F1 score weighed
        outcome = cross_val_score(svcTest, tfidf_X, labels, cv=20, scoring='f1_weighted')
        avg[cnt] = np.mean(outcome)
        minim[cnt] = np.min(outcome)
        maxim[cnt] = np.max(outcome)

        cnt = cnt + 1
        print("\rComputing... " + str(int((cnt / ((stop - start) / step)) * 100)) + "%", end="", flush=True)
    print()

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(cRange, minim, c='red', label='min(f1_weighted) of cross validation')
    ax.plot(cRange, maxim, c='green', label='max (f1_weighted) of cross validation')
    ax.plot(cRange, avg, c='blue', label='avg (f1_weighted) of cross validation')

    plt.xlabel('C range')
    plt.ylabel('F1 score')
    plt.title('F1 score varying with C')
    plt.legend(loc="lower right")
    ax.legend()
    plt.show()

    print("==========CROSS-VALIDATION END==========")
    print()
    return


# explore a range of max_iteration values
# get the F1_weighted score from cross validation with 20 folds
def tuneMaxIter(tfidf_X, labels):
    print("==========CROSS-VALIDATION START==========")

    print("--- Cross validating for max_iter parameter:")

    start = 1
    stop = 10
    step = 1
    itRange = np.arange(start, stop, step)
    minim = np.zeros(np.shape(itRange))
    maxim = np.zeros(np.shape(itRange))
    avg = np.zeros(np.shape(itRange))
    cnt = 0
    for it in itRange:
        svcTest = LinearSVC(dual=False, max_iter=it, C=0.3125)
        outcome = cross_val_score(svcTest, tfidf_X, labels, cv=20, scoring='f1_weighted')
        avg[cnt] = np.mean(outcome)
        minim[cnt] = np.min(outcome)
        maxim[cnt] = np.max(outcome)
        cnt = cnt + 1
        print("\rComputing... " + str(int((cnt / ((stop - start) / step)) * 100)) + "%", end="", flush=True)
    print()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(itRange, minim, c='red', label='min')
    ax.plot(itRange, maxim, c='green', label='max')
    ax.plot(itRange, avg, c='blue', label='avg')
    plt.xlabel('max_iterations')
    plt.ylabel('F1 score')
    plt.title('F1 score varying with max_iterations')
    plt.legend(loc="lower right")
    ax.legend()
    plt.show()

    print("==========CROSS-VALIDATION END==========")
    print()
    return


# print the ROCs of the two classes 'pos' and 'neg' only
def generateROC(tfidf_X, labels):
    print("==========ROC CURVE START==========")

    print("--- Printing ROC curve:")

    y = label_binarize(labels, classes=['pos', 'neg'])
    # print(y)

    x_train, x_test, y_train, y_test = train_test_split(tfidf_X, y, test_size=0.05)

    svcTest = LinearSVC(dual=False, max_iter=10000, C=0.3125)

    svcTest.fit(x_train, y_train)
    y_score = svcTest.predict(x_test)  # y_test = ground truth   |   y_score = predicted

    y_test_shped = np.zeros((np.size(y_test), 2), dtype=int)
    counter = 0
    for lab in y_test:
        if lab == 0:
            y_test_shped[counter, 0] = 1
        else:
            y_test_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_test_shped)

    y_score_shped = np.zeros((np.size(y_score), 2), dtype=int)
    counter = 0
    for lab in y_score:
        if lab == 0:
            y_score_shped[counter, 0] = 1
        else:
            y_score_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_score_shped)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):  # 2 classes
        fpr[i], tpr[i], _ = roc_curve(y_test_shped[:, i], y_score_shped[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_shped.ravel(), y_score_shped.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    colors = ['darkorange', 'red']
    classes = ['pos', 'neg']
    for i, color in zip(range(2), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()

    print("==========ROC CURVE END==========")
    print()
    return


# print the confusion matrix only
def generateConfMatrix(tfidf_X, labels):
    print("==========CONFUSION MATRIX START==========")

    print("--- Generating confusion matrix:")

    y = label_binarize(labels, classes=['pos', 'neg'])
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(tfidf_X, y, test_size=0.15)

    svcTest = LinearSVC(dual=False, max_iter=10000, C=0.3125)

    svcTest.fit(x_train, y_train)

    disp = plot_confusion_matrix(svcTest, x_test, y_test, normalize=None, display_labels=['pos', 'neg'])
    disp.ax_.set_title("Confusion matrix not normalized")

    print("--- Confusion matrix not normalized:")
    print(disp.confusion_matrix)

    plt.show()

    print("==========CONFUSION MATRIX END==========")
    print()
    return


# prints ROCs for the two classes and confusion matrix (on same data)
def generateConfMatrixAndROC(tfidf_X, labels):
    print("==========ROC CURVE AND CONFUSION MATRIX START==========")

    y = label_binarize(labels, classes=['pos', 'neg'])
    # print(y)

    x_train, x_test, y_train, y_test = train_test_split(tfidf_X, y, test_size=0.05)

    svcTest = LinearSVC(dual=False, max_iter=10000, C=0.3125)

    svcTest.fit(x_train, y_train)
    y_score = svcTest.predict(x_test)  # y_test = ground truth   |   y_score = predicted

    disp = plot_confusion_matrix(svcTest, x_test, y_test, normalize=None, display_labels=['pos', 'neg'])
    disp.ax_.set_title("Confusion matrix not normalized")

    print("--- Confusion matrix not normalized:")
    print(disp.confusion_matrix)
    print()

    plt.show()

    print("--- Printing ROC curve")

    y_test_shped = np.zeros((np.size(y_test), 2), dtype=int)
    counter = 0
    for lab in y_test:
        if lab == 0:
            y_test_shped[counter, 0] = 1
        else:
            y_test_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_test_shped)

    y_score_shped = np.zeros((np.size(y_score), 2), dtype=int)
    counter = 0
    for lab in y_score:
        if lab == 0:
            y_score_shped[counter, 0] = 1
        else:
            y_score_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_score_shped)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):  # 2 classes
        fpr[i], tpr[i], _ = roc_curve(y_test_shped[:, i], y_score_shped[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_shped.ravel(), y_score_shped.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    colors = ['darkorange', 'red']
    classes = ['pos', 'neg']
    for i, color in zip(range(2), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) curve')
    plt.legend(loc="lower right")
    plt.show()
    print("done")

    print("==========ROC CURVE AND CONFUSION MATRIX END==========")
    print()
    return


# prints the chart with the comparison of many ROCs of different classifiers
def compareClassifiers(tfidf_X, labels):
    print("==========MODEL COMPARISON START==========")

    print("--- Printing ROC curves:")

    y = label_binarize(labels, classes=['pos', 'neg'])
    # print(y)

    x_train, x_test, y_train, y_test = train_test_split(tfidf_X, y, test_size=0.05)

    svcTest = LinearSVC(dual=False, max_iter=10000, C=0.3125)
    randForest = RandomForestClassifier(n_jobs=5)
    ruleBased = DummyClassifier(strategy='stratified')
    knn = KNeighborsClassifier()
    cnb = MultinomialNB()
    decTree = DecisionTreeClassifier()

    # 1. SVC
    svcTest.fit(x_train, y_train)
    y_score = svcTest.predict(x_test)  # y_test = ground truth   |   y_score = predicted

    y_test_shped = np.zeros((np.size(y_test), 2), dtype=int)
    counter = 0
    for lab in y_test:
        if lab == 0:
            y_test_shped[counter, 0] = 1
        else:
            y_test_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_test_shped)

    y_score_shped = np.zeros((np.size(y_score), 2), dtype=int)
    counter = 0
    for lab in y_score:
        if lab == 0:
            y_score_shped[counter, 0] = 1
        else:
            y_score_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_score_shped)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr[0], tpr[0], _ = roc_curve(y_test_shped[:, 0], y_score_shped[:, 0])
    roc_auc[0] = auc(fpr[0], tpr[0])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_shped.ravel(), y_score_shped.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    color = 'darkorange'
    plt.plot(fpr[0], tpr[0], color=color, lw=lw, label='ROC curve SVC (area = {0:0.2f})'.format(roc_auc[0]))

    # 2. Random forest
    randForest.fit(x_train, y_train)
    y_score = randForest.predict(x_test)

    y_test_shped = np.zeros((np.size(y_test), 2), dtype=int)
    counter = 0
    for lab in y_test:
        if lab == 0:
            y_test_shped[counter, 0] = 1
        else:
            y_test_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_test_shped)

    y_score_shped = np.zeros((np.size(y_score), 2), dtype=int)
    counter = 0
    for lab in y_score:
        if lab == 0:
            y_score_shped[counter, 0] = 1
        else:
            y_score_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_score_shped)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr[0], tpr[0], _ = roc_curve(y_test_shped[:, 0], y_score_shped[:, 0])
    roc_auc[0] = auc(fpr[0], tpr[0])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_shped.ravel(), y_score_shped.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    color = 'red'
    plt.plot(fpr[0], tpr[0], color=color, lw=lw, label='ROC curve of Random Forest (area = {0:0.2f})'.format(roc_auc[0]))

    # 3. Rule based classifier
    decTree.fit(x_train, y_train)
    y_score = decTree.predict(x_test)

    y_test_shped = np.zeros((np.size(y_test), 2), dtype=int)
    counter = 0
    for lab in y_test:
        if lab == 0:
            y_test_shped[counter, 0] = 1
        else:
            y_test_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_test_shped)

    y_score_shped = np.zeros((np.size(y_score), 2), dtype=int)
    counter = 0
    for lab in y_score:
        if lab == 0:
            y_score_shped[counter, 0] = 1
        else:
            y_score_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_score_shped)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr[0], tpr[0], _ = roc_curve(y_test_shped[:, 0], y_score_shped[:, 0])
    roc_auc[0] = auc(fpr[0], tpr[0])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_shped.ravel(), y_score_shped.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    color = 'green'
    plt.plot(fpr[0], tpr[0], color=color, lw=lw, label='ROC curve of Decision tree classifier (area = {0:0.2f})'.format(roc_auc[0]))

    # 4. K-NN
    knn.fit(x_train, y_train)
    y_score = knn.predict(x_test)

    y_test_shped = np.zeros((np.size(y_test), 2), dtype=int)
    counter = 0
    for lab in y_test:
        if lab == 0:
            y_test_shped[counter, 0] = 1
        else:
            y_test_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_test_shped)

    y_score_shped = np.zeros((np.size(y_score), 2), dtype=int)
    counter = 0
    for lab in y_score:
        if lab == 0:
            y_score_shped[counter, 0] = 1
        else:
            y_score_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_score_shped)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr[0], tpr[0], _ = roc_curve(y_test_shped[:, 0], y_score_shped[:, 0])
    roc_auc[0] = auc(fpr[0], tpr[0])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_shped.ravel(), y_score_shped.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    color = 'blue'
    plt.plot(fpr[0], tpr[0], color=color, lw=lw, label='ROC curve of K-NN (area = {0:0.2f})'.format(roc_auc[0]))

    # 5. Naive Bayes
    cnb.fit(x_train, y_train)
    y_score = cnb.predict(x_test)

    y_test_shped = np.zeros((np.size(y_test), 2), dtype=int)
    counter = 0
    for lab in y_test:
        if lab == 0:
            y_test_shped[counter, 0] = 1
        else:
            y_test_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_test_shped)

    y_score_shped = np.zeros((np.size(y_score), 2), dtype=int)
    counter = 0
    for lab in y_score:
        if lab == 0:
            y_score_shped[counter, 0] = 1
        else:
            y_score_shped[counter, 1] = 1
        counter = counter + 1
    # print(y_score_shped)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr[0], tpr[0], _ = roc_curve(y_test_shped[:, 0], y_score_shped[:, 0])
    roc_auc[0] = auc(fpr[0], tpr[0])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_shped.ravel(), y_score_shped.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    lw = 2
    color = 'darkturquoise'
    plt.plot(fpr[0], tpr[0], color=color, lw=lw, label='ROC curve of Naive Bayes (area = {0:0.2f})'.format(roc_auc[0]))

    # conclude plot
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Classifiers comparison (ROC curves)')
    plt.legend(loc="lower right")
    plt.show()

    print("==========MODEL COMPARISON END==========")
    print()
    return


"""  |   =======================================================================================   |
     |   ----- NOT USED AT EACH RUN, BUT USED IN DATA EXPLORATION AT DEVELOPMENT BEGINNING -----   |
     V   =======================================================================================   V
"""


# W.I.P. --> decide if use Stemmer, Lemmatizer or an hybrid (right now this is hybrid)
class LemmaTokenizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = SnowballStemmer("italian")

    def __call__(self, document):
        lemmas = []
        re_digit = re.compile("[0-9]")  # regular expression to filter digit, tokens

        for t in word_tokenize(document, language="italian"):
            t = t.strip()
            lemma = self.lemmatizer.lemmatize(t)

            # remove tokens with only punctuation chars and digits
            if lemma not in string.punctuation and len(lemma) > 1 and len(lemma) < 16 and not re_digit.match(lemma):
                lemmas.append(self.stemmer.stem(lemma))

        return lemmas


# Function for the creation of a file with each word and its frequency formatted as:    word freq\n
# it builds it from both the documents
def printStopwordsFile():
    print("==========WORD FREQUENCY FILE CREATION START==========")
    print("---------- {developmentImproved.csv} ----------")
    with io.open("../developmentImproved.csv", "r", encoding="utf8") as opened_file:      # open (use utf-8 due to Emoji
        development = opened_file.read()                                                  # read entire file content

    print("---------- {evaluationImproved.csv} ----------")
    with open("../evaluationImproved.csv", encoding="utf8") as opened_file:               # open (use utf-8 due to Emoji
        evaluation = opened_file.read()                                                   # read entire file content

    method = 1
    # 1: tokenize with nltk.word_tokenizer and use the italian snowball stemmer (seems to be better for now)
    # 2: use the nltk wordnet lemmatizer
    if method == 1:
        # method 1
        allTokens = word_tokenize(development, language="italian")
        tokenDict = {}

        stemmer = SnowballStemmer("italian")

        print("File 1 ...")

        stat = 0

        for t in allTokens:
            t = t.lower().strip()
            s = re.compile(",|\.|'|!|`|\(|\)|/|:|;|&|%|\"|=|\?|@|\^|“|”|\\|_|…").split(t)
            stat = stat + len(s) - 1
            for r in s:
                r = stemmer.stem(r)
                if r in tokenDict.keys():
                    tokenDict[r] = tokenDict[r]+1
                else:
                    tokenDict[r] = 1

        print("File 2 ...")
        allTokens = word_tokenize(evaluation, language="italian")
        for t in allTokens:
            t = t.lower().strip()
            s = re.compile(",|\.|'|!|`|\(|\)|/|:|;|&|%|\"|=|\?|@|\^|“|”|\\|_|…").split(t)
            stat = stat + len(s) - 1
            for r in s:
                r = stemmer.stem(r)
                if r in tokenDict.keys():
                    tokenDict[r] = tokenDict[r] + 1
                else:
                    tokenDict[r] = 1

        print("Finalising ...")
        # Converting into list of tuple
        listTknCnt = [(k, v) for k, v in tokenDict.items()]
        # sorting
        listTknCnt.sort(key=lambda x: x[1], reverse=True)
        # print into a file
        with io.open("stopwordsStemImproved.txt", "w", encoding="utf8") as f:
            for k, v in listTknCnt:
                f.write(str(k) + " " + str(v) + "\n")
        print("Additional splits: "+str(stat))
    else:
        # method 2
        tokenDict = {}

        print("File 1 ...")
        tokenizer = LemmaTokenizer()
        allTokens2 = tokenizer(development)

        stat = 0

        for t in allTokens2:
            t = t.lower().strip()
            s = re.compile(",|\.|'|!|`|\(|\)|/|:|;|&|%|\"|=|\?|@|\^|“|”|\\|_|…").split(t)
            stat = stat + len(s) - 1
            for r in s:
                if r in tokenDict.keys():
                    tokenDict[r] = tokenDict[r] + 1
                else:
                    tokenDict[r] = 1

        print("File 2 ...")
        allTokens2 = tokenizer(evaluation)
        for t in allTokens2:
            t = t.lower().strip()
            s = re.compile(",|\.|'|!|`|\(|\)|/|:|;|&|%|\"|=|\?|@|\^|“|”|\\|_|…").split(t)
            stat = stat + len(s) - 1
            for r in s:
                if r in tokenDict.keys():
                    tokenDict[r] = tokenDict[r] + 1
                else:
                    tokenDict[r] = 1

        print("Finalising ...")
        # Converting into list of tuple
        listTknCnt = [(k, v) for k, v in tokenDict.items()]
        # sorting
        listTknCnt.sort(key=lambda x: x[1], reverse=True)
        # print into a file
        with io.open("stopwordsStemImproved.txt", "w", encoding="utf8") as f:
            for k, v in listTknCnt:
                f.write(str(k) + " " + str(v) + "\n")
        print("Additional splits: " + str(stat))

    print("==========WORD FREQUENCY FILE CREATION START==========")


# function that looks for a substring (lookfor) inside the reviews
# returns how many times it appears in positive and negative reviews
def explore(lookfor, reviews, labels, ratio=False):
    pos = 0
    neg = 0
    c = 0

    for i in reviews:
        if lookfor in i:
            if labels[c] == 'pos':
                pos = pos + i.lower().count(lookfor)
            elif labels[c] == 'neg':
                neg = neg + i.lower().count(lookfor)
            else:
                print("OPS")
        c = c + 1
    if ratio:
        if pos+neg != 0 and (pos/(pos+neg)>=0.7 or pos/(pos+neg)<=0.3):
            print(lookfor + " " + str(pos / (pos + neg)))
            return lookfor
        else:
            return None
    else:
        print("String: "+lookfor)
        print("#pos: "+str(pos))
        print("#neg: "+str(neg))


# function to plot the density of the words starting from the word frequency file (filename)
# on x-axis: the frequency of a word
# on y-axis: how many words there are with that frequency
# start: from which freq start to plot
# end: up to which freq to plot
def plotStopwordsDensityDistribution(start, end, filename="stopwordsStemImproved.txt"):
    fig, ax = plt.subplots(figsize=(7, 5))

    firstTime = True
    x = None
    y = None
    f = open(filename, "r", encoding="utf8")
    for line in f:
        if firstTime:
            x = np.arange(int(line.split(" ")[1]) + 1)
            y = np.zeros(int(line.split(" ")[1]) + 1)
            firstTime = False
        if int(line.split(" ")[1]) <= 0:  # this should not happen
            break
        y[int(line.split(" ")[1])] = y[int(line.split(" ")[1])] + 1
    f.close()

    # ax.plot(x[:10], y[:10])
    ax.plot(x[start:end], y[start:end])
    plt.xlabel('Words frequency')
    plt.ylabel('Amount of words for each frequency value')
    plt.title('Frequency density plot')

    plt.show()


"""  |   =======================================================================================   |
     |   -------------------------------------- DEPRECATED -------------------------------------   |
     V   =======================================================================================   V
"""


def stopwordRemoving(reviews, stopwords):

    size = len(reviews)

    counter = 0
    wordcount = 0
    remcount = 0

    lightenedReviews = []

    for rev in reviews:
        tmpRev = rev
        for word in tmpRev:
            wordcount = wordcount + 1
            if word in stopwords or word == '':
                remcount = remcount + 1
                tmpRev.remove(word)
        lightenedReviews.append(tmpRev)
        counter = counter + 1
        print("\rComputing... " + str(int((counter / size) * 100)) + "%", end="", flush=True)

    print()
    print(f"Scanned {wordcount} words, removed {remcount}, remaining: {wordcount-remcount}")
    print()
    return lightenedReviews


def ngramming(reviews, min=1, max=3):

    size = len(reviews)
    counter = 0

    ngramcnt = 0

    ngrammedReviews = []

    for rev in reviews:
        newRev = []                             # new review that will be assembled as a list of sets
        start = max                             # assumes all the values in range [min,max] (min,max included)
        while start >= min:
            tmp = generateNgrams(rev, start)    # generates the ngrams of that length
            start = start - 1
            newRev = newRev + tmp               # add the list of sets to the new review (tot list of sets of that rev)
        ngrammedReviews.append(newRev)          # new review is complete, add it to the list of reviews
        ngramcnt = ngramcnt + len(newRev)
        counter = counter + 1
        print("\rComputing... " + str(int((counter / size) * 100)) + "%", end="", flush=True)
    print()
    print(f"Successfully created {ngramcnt} n-grams")
    print()
    return ngrammedReviews


def generateNgrams(rev, size):
    # builds a list of sets (list of ngrams of length "size" of review "rev")
    # ngrams = []                             # list of ngrams under construction
    ng = set()

    pos = 0                                 # start position in list of words of rev
    while pos+size-1 < len(rev)-1:          # while max_index_accessed < max_index_of_rev
        cnt = size                          # how many elements to put in the ngram
        tmpList = []                        # temporary list of ngrams to be used to create the ngram as a set
        while cnt > 0:                      # loop until taken #size elements
            tmpList.append(rev[pos+cnt-1])  # take 3rd, 2nd, 1st
            cnt = cnt - 1
        newSet = frozenset(tmpList)         # build the ngram as a set (from the list)
        # ngrams.append(newSet)               # add the ngram to the list of ngrams
        ng.add(newSet)                      # add the ngram to the set of ngrams
        pos = pos + 1
    return list(ng)


'''
# version to work with the ngrams
def genVocabulary(ngrammedRev, labels, ngrammedTar, maxFeatures=20000):

    """
    1. Put ngrams all together (no repetitions, all ngrams from DEVELOPMENT file only)
    2. Assign goodness value to each ngram
    3. Sort by descending fitness
    4. Keep only top maxFeatures
    5. Return vocabulary
    """

    size = len(ngrammedRev)

    vocabulary = dict()

    occ = dict()

    counter = 0

    statistics = 0

    for rev in ngrammedRev:
        for ngr in rev:
            if ngr not in vocabulary.values():
                vocabulary[counter] = ngr
                occ[counter] = 1
                counter = counter + 1
            else:
                for k, v in vocabulary.items():
                    if v == ngr:
                        occ[k] = occ[k] + 1
        statistics = statistics + 1
        print("\rComputing... " + str(int((statistics / size) * 100)) + "%", end="", flush=True)

    max = 0
    n = None
    for k, v in occ.items():
        if v > max:
            max = v
            n = k
    print()
    print(f"There are {counter} unique ngrams")
    print(f"Most frequent: {vocabulary[k]} appears {occ[k]} times ({max})")

    print(vocabulary)
    print(occ)


    return vocabulary
'''


def genVocabulary(ngrammedRev, labels, ngrammedTar, maxFeatures=20000):

    """
    1. Put words all together (no repetitions, all ngrams from DEVELOPMENT file only)
    2. Assign goodness value to each word
    3. Sort by descending fitness
    4. Keep only top maxFeatures
    5. Return vocabulary
    """

    size = len(ngrammedRev)

    vocabulary = dict()

    counter = 0

    statistics = 0

    for rev in ngrammedRev:
        for word in rev:
            if word not in vocabulary and word != '':
                vocabulary[word] = 1
                counter = counter + 1
            elif word != '':
                vocabulary[word] = vocabulary[word] + 1
        statistics = statistics + 1
        print("\rComputing... " + str(int((statistics / size) * 100)) + "%", end="", flush=True)
    print()

    '''
    # Converting into list of tuple
    lst = [(k, v) for k, v in vocabulary.items()]
    # sorting
    lst.sort(key=lambda x: x[1], reverse=True)
    # print into a file
    with io.open("vocabulary.txt", "w", encoding="utf8") as f:
        for k, v in lst:
            f.write(str(k) + " " + str(v) + "\n")
    '''

    counter = 0
    l = []
    for k, v in vocabulary.items():
        if v <= 2:
            l.append(k)

    for el in l:
            vocabulary.pop(el)

    for k, v in vocabulary.items():
        vocabulary[k] = counter
        counter = counter + 1

    return vocabulary


def getCountMatrix(reviews, vocabulary):

    rows = len(reviews)
    cols = len(vocabulary)

    # countMatrix = np.zeros((rows, cols))

    countMatrix = csr_matrix((rows, cols), dtype='float')

    i = 0
    for rev in reviews:
        for word in rev:
            if word in vocabulary:
                countMatrix[i, vocabulary[word]] = countMatrix[i, vocabulary[word]] + 1
        i = i + 1
        print("\rComputing... " + str(int((i / rows) * 100)) + "%", end="", flush=True)
    print()
    return countMatrix


# once I erroneously printed a resoult file with 0,1 instead of pos,neg
# this function does the conversion to the correct format
def debug():
    res = []
    f = True
    with io.open("results.csv", "r", encoding="utf8") as opened_file:  # open (use utf-8 due to Emoji
        for line in opened_file:
            if f:
                f = False
                continue
            print(line)
            print(line.split(',')[0])
            print(line.split(',')[1])
            if int(line.split(',')[1]) == 0:
                res.append('pos')
            else:
                res.append('neg')

    with open("resultsNN.csv", "w") as writeFile:
        writeFile.write("Id,Predicted\n")
        counter = 0
        for v in res:
            writeFile.write(str(counter)+','+str(v)+'\n')
            counter = counter + 1
    writeFile.close()

    return


"""  |   =======================================================================================   |
     |   =======================================================================================   |
     |   =======================================================================================   |
     |   ---------------------------------------- MAIN -----------------------------------------   |
     |   =======================================================================================   |
     |   =======================================================================================   |
     V   =======================================================================================   V
"""


class StemTokenizer(object):
    def __init__(self):
        self.stemmer = SnowballStemmer("italian")

    def __call__(self, document):
        re_digit = re.compile("[0-9]")  # regular expression to filter digit, tokens

        new_tokenized_doc = []
        for t in word_tokenize(document, language="italian"):
            s = re.compile(",|\.|'|!|`|\(|\)|/|:|;|&|%|\"|=|\?|@|\^|“|”|\\|_|…").split(t)
            for word in s:
                word = self.stemmer.stem(word.lower().strip())
                if word not in string.punctuation and len(word) > 2 and len(word) < 16 and not re_digit.match(word):
                    new_tokenized_doc.append(word)
        return new_tokenized_doc


def main():

    print("==========READ DEVELOPMENT FILE START==========")

    reviews, labels = readDevfile(fname="../developmentImproved.csv")
    if reviews is None or labels is None:
        exit(1)

    print("==========READ DEVELOPMENT FILE END==========")
    print()
    print("==========READ EVALUATION FILE START==========")

    targets = readEvalFile(fname="../evaluationImproved.csv")
    if targets is None:
        exit(2)

    print("==========READ EVALUATION FILE END==========")
    print()
    print("==========PREPROCESSING START==========")

    '''
    0. handmade improvements (already done)
    1. review --> split in token list
    2. remove len<=2
    3. stem
    
    --steps 4, 5, 6, 7, 8: removed (too computationally complex --> 9.3hours to run --> score of 93.7)
    4. remove stopwords
    5. generate n-gram (len 3?) as set (reviews now represented as their set of n-grams)
    6. find vocabulary:
        - all n-grams of all reviews (development) together, without duplicates and represented by unique ID
        - pick the top N according to a goodness measure (most unbalanced separation weighted with their frequency)
        - consider weighting also by their presence in the eval set
    7. count matrix computation
    8. skl transform: sklearn.feature_extraction.text.TfidfTransformer (idf=false)
    --
    
    4'. re-build as strings from tokens
    5'. vectorize (some of the skipped step end un in this one)
    '''

    # 1. Tokenizing:
    print("--- Tokenizing:")
    tokenizedReviews = tokenizing(reviews)
    if tokenizedReviews is None:
        exit(3)
    tokenizedTargets = tokenizing(targets)
    if tokenizedTargets is None:
        exit(4)

    # 2. Cleaning
    print("--- Removing short words:")
    cleanedReviews = clean(tokenizedReviews)
    cleanedTargets = clean(tokenizedTargets)

    # 3. Stemming
    print("--- Stemming:")
    stemmedReviews = stemming(cleanedReviews)
    if stemmedReviews is None:
        exit(5)
    stemmedTargets = stemming(cleanedTargets)
    if stemmedTargets is None:
        exit(6)

    # 4. Stop-word generation
    print("--- Generating stop words:")
    stopw = getStopwordlist(threshold=20000)
    print()

    '''
    print("--- Removing stop words:")
    lightenedReviews = stopwordRemoving(stemmedReviews, stopw)
    lightenedTargets = stopwordRemoving(stemmedTargets, stopw)
    
    # print("--- Generating n-grams:")                              
    # THIS WAS A PATH TO FOLLOW
    # ngrammedReviews = ngramming(lightenedReviews)
    # ngrammedTargets = ngramming(lightenedTargets)

    print("--- Generating vocabulary:")
    voc = genVocabulary(lightenedReviews, labels, lightenedTargets)

    print("--- Generating count matrix:")
    cmR = getCountMatrix(lightenedReviews, voc)
    cmT = getCountMatrix(lightenedTargets, voc)
    print(cmR)

    print("--- Vectorizing:")
    vectorizer = TfidfTransformer()
    tf_x = vectorizer.fit_transform(cmR)
    tf_t = vectorizer.transform(cmT)
    '''

    print("--- Rebuilding reviews:")
    preprocessedReviews = assemble(stemmedReviews)
    preprocessedTargets = assemble(stemmedTargets)

    print("--- Vectorizing:")
    stemTokenizer = StemTokenizer()
    vectorizer = TfidfVectorizer(strip_accents='unicode', tokenizer=stemTokenizer, max_features=20000, analyzer='word',
                                 stop_words=stopw, min_df=2, max_df=0.35, ngram_range=(1, 2))

    print("\rComputing... 0%", end="", flush=True)
    tfidf_X = vectorizer.fit_transform(preprocessedReviews)
    print("\rComputing... 50%", end="", flush=True)
    tfidf_T = vectorizer.transform(preprocessedTargets)
    print("\rComputing... 100%", end="", flush=True)
    print()

    print("==========PREPROCESSING END==========")
    print()
    print("==========CLASSIFICATION START==========")

    active = False
    if active:
        tuneC(tfidf_X, labels)

    active = False
    if active:
        tuneMaxIter(tfidf_X, labels)

    active = False
    if active:
        generateROC(tfidf_X, labels)

    active = False
    if active:
        generateConfMatrix(tfidf_X, labels)

    active = False
    if active:
        generateConfMatrixAndROC(tfidf_X, labels)

    active = False
    if active:
        compareClassifiers(tfidf_X, labels)

    print("--- Classifying:")
    """
    clf = RandomForestClassifier(n_jobs=5)
    clf.fit(tfidf_X, labels)
    result = clf.predict(tfidf_T)
    """
    svc = LinearSVC(dual=False, C=0.3125)  # c=0.3125
    svc.fit(tfidf_X, labels)
    result = svc.predict(tfidf_T)
    print("done")

    print("==========CLASSIFICATION END==========")
    print()

    with open("results.csv", "w") as writeFile:
        writeFile.write("Id,Predicted\n")
        counter = 0
        for v in result:
            writeFile.write(str(counter)+','+str(v)+'\n')
            counter = counter + 1
    writeFile.close()

    print("==========COMPLETE==========")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
