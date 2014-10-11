# -*- coding: cp1252 -*-
#!/usr/bin/env python
#coding: utf8
from flask import Flask, request, json, current_app

import os,re
import pickle 
import pypyodbc as pyo
import numpy as np
import pandas as pd 

from time import time
from codecs import open as op

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.grid_search import GridSearchCV

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import RegexpTokenizer


#import grantAutoClassification.psykit_classifier_1
app = Flask(__name__)
classifier = None
english_stops = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
rTokenizer = RegexpTokenizer('\w+|\$[\d\.]+')#\w+|\$[\d\.]+|\S+

picklefile = 'rows.pkl'


def __pad(strdata):
    """ Pads `strdata` with a Request's callback argument, if specified, or does
    nothing.
    """
    if request.args.get('callback'):
        return "%s(%s);" % (request.args.get('callback'), strdata)
    else:
        return strdata

def __mimetype():
    if request.args.get('callback'):
        return 'application/javascript'
    else:
        return 'application/json'

def __dumps(*args, **kwargs):
    """ Serializes `args` and `kwargs` as JSON. Supports serializing an array
    as the top-level object, if it is the only argument.
    """
    indent = None
    if (current_app.config.get('JSONIFY_PRETTYPRINT_REGULAR', False)
            and not request.is_xhr):
        indent = 2
    return json.dumps(args[0] if len(args) is 1 else dict(*args, **kwargs),
                      indent=indent)


def lematizeWord(word):
    """ Lemmatizes word. Doesn't distinguish tags. Changes word to lower case.
    """
    return lemmatizer.lemmatize(lemmatizer.lemmatize(word.lower(), pos='v'), pos='n')

def lemaTokenizer(text):
    """ Tokenizes text, lemmatizes words in text and returns the non-stop words (pre-lemma).
    """
    return [lematizeWord(word.lower()) for word in rTokenizer.tokenize(re.sub(r'á', ' ', text.replace("'s", ""))) if(word.lower()) not in english_stops]

def loadfromdb():
    t0 = time()
    cnxn = pyo.connect(driver='{FreeTDS}',server='172.16.8.60',port='1433',uid='user',pwd='textmining',database='text_classification')
    cursor = cnxn.cursor()
    cursor.execute("select * from set1 order by 1 ")
    rows = cursor.fetchall()
    task_time = time() - t0
    print("Query time: %0.3fs" % task_time)    
    
    rowstrip =[]
    for row in rows:
        rowstrip.append([el for el in row])
    rowdata = {'rows': rowstrip, 't0': t0}
    with open(picklefile, 'wb') as fh:
        pickle.dump(rowdata, fh)
    
    return rows 

def loadfromfile():
    with open(picklefile, 'rb') as fh:
        rowdata = pickle.load(fh)
    return rowdata['rows'], rowdata['t0']

def __train(force_load = False, param_search = False, reload_time = 1, nquantiles = 100):
    """ Creats model; Uses bi-gram; tfidf weighting; one-vs-all classifier.
        Exception not handled here... assumes everything works as written.
    """
    
    # Load data
    try:
        rowf, t0 = loadfromfile()      
        if time() - t0 < reload_time * 60*60*24 and force_load==False:
            rows = rowf
        else:
            rows = loadfromdb()          
    except: 
        rows = loadfromdb()
        
    #Slice into training and test sets
    t0 = time()
    trainX, trainY = [],[]
    testX, testY = [], []
    for r in rows:
        if(r[3]!="test"):
            trainX.append(r[2])#+ (" " + r[4].decode('latin')).replace(" "," @")
            trainY.append(r[1])
        else:
            testX.append(r[2])#+ (" " + r[4].decode('latin')).replace(" "," @")
            testY.append(r[1])
    print("Total Records: %0.0f" %len(trainX))        
    task_time = time() - t0
    print("Slice time: %0.3fs" % task_time)
    t0 = time()
    
    # 
    classifier = Pipeline([ ('vectorizer', CountVectorizer(ngram_range=(1, 2), min_df=1,tokenizer=lemaTokenizer)), ('tfidf', TfidfTransformer()),('clf', OneVsRestClassifier(LinearSVC()))])
    classifier.fit(trainX, trainY)
    
    if param_search:    
        params = dict(clf__estimator__C=[0.1, 1, 10, 100])
        grid_search = GridSearchCV(classifier, param_grid=params)   
        grid_search.fit(trainX, trainY)
        classifier = grid_search.best_estimator_
        
    margins = classifier.decision_function(testX)
    isCorrect = (classifier.predict(testX) == testY)
    
    df=pd.DataFrame({'margins':np.max(margins,1), 'isCorrect':isCorrect})
    q, bins = pd.qcut(df['margins'],nquantiles,retbins=True)
    df['quantile'] = q.labels
    
    gb=df.groupby('quantile')
    qm=gb['isCorrect'].mean()
    
    task_time = time() - t0
    print("Learn time: %0.3fs" % task_time)
    t0 = time()
    return classifier
    
@app.route('/')
def test_function():
    """ root returns test data ({"data": [1,2,3,4,"test1","test2","test3","test4"]}.).
        used for checking if service is running
    """
    return current_app.response_class(__pad(__dumps(data=[i for i in [1,2,3,4,'test1','test2','test3','test4'] ])),mimetype=__mimetype())

@app.route('/svm_autoclassify/<grantDescription>')
def svm_classify(grantDescription=None):
    """ main end-point. Given a text retruns the text and class.
    """
    print([grantDescription])
    print(classifier)
    t0 = time()
    
    predicted = classifier.predict([grantDescription])
    margin = classifier.decision_function([grantDescription])
    
    mbin = pd.cut([margin], bins, labels=False)[0]
    if margin < bin[0]:
        mbin = 0
    elif margin > bin[-1]:
        mbin = nquantiles - 1 
    prob = qm[mbin]
    
    task_time = time() - t0
    print("Prediction time: %0.3fs" % task_time)
    return current_app.response_class(__pad(__dumps(data=[i for i in [predicted[0],grantDescription,prob] ])),mimetype=__mimetype())

classifier = __train()
if __name__ == '__main__':
    app.run(host='0.0.0.0',port =9090) 
