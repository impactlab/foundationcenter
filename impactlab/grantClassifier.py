# -*- coding: utf-8 -*-
# asdf asdf asdf 
from flask import Flask, request, json, current_app

import os,re,datetime
import cPickle as pickle
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
from statsmodels.nonparametric.kernel_regression import KernelReg
                     
class grantData:
    #Handles fetching of grant data from database or file. 
    def __init__(self, dbSettings = None, dbFields = None, 
                 holdout_frac = 0.1, picklefile = None, 
                 reload_time = datetime.timedelta(days=1),
                 sqla_connection = None):
        self.dbSettings = dbSettings
        self.dbFields = dbFields
        self.reload_time = reload_time #number of days before data saved to file becomes stale
        self.picklefile = picklefile
        self.holdout_frac = holdout_frac #fraction of data to be held out for training
        self.sqla_connection = sqla_connection

        
    def fetchFromDB(self):
        topstr = ('TOP ' + str(self.dbFields['maxRows']) + ' ') if self.dbFields['maxRows'] != None else '' 
        fieldstr = ','.join(filter(None,[self.dbFields['desc'],self.dbFields['org'],self.dbFields['label']]))
        querystr = "select " + topstr + fieldstr + " from " + self.dbFields['table'] 
        
        try:
            if self.sqla_connection is not None:
                rows = self.sqla_connection.execute(querystr)
                rowstrip = []
                try:
                    for row in rows:
                        rowstrip.append([row[self.dbFields['desc']]+u' '+row[self.dbFields['org']], 
                                         row[self.dbFields['label']]])
                except:
                    for row in rows:
                        rowstrip.append([row[self.dbFields['desc']], row[self.dbFields['label']]])                    
                                          
            else:
                cnxn = pyo.connect(**self.dbSettings)           
                cursor = cnxn.cursor()
                cursor.execute(querystr)
                rows = cursor.fetchall()
            
                field_names = [d[0] for d in cursor.description]
            
                try: 
                    desc_ix = field_names.index(self.dbFields['desc'])
                    label_ix = field_names.index(self.dbFields['label'])
                except ValueError:
                    return [[]]
            
                #Check if table has organization names, 
                #if so append them to the grant descriptions
                rowstrip =[]
                try:
                    org_ix = field_names.index(self.dbFields['org'])               
                    for row in rows:
                        rowstrip.append([row[desc_ix] + ' ' + row[org_ix], row[label_ix]])  
                except valueError:
                    for row in rows:
                        rowstrip.append([row[desc_ix], row[label_ix]])                
            
            #pickle the database query
            self.loadTime = time()
            try:
                pickle.dump(rowstrip, open(self.picklefile, 'wb'),2)
            except:
                pass
              
            return rowstrip
        except: 
            return [[]]
            
    def load(self): 
        #load data and split into training and test sets
        rows = loadNonStaleFile(self.picklefile, self.fetchFromDB, max_age = self.reload_time)
        if rows is None:
            return False
        
        #Slice into training and test sets
        np.random.shuffle(rows)
        cutoff = int(len(rows)*self.holdout_frac)
        self.trainX, self.testX, self.trainY, self.testY = (),(),(),()
        try:
            self.trainX, self.trainY = [list(l for l in zip(*rows[cutoff:]))][0]
            self.testX, self.testY = [list(l for l in zip(*rows[:cutoff]))][0]
        except ValueError:
            pass
    def appendData(self, moreData):
        self.trainX = self.trainX + moreData.trainX
        self.trainY = self.trainY + moreData.trainY
        self.testX = self.testX + moreData.testX
        self.testY = self.testY + moreData.testY

def loadNonStaleFile(picklefile, loader, max_age = datetime.timedelta(days=1)):
    # Loads object from disk if the file is not older than max_age
    # Otherwise loads object by calling loader()
    
    if not isinstance(picklefile, basestring):
        picklefile = ''
    
    if os.path.isfile(picklefile):
        filetime = datetime.datetime.fromtimestamp(os.path.getmtime(picklefile))
        if datetime.datetime.now() - filetime < max_age:
            try: 
                return pickle.load(open(picklefile,'rb'),2)
            except:
                pass  
    try: 
        return loader()
    except: 
        pass
    
    try: 
        return pickle.load(open(picklefile,'rb'),2)
    except: 
        return None

class grantClassifier:
    #scikit-learn classifier for labeling grant descriptions with grant categories. 

    def __init__(self, data = None, nquantiles = 25, picklefile = None, 
                 reload_time = datetime.timedelta(days=1), grid_search = False,
                 kernel_reg = False):
        self.classifier = None 
        self.data = data
        self.nquantiles = nquantiles
        self.picklefile = picklefile 
        self.reload_time = reload_time 
        self.grid_search = grid_search
        self.kernel_reg = kernel_reg
    
    def load(self):
        #Loads classifier from disk if file exists and is not older than reload_time. 
        #Otherwise trains from data 
        
        self.classifier = loadNonStaleFile(self.picklefile, self.train, max_age = self.reload_time)
        self.binnedProbs()

    def train(self):
        #Trains classifier from data
        svmClassifier = Pipeline([ ('vectorizer', CountVectorizer(ngram_range=(1, 2),min_df=1,
                                                                  tokenizer=lemaTokenizer)),
                                   ('tfidf', TfidfTransformer()),
                                   ('clf', OneVsRestClassifier(LinearSVC()))])
        
        if self.grid_search:
            #Option to grid-search the regularization parameter C
            params = {'clf__estimator__C':(0.1, 1, 10, 1000)}
            grid_search = GridSearchCV(svmClassifier, param_grid=params)   
            grid_search.fit(self.data.trainX, self.data.trainY)
            
            clf = grid_search.best_estimator_
        else:
            svmClassifier.fit(self.data.trainX, self.data.trainY)        
            clf = svmClassifier
        
        if self.picklefile is not None:
            pickle.dump(clf, open(self.picklefile, 'wb'),2)  
        
        return clf         
        
    def test(self):
        #Computes accuracy of classifer on the holdout set
        return self.classifier.score(self.data.testX, self.data.testY)
    
    def predict(self, X):
        predicted = self.classifier.predict(X)
        
        #Calculate accuracy estimate based on binned probabilities
        margins = np.max(self.classifier.decision_function(X),1)        
        prob = self.calcProb(margins)
            
        return predicted, prob
            
    def predict_multiple(self, X, npredict = 5):
        #Return npredict highest classes and scores
        #scores should no longer be interpreted as probabilities 
        
        predicted = self.classifier.predict([X])
        
        margins = np.hstack(self.classifier.decision_function([X]))
        top_id = np.argsort(margins)[::-1][:npredict]
        
        class_names = self.classifier.steps[2][1].classes_
        
        top_classes = [class_names[ix] for ix in top_id[:npredict]]
        top_scores = self.calcProb([margins[ix] for ix in top_id[:npredict]])
            
        return zip(top_classes, top_scores)
        
    def calcProb(self, margins):
        #Given a margin, look up the quantile and then 
        #the accuracy of that quantile
        if self.kernel_reg == True:
            return self.kr.fit(margins)[0]
            
        mbins = pd.cut(margins, self.marginBins, labels=False)
        mbins[margins<self.marginBins[0]] = 0 
        mbins[margins>self.marginBins[-1]] = self.nquantiles - 1 
        
        return [self.quantileProbs[mb] for mb in mbins]    
    
    def binnedProbs(self):
        #divide the SVM margins into quantiles and calculate the accuracy for each quantile
        margins = self.classifier.decision_function(self.data.testX)
        isCorrect = (self.classifier.predict(self.data.testX) == self.data.testY)
        maxmargin = np.max(margins,1)
        
        df=pd.DataFrame({'margins':maxmargin, 'isCorrect':isCorrect})
        q, self.marginBins = pd.qcut(df['margins'],self.nquantiles,retbins=True)
        df['quantile'] = q.labels
        
        gb=df.groupby('quantile')
        self.quantileProbs=gb['isCorrect'].mean() 
        
        if self.kernel_reg == True:
            self.kr = KernelReg(isCorrect, maxmargin, 'c')
            
# Tokenization and lemmatization
# Isn't convenient to put inside the class 
# because of issues with deep copying.  
    
lemmatizer = WordNetLemmatizer()
rTokenizer = RegexpTokenizer('\w+|\$[\d\.]+')#\w+|\$[\d\.]+|\S+
english_stops = stopwords.words('english')

def lematizeWord(word):
    # Lemmatizes word. Doesn't distinguish tags. Changes word to lower case.   
    return lemmatizer.lemmatize(
        lemmatizer.lemmatize(word.lower(), pos='v'), pos='n')

def lemaTokenizer(text):
    # Tokenizes text, lemmatizes words in text and returns the non-stop words (pre-lemma).  
    return [lematizeWord(word.lower()) 
            for word in rTokenizer.tokenize(re.sub(r'á', ' ', text.replace("'s", ""))) 
            if(word.lower()) not in english_stops]
  
        
        

