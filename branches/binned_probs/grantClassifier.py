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

DB_SETTINGS = {'driver': '{FreeTDS}',
               'server': '172.16.8.60',
               'port': '1433',
               'uid': 'user',
               'pwd': 'textmining',
               'database':'text_classification'}

DB_TABLE = 'set1'

class grantData:
    def __init__(self, dbSettings = DB_SETTINGS, dbTable = DB_TABLE, pickleFile = None):
        self.dbSettings = dbSettings
        self.dbTable = dbTable
        self.loadTime = 0
        self.pickleFile = pickleFile
        
        self.load()
        
    def fetchFromDB(self):
        try:
            cnxn = pyo.connect(**self.dbSettings)
            
            cursor = cnxn.cursor()
            cursor.execute("select * from " + self.dbTable + " order by 1 ")
            rows = cursor.fetchall()
            
            #convert pypyodbc object to generic list
            rowstrip =[]
            for row in rows:
                rowstrip.append([el for el in row])
            
            #pickle the database query
            self.loadTime = time()
            try:
                with open(self.picklefile, 'wb') as fh:
                    pickle.dump({'rows':rowstrip, 'fileTime': self.loadTime}, fh)
            except:
                pass
              
            return rowstrip
        except: 
            return [[]]
            
    def fetchFromFile(self):
        try:
            with open(self.picklefile, 'rb') as fh:
                rowdata = pickle.load(fh)
            return rowdata['rows'], rowdata['fileTime']
        except: 
            return [[]], 0
        
    def fetch(self):    
        if self.pickleFile is not None:
            rowf, fileTime = self.fetchFromFile()     
            if time() - fileTime < reload_time * 60*60*24:
                rows = rowf
                self.loadTime = fileTime
            else:
                rows = self.fetchFromDB()
        else:
            rows = self.fetchFromDB()
        return rows
     
    def load(self): 
        rows = self.fetch()
        if rows is None:
            return False
        
        #Slice into training and test sets
        self.trainX, self.trainY = [],[]
        self.testX, self.testY = [],[]
        for r in rows:
            if(r[3]!="test"):
                self.trainX.append(r[2])#+ (" " + r[4].decode('latin')).replace(" "," @")
                self.trainY.append(r[1])
            else:
                self.testX.append(r[2])#+ (" " + r[4].decode('latin')).replace(" "," @")
                self.testY.append(r[1])      

class grantClassifier:
    def __init__(self, data = None, nquantiles = 25):
        self.classifier = None
        
        self.english_stops = stopwords.words('english')
        self.lemmatizer = WordNetLemmatizer()
        self.rTokenizer = RegexpTokenizer('\w+|\$[\d\.]+')#\w+|\$[\d\.]+|\S+
        
        self.data = data
        self.nquantiles = nquantiles
        
    def lematizeWord(self,word):
        """ Lemmatizes word. Doesn't distinguish tags. Changes word to lower case.
        """
        return self.lemmatizer.lemmatize(
            self.lemmatizer.lemmatize(word.lower(), pos='v'), pos='n')
    
    def lemaTokenizer(self,text):
        """ Tokenizes text, lemmatizes words in text and returns the non-stop words (pre-lemma).
        """
        return [self.lematizeWord(word.lower()) 
                for word in self.rTokenizer.tokenize(re.sub(r'á', ' ', text.replace("'s", ""))) 
                if(word.lower()) not in self.english_stops]

    def train(self, grid_search = False):
        svmClassifier = Pipeline([ ('vectorizer', CountVectorizer(ngram_range=(1, 2), min_df=1,tokenizer=self.lemaTokenizer)), ('tfidf', TfidfTransformer()),('clf', OneVsRestClassifier(LinearSVC()))])
        
        if grid_search:
            params = dict(clf__estimator__C=[0.1, 1, 10, 100])
            grid_search = GridSearchCV(svmClassifier, param_grid=params)   
            grid_search.fit(self.data.trainX, self.data.trainY)
            
            self.classifier = grid_search.best_estimator_            
        else:
            svmClassifier.fit(self.data.trainX, self.data.trainY)        
            self.classifier = svmClassifier
            
        self.binnedProbs()
    
    def predict_single(self, X):
        predicted = self.classifier.predict([X])
        
        margin = np.max(self.classifier.decision_function([X]))        
        mbin = pd.cut([margin], self.marginBins, labels=False)[0]
        if margin < self.marginBins[0]:
            mbin = 0
        elif margin > self.marginBins[-1]:
            mbin = self.nquantiles - 1 
        prob = self.quantileProbs[mbin]
            
        return predicted[0], prob
    
    def binnedProbs(self):
        #divide the SVM margins into quantiles and calculate the accuracy for each quantile
        margins = self.classifier.decision_function(self.data.testX)
        isCorrect = (self.classifier.predict(self.data.testX) == self.data.testY)
        
        df=pd.DataFrame({'margins':np.max(margins,1), 'isCorrect':isCorrect})
        q, self.marginBins = pd.qcut(df['margins'],self.nquantiles,retbins=True)
        df['quantile'] = q.labels
        
        gb=df.groupby('quantile')
        self.quantileProbs=gb['isCorrect'].mean()        
        

