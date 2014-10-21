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

#Settings for connection to database server
DB_SETTINGS = {'driver': '{FreeTDS}',
               'server': '172.16.8.60',
               'port': '1433',
               'uid': 'user',
               'pwd': 'textmining',
               'database':'text_classification'}

#Settings for database table. 
#'table' is the table name, plus the following specific fields
#desc: grant description text
#label: three-character grant label 
#org: organization name, currently this is concatenated with the text description
DB_FIELDS = {'table':'grants', 
             'desc': 'description', 
             'org': 'recipient_name',
             'label': 'activity_override3',
             'maxRows': None}
                          
class grantData:
    #Handles fetching of grant data from database or file. 
    def __init__(self, dbSettings = DB_SETTINGS, dbFields = DB_FIELDS, 
                 holdout_frac = 0.1, pickleFile = None, reload_time = 1):
        self.dbSettings = dbSettings
        self.dbFields = dbFields
        self.reload_time = 0. #number of days before data saved to file becomes stale
        self.pickleFile = pickleFile
        self.holdout_frac = holdout_frac #fraction of data to be held out for training
        
        self.loadTime = 0 
        
        #self.load()
        
    def fetchFromDB(self):
        try:
            cnxn = pyo.connect(**self.dbSettings)           
            cursor = cnxn.cursor()
            
            topstr = ('TOP ' + str(DB_FIELDS['maxRows']) + ' ') if DB_FIELDS['maxRows'] != None else '' 
            fieldstr = ','.join([self.dbFields['desc'],self.dbFields['org'],self.dbFields['label']])
            cursor.execute("select " + fieldstr + " from " + self.dbFields['table'])
            rows = cursor.fetchall()
            
            field_names = [d[0] for d in cursor.description]
            
            try: 
                desc_ix = field_names.index(self.dbFields['desc'])
                label_ix = field_names.index(self.dbFields['label'])
            except valueError:
                return [[]]
            
            #Check if table has organization names            
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
        #chooses whether to fetch from file or database
        if self.pickleFile is not None:
            #check if data is stale, if so load from DB
            rowf, fileTime = self.fetchFromFile()     
            if time() - fileTime < self.reload_time * 60*60*24:
                rows = rowf
                self.loadTime = fileTime
            else:
                rows = self.fetchFromDB()
        else:
            rows = self.fetchFromDB()
        return rows
     
    def load(self): 
        #load data and split into training and test sets
        rows = self.fetch()
        if rows is None:
            return False
        
        #Slice into training and test sets
        np.random.shuffle(rows)
        cutoff = int(len(rows)*self.holdout_frac)
        self.trainX, self.trainY = [list(l for l in zip(*rows[cutoff:]))][0]
        self.testX, self.testY = [list(l for l in zip(*rows[:cutoff]))][0]

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
        svmClassifier = Pipeline([ ('vectorizer', CountVectorizer(ngram_range=(1, 2),
                                                                  min_df=1,tokenizer=self.lemaTokenizer)),
                                   ('tfidf', TfidfTransformer()),
                                   ('clf', OneVsRestClassifier(LinearSVC()))])
        
        if grid_search:
            #option to grid-search the regularization parameter C
            params = dict(clf__estimator__C=[0.1, 1, 10, 100])
            grid_search = GridSearchCV(svmClassifier, param_grid=params)   
            grid_search.fit(self.data.trainX, self.data.trainY)
            
            self.classifier = grid_search.best_estimator_            
        else:
            svmClassifier.fit(self.data.trainX, self.data.trainY)        
            self.classifier = svmClassifier
            
        self.binnedProbs()
        
    def test(self):
        return self.classifier.score(self.data.testX, self.data.testY)
    
    def predict_single(self, X):
        predicted = self.classifier.predict([X])
        
        #calculate accuracy estimate based on binned probabilities
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
        

