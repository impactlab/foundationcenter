#coding: utf8

from flask import Flask, request, json, current_app

import os,re, pickle, datetime
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
                 holdout_frac = 0.1, picklefile = None, 
                 reload_time = datetime.timedelta(days=1)):
        self.dbSettings = dbSettings
        self.dbFields = dbFields
        self.reload_time = reload_time #number of days before data saved to file becomes stale
        self.picklefile = picklefile
        self.holdout_frac = holdout_frac #fraction of data to be held out for training
        
    def fetchFromDB(self):
        try:
            cnxn = pyo.connect(**self.dbSettings)           
            cursor = cnxn.cursor()
            
            topstr = ('TOP ' + str(DB_FIELDS['maxRows']) + ' ') if DB_FIELDS['maxRows'] != None else '' 
            fieldstr = ','.join([self.dbFields['desc'],self.dbFields['org'],self.dbFields['label']])
            cursor.execute("select " + topstr + fieldstr + " from " + self.dbFields['table'])
            rows = cursor.fetchall()
            
            field_names = [d[0] for d in cursor.description]
            
            try: 
                desc_ix = field_names.index(self.dbFields['desc'])
                label_ix = field_names.index(self.dbFields['label'])
            except valueError:
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
                pickle.dump(rowstrip, open(self.picklefile, 'wb'))
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
        self.trainX, self.trainY = [list(l for l in zip(*rows[cutoff:]))][0]
        self.testX, self.testY = [list(l for l in zip(*rows[:cutoff]))][0]

def loadNonStaleFile(picklefile, loader, max_age = datetime.timedelta(days=1)):
    if os.path.isfile(picklefile):
        filetime = datetime.datetime.fromtimestamp(os.path.getmtime(picklefile))
        if datetime.datetime.now() - filetime < max_age:
            try: 
                return pickle.load(open(picklefile,'rb'))
            except:
                pass  
    try: 
        return loader()
    except: 
        pass
    
    try: 
        return pickle.load(picklefile)
    except: 
        return None

class grantClassifier:
    #scikit-learn classifier for labeling grant descriptions with grant categories. 
    def __init__(self, data = None, nquantiles = 25, picklefile = None, 
                 reload_time = datetime.timedelta(days=1)):
        self.classifier = None 
        self.data = data
        self.nquantiles = nquantiles

    def train(self, grid_search = False):
        svmClassifier = Pipeline([ ('vectorizer', CountVectorizer(ngram_range=(1, 2),
                                                                  min_df=1,tokenizer=lemaTokenizer)),
                                   ('tfidf', TfidfTransformer()),
                                   ('clf', OneVsRestClassifier(LinearSVC()))])
        
        if grid_search:
            #option to grid-search the regularization parameter C
            params = {'clf__estimator__C':(0.1, 1, 10, 1000)}
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
  
        
        

