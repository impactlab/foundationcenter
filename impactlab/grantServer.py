
from flask import Flask, request, current_app, jsonify
from flask.ext.sqlalchemy import SQLAlchemy

import datetime, os
import simplejson as json
import threading, Queue
import numpy as np

import grantClassifier 

#Parameters

DATABASE_URI = 'mssql+pymssql://user:textmining@172.16.8.60:1433/text_classification'
PICKLE_FILE = 'gd.pkl'

#Settings for database table. 
#'table' is the table name, plus the following specific fields
#desc: grant description text
#label: three-character grant label 
#org: organization name, currently this is concatenated with the text description

DB_FIELDS = {'table':'grants', 
             'desc': 'description', 
             'org': 'recipient_name',
             'label': 'activity_override3',
             'pk': 'grant_key',
             'retrain_count': 'number_retrains',
             'maxRows': 50000}

RETRAIN_FIELDS = {'table':'retrained_grants',
                  'desc':'description',
                  'org': '',
                  'label':'activity',
                  'text_key': 'grant_id',
                  'maxRows': None}

TOPN_DEFAULT = 5

#

MAXIMUM_OFFSET = 3000
QUERY_ROWS = 50
STORED_ROWS = 50
RELOAD_FRAC = 0.75

#

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_ECHO'] = False

db = SQLAlchemy(app)

myData = grantClassifier.grantData(picklefile = PICKLE_FILE, 
                                   sqla_connection = db.engine.connect(), 
                                   dbFields = DB_FIELDS)

retrainData = grantClassifier.grantData(sqla_connection = db.engine.connect(), 
                                        dbFields = RETRAIN_FIELDS)

myClassifier = grantClassifier.grantClassifier(myData,kernel_reg=True)

stored_texts = Queue.Queue()
lock = threading.Lock()

class Grant(db.Model):
    __table__ = db.Table(DB_FIELDS['table'], db.metadata, autoload=True, autoload_with=db.engine)
    
    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}    
        
class retrainedGrants(db.Model):
    __tablename__ = 'retrained_grants'
    rowid = db.Column(db.Integer, primary_key = True)
    
    user_id = db.Column(db.Integer)
    entry_date = db.Column(db.DateTime) 
    grant_id = db.Column(db.Integer)
    
    description = db.Column(db.String(1000))
    activity = db.Column(db.String(10))
    
    activity_svm = db.Column(db.String(10))
    activity_rule = db.Column(db.String(10))
    model_version = db.Column(db.String(255))

    @classmethod
    def load(cls, inDict):
        inDict['entry_date'] = datetime.datetime.now()
        return cls(**inDict)

@app.route('/text', methods=['GET'])
def text():
    # TODO: ALlow reading from wherever text is to be store """
    """ GET returns a json containing {'text' : '<your text>'}, to be labeled by a human """
    print "GET /text"
    text = _text()
    if not text:
        return '', 400 # error
    text['text'] = text.pop(DB_FIELDS['desc'])
    return json.dumps(text)
    
def _text():
    """ returns text to be labeled
    
    input: n/a
    output: dict (on success) or None (on failure)
    """
    if stored_texts.qsize() < (STORED_ROWS * RELOAD_FRAC):
        if lock.acquire(False):
            t = threading.Thread(target=_fetch_texts, args = (stored_texts, lock))
            t.daemon = True
            t.start()        
    
    if not stored_texts.empty():
        return stored_texts.get().as_dict()  
    else:
        return None

def _fetch_texts(q, lock):    
    try:
        max_offset = Grant.query.filter(Grant.number_retrains == None).count()
        if max_offset == 0:
            min_retrains = db.session.query(db.func.min(Grant.number_retrains)).scalar()
            max_offset = Grant.query.filter(Grant.number_retrains == min_retrains).count()
        max_offset = min(max_offset, MAXIMUM_OFFSET)    
        offset = int(np.random.rand()*max_offset)
        result = Grant.query \
            .order_by(Grant.number_retrains) \
            .offset(offset) \
            .limit(QUERY_ROWS) \
            .all()
        res_sub = sorted(result, key = lambda rr: myClassifier.predict([getattr(rr,DB_FIELDS['desc'])])[1])[:STORED_ROWS]  
        np.random.shuffle(res_sub)
        
        map(q.put, res_sub)
    finally:
        lock.release()
        return True        
        
    
@app.route('/label', methods=['POST'])
def label():
    """ POST with json body, returns 200 if saved successfully

    For example, to save a text label using row ids from a database, one might write

    $ curl -X POST -H "Content-Type: application/json" -d '{"text_id":5,"label_id":19}' localhost:9090/label
    """
    j = request.get_json()
    print "POST /label", j
    if not _label(j):
        return '', 400 # error
    return ''

def _label(j={}):
    """ saves label and meta-data for text. returns True/False on success/failure.

    input: user-defined dict
    output: boolean (True on success) or None (on Failure)

    For example, one might save properties such as

    {
        // essential - mapping of text to label
        'text_id' : 5,
        'label_id' : 19,

        // optional - metadata
        'user_id' : foo,
        'timestamp' : bar,
        'model_version' : baz,
    }
    """
    try:
        newRow = retrainedGrants.load(j)
        db.session.add(newRow)
        db.session.commit()
        return True
    except:
        return None

@app.route('/classify', methods=['POST'])
def classify():
    """ POST with json body of {'text':'<your text>'}, returns a dict of classifications

    Example:

    $ curl -X POST -H "Content-Type: application/json" -d '{"text":"classify me!"}' localhost:9090/classify
    {
        ... // classification details
    }
    """
    j = request.get_json()
    print "post /classify", j
    c = _classify(j)
    if not c:
        return '', 400 # error
    return json.dumps(c)

def _classify(j={}):
    """ given text, returns a dict describing machine classifications.

    input: string
    output: dict (on success) or None (on error)

    For example, one might save properties such as

    {
        'svm' : {
            'category' : 'bob',
            'confidence' : .97
        },
        'svm_multiple' : {
            // etc -
        }
    }
    """  
    
    try: 
        npredict = int(j['npredict'])
    except:
        npredict = TOPN_DEFAULT    
        
    try:
        top_classes = myClassifier.predict_multiple(j['text'], npredict=npredict)
        
        #Format output
        c = {'svm' : []}  
        for ii in range(len(top_classes)):
            item = {'category': top_classes[ii][0],
                         'confidence': top_classes[ii][1],
                         'rank': ii}
            c['svm'].append(item)        
        return c
    except:
        return None

@app.route('/train')
def train():
    """ URL hook that initiates retraining of the classifier

    TODO: shouldn't be public - maybe this a hook is only accessible via SECRET_KEY?
    """
    if not _train(myData, retrainData, myClassifier):
        return '', 400
    return ''

def _train(data=None, retrain_data=None, classifier=None):
    """ (re)train the classifer given new labeled data

    input: n/a
    output: bool (on success) or None (on failure)
    In the _train() method, this process might look like:
        - run classifier(s) training method on all labeled data
        - output a pickle file of newly built classifer, with data in filename
        NOTE: It's probably better to do this in another process so as to not hit
        100 percent CPU in the API's process due to retraining.

    Externally, one might manage this via:
        - have a cronjob that runs daily and hits this hook to do the retraining
        - have a cronjob that runs daily (later, after retraining) and restarts the server.
            alternately use a unix utility like `inotifywait` in order to watch for file changes
            and restart... http://superuser.com/questions/181517/how-to-execute-a-command-whenever-a-file-changes
        - when the server restarts, make sure which will always read from the most recent
            classifier file (ideas: find most recent file. backup/overwrite old classifier.)
    """
    
    data.load()
    retrain_data.load()
    
    data.appendData(retrain_data)
    
    classifier.load()
    
    return True

@app.route('/count_retrains')
def count_retrains():
    """ URL hook that initiates retraining of the classifier

    TODO: shouldn't be public - maybe this a hook is only accessible via SECRET_KEY?
    """
    if not _count_retrains():
        return '', 400
    return ''

def _count_retrains():
    querystr = "UPDATE " + DB_FIELDS['table'] + ' SET ' 
    querystr += DB_FIELDS['table'] + '.' + DB_FIELDS['retrain_count'] + ' = count_table.counts FROM ' 
    querystr += DB_FIELDS['table'] + ' JOIN (SELECT ' + RETRAIN_FIELDS['text_key'] + ', count(' 
    querystr += RETRAIN_FIELDS['text_key'] + ') counts FROM ' + RETRAIN_FIELDS['table'] 
    querystr += ' GROUP BY ' + RETRAIN_FIELDS['text_key'] + ') AS count_table ON ' 
    querystr += DB_FIELDS['table']+ '.' + DB_FIELDS['pk'] + ' = count_table.grant_id'
    
    try:
        conn = db.engine.connect()
        result = conn.execute(querystr)
        return True
    except:
        return None

if __name__ == '__main__':
    debug = os.environ.get('DEBUG', True)
    port = os.environ.get('PORT', 9090)
    
    train_success = train()
    
    lock.acquire()
    t = threading.Thread(target=_fetch_texts, args = (stored_texts, lock))
    t.daemon = True
    t.start()        
    
    app.run(host='0.0.0.0',port=port, debug=debug)
