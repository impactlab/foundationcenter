
from flask import Flask, request, current_app, jsonify
from flask.ext.sqlalchemy import SQLAlchemy
import grantClassifier 
import datetime, os
import simplejson as json
import numpy as np

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
             'maxRows': 50000}

TOPN_DEFAULT = 5
MAXIMUM_OFFSET = 1000

#

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)

myData = grantClassifier.grantData(picklefile = PICKLE_FILE, 
                                   sqla_connection = db.engine.connect(), 
                                   dbFields = DB_FIELDS)
myData.load()

myClassifier = grantClassifier.grantClassifier(myData)
myClassifier.load()

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
    return json.dumps(text)
    
def _text():
    """ returns text to be labeled
    
    input: n/a
    output: dict (on success) or None (on failure)
    """
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
            .first() 
        return result.as_dict()
    except:
        return None
    
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
    except ValueError:
        npredict = TOPN_DEFAULT    
        
    try:
        top_classes = myClassifier.predict_multiple(j['text'], npredict=npredict)
        
        #Format output
        c = {'svm' : 
             {'category': top_classes[0][0],
              'confidence': top_classes[0][1], 
              'alternates': []}  
            }  
        for ii in range(1, len(top_classes)):
            alternate = {'category': top_classes[ii][0],
                         'confidence': top_classes[ii][1],
                         'rank': ii}
            c['svm']['alternates'].append(alternate)        
        return c
    except:
        return None

if __name__ == '__main__':
    debug = os.environ.get('DEBUG', True)
    port = os.environ.get('PORT', 9090)

    app.run(host='0.0.0.0',port=port, debug=debug)
