"""
Prototype API for Foundation Center

Loads data from database or file, 
trains SVM classifier.

Given a grant description as string, 
returns either 
(1) the predicted grant type and estimated
probability of being correct
or (2) the top n classes and scores 
for each class
"""

from flask import Flask, request, json, current_app
from flask.ext.sqlalchemy import SQLAlchemy
import grantClassifier as grantClassifier
import json, datetime

#Parameters

TOPN_DEFAULT = 5

#Text formatting functions:

def __pad(strdata):
    # Pads `strdata` with a Request's callback argument, if specified, or does
    # nothing.
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
    # Serializes `args` and `kwargs` as JSON. Supports serializing an array
    # as the top-level object, if it is the only argument.  
    indent = None
    if (current_app.config.get('JSONIFY_PRETTYPRINT_REGULAR', False)
            and not request.is_xhr):
        indent = 2
    return json.dumps(args[0] if len(args) is 1 else dict(*args, **kwargs),
                      indent=indent)

#Main program

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pymssql://user:textmining@172.16.8.60:1433/text_classification'
app.config['SQLALCHEMY_ECHO'] = True
db = SQLAlchemy(app)

myData = grantClassifier.grantData(picklefile = 'gd.pkl')
myData.load()

myClassifier = grantClassifier.grantClassifier(myData)
myClassifier.load()

#SQLAlchemy models

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
    def loadfromjson(cls, inStr):
        inDict = json.loads(inStr)
        inDict['entry_date'] = datetime.datetime.now()
        return cls(**inDict)

#Flask routes

@app.route('/svm_autoclassify/<grantDescription>')
def svm_classify(grantDescription = None):
    # Main end-point. Given a text returns the text, accuracy, and predicted class 
    prediction, prob = myClassifier.predict_single(grantDescription)
    return current_app.response_class(__pad(__dumps(
        data=[i for i in [prediction,prob,grantDescription] ])),mimetype=__mimetype())

@app.route('/svm_multiple/<grantDescription>/<topn>')
def svm_multiple(grantDescription = None, topn = None):
    # Alternate end-point. Given a text returns the text, plus the accuracy, and predicted class 
    # of top-scoring classes
    try: 
        npredict = int(topn)
    except ValueError:
        npredict = TOPN_DEFAULT
    top_classes = myClassifier.predict_multiple(grantDescription, npredict=npredict)
    return current_app.response_class(__pad(json.dumps([grantDescription,top_classes])),mimetype=__mimetype())

@app.route('/add_label/<something>', methods=['GET', 'POST'])
def add_label(something = None):
    data = request.data
    newRow = retrainedGrants.loadfromjson(data)
    
    db.session.add(newRow)
    db.session.commit()
    return 'Added row to database\n' + data
    
if __name__ == '__main__':
    app.run(host='0.0.0.0',port =9090) 
