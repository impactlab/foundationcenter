from flask import Flask, request, json, current_app
<<<<<<< HEAD
import grantClassifier as grantClassifier
=======
import grantClassifier2 as grantClassifier
>>>>>>> 1fdf93692daef08477da62d59a848b60e0e23ef4

app = Flask(__name__)

myData = grantClassifier.grantData(picklefile = 'gd.pkl')
myData.load()

myClassifier = grantClassifier.grantClassifier(myData)
myClassifier.load()

@app.route('/svm_autoclassify/<grantDescription>')
def svm_classify(grantDescription=None):
    # Main end-point. Given a text returns the text, accuracy, and predicted class 

    prediction, prob = myClassifier.predict_single(grantDescription)
    return current_app.response_class(__pad(__dumps(
        data=[i for i in [prediction,prob,grantDescription] ])),mimetype=__mimetype())

if __name__ == '__main__':
    app.run(host='0.0.0.0',port =9090) 

#Formatting functions:

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
