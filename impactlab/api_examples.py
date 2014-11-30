import urllib2
import json

ADD_URL = 'http://localhost:9090/label'
def example_add(url = ADD_URL):
    testdata = {
    'user_id': '90001',
    'grant_id': '90001',
    'description': 'Sunday test',
    'activity': 'W31',
    'activity_svm': 'W21',
    'activity_rule': 'W11',
    'model_version': '0.07'
    }
    
    jdata = json.dumps(testdata)
    print jdata
    req = urllib2.Request(url, jdata, {'Content-Type': 'application/json'})
    urllib2.urlopen(req)

CLASSIFY_URL = 'http://localhost:9090/classify'
def example_classify(url = CLASSIFY_URL):
    testdata = {'text': 'For Chicago Make a Ballet on Tour and Healthy Bodies Program',
                'npredict': '10'
    }
    jdata = json.dumps(testdata)
    print jdata
    req = urllib2.Request(url, jdata, {'Content-Type': 'application/json'})
    response = urllib2.urlopen(req)
    print response.read()
    


BATCH_URL = 'http://localhost:9090/svm_batch/example'
def example_batch(url = BATCH_URL):
    testdata = ['education', 'ballet', 'diabetes']
    jdata = json.dumps(testdata)
    print jdata
    req = urllib2.Request(url, jdata, {'Content-Type': 'application/json'})
    response = urllib2.urlopen(req)    
    print response.read()
    