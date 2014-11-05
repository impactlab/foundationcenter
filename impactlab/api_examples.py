import urllib2
import json

ADD_URL = 'http://localhost:9090/add_label/example'
def example_add(url = ADD_URL):
    testdata = {
    'user_id': '20001',
    'grant_id': '00001',
    'description': 'For enabiling the proliferation of even more awesome wombats',
    'activity': 'W31',
    'activity_svm': 'W21',
    'activity_rule': 'W11',
    'model_version': '0.07'
    }
    
    jdata = json.dumps(testdata)
    print jdata
    req = urllib2.Request(url, jdata, {'Content-Type': 'application/json'})
    urllib2.urlopen(req)

BATCH_URL = 'http://localhost:9090/svm_batch/example'
def example_batch(url = BATCH_URL):
    testdata = ['education', 'ballet', 'diabetes']
    jdata = json.dumps(testdata)
    print jdata
    req = urllib2.Request(url, jdata, {'Content-Type': 'application/json'})
    response = urllib2.urlopen(req)    
    print response.read()
    