import urllib2
import json

ADD_URL = 'http://localhost:9090/add_label/test'
def test_add(url = ADD_URL):
    testdata = {
    'user_id': '20001',
    'grant_id': '00001',
    'description': 'For enabiling the proliferation of even more awesomeness',
    'activity': 'A31',
    'activity_svm': 'A21',
    'activity_rule': 'A11',
    'model_version': '0.07'
    }
    
    jdata = json.dumps(testdata)
    print jdata
    req = urllib2.Request(url, jdata, {'Content-Type': 'application/json'})
    urllib2.urlopen(req)
    