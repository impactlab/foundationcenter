Foundation Center automated classification API
================
The Foundation Center (http://foundationcenter.org) is a NYC-based nonprofit that maintains a 
database of philanthropic grants. The Center has produced a taxonomy (http://taxonomy.foundationcenter.org/) that
they use to classify grants into different categories, for example 'public health' or 'natural resources 
conservation and protection'. 

At present the grants are classified by human experts. The goal of the project is to supplement the human classifications 
with automated classifications in two ways: first, to offer suggestions to the human classifiers so that they can classify 
grants more efficiently, and second, to classify lower-priority grants in a completely automated way. In both cases it is important 
that the machine classifier produce a confidence score in addition to a category. 

##Interface
Implements the following API for machine-assisted classification of short grant descriptions. 

**/text**
Returns a new text for potential human classification, with accompanying metdata. 

Output format: JSON dictionary where the key 'text' maps to the text to be classified, 
additional metadata comes from the other fields in the databae table where the texts reside. 
```
{'text': 'grant description to be classified'
'grant_id': 8161203
'field': 'value' 
...}
```

**/classify**
Given a text as input, returns machine classifications of the text category, 
selecting from texts with the fewest human classifications. 
In the application settings this can be set to return primarily texts where the machine classification has low confidence. 

Additional input (HTTP POST with content-type set to application/json): 
JSON dictionary where 'text' maps to the text to be classified and 'npredict' maps to 
the number of suggestions to return
```
{'text': 'text to be classified goes here'
'npredict': 10}
```

Output format: JSON dictionary, each key is the name of a classifier and maps to a list of npredict suggestions. 
```
{"svm": 
[{"category": "A63", "confidence": 1.0, "rank": 0}, 
{"category": "A6E", "confidence": 0.94999999999999996, "rank": 1}, 
{"category": "B2R", "confidence": 0.41791044776119401, "rank: 2}, ...]
```
**/label**
Updates the database with a human classification 

Additional input (HTTP POST with content-type set to application/json): 
JSON dictionary where keys correspond to field names in the database table for annotations. 
```
{'user_id': '90001',
'grant_id': '8571001',
'description': 'text that was classified',
'activity': 'W31'}
```
Returns: Empty page on success, HTTP 400 on failure 

**/train**
Initiates retraining of the classifier, to be called by a task scheduler or cronjob
Returns: Empty page on success, HTTP 400 on failure 

**/count_retrains**
Initiates retraining of the classifier, to be called by a task scheduler or cronjob
Returns: Empty page on success, HTTP 400 on failure 

## Build 
TODO
