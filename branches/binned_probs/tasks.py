from celery import Celery
from celery.schedules import crontab 
import urllib2

RETRAIN_URL = 'http://localhost:9090/retrain'
TIME_ZONE = 'America/Los_Angeles'
RETRAIN_HOUR = 3 # 3 AM

"""
Initiates nightly retraining. 
Requires celery and redis. 

Run with:
celery -A tasks worker --loglevel=info --beat

Can be replaced with cron or at. 
"""

celery_app = Celery('tasks',broker='redis://localhost:6379')
celery_app.conf.update(CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379',
    CELERY_TIMEZONE = TIME_ZONE,
    CELERYBEAT_SCHEDULE = {
        'every-minute': {
            'task': 'tasks.retrain',
            'schedule': crontab(minute=0, hour=RETRAIN_HOUR),
            'args': None,
        },
    })

@celery_app.task
def retrain():
    response = urllib2.urlopen(RETRAIN_URL)