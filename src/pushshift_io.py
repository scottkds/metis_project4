# Imports
import requests
import json
from math import ceil, floor
from datetime import datetime, date, time, timedelta
from collections import namedtuple

post = namedtuple('post', ['id', 'author', 'title', 'url', 'img', 'created'])
post('123', 'Jim', 'Stuff and things', 'here', 'there', 'today')

# Function to convert a timestamp to a unix timestamp
def unix_time(date_time):
    epoch = datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0)
    # return int((datetime.strptime(date_time, '%m/%d/%Y %H:%M:%S') - epoch).total_seconds())
    return floor((datetime.strptime(date_time, '%m/%d/%Y %H:%M:%S')).timestamp())

def get_posts(subreddit, before_str, after_str, count):
    """Returns a list of reponses from the Pushshift.io API"""
    request_string = 'https://api.pushshift.io/reddit/search/submission?subreddit={}&before={}&after={}&size=25'
    request_string = request_string.format(subreddit, unix_time(before_str), unix_time(after_str))
    response = requests.get(request_string)
    assert response.status_code == 200
    return json.loads(response.content)

rs = get_posts('FoodPorn', '11/01/2019 00:00:00','01/01/2018 00:00:00', 25)


l_data = []
for p in rs['data']:
    l_data.append(post(p['id'], p['author'], p['title'], p['permalink'], p['url'], p['created_utc']))
l_data

(datetime.now().timestamp())
# timestamp 1572591600
# minus epoch 1572566400

datetime.fromtimestamp(unix_time('11/01/2019 00:00:00'))

datetime(year=2019, month=1, day=1) - timedelta(seconds=1)

unix_time('11/01/2019 00:00:00')

before = unix_time('01/01/2018 00:00:00')
before

after = unix_time('01/01/2019 00:00:00')
after
before = unix_time('11/05/2019 00:00:00')

subreddit = 'FoodPorn'

request_string = 'https://api.pushshift.io/reddit/search/submission?subreddit={}&before={}&after={}&size=25'

request_string
request_string = request_string.format(subreddit, before, after)
request_string
r = requests.get(request_string)
r
response_json = json.loads(r.content)
type(response_json)
response_json['data']
