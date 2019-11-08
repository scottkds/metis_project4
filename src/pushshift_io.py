# Imports
import requests
import json
from math import ceil, floor
from datetime import datetime, date, timedelta
import time
import pickle
from collections import namedtuple
import pandas as pd

post = namedtuple('post', ['id', 'author', 'title', 'url', 'img', 'created'])
pt = post('123', 'Jim', 'Stuff and things', 'here', 'there', 'today')
pt[0]

# Function to convert a timestamp to a unix timestamp
def unix_time(date_time):
    # epoch = datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0)
    # return int((datetime.strptime(date_time, '%m/%d/%Y %H:%M:%S') - epoch).total_seconds())
    return floor((datetime.strptime(date_time, '%m/%d/%Y %H:%M:%S')).timestamp())

def get_posts(subreddit, before, after, count):
    """Returns a list of reponses from the Pushshift.io API"""
    response_list = []
    while not response_list:
        request_string = 'https://api.pushshift.io/reddit/search/submission?subreddit={}&before={}&after={}&size={}'
        request_string = request_string.format(subreddit, before, after, count)
        response = requests.get(request_string)
        assert response.status_code == 200
        response_list = json.loads(response.content)['data']
        if response_list:
            return response_list
        else:
            time.sleep(2)
    return ['Error']


rs = get_posts('FoodPorn', unix_time('11/01/2019 00:00:00'), unix_time('01/01/2018 00:00:00'), 25)


l_data = []
keep_going = True
last_post = -1
start = unix_time('01/01/2018 00:00:00')
stop = unix_time('11/01/2019 00:00:00')
while keep_going:
    rs = get_posts('FoodPorn', stop, start, 100)
    for p in rs:
        l_data.append(post(p['id'], p['author'], p['title'], p['permalink'], p['url'], p['created_utc']))
    if len(rs) < 25:
        keep_going = False
    elif len(l_data) % 1000 == 0:
        with open('../data/raw/posts.pkl', 'wb') as f:
            pickle.dump(l_data, f)
    else:
        start = rs[-1]['created_utc'] + 1
        print('Entries so far:', len(l_data))

l_data[-1]

!pwd
datetime.fromtimestamp(1572590667)

with open('../data/raw/posts.pkl', 'wb') as f:
    pickle.dump(l_data, f)


l_data[-10:]

def make_frame(data):
    columns = ('id', 'author', 'title', 'url', 'img', 'created')
    data_dict = {'id': [], 'author': [], 'title': [], 'url': [], 'img': [], 'created': []}
    for item in data:
        for idx, name in enumerate(columns):
            data_dict[name].append(item[idx])
    return pd.DataFrame(data_dict)

posts_df = make_frame(l_data)
posts_df.head()
posts_df.shape
posts_df.to_csv('../data/interim/fp_posts.csv', index=False)

posts_df.head(10)
