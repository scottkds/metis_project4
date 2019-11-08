import requests
import json
from math import ceil, floor
from datetime import datetime, date, timedelta
import time
import pickle
from collections import namedtuple
import pandas as pd
import pprint

def unix_time(date_time):
    """Returns the unix timestamp for a formatted date string."""
    return floor((datetime.strptime(date_time, '%m/%d/%Y %H:%M:%S')).timestamp())

def format_request(subreddit, before, after, count):
    return 'https://api.pushshift.io/reddit/search/submission?subreddit={}&before={}&after={}&size={}' \
        .format(subreddit, unix_time(before), unix_time(after), count)

def get_posts(subreddit, before, after, count):
    return json.loads(requests.get(format_request(subreddit, before, after, count)).content)['data']

def get_commets():
    return json.loads(requests.get('https://api.pushshift.io/reddit/search/comment/?subreddit=FoodPorn&after=1d&size=1').content)['data']

def get_commets_by_id(id, count):
    url = 'https://api.pushshift.io/reddit/search/comment/?subreddit=FoodPorn&link_id=t3_{}&size=20'.format(id)
    return json.loads(requests.get(url).content)['data']

def get_comment_parts(id, count, min_score=5):
    return map(lambda elem: (elem['link_id'], elem['body']), 
               filter(lambda elem: elem['score'] > min_score, get_commets_by_id(id, count)))

def get_comments_from_ids(ids, count, min_score=5):
    list_of_comments = []
    for id in list(ids):
        # print(id, len(id))
        list_of_comments.append(get_comment_parts(id, count, min_score=5))
    return list_of_comments




if __name__ == '__main__':
    pp = pprint.PrettyPrinter(indent=4)
    # print('Test request format:', format_request('FoodPorn', '11/01/2019 00:00:00', '01/01/2018 00:00:00', 1))
    # print('Test request output:', get_posts('FoodPorn', '11/01/2019 00:00:00', '01/01/2018 00:00:00', 1))
    # print('Test request comment:', get_commets_by_id('7netog', 20))
    # out = list(get_comment_parts('7netog', 20))
    # print(out[0][1])

    posts_df = pd.read_csv('/Users/scott/p4/data/interim/fp_posts.csv')
    print(posts_df.head())

    comments = get_comments_from_ids(posts_df['id'], 20, min_score=1)

    with open('/Users/scott/p4/data/interim/cooments.pkl', 'wb') as f:
        pickle.dump(comments, f)
    
