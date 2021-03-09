import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
from pytrends.request import TrendReq
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
today = datetime.today()
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

import ML

today = datetime.today()

ticker = 'MSFT'

script = "9ho5HG7o00PT-g"
secret = "2CQTFbYyYp5aLEN7bHkKGO8X4E3YHQ"

beginDate = '2020-12-01'
endDate = datetime.today().strftime('%Y-%m-%d')

def df_from_response(res):
    df = pd.DataFrame()

    for post in res.json()['data']['children']:
        df = df.append({
            'subreddit': post['data']['subreddit'],
            'title': post['data']['title'],
            'selftext': post['data']['selftext'],
            'num_comments': post['data']['num_comments'],
            'upvote_ratio': post['data']['upvote_ratio'],
            'date': datetime.fromtimestamp(post['data']['created_utc']).strftime('%Y-%m-%d'),
            'ups': post['data']['ups'],
            'downs': post['data']['downs'],
            'score': post['data']['score'],
            'kind': post['kind'],
            'id': post['data']['id'],
        }, ignore_index=True)
    return df

auth = requests.auth.HTTPBasicAuth(script, secret)
data = {'grant_type': 'password',
        'username': 'NoShare8264',
        'password': 'NewPass227$'}

headers = {'User-Agent': 'Final_Project/0.0.1'}

request = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)
token = f"bearer {request.json()['access_token']}"
headers = {**headers, **{'Authorization': token}}

posts = pd.read_csv("trimmed_posts.csv")
selected_cols = ['title','selftext']

def stock_info(ticker):
    try:
        new_df = posts[posts[selected_cols].apply(lambda x: x.str.contains(ticker)).all(axis=1)]
        data = new_df
        data['Ticker'] = ticker
        group_df = new_df.groupby('created_utc') \
        .agg({'id':'count', 'num_comments':'mean', 'score':'mean'}) \
        .rename(columns={'id':'post_count','num_comments':'avg_comments', 'score':'avg_score'}) \
        .reset_index()
        group_df['Ticker'] = ticker
    except:
        places = {'created_utc': 0, 'post_count': 0, 'avg_comments': 0, 'avg_score': 0, 'ticker': ticker}
        group_df = pd.DataFrame.from_dict(places)

    super_posts_df = data.loc[data['score'] > 300]
    if not super_posts_df.empty:
        super_posts_df = super_posts_df.groupby(['created_utc', 'Ticker'])['score'].apply(lambda x: (x>=300).sum()).reset_index(name='Count_Score_300')

    try:
        pytrend = TrendReq(hl='en-US', tz=360)
        pytrend.build_payload(kw_list=[ticker], timeframe=beginDate + ' ' + endDate, geo='US')
        df = pytrend.interest_over_time()
        df['Noise'] = df[ticker]
        df[ticker]= ticker
        df.index.names = ['Date']
        df.columns = ['Ticker','isPartial', 'Noise']
        mergedNoise = df
    except:
        noise = {'Ticker': ticker, 'isPartial': 'No', 'Noise': 0}
        mergedNoise = pd.DataFrame.from_dict(noise)

    group_df['created_utc'] = pd.to_datetime(group_df['created_utc'])
    if not super_posts_df.empty:
        super_posts_df['created_utc'] = pd.to_datetime(super_posts_df['created_utc'])
    merged = group_df.merge(mergedNoise, left_on=['created_utc', 'Ticker'], right_on=['Date', 'Ticker'], how='left')
    if not super_posts_df.empty:
        merged = merged.merge(super_posts_df, left_on=['created_utc', 'Ticker'], right_on=['created_utc', 'Ticker'], how="left")
    else:
        merged['Count_Score_300'] = 0
    merged.drop(columns=['isPartial'])

    stockData = yf.download(ticker, start=beginDate, end=endDate)
    stockData['Ticker'] = ticker
    stockReport = pd.DataFrame(stockData, columns= ['Ticker','Adj Close','Volume'])
    merged = merged.merge(stockReport, left_on=['created_utc', 'Ticker'], right_on=['Date', 'Ticker'], how='left')
    merged['post_count_change'] = merged['post_count'].pct_change()
    merged['avg_score_change'] = merged['avg_score'].pct_change()
    merged['Adj Close_change'] = merged['Adj Close'].pct_change()
    merged['Volume_change'] = merged['Volume'].pct_change()
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged.replace([np.inf, -np.inf], np.nan).dropna(subset=['post_count_change', 'avg_score_change', 'Volume_change'], how="all")
    merged = merged.dropna()

    expected_posts = merged['post_count'].mean()
    expected_avg_comments = merged['avg_comments'].mean()
    expected_volume_change = merged['Volume'].mean()
    expected_300_count = merged['Count_Score_300'].mean()

    data = pd.DataFrame()
    length = 0
    try:
        params = {'limit': 100,
                'q': ticker,
                'restrict_sr': True}
        res = requests.get("https://oauth.reddit.com/r/WallStreetBets/search",
                        headers=headers,
                        params=params)

        new_df = df_from_response(res)
        new_df.sort_values(by=['date'], inplace=True, ascending=False, axis=0)
        row = new_df.iloc[len(new_df)-1]
        fullname = row['kind'] + '_' + row['id']
        params['after'] = fullname
        data = data.append(new_df, ignore_index=True)
    except:
        data

    data['date'] = pd.to_datetime(data['date'])
    try:
        data['Ticker'] = ticker
        live_group = data.groupby('date') \
        .agg({'id':'count', 'num_comments':'mean', 'score':'mean'}) \
        .rename(columns={'id':'post_count','num_comments':'avg_comments', 'score':'avg_score'}) \
        .reset_index()
        live_group['Ticker'] = ticker
    except:
        places = {'created_utc': 0, 'post_count': 0, 'avg_comments': 0, 'avg_score': 0, 'ticker': ticker}
        live_group = pd.DataFrame.from_dict(places)

    live_group = live_group.merge(stockReport, left_on=['date', 'Ticker'], right_on=['Date', 'Ticker'], how='left')
    live_group['Adj Close_change'] = live_group['Adj Close'].pct_change()
    live_group['Volume_change'] = live_group['Volume'].pct_change()
    live_group['post_count_change'] = live_group['post_count'].pct_change()
    live_group['avg_score_change'] = live_group['avg_score'].pct_change()


    xfits = live_group[live_group.date > datetime.datetime.now() - pd.to_timedelta("3day")]

    return({})


