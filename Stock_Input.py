import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from pandas_datareader.data import DataReader
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay 
from datetime import datetime
from pytrends.request import TrendReq
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from ML import *


def stock_info(ticker):

    
    
    today = datetime.today()
    beginDate ='2020-01-01'
    endDate = datetime.datetime.now().date()
    
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



    df = DataReader(ticker, 'yahoo', beginDate, endDate)
    df['Close'] = df['Adj Close']
    df = df.drop(columns=['Adj Close','High','Low','Open'], axis=1)
    modelData = df['Close'].to_frame()
    five_rolling = modelData.rolling(window=5).mean()
    ten_rolling = modelData.rolling(window=10).mean()
    twenty_rolling = modelData.rolling(window=20).mean()
    fifty_rolling = modelData.rolling(window=50).mean()
    hundred_rolling = modelData.rolling(window=100).mean()

    futureDays = 10
    modelData['Target'] = modelData['Close'].shift(-futureDays)
    
    X = np.array(modelData.drop(['Target'], 1))[:-futureDays]
    y = np.array(modelData['Target'])[:-futureDays]

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25)
    Xfuture = modelData.drop(['Target'], 1)[:-futureDays]
    Xfuture = Xfuture.tail(futureDays)
    Xfuture = np.array(Xfuture)
    
    train_data, test_data = df[0:int(len(df)*0.7)], df[int(len(df)*0.75):]
    training_data = train_data['Close'].values
    test_data = test_data['Close'].values
    history = [x for x in training_data]
    model_predictions = []
    N_test_observations = len(test_data)
    for time_point in range(N_test_observations):
        model = ARIMA(history, order=(4,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        model_predictions.append(yhat)
        true_test_value = test_data[time_point]
        history.append(true_test_value)
    MSE_error = mean_squared_error(test_data, model_predictions)
    
    linear = LinearRegression().fit(Xtrain, ytrain)
    linearPrediction = linear.predict(Xfuture)
    linearResult = linear.score(Xtrain, ytrain)
    
    valid = modelData[X.shape[0]:]
    valid['Target'] = predictions
    
    tree = DecisionTreeRegressor().fit(Xtrain, ytrain)
    treePrediction = tree.predict(Xfuture)
    treeResult = tree.score(Xtrain, ytrain)
    predictions = treePrediction
    valid = modelData[X.shape[0]:]
    valid['Predict'] = predictions
    
    todaysDate = datetime.datetime.now().date()
    futureDays = 10
    us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    futureDates = pd.date_range(todaysDate, periods = futureDays, freq=us_bd)
    combinedDFcol =['Close','Predict','SM1','SM2','SM3','SM4'] 
    futureDF=pd.DataFrame(index=futureDates, columns=combinedDFcol)
    futureDF['Predict']=model_fit.forecast(steps=futureDays)[0]
    combinedDF=df.append(futureDF, ignore_index = False)
    combinedDF.index.names = ['Date']
    currInfo=yf.Ticker(ticker).info
    
    currInfo=yf.Ticker(ticker).info
    infoDict={
        'longName: ':currInfo['symbol'],
        'Current Ask/Bid: ': str(currInfo['ask'])+'/'+str(currInfo['bid']),
        'Open Price: ': str(round(currInfo['open'],2)),
        'High/Low Price: ': str(currInfo['dayHigh'])+'/'+str(currInfo['dayLow']),
        'Avg Volume: ': str(currInfo['averageVolume']), 
        'Volume: ': str(currInfo['volume']),
        '52w High: ': str(round(currInfo['fiftyTwoWeekHigh'],2)),
        '52w Low: ': str(round(currInfo['fiftyTwoWeekLow'],2)),
        'MorningStar Rating: ':str(currInfo['morningStarOverallRating']),
        'Short Ratio: ': str(currInfo['shortRatio'])
        }    
        
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
    
    super_posts_live = data.groupby('date')['score'].apply(lambda x: (x >= 300).sum()).reset_index(name='Count_Score_300')
    if not super_posts_live.empty:
        super_posts_live = super_posts_live
    else:
        super_posts_live['Count_Score_300'] = 0
        
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
        
    live_group = live_group.merge(super_posts_live, left_on=['date'], right_on=['date'], how="left")
    live_group = live_group.merge(stockReport, left_on=['date', 'Ticker'], right_on=['Date', 'Ticker'], how='left')
    live_group['Adj Close_change'] = live_group['Adj Close'].pct_change()
    live_group['Volume_change'] = live_group['Volume'].pct_change()
    live_group['post_count_change'] = live_group['post_count'].pct_change()
    live_group['avg_score_change'] = live_group['avg_score'].pct_change()


    xfits = live_group[live_group.date > datetime.now() - pd.to_timedelta("3day")]
    xfits_dates = xfits['date']
    xfittings = xfits[['post_count_change', 'avg_score_change', 'Volume_change', 'Count_Score_300']]
    
    X_fits_scaled = X_scaler.transform(xfittings)
    social_predictions = model.predict(X_fits_scaled)
    social_predictions = pd.DataFrame(social_predictions.reshape(-1,1))
    Xnew, _ = make_regression(n_samples=10, n_features=4, noise=0.01, random_state=1)
    ynew = model.predict(Xnew)
    future_predict_df = pd.DataFrame(ynew.reshape(-1,1))
    live_group.sort_values(by=['date'], inplace=True, ascending=False)
    
    future_dates = pd.date_range(start=today, periods=10).strftime('%Y-%m-%d')
    futureDates = pd.date_range(todaysDate, periods = futureDays, freq=us_bd)
    futureDates['SMPredict']=df.append(future_predict_df, ignore_index=False)
    
    SMPredict=pd.DataFrame(index=future_dates, columns=combinedDFcol)
    future_predict_df = pd.DataFrame(ynew.reshape(-1,1), future_dates)
    future_predict_df.rename(columns={0: "SMPredict"})
    SMPredict = SMPredict.merge(future_predict_df, left_index=True, right_index=True, how="left")
    
    xfits = xfits.drop(columns=['Ticker', 'Adj Close_change', 'Volume_change', 'post_count_change', 'avg_score_change'])
    xfits['SMPredictions'] = social_predictions
    xfits = xfits.set_index('date')
    xfits=df.append(SMPredict, ignore_index = False)
    xfits = xfits.drop(columns=['Ticker', 'isPartial', 'Noise'])
    xfits.index.name = "Date"
    
    combinedDFcol =['Close','Predict','SM1','SM2','SM3','SM4'] 
    futureDF=pd.DataFrame(index=futureDates, columns=combinedDFcol)
    futureDF['Predict']=model_fit.forecast(steps=futureDays)[0]
    combinedDF=df.append(futureDF, ignore_index = False)
    combinedDF.index.name = ['Date']
    
    today_dict = xfits.to_dict()
    futureSM_prediction = SMPredict.to_dict()
    carls_dict = combinedDF.to_dict()

    return({today_dict, futureSM_prediction, carls_dict})


