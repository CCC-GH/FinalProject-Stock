import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt 
from pandas_datareader.data import DataReader
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from mplfinance import candlestick_ohlc
import matplotlib.dates as mdates
import pprint
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import accuracy_score
beginDate ='2021-01-01'
endDate = datetime.datetime.now().date()     
#
#df = pd.read_csv('XXXXXX.csv',index_col=0)
# note: if using line above, loading from csv, comment out section below
#
# User input and load ticker data from yahoo finance
while True:
    try: 
        ticker = input('Enter Stock Ticker: ').upper()
        df = DataReader(ticker, 'yahoo', beginDate, endDate)
        #df = yf.Ticker(ticker, start=beginDate, end=endDate)
    except:
        print('\nStock Ticker Symbol does not exist!\n')
        continue;
    break
#
# Use Adj Close instead of Close
df['Close'] = df['Adj Close']
df = df.drop(columns=['Adj Close','High','Low','Open'], axis=1)
print(f'\n{df.describe()}\n')
df['Moving_av']= df['Close'].rolling(window=50,min_periods=0).mean()
df['Moving_av'].plot()

i=1
rate_increase_in_vol=[0]
rate_increase_in_close=[0]
while i<len(df):
    rate_increase_in_vol.append(df.iloc[i]['Volume']-df.iloc[i-1]['Volume'])
    rate_increase_in_close.append(df.iloc[i]['Close']-df.iloc[i-1]['Close'])
    i+=1
df['Increase_in_vol']=rate_increase_in_vol
df['Increase_in_close']=rate_increase_in_close
df['Increase_in_vol'].plot()
df['Increase_in_close'].plot()

df_ohlc= df['Close'].resample('10D').ohlc()
df_volume=df['Volume'].resample('10D').sum()
df_ohlc.reset_index(inplace=True)
df_ohlc['Date']=df_ohlc['Date'].map(mdates.date2num)
ax1=plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2=plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1 , sharex=ax1)
ax1.xaxis_date()
candlestick_ohlc(ax1,df_ohlc.values, width=2, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)
#
#
df3=df
df3.fillna(0, inplace=True)
y_df=df3[['Close','Volume']]
col_y=y_df.columns
y_df=df3[['Close','Volume']]
y_df_mod=y_df.drop(['Close','Volume'],axis=1)
y_df_mod.column
Drop_cols=col_y
Drop_cols=Drop_cols.tolist()
Drop_cols.append('Date')
X_df=df3.drop(Drop_cols,axis=1)
X_df.columns
X=X_df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def model():
    mod=Sequential()
    mod.add(Dense(32, kernel_initializer='normal',input_dim = 200, activation='relu'))
    mod.add(Dense(64, kernel_initializer='normal',activation='relu'))
    mod.add(Dense(128, kernel_initializer='normal',activation='relu'))
    mod.add(Dense(256, kernel_initializer='normal',activation='relu'))
    mod.add(Dense(4, kernel_initializer='normal',activation='linear'))
    mod.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy','mean_absolute_error'])
    mod.summary()
    return mod
#
#
#
X_test_scaled=sc_2.transform(X_test)
y_pred=model_ANN.predict(X_test)

y_pred_mod=[]
y_test_mod=[]
for i in range(0,2):
    j=0
    y_pred_temp=[]
    y_test_temp=[]
    while(j<len(y_test)):
        y_pred_temp.append(y_pred[j][i])
        y_test_temp.append(y_test[j][i])
        j+=1
    y_pred_mod.append(np.array(y_pred_temp))
    y_test_mod.append(np.array(y_test_temp))
df_res=pd.DataFrame(list(zip(y_pred_mod[0],y_pred_mod[1],y_test_mod[0],y_test_mod[1])),columns=['Pred_high','Pred_low','Actual_high','Actual_low'])
print(df_res.head())

import matplotlib.pyplot as plt
ax1=plt.subplot2grid((4,1), (0,0), rowspan=5, colspan=1)
ax1.plot(df_res_2.index, df_res_2['Pred_high'], label="Pred_high")
ax1.plot(df_res_2.index, df_res_2['Actual_high'], label="Actual_high")
plt.legend(loc="upper left")
plt.xticks(rotation=90)
plt.show()

ax1=plt.subplot2grid((4,1), (0,0), rowspan=5, colspan=1)
ax1.plot(df_res_2.index, df_res_2['Pred_low'], label="Pred_low")
ax1.plot(df_res_2.index, df_res_2['Actual_low'], label="Actual_low")
plt.legend(loc="upper left")
plt.xticks(rotation=90)
plt.show()

df_res_2=df_res[200:300]
ax1=plt.subplot2grid((4,1), (0,0), rowspan=5, colspan=1)
ax1.plot(df_res_2.index, df_res_2['Pred_high'], label="Pred_high")
ax1.plot(df_res_2.index, df_res_2['Actual_high'], label="Actual_high")
plt.legend(loc="upper left")
plt.xticks(rotation=90)
plt.show()

ax1=plt.subplot2grid((4,1), (0,0), rowspan=5, colspan=1)
ax1.plot(df_res_2.index, df_res_2['Pred_low'], label="Pred_low")
ax1.plot(df_res_2.index, df_res_2['Actual_low'], label="Actual_low")
plt.legend(loc="upper left")
plt.xticks(rotation=90)
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test_mod[1], y_pred_mod[1])
ax.plot([y_test_mod[1].min(),y_test_mod[1].max()], [y_test_mod[1].min(), y_test_mod[1].max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

fig, ax = plt.subplots()
ax.scatter(y_test_mod[0], y_pred_mod[0])
ax.plot([y_test_mod[0].min(),y_test_mod[0].max()], [y_test_mod[0].min(), y_test_mod[0].max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

# import bs4 as bs
# import pickle
# import requests
# def save_tickers():
# 	resp=requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# 	soup=bs.BeautifulSoup(resp.text)
# 	table=soup.find('table',{'class':'wikitable sortable'})
# 	tickers=[]
# 	for row in table.findAll('tr')[1:]:
# 		ticker=row.findAll('td')[0].text[:-1]
# 		tickers.append(ticker)
# 	with open("tickers.pickle",'wb') as f:
# 		pickle.dump(tickers, f)
# 	return tickers
# save_tickers()