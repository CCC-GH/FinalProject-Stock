# -*- coding: utf-8 -*-
'''
Created on Sat Feb 27 14:21:41 2021
@author: coffm
$ python3 -m install pyfinance
$ pip install yfinance --upgrade --no-cache-dir
https://pypi.org/project/pyfinance/
https://pandas-datareader.readthedocs.io/en/latest/#
'''

import pandas as pd
#import yfinance as yf
import numpy as np

import matplotlib.pyplot as plt 
#import seaborn as sns
#sns.set_style('whitegrid')
plt.style.use('fivethirtyeight') 
from datetime import datetime
%matplotlib inline


# Read stock data from yahoo
from pandas_datareader.data import DataReader

beginDate ='2021-01-01'
endDate ='2021-02-28'       
#tickerList=['AAPL','GOOG','MSFT','AMZN']
tickerList=['AAPL','GOOG','MSFT','AMZN','DIS','TSLA','GME','AMC']

# Set DataFrame names
for ticker in tickerList:
    globals()[ticker] = DataReader(ticker,'yahoo', beginDate, endDate)
print(globals())

#coList = [AAPL, GOOG, MSFT, AMZN]
#coName = ['Apple','Google','Microsoft','Amazon']
coList = [AAPL, GOOG, MSFT, AMZN, DIS, TSLA, GME, AMC]
coName = ['Apple','Google','Microsoft','Amazon','Disney','Tesla','GameStop','AMC']


for co, i in zip(coList, coName):
    co['coName'] = i
df = pd.concat(coList, axis=0)
df.tail(10)

# plt.figure(figsize=(12, 8))
# plt.subplots_adjust(top=1.25, bottom=1.2)
# for i, company in enumerate(coList, 1):
#     plt.subplot(2, 2, i)
#     company['Adj Close'].plot()
#     plt.ylabel('Adj Close')
#     plt.xlabel(None)
#     plt.title(f'{tickerList[i - 1]}')
 
daysMovAvg= [5, 10, 20, 40]
for ma in daysMovAvg:
    for co in coList:
        columnName = f'MA-{ma} days'
        co[columnName] = co['Adj Close'].rolling(ma).mean()

#df.groupby('coName').hist(figsize=(12, 12));

fig, axes = plt.subplots(nrows=2, ncols=4)
fig.set_figheight(8)
fig.set_figwidth(15)
AAPL[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[0,0])
axes[0,0].set_title('Apple')
GOOG[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[0,1])
axes[0,1].set_title('Google')
MSFT[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[0,2])
axes[0,2].set_title('Microsoft')
AMZN[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[0,3])
axes[0,3].set_title('Amazon')
DIS[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[1,0])
axes[1,0].set_title('Disney')
TSLA[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[1,1])
axes[1,1].set_title('Tesla')
GME[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[1,2])
axes[1,2].set_title('Gamestop')
AMC[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[1,3])
axes[1,3].set_title('AMC')
fig.tight_layout()

df = DataReader('AAPL', 'yahoo', beginDate, endDate)
data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
training_data_len = int(np.ceil( len(dataset) * .95 ))
training_data_len

# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data

# Create the training data set 
# Create the scaled training data set
train_data = scaled_data[0:int(training_data_len), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if i<= 61:
        print(x_train)
        print(y_train)
        print() 
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)
# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# x_train.shape

from keras.models import Sequential
from keras.layers import Dense, LSTM
# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
# Convert the data to a numpy array
x_test = np.array(x_test)
# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
# Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(rmse)

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
# Show the valid and predicted prices
print(valid)









   
# print(ticker.institutional_holders)
# print(history.head())
# history.to_csv('yFinance_StockReport.csv')

# history = history.reset_index()
# for i in ['Open','High','Close','Low']: 
#      history[i]=history[i].astype('float64')

# plt.figure(figsize=(10,5))
# history['Date'] = pd.to_datetime(history.Date,format='%Y/%m/%d')
# history.index = history['Date']
# plt.plot(history['Close'])