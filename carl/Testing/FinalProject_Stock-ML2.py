'''
$ python3 -m install pyfinance
$ pip install yfinance --upgrade --no-cache-dir
$ pip install tensorflow
https://pypi.org/project/pyfinance/
https://pandas-datareader.readthedocs.io/en/latest/#
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from datetime import datetime
%matplotlib inline
from pandas_datareader.data import DataReader

beginDate ='2020-03-04'
endDate ='2021-03-04'       
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
print(df.tail(5))
print(AAPL.describe())

daysMovAvg= [5, 10, 20, 40]
for ma in daysMovAvg:
    for co in coList:
        columnName = f'MA-{ma} days'
        co[columnName] = co['Adj Close'].rolling(ma).mean()

fig, axes = plt.subplots(nrows=2, ncols=4)
fig.set_figheight(8)
fig.set_figwidth(15)
AAPL[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[0,0])
axes[0,0].set_title('Apple')
GOOG[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[0,1])
axes[0,1].set_title('Google')
MSFT[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[1,0])
axes[1,0].set_title('Microsoft')
AMZN[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[1,1])
axes[1,1].set_title('Amazon')
DIS[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[0,2])
axes[0,2].set_title('Disney')
TSLA[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[0,3])
axes[0,3].set_title('Tesla')
GME[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[1,2])
axes[1,2].set_title('Gamestop')
AMC[['Adj Close','MA-5 days','MA-10 days','MA-20 days','MA-40 days']].plot(ax=axes[1,3])
axes[1,3].set_title('AMC')
fig.tight_layout()
#
# Machine Learning
#
ticker = 'AAPL'
beginDate ='2011-03-04'
df = DataReader(ticker, 'yahoo', beginDate, endDate)

data = df.filter(['Close'])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
trainingDataLen = int(np.ceil( len(dataset) * .95 ))
print('\nTraining Length:', trainingDataLen, '\n')

# Scale the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(dataset)
print('\nScaled Data:\n', scaledData,'\n')

# Create the scaled training data set
trainData = scaledData[0:int(trainingDataLen), :]
# Split the data into x_train and y_train data sets
x_train = []
y_train = []
for i in range(60, len(trainData)):
    x_train.append(trainData[i-60:i, 0])
    y_train.append(trainData[i, 0])
    if i<= 31:
        print(x_train)
        print(y_train)
        print() 
# Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)
# Reshape the data
print(x_train)
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
# Train the modelhttps://www.kaggle.com/faressayah/stock-market-analysis-prediction-using-lstm#2.-What-was-the-moving-average-of-the-various-stocks?
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002 
testData = scaledData[trainingDataLen - 60: , :]
# Create the data sets x_test and y_test
x_test = []
y_test = dataset[trainingDataLen:, :]
for i in range(60, len(testData)):
    x_test.append(testData[i-60:i, 0])
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
train = data[:trainingDataLen]
actual = data[trainingDataLen:]
actual['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(12,6))
plt.title(f'Training Model from {ticker}')
plt.xlabel('Close Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot()
plt.plot(train['Close'])
plt.plot(actual[['Close']],'g')
plt.plot(actual[['Predictions']],'r')
plt.legend(['Train', 'Actual', 'Predicted'], loc='middle left', fontsize=18)
plt.show()

plt.figure(figsize=(12,6))
plt.title(f'Actual from {ticker}')
plt.xlabel('Close Date', fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.plot()
plt.plot(df['Close'])
plt.legend(['Actual'], loc='middle left', fontsize=18)
plt.show()

# Show the actual and predicted prices
print(actual)