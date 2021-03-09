import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt 
from pandas.plotting import lag_plot
from pandas_datareader.data import DataReader
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import pprint
from ML import *
from Stock_Input_Copy import *
beginDate ='2020-01-01'
endDate = datetime.now().date()-datetime.timedelta(days=1)   
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
modelData = df['Close'].to_frame()
print(f'\n{modelData.describe()}\n')
#
# Calculate the 10, 30, 60 days moving averages of the closing prices
#df2 = DataReader(ticker, 'yahoo', beginDate, endDate).adjclose
five_rolling = modelData.rolling(window=5).mean()
ten_rolling = modelData.rolling(window=10).mean()
twenty_rolling = modelData.rolling(window=20).mean()
fifty_rolling = modelData.rolling(window=50).mean()
hundred_rolling = modelData.rolling(window=100).mean()
# Plot rolling averages with matplotlib 
plt.figure(figsize=(10, 6))
plt.plot(modelData.index, modelData, label='Adj Closing')
plt.title(f'{ticker} Stock Price - 5/10/20/50/100 Day Moving Avg')
plt.plot(five_rolling.index, five_rolling, label='5 days rolling')
plt.plot(ten_rolling.index, ten_rolling, label='10 days rolling')
plt.plot(twenty_rolling.index, twenty_rolling, label='20 days rolling')
plt.plot(fifty_rolling.index, fifty_rolling, label='50 days rolling')
plt.plot(hundred_rolling.index, hundred_rolling, label='100 days rolling')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.legend()
plt.savefig(f'.\output\{ticker}-MovingAvgs_{beginDate}_{endDate}')
plt.savefig('.\output\MovingAvgs')
plt.show()
#
# Setup number of days in training model; sample/shape(X,1)and Target(y) data
futureDays = 10
modelData['Target'] = modelData['Close'].shift(-futureDays)
#modelData = modelData.dropna()
print(modelData)
X = np.array(modelData.drop(['Target'], 1))[:-futureDays]
y = np.array(modelData['Target'])[:-futureDays]
#
# Split data 75% training, 25% testing
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25)
Xfuture = modelData.drop(['Target'], 1)[:-futureDays]
Xfuture = Xfuture.tail(futureDays)
Xfuture = np.array(Xfuture)
#
# AutoRegressive Integrated Moving Average ARIMA(history=list, order=(p,d,q))
#   Note: (p-lag observations, d-degree of differencing, q-size of moving avg window)
plt.figure(figsize=(10, 6))
lag_plot(df['Close'], lag=3)
plt.title(f'{ticker} Stock - Autocorrelation plot with lag=3')
plt.savefig(f'.\output\{ticker}-Autocorrelation_{beginDate}_{endDate}')
plt.savefig('.\output\Autocorrelation')
plt.show()
# Setup ARIMA training model
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
print('Testing Mean Squared Error is {}'.format(MSE_error))
# Plot AIMA prediction model results
test_set_range = df[int(len(df)*0.75):].index
plt.figure(figsize=(10, 6))
plt.plot(test_set_range, test_data, label='Actual')
plt.plot(test_set_range, model_predictions, linestyle='dashed', label='Predicted')
plt.title(f'{ticker} Stock Price Prediction - ARIMA Model Perforance')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.legend(loc='upper left')
plt.savefig(f'.\output\{ticker}-ARIMA_{beginDate}_{endDate}')
plt.savefig('.\output\ARIMA')
plt.show()
# print summary of ARIMA fit model
print(model_fit.summary())
#
# Linear Regression Model
# Fit model with Linear Regression prediction model
linear = LinearRegression().fit(Xtrain, ytrain)
linearPrediction = linear.predict(Xfuture)
linearResult = linear.score(Xtrain, ytrain)
print("Linear Accuracy: %.3f%%" % (linearResult*100.0))
print('Linear Regression prediction =',linearPrediction)
predictions = linearPrediction
#mean_absolute_error(ytest, linearPrediction)
valid = modelData[X.shape[0]:]
valid['Target'] = predictions
plt.figure(figsize=(10, 6))
plt.title(f'{ticker} Stock Price Prediction - Linear Regression Model Performance')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(modelData['Close'])
plt.plot(valid[['Close', 'Target']])
plt.legend(['Original', 'Actual', 'Predicted'])
plt.savefig(f'.\output\{ticker}-LinearRegression_{beginDate}_{endDate}')
plt.savefig('.\output\LinearRegression')
plt.show()

rSqLinear=linear.score(X,y)
print('\ncoefficient of determination:', rSqLinear)
print('intercept:', linear.intercept_)
print('slope: ', linear.coef_)
#
# Decision Tree Model
# Fit model with Decision Tree prediction model
tree = DecisionTreeRegressor().fit(Xtrain, ytrain)
treePrediction = tree.predict(Xfuture)
treeResult = tree.score(Xtrain, ytrain)
print("Tree Accuracy: %.3f%%" % (treeResult*100.0))
print('Decision Tree prediction =',treePrediction)
predictions = treePrediction
valid = modelData[X.shape[0]:]
valid['Predict'] = predictions
plt.figure(figsize=(10, 6))
plt.title(f'{ticker} Stock Price Prediction - Decision Tree Regressor Model Performance')
plt.xlabel('Date')
plt.ylabel('Close Price USD')
plt.plot(modelData['Close'])
plt.plot(valid[['Close', 'Predict']])
plt.legend(['Original', 'Actual', 'Predicted'])
plt.savefig(f'.\output\{ticker}-DecisionTree_{beginDate}_{endDate}')
plt.savefig('.\output\DecisionTree')
plt.show()
#
# Create new DataFrame with future business days-closing price populated
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
#X_new = np.array(futureDates).astype(float).reshape(-1, 1)
#finalPredict = linear.predict(X_new)
#print(finalPredict)
print(combinedDF)
#
# Write to CSV
combinedDF.to_csv(f'.\output\{ticker}-CombinedDF_{beginDate}_{endDate}.csv')
combinedDF.to_csv('.\output\combinedDF.csv')
# 
# Stock Information (upper-right box), 
currInfo=yf.Ticker(ticker).info
pprint.pprint(currInfo)
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
print('\nCurrent-Key Stock Information:\n')
print(f"Company: {currInfo['longName']} ({currInfo['symbol']})")
print(f"Current Ask/Bid (USD): {currInfo['ask']}/{currInfo['bid']}")
print(f"Open Price: {round(currInfo['open'],2)}")
print(f"High/Low Price: {currInfo['dayHigh']}/{currInfo['dayLow']}")
print(f"Avg Volume: {currInfo['averageVolume']}")
print(f"Volume: {currInfo['volume']}")
print(f"52wk High: {round(currInfo['fiftyTwoWeekHigh'],2)}")
print(f"52wk Low: {round(currInfo['fiftyTwoWeekLow'],2)}")
print(f"MorningStar Rating: {currInfo['morningStarOverallRating']}")
print(f"Short Ratio: {currInfo['shortRatio']}")
