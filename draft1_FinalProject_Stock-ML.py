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
import pprint
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
plt.ylabel('Price USD')
plt.legend()
plt.show()
#
# Setup number of days to predict, shift dates
futureDays = 10
modelData['Predict'] = modelData['Close'].shift(-futureDays)
modelData = modelData.dropna()
print(modelData)
x = np.array(modelData.drop(['Predict'], 1))[:-futureDays]
y = np.array(modelData['Predict'])[:-futureDays]
#
# Split data 75% training, 25% testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
xfuture = modelData.drop(['Predict'], 1)[:-futureDays]
xfuture = xfuture.tail(futureDays)
xfuture = np.array(xfuture)

# Decision Tree and Linear Regression models
#print('Decision Tree prediction =',treePrediction)
#print('Linear Regression prediction =',linearPrediction)
#
# Plot Linear Regression prediction
linear = LinearRegression().fit(xtrain, ytrain)
#linearResult = linear.score(xtrain, ytrain)
#print("Linear Accuracy: %.3f%%" % (linearResult*100.0))
linearPrediction = linear.predict(xfuture)
predictions = linearPrediction
#mean_absolute_error(ytest, linearPrediction)
valid = modelData[x.shape[0]:]
valid['Predict'] = predictions
plt.figure(figsize=(10, 6))
plt.title(f'{ticker} Stock Price Prediction - Linear Regression Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD')
plt.plot(modelData['Close'])
plt.plot(valid[['Close', 'Predict']])
plt.legend(['Original', 'Actual', 'Predicted'])
plt.show()
# Linear Regression Model
rSqLinear=linear.score(x,y)
print('\ncoefficient of determination:', rSqLinear)
print('intercept:', linear.intercept_)
print('slope: ', linear.coef_)
#
# Plot Decision Tree prediction
tree = DecisionTreeRegressor().fit(xtrain, ytrain)
#treeResult = tree.score(xtrain, ytrain)
#print("Tree Accuracy: %.3f%%" % (treeResult*100.0))
treePrediction = tree.predict(xfuture)
predictions = treePrediction
valid = modelData[x.shape[0]:]
valid['Predict'] = predictions
plt.figure(figsize=(10, 6))
plt.title(f'{ticker} Stock Price - Decision Tree Regressor Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD')
plt.plot(modelData['Close'])
plt.plot(valid[['Close', 'Predict']])
plt.legend(['Original', 'Actual', 'Predicted'])
plt.show()
#
# Create new DataFrame with future business days-closing price populated
todaysDate = datetime.datetime.now().date()
futureDays = 10
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
futureDates = pd.date_range(todaysDate, periods = futureDays, freq=us_bd)
combinedDFcol =['Close','Predict','SM1','SM2','SM3','SM4'] 
futureDF=pd.DataFrame(index=futureDates, columns=combinedDFcol)
combinedDF=df.append(futureDF, ignore_index = False)
combinedDF.index.names = ['Date']
currInfo=yf.Ticker(ticker).info
combinedDF['Predict'].loc[-futureDays:]=(currInfo['bid']+currInfo['ask'])/2
print(combinedDF)
#
# Write to CSV
combinedDF.to_csv(f'.\output\{ticker}-CombinedDF_{beginDate}_{endDate}.csv')
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
