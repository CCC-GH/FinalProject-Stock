import pandas as pd    
import numpy as np
import matplotlib.pyplot as plt 
from pandas_datareader.data import DataReader
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

beginDate ='2020-03-05'
endDate ='2021-03-04'       
#
#df = pd.read_csv('XXXXXX.csv',index_col=0)
# note: if using line above, loading from csv, comment out section below
#
# User input and load ticker data from yahoo finance
while True:
    try: 
        ticker = input('Enter Stock Ticker: ').upper()
        df = DataReader(ticker, 'yahoo', beginDate, endDate)
    except:
        print('\nStock Ticker Symbol does not exist!\n')
        continue;
    break
#
# Use Adj Close instead of Close
df['Close'] = df['Adj Close']
df = df.drop('Adj Close',axis=1)

# Future Days - number of days to predict
modelData = df['Close'].to_frame()
futureDays = 30
modelData['Predict'] = modelData['Close'].shift(-futureDays)
modelData = modelData.dropna()
print(modelData)
x = np.array(modelData.drop(['Predict'], 1))[:-futureDays]
#print(x)
y = np.array(modelData['Predict'])[:-futureDays]
#print(y)
#
# Split data 75% training, 25% testing
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)
tree = DecisionTreeRegressor().fit(xtrain, ytrain)
linear = LinearRegression().fit(xtrain, ytrain)
xfuture = modelData.drop(['Predict'], 1)[:-futureDays]
xfuture = xfuture.tail(futureDays)
xfuture = np.array(xfuture)
#
# Decision Tree and Linear Regression models
treePrediction = tree.predict(xfuture)
#print('Decision Tree prediction =',treePrediction)
linearPrediction = linear.predict(xfuture)
#print('Linear Regression prediction =',linearPrediction)
#
# Plot Linear Regression prediction
predictions = linearPrediction
valid = modelData[x.shape[0]:]
valid['Predict'] = predictions
plt.figure(figsize=(10, 6))
plt.title(f'{ticker} - Stock Price Prediction - Linear Regression Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD')
plt.plot(modelData['Close'])
plt.plot(valid[['Close', 'Predict']])
plt.legend(['Original', 'Actual', 'Predicted'])
plt.show()
#
# Plot Decision Tree prediction
predictions = treePrediction
valid = modelData[x.shape[0]:]
valid['Predict'] = predictions
plt.figure(figsize=(10, 6))
plt.title(f'{ticker} - Stock Price - Decision Tree Regressor Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD')
plt.plot(modelData['Close'])
plt.plot(valid[['Close', 'Predict']])
plt.legend(['Original', 'Actual', 'Predicted'])
plt.show()
#
# Create new DataFrame with future business days-closing price populated
combinedDF=df['Close'].to_frame()
todaysDate = datetime.datetime.now().date()
futureDays = 10
us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())
futureDates = pd.date_range(todaysDate, periods = futureDays, freq=us_bd)
combinedDFcol =['Close','Predict','SM1','SM2','SM3','SM4'] 
futureDF=pd.DataFrame(index=futureDates, columns=combinedDFcol)
combinedDF=combinedDF.append(futureDF, ignore_index = False)
combinedDF.index.names = ['Date']
print(combinedDF)
#predictDF=predictDF.fillna(0)
#
# Write to CSV
combinedDF.to_csv(f'.\output\{ticker}-CombinedDF_{beginDate}_{endDate}.csv')
