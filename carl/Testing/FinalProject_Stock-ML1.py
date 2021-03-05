'''
$ python3 -m install pyfinance
$ pip install yfinance --upgrade --no-cache-dir
$ pip install tensorflow
https://pypi.org/project/pyfinance/
https://pandas-datareader.readthedocs.io/en/latest/#
'''
# Stock data from yahoo
import pandas as pd    
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10) 
import numpy as np
import matplotlib.pyplot as plt 
from pandas_datareader.data import DataReader
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import pprint

beginDate ='2011-03-04'
endDate ='2021-03-04'  
stock='AAPL'
df = DataReader(stock, 'yahoo', beginDate, endDate)
#df = pd.read_csv('MSFT.csv',index_col=0)
df['Close'] = df['Adj Close']
df = df.drop('Adj Close',axis=1)

# Moving averages with periods 5,10,20,50,100,200 days
for ma_period in [5,10,20,50,100,200]:
    indicator_name = 'ma_%d' % (ma_period)
    df[indicator_name] = df['Close'].rolling(ma_period).mean()

# Bollinger bands (the moving average plus and minus 1 and 2 standard deviations)
df['Boll_Up_20_2'] = df['Close'].rolling(20).mean() + 2*df['Close'].rolling(20).std()
df['Boll_Down_20_2'] = df['Close'].rolling(20).mean() - 2*df['Close'].rolling(20).std()
df['Boll_Up_20_1'] = df['Close'].rolling(20).mean() + df['Close'].rolling(20).std()
df['Boll_Down_20_1'] = df['Close'].rolling(20).mean() - df['Close'].rolling(20).std()
df['Boll_Up_10_1'] = df['Close'].rolling(10).mean() + df['Close'].rolling(10).std()
df['Boll_Down_10_1'] = df['Close'].rolling(10).mean() - df['Close'].rolling(10).std()
df['Boll_Up_10_2'] = df['Close'].rolling(10).mean() + 2*df['Close'].rolling(10).std()
df['Boll_Down_10_2'] = df['Close'].rolling(10).mean() - 2*df['Close'].rolling(10).std()

# Donchian channels - rolling maximum and minimum prices during the same periods as moving avg
for channel_period in [5,10,20,50,100,200]:
    up_name = 'Don_Ch_Up_%d' % (channel_period)
    down_name = 'Don_Ch_Down_%d' % (channel_period)
    df[up_name] = df['High'].rolling(channel_period).max()
    df[down_name] = df['Low'].rolling(channel_period).min()
    
# Shifted into time lags, 1-10 days prior
newdata = df['Close'].to_frame()
for lag in [1,2,3,4,5,6,7,8,9,10]:
    shift = lag
    shifted = df.shift(shift)
    shifted.columns = [str.format('%s_shift_by_%d' % (column,shift)) for column in shifted.columns]
    newdata = pd.concat((newdata,shifted),axis=1)                                                                                        

# Future Days - target days to predict
forward_lag = 5
newdata['target'] = newdata['Close'].shift(-forward_lag)
newdata = newdata.drop('Close',axis=1)
newdata = newdata.dropna()
pprint.pprint(newdata, width=80)

X = newdata.drop('target',axis=1)
Y = newdata['target']
train_size = int(X.shape[0]*0.7)
X_train = X[0:train_size]
y_train = Y[0:train_size]
X_test = X[train_size:]
y_test = Y[train_size:]

correlations = np.abs(X_train.corrwith(y_train))
features =  list(correlations.sort_values(ascending=False)[0:50].index)
X_train = X_train[features]
X_test = X_test[features]
pprint.pprint(features)
#
# Linear Regression model
#
# Train model
lr = LinearRegression()
lr.fit(X_train,y_train)
# Calc prediction of the model
y_pred = lr.predict(X_test)
# Compare to real test set
print('\nMean Absolute Error:', mean_absolute_error(y_test,y_pred),'\n')

# Plot real values of test set and the predicted value
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'{stock} - Linear Regression')
plt.show()

























