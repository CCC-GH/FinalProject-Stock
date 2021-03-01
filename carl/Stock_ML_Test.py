# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:38:31 2021

@author: coffm
"""

#import packages
import pandas as pd
import numpy as np

#to plot within notebook
import matplotlib.pyplot as plt
#%matplotlib inline

#setting figure size
#from matplotlib.pylab import rcParams
#rcParams['figure.figsize'] = 20,10

#for normalizing data
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))

#read the file
df = pd.read_csv('NSE-TATAGLOBAL11.csv')
df = df.drop(['Open', 'High', 'Low', 'Last', 'Turnover (Lacs)', 'Total Trade Quantity'], axis=1)
#print the head
print(df.head())

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%m/%d/%Y')
df.index = df['Date']

#plot
plt.figure(figsize=(10,5))
plt.plot(df['Close'])
#
#
print('\n Shape of the data:', df.shape)

#creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
     new_data['Date'][i] = data['Date'][i]
     new_data['Close'][i] = data['Close'][i]

# NOTE: While splitting the data into train and validation set, we cannot
#   use random splitting since that will destroy the time component. 
#   So here we have set the last year’s data into validation and the 
#   4 years’ data before that into train set.

# splitting into train and validation
train = new_data[:987]
valid = new_data[987:]

# shapes of training set
print('\n Shape of training set:', train.shape)

# shapes of validation set
print('\n Shape of validation set:', valid.shape)

# In the next step, we will create predictions for the validation set and
#    check the RMSE using the actual values.
# making predictions
preds = []
for i in range(0,valid.shape[0]):
    a = train['Close'][len(train)-248+i:].sum() + sum(preds)
    b = a/248
    preds.append(b)

# checking the results (RMSE value)
rms=np.sqrt(np.mean(np.power((np.array(valid['Close'])-preds),2)))
print('\n RMSE value on validation set:', round(rms,2),'\n\n')
#
#
#plot
valid['Predictions'] = 0
valid['Predictions'] = preds
plt.figure(figsize=(10,5))
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])


# #setting index as date values
# df['Date'] = pd.to_datetime(df.Date,format='%m/%d/%Y')
# df.index = df['Date']

# #sorting
# data = df.sort_index(ascending=True, axis=0)

# #creating a separate dataset
# new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

# for i in range(0,len(data)):
#     new_data['Date'][i] = data['Date'][i]
#     new_data['Close'][i] = data['Close'][i]
    
# new_data['mon_fri'] = 0
# for i in range(0,len(new_data)):
#     if (new_data['Dayofweek'][i] == 0 or new_data['Dayofweek'][i] == 4):
#         new_data['mon_fri'][i] = 1
#     else:
#         new_data['mon_fri'][i] = 0
        
# #split into train and validation
# train = new_data[:987]
# valid = new_data[987:]

# x_train = train.drop('Close', axis=1)
# y_train = train['Close']
# x_valid = valid.drop('Close', axis=1)
# y_valid = valid['Close']

# #implement linear regression
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(x_train,y_train)

# #make predictions and find the rmse
# preds = model.predict(x_valid)
# rms=np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(preds)),2)))
# print(rms)