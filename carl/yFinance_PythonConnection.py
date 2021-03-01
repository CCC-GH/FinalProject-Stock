# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:21:41 2021
@author: coffm
$ python3 -m install pyfinance
https://pypi.org/project/pyfinance/
"""

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

beginDate = '2021-01-01'
endDate = '2021-02-26'       
ticker_list=['MSFT']

ticker = yf.Ticker(ticker_list[0])
history=ticker.history(start=beginDate,end=endDate)
print(history.head())
history.to_csv('yFinance_StockReport.csv')

history = history.reset_index()
for i in ['Open', 'High', 'Close', 'Low']: 
     history[i]=history[i].astype('float64')

plt.figure(figsize=(10,5))
history['Date'] = pd.to_datetime(history.Date,format='%Y/%m/%d')
history.index = history['Date']
plt.plot(history['Close'])