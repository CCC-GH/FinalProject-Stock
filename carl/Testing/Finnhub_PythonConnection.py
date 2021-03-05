# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 00:25:54 2021

https://finnhub.io/docs/api/company-basic-financials
https://github.com/Finnhub-Stock-API/finnhub-python
https://finnhub.io/docs/api
key: c0utkd748v6or05b7jt0


@author: coffm
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 14:21:41 2021
@author: coffm
https://finnhub.io/docs/api
https://github.com/Finnhub-Stock-API/finnhub-python
key: c0utkd748v6or05b7jt0
pip install finnhub-python
"""


import pandas as pd
import finnhub
import matplotlib.pyplot as plt

beginDate = '2020-01-01'
endDate = '2021-03-02'       
ticker_list=['MSFT','TSLA']
#ticker_list=['MSFT','TSLA','AMZN','AAPL','JPM','CRM','ZM','PTON','DIS','MCD','PRTY','CSCO','GOOGL','ORCL','GME','AMC','BB','NOK','BBBY','BBY','KOSS','EXPR']


key='c0utkd748v6or05b7jt0'

finnhub_client=finnhub.Client(api_key=key)
# Basic financials
#print(finnhub_client.company_basic_financials('MSFT', 'all'))

# Upgrade downgrade
#print(finnhub_client.upgrade_downgrade(symbol='MSFT', _from='2021-03-01', to='2021-03-02'))

# Company News
# Need to use _from instead of from to avoid conflict
#news=finnhub_client.company_news('MSFT', _from='2021-03-02', to='2021-03-02')
# print(news)

# history=ticker.history(start=beginDate,end=endDate)
# print(ticker.institutional_holders)
# print(history.head())
# history.to_csv('yFinance_StockReport.csv')

# history = history.reset_index()
# for i in ['Open', 'High', 'Close', 'Low']: 
#      history[i]=history[i].astype('float64')

# plt.figure(figsize=(10,5))
# history['Date'] = pd.to_datetime(history.Date,format='%Y/%m/%d')
# history.index = history['Date']
# plt.plot(history['Close'])