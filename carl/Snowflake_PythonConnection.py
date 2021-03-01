# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 09:34:36 2021
@author: coffm
$ pip install snowflake-connector-python[pandas]
https://pypi.org/project/snowflake-connector-python/
"""

import snowflake.connector
import pandas as pd
conn=snowflake.connector.connect(
                user='COFFMANDATA',
                password='FinalProject1',
                account='gl21222.east-us-2.azure',
                warehouse='COMPUTE_WH',
                database='CMACCXK_AZURE_EASTUS2_US_STOCKS_DAILY',
                schema='PUBLIC'
                )
query = "select * from stock_history where symbol='MSFT'"
df=pd.read_sql(query, conn)
print(df.head())
df.to_csv('Snowflake_StockReport.csv')