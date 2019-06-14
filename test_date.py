
import csv
import sys
import re
import pandas as pd
#import pandas.io.data
from pandas_datareader import data, wb
import datetime
import matplotlib.pyplot as plt
from stockstats import StockDataFrame
import os

stocks = []
path = '/Users/omnisciemployee/Documents/Data/Stocks_2/'
for f in os.listdir(path):
        filename = f
        name_parts = re.split("_copy", filename)
        sym = name_parts[0]
        try:
                stocks.append(sym)
        except Exception as e:
                print(e)
                continue


for i in stocks:
	try:
		data = pd.read_csv("/Users/omnisciemployee/Documents/Data/Stocks_2/"+i+"_copy.csv")
		data1 = data[['Date', 'Symbol']]
		data1.to_csv("/Users/omnisciemployee/Documents/Data/Stocks_date/"+i+"_date.csv", index=False)
	except Exception as e:
		print(i)
		print(e)
		continue
