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
		data1 = data['Date']
		data1.to_csv("/Users/omnisciemployee/Documents/Data/Stocks_date/"+i+"_date.csv", index=False)
#		print(data1.head())
		stock = StockDataFrame.retype(data)
		data['rsi']=stock['rsi_14']
		data['wr']=stock['wr_14']
		data['kdjk']=stock['kdjk']
		data['kdjd']=stock['kdjd']
		data['kdjj']=stock['kdjj']
		data['macd']=stock['macd']
#		data['sma']=stock['sma']
#		data['ema']=stock['ema']
		data['proc']=stock['open_-2_r']
		
		
		
		del data['close_-1_s']
		del data['close_-1_d']
		del data['rs_14']
		del data['rsi_14']
		
#		for j in data['rsi']:
#			print(i)
		
		try:
#			new_path = '/Users/omnisciemployee/Documents/Data/Stocks_3/'
#			data.to_csv(new_path+i+'_full.csv', index=False)
#			temp = pd.read_csv("/Users/omnisciemployee/Documents/Data/Stocks_3/"+i+"_full.csv")
#			data1, temp = [d.reset_index(drop=True) for d in (data1, temp)]
#			data.join(data1)
			new_path = '/Users/omnisciemployee/Documents/Data/Stocks_last/'
			data.to_csv(new_path+i+'_full.csv', index=False)
		except Exception as e:
			print(e)
			print(i)
			continue
	
	except Exception as e:
		print(i)
		print(e)
		continue
