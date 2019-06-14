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
		d1 = pd.read_csv("/Users/omnisciemployee/Documents/Data/Stocks_last/"+i+"_full.csv")
		d2 = pd.read_csv("/Users/omnisciemployee/Documents/Data/Stocks_date/"+i+"_date.csv")
		d1 = pd.concat([d1, d2], axis=1)

		try:
			new_path = '/Users/omnisciemployee/Documents/Data/Stocks_done/'
			d1.to_csv(new_path+i+'_done.csv', index=False)
		except Exception as ex:
			print(i)
			print(e)
			continue

	except Exception as e:
		print(i)
		print(e)
		continue

