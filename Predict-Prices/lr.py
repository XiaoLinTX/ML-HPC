'''
Descripttion: Process data
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-23 21:50:58
LastEditors: ZhangHongYu
LastEditTime: 2021-01-23 21:55:14
'''
import pandas as pd

train = pd.read_csv('Predict-Prices/data/train.csv')
print(train.columns)
print(train.head)