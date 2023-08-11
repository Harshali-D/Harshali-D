# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 11:23:57 2023

@author: harsahli
"""
#the code works but accuracy is yet to be measured 


import yfinance as yf
import numpy as np
import pandas as pd 
import os 
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


ticker='^NSEI'
start_date='2021-07-03'
end_date='2023-08-03'

data= yf.download(ticker,start=start_date,end=end_date)

df=pd.DataFrame(data)

df['date']=pd.to_datetime(df.index)
import seaborn as sns
sns.pairplot(df,hue="Close")
figure=go.Figure(data=[go.Candlestick(x=df['date'],
                                    open=df['Open'],
                                    close=df['Close'],
                                    high=df['High'],
                                    low=df['Low'])])

pyo.plot(figure,filename='nifty_chart.html')

df.drop(['date', 'Volume'],axis=1,inplace=True)

df.reset_index(drop=True,inplace=True)

df.plot.line(y='Close',use_index=True)



#now we split the data in training and testing 

X=df[['Open','High','Low','Adj Close']]
 #this is independent variables 
y=df['Close'] #this is dependent variable , i.e we will predict Y 



X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3 , random_state=42)

rf=RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(X_train, y_train)

y_pred= rf.predict(X_test)

mse=mean_squared_error(y_test, y_pred)

print("mean squared eroor is ->",mse)



new_data=np.array([[19655.4,19678.2,19423.6,19526.6]])

predicted_price=rf.predict(new_data)
print("the predicted price or tommrow is->",predicted_price)


