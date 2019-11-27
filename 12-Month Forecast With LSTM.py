# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('~/time-series-1/Data/AirPassengers.csv')
print(df.info())
df.head()
# -

# The column `Month` is of object type. It needs to be in datetime format.

df.Month = pd.to_datetime(df.Month)
df = df.set_index("Month")
df.head()

# The raw data contains two columns, one of which is converted to index column.  
# After a little effortless data preprocessing, plotting the time series data is  
# our first step to begin any time series data analysis.

df.plot()

# Two major observations: (1) the time series data is trending upward and (2)  
# the amplitude of the variation is getting large over time. 

# These two features suggest that the time series data could be fit by a  
# multiplicative time series model.

# Before moving onto serious modeling, we need to split our data into the  
# training and testing sets for model validation.

train, test = df[:-12], df[-12:]

# As you can see, we perform one of the most important transformations we need  
# to our data: feature scaling. 

# Machine learning algorithms don't perform well when the input numerical  
# attributes have very different scales.

scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

# Two common ways to get all attributes have the same scale: `min-max scaling`  
# and `standardization`. We chose the former to apply.

# **Note:** We train the scaler with the training data only, not with the full  
# dataset (including the test set). Only then can we use them to transform the  
# training set and the test set (and new data).

n_input = 12
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

# length: # of lag observations to use in the input portion of each sample  
# batch_size: # of samples to return on each iteration  

# Suppose we have a univariate data [1,2,3,4,5]. If `length=2` and `batch_size=3`,  
# then x1 = [1,2], and y1 = [[2,3],[3,4],[4,5]].

# Reference: 
# - [How to use the TimeseriesGenerator](https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/)  
#   Check `univariate-one-step-TimeSeriesGenerator.py` for instance.   
# - [Difference between a batch and an epoch](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(Dropout(0.15)) # a dropout layer to ease overfitting concern
model.add(Dense(1)) # a dense layer making the prediction
model.compile(optimizer='adam', loss='mse')

# LSTM expects data input to have the shape in the following format:  
# `[batch, timesteps, features]`.  

# `batch`: # of samples.  
# `timesteps`: # of lags plus one (one represents for the current value)  
# `features`: One for an univariate series

# Suppose a univariate time series [1,2,3,4,5,6] is splitted into  
# [x1,y1] = [[1,2,3,4],[5]] and [x2,y2] = [[2,3,4,5],[6]]. Then,  
# In this example, we have `[batch, timesteps, features]=[2,4,1]`.

# Check `univariate-one-step-with-LSTM.py` for instance.

# Note: If we want to create a multiple-layer-perceptron model, then we can  
# simplify the LSTM model by substituting the LSTM line with the following code:  
# `model.add(Dense(100, activation='relu', input_dim=n_input)),` for example.


model.fit_generator(generator,epochs=90)

# +
pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

# +
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=df[-n_input:].index, columns=['Prediction'])

df_test = pd.concat([df,df_predict], axis=1)
# -

plt.figure(figsize=(20, 5))
plt.plot(df_test.index, df_test['AirPassengers'])
plt.plot(df_test.index, df_test['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()

pred_actual_rmse = rmse(df_test.iloc[-n_input:, [0]], df_test.iloc[-n_input:, [1]])
print("rmse: ", pred_actual_rmse)

train = df

scaler.fit(train)
train = scaler.transform(train)

n_input = 12
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

model.fit_generator(generator,epochs=90)

# +
pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
# -

from pandas.tseries.offsets import DateOffset
add_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,13) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)

# +
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, columns=['Prediction'])

df_proj = pd.concat([df,df_predict], axis=1)
# -

plt.figure(figsize=(20, 5))
plt.plot(df_proj.index, df_proj['AirPassengers'])
plt.plot(df_proj.index, df_proj['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()

# For a more compplicated case in splitting multivariate time series, see   
# `multivariate-one-step-TimeSeriesGenerator.py` and  
# `multivariate-one-step-with-LSTM.py` for example. 

# `multivariate-input-and-dependent-I.py`
# `multivariate-input-and-dependent-II.py`
# `multistep-forecasts.py`



