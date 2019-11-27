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
import time
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

def run_sequence_plot(x, y, title='', xlabel="time", ylabel="series"):
    plt.plot(x, y, 'b-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.show();    

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

run_sequence_plot(df.index.values, df.AirPassengers, title="Air Passengers")

# Two major observations: (1) the time series data is trending upward and (2)  
# the amplitude of the variation is getting large over time. 

# These two features suggest that the time series data could be fit by a  
# multiplicative time series model.

# Before moving onto serious modeling, we need to split our data into the  
# training and testing sets for model validation.

train, test = df[:-12], df[-12:]

# # Feature Scaling

# As you will see, we perform one of the most important transformations we need  
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

# # Sampling for Supervised TS Prediction

n_input = 12
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

# `length`: the number of lagging time steps from the 1st argument.  
# `batch_size`: the number of looking forward time steps from the 2nd argument.

# ## Cases of Time Series Sampling

# To have a concrete idea of what `length` and `batch_size` stand for, experiment  
# the following .py files from Jason Brownlee's [website](https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/):
# - *multivariate-one-step-TimeSeriesGenerator.py*: Splitting a typical  
# multivariate time series with a univariate series.  
# - *multistep-forecasts.py*: multivariate input with multivariate output.  
# - *multivariate-input-and-dependent-I.py*: A case where input data at time  
# `t` is mapped to output data at the same time `t` (i.e., neither of the series  
# are time series data). 
# - *multivariate-input-and-dependent-II.py*: Similar to the above except for  
# using `insert` along with `TimeSeriesGenerator` to form the batch series for  
# convenience.

# Reference: 
# - [How to use the TimeseriesGenerator](https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/)  
#   Check `univariate-one-step-TimeSeriesGenerator.py` for instance.   
# - [Difference between a batch and an epoch](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)

# # Forming LSTM Layer

model = Sequential()
model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
model.add(Dropout(0.15)) # a dropout layer to ease overfitting concern
model.add(Dense(1)) # a dense layer making the prediction
model.compile(optimizer='adam', loss='mse')

# LSTM expects data input to have the shape in the following format:  
# `[batch, timesteps, features]`, where:  
# - *batch*: number of neurons
# - *timesteps*: number of lags
# - *features*: number of series (i.e., dimension of output series data), for  
# instance, the value is one for a univariate time series.

# Suppose a univariate time series [1,2,3,4,5,6,7] is splitted into  
# t1 = [x1,y1] = [[1,2,3,4],[5]], t2 = [x2,y2] = [[2,3,4,5],[6]], and  
# t3 = [x3,y3] = [[3,4,5,6],[7]].
# In this example, we have `[timesteps, features]=[4,1]`.   

# Note: If we want to create a multiple-layer-perceptron model, then we can  
# simplify the LSTM model by substituting the LSTM line with the following code:  
# `model.add(Dense(100, activation='relu', input_dim=n_input)),` for example.  

# Check `univariate-one-step-with-LSTM.py` for instance.

# Timing the training
start_time = time.time()
model.fit_generator(generator,epochs=90)
print('Generator fitting took {} seconds'.format(time.time()-start_time))

# +
pred_list = []
# Make a prediction on the last window of size `n_input`.
batch = train[-n_input:].reshape((1, n_input, n_features))

# -

# How do we figure our what the parameters represent for?  
# Back to our univariate time series [1,2,...,7].  
# If we plan to make a prediction based on two latest samples, t2 and t3, then  
# the combination of arguments [batch, timesteps, features] should be [2,4,1].  
# In the code here, the prediction is based off the last window, so the  
# combination of parameters is [1, 12, 1] where 12 is the value of `n_input`.  

# Next, make a new prediction and then shift forward the window by one timestep for  
# every iteration. In the end of each iteration, we update the batch variable  
# so that the newly obtained scalar is the latest datapoint of the window.

# +

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)

# Scaling back our prediction to the original scale.
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

# Measuring the Error
pred_actual_rmse = rmse(df_test.iloc[-n_input:, [0]], df_test.iloc[-n_input:, [1]])
print("rmse: ", pred_actual_rmse)

# Suppose the model outperforms other predictive models. Then we would retrain  
# the model on the full training data before make new predictions in the  
# production phase. 

# + 
train = df
scaler.fit(train)
train = scaler.transform(train)

n_input = 12
n_features = 1
generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)

start_time = time.time()
model.fit_generator(generator,epochs=90)
print('Generator fitting took {} seconds'.format(time.time()-start_time))

# -

# Reference:
# - [The relation between `step_per_epoch` and `batch_size`](https://datascience.stackexchange.com/questions/47405/what-to-set-in-steps-per-epoch-in-keras-fit-generator#content)  

# +
pred_list = []

batch = train[-n_input:].reshape((1, n_input, n_features))

for i in range(n_input):   
    pred_list.append(model.predict(batch)[0]) 
    batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
# -

# Since we are going to make a prediction beyond the training data, we create  
# a index column for the new datetime series.

from pandas.tseries.offsets import DateOffset
add_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,13) ]
future_dates = pd.DataFrame(index=add_dates[1:],columns=df.columns)
df_predict = pd.DataFrame(scaler.inverse_transform(pred_list),
                          index=future_dates[-n_input:].index, 
                          columns=['Prediction'])

# Plot the multistep prediction.                        
df_proj = pd.concat([df,df_predict], axis=1)
plt.figure(figsize=(20, 5))
plt.plot(df_proj.index, df_proj['AirPassengers'])
plt.plot(df_proj.index, df_proj['Prediction'], color='r')
plt.legend(loc='best', fontsize='xx-large')
plt.xticks(fontsize=18)
plt.yticks(fontsize=16)
plt.show()

# See *multivariate-one-step-with-LSTM.py* for predicting multivariate time series.

# # Reference

# [1] [A Quick Example of Time-Series Prediction Using Long Short-Term Memory (LSTM) Networks](https://medium.com/swlh/a-quick-example-of-time-series-forecasting-using-long-short-term-memory-lstm-networks-ddc10dc1467d) by Ian Felton  
# [2] [How to Use the TimeseriesGenerator for Time Series Forecasting in Keras](https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/) by Jason Brownless