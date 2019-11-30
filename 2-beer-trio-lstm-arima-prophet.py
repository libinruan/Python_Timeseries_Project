# -*- coding: utf-8 -*-
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

# # EDA

# +
import time
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose 
from pmdarima import auto_arima                        
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import warnings
# plt.style.use('classic')
warnings.filterwarnings("ignore")


# -

def run_sequence_plot(x, y, title='', xlabel="time", ylabel="series"):
    plt.plot(x, y, 'b-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.show(); 

path = './Data/monthly-beer-production-in-austr.csv'
df = pd.read_csv(path, skipfooter=2)
df.head()

# Three issues in the raw data needs to be fixed: (1) the 2nd column's name is  
# lengthy, and (2) the values in the 2nd column is not in numeric format

# Fix problem No. 1
new_columns = df.columns.values
new_columns[1] = "Production"
df.columns = new_columns
df.columns

# Fix problem No. 2: convert index of datetimes to a DatetimeIndex for time  
# series operations, like resample.
df.Month = pd.to_datetime(df.Month)
df = df.set_index("Month")
df.tail()

df.index

# Note that the newly generated index column is of no frequency. We can assign  
# it an appropriate one: "MS" which stands for the start of every month.  
# For a more general treatment, check this [post](https://stackoverflow.com/questions/31517728/python-pandas-detecting-frequency-of-time-series) for details.

df.index.freq = 'MS'

# Check to see if there is any missing value
all(df.Production.notna()) # answer: True, there is no missing value.

# Data splitting for validation
train_data = df[:len(df)-12]
test_data = df[len(df)-12:]

# plt.figure(figsize = (16,12))
plt.rcParams['figure.figsize'] = [16, 12]
a = seasonal_decompose(df["Production"], model = "additive")
a.plot();

# # SARIMA Time Series Model

# ## Automatic Grid Search for the Best Combination

# To my experience, this method dramatically reduced the run time in searching  
# for the optimal combination given the same parameter space for the algorithm  
# to roam on.

# Model Selection.
start_time = time.time()
auto_result = auto_arima(df['Production'], seasonal=True, m=12,max_p=2, max_d=2,max_q=2, max_P=2, max_D=2,max_Q=2)
print('Model fitting took {} seconds'.format(time.time()-start_time))  

print(auto_result.summary())

# Out-of-Sample Forecasts
arima_model = SARIMAX(train_data['Production'], order = (2,1,2), seasonal_order = (1,0,2,12))
arima_result = arima_model.fit()
print(arima_result.summary())

plt.rcParams['figure.figsize'] = [15, 12] 
arima_result.plot_diagnostics()
plt.tight_layout()

# The parameter combination, (p,d,q)X(P,D,Q)S, varies across the two methods  
# I tested above. I cannot figure out the reason that leads to this divergence  
# at the moment. However, the combination obtained from the automatic method  
# looks more reasonable to me. Either of the seasonal and non-seasonal degree  
# of integration is equal to or smaller than 1, which is a rule mentioned by  
# Robert Nau.

arima_pred = arima_result.predict(start = len(train_data), end = len(df)-1, typ="levels").rename("ARIMA Predictions")
arima_pred

test_data['Production'].plot(figsize = (16,5), legend=True)
arima_pred.plot(legend = True);

# +
arima_rmse_error = rmse(test_data['Production'], arima_pred)
arima_mse_error = arima_rmse_error**2
mean_value = df['Production'].mean()

print('MSE Error: {} \nRMSE Error: {} \nMean: {}'.format(arima_mse_error, arima_rmse_error, mean_value))
# -

test_data['ARIMA_Predictions'] = arima_pred


# # LSTM Recurrent Neural Network 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 12
n_features= 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)

# +
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

lstm_model = Sequential()
lstm_model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

print(lstm_model.summary())
# -

start_time = time.time()
lstm_model.fit_generator(generator,epochs=25)
print('Model fitting took {} seconds'.format(time.time()-start_time))  

losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(0,25,1), np.arange(1,26,1))
plt.plot(range(len(losses_lstm)),losses_lstm);


# +
# ## Out-of-Sample Forecasts with Dynamic Updates
# +
lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
# -

lstm_predictions_scaled

lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

lstm_predictions

test_data['LSTM_Predictions'] = lstm_predictions

test_data

test_data['Production'].plot(figsize = (16,5), legend=True)
test_data['LSTM_Predictions'].plot(legend = True);

# +
lstm_rmse_error = rmse(test_data['Production'], test_data["LSTM_Predictions"])
lstm_mse_error = lstm_rmse_error**2
mean_value = df['Production'].mean()

print('MSE Error: {} \nRMSE Error: {} \nMean: {}'.format(lstm_mse_error, lstm_rmse_error, mean_value))
# -

# # Facebook Prophet Additive Model

# As quoted from the official webpage of Prophet on GitHub, we know:
# > Prophet is a procedure for forecasting time series data based on an additive  
# model where non-linear trends are fit with yearly, weekly, and daily  
# seasonality, plus holiday effects. It works best with time series that have  
# strong seasonal effects and several seasons of historical data. Prophet is  
# robust to missing data and shifts in the trend, and typically handles outliers  
# well.

df.info()

# Predictions are then made on a dataframe with a column ds containing the dates  
# for which a prediction is to be made. You can get a suitable dataframe that  
# extends into the future a specified number of days using the helper method  
# Prophet.make_future_dataframe. By default it will also include the dates from  
# the history, so we will see the model fit as well.  

# Convert series to dataframe with a new column from the original index column:  

df_pr = df.copy()
df_pr = df_pr.reset_index()
df_pr.columns = ['ds','y']

train_data_pr = df_pr[:len(df)-12]
test_data_pr = df_pr[len(df)-12:]

from fbprophet import Prophet

# Instantiating a new Prophet object:

m = Prophet()

# Calling its `fit` method and pass in the historical dataframe:

m.fit(train_data_pr)

# Predictions are made on a dataframe with a column `ds` containing the dates  
# for which a prediction is to be made. You can get a suitable dataframe that  
# extends into the future a specified number of days using the helper method  
# `Prophet.make_future_dataframe`. By default it will also include the dates from  
# the history, so we will see the model fit as well.

future = m.make_future_dataframe(periods=12,freq='MS')
prophet_pred = m.predict(future)

print(prophet_pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# The predict method will assign each row in future a predicted value which it  
# names `yhat`.

prophet_pred = pd.DataFrame({"Date" : prophet_pred[-12:]['ds'], "Pred" : prophet_pred[-12:]["yhat"]})
prophet_pred = prophet_pred.set_index("Date")
prophet_pred.index.freq = "MS"
test_data["Prophet_Predictions"] = prophet_pred['Pred']

# Comparison of forecasts with real data:

import seaborn as sns
plt.figure(figsize=(16,5))
ax = sns.lineplot(x= test_data.index, y=test_data["Production"])
sns.lineplot(x=test_data.index, y = test_data["Prophet_Predictions"]);

# +
prophet_rmse_error = rmse(test_data['Production'], test_data["Prophet_Predictions"])
prophet_mse_error = prophet_rmse_error**2
mean_value = df['Production'].mean()

print('MSE Error: {} \nRMSE Error: {} \nMean: {}'.format(prophet_mse_error, prophet_rmse_error, mean_value))
# -

# Table of Out-of-Sample Forecasting Errors across Models:

rmse_errors = [arima_rmse_error, lstm_rmse_error, prophet_rmse_error]
mse_errors = [arima_mse_error, lstm_mse_error, prophet_mse_error]
errors = pd.DataFrame({"Models" : ["ARIMA", "LSTM", "Prophet"],"RMSE Errors" : rmse_errors, "MSE Errors" : mse_errors})
print(errors)

# Plot of Out-of-Sample Forecasts:

plt.figure(figsize=(16,9))
plt.plot_date(test_data.index, test_data["Production"], linestyle="-")
plt.plot_date(test_data.index, test_data["ARIMA_Predictions"], linestyle="-.")
plt.plot_date(test_data.index, test_data["LSTM_Predictions"], linestyle="--")
plt.plot_date(test_data.index, test_data["Prophet_Predictions"], linestyle=":")
plt.title("Forecasts")
plt.legend() 
plt.show()


# # Reference
# [1] [Time Series Forecasting — ARIMA, LSTM, Prophet with Python](https://medium.com/@cdabakoglu/time-series-forecasting-arima-lstm-prophet-with-python-e73a750a9887) by Caner Dabakoglu  
# [2] [Quick Start | Prophet](https://facebook.github.io/prophet/docs/quick_start.html)  
# [3] [Implementing Facebook Prophet efficiently](https://towardsdatascience.com/implementing-facebook-prophet-efficiently-c241305405a3) by Ruan van der Merwe  
# [4] [Forecasting multiple time-series using Prophet in parallel](https://medium.com/spikelab/forecasting-multiples-time-series-using-prophet-in-parallel-2515abd1a245)  
#
#

