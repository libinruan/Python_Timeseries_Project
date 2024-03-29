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

# # Time Series Forecasting with Python (ARIMA, LSTM, Prophet)

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf 
from statsmodels.tsa.seasonal import seasonal_decompose 
from pmdarima import auto_arima                        
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings("ignore")

# In this article we will try to forecast a time series data basically. We'll build three different model with Python and inspect their results. Models we will use are ARIMA (Autoregressive Integrated Moving Average), LSTM (Long Short Term Memory Neural Network) and Facebook Prophet. Let's jump in and start with ARIMA.

# ## ARIMA (Autoregressive Integrated Moving Average)

# ARIMA is a model which is used for predicting future trends on a time series data. It is model that form of regression analysis. 
# * **AR (Autoregression) :** Model that shows a changing variable that regresses on its own lagged/prior values.
# * **I (Integrated) :**  Differencing of raw observations to allow for the time series to become stationary
# * **MA (Moving average) :** Dependency between an observation and a residual error from a moving average model
#
# For ARIMA models, a standard notation would be ARIMA with p, d, and q, where integer values substitute for the parameters to indicate the type of ARIMA model used.
#
# * **p:** the number of lag observations in the model; also known as the lag order.
# * **d:** the number of times that the raw observations are differenced; also known as the degree of differencing.
# * **q:** the size of the moving average window; also known as the order of the moving average.
#
# For more information about ARIMA you can check:
# <br>
# [What is ARIMA](https://www.quora.com/What-is-ARIMA)
# <br>
# [Autoregressive Integrated Moving Average (ARIMA)](https://www.investopedia.com/terms/a/autoregressive-integrated-moving-average-arima.asp)

# ## LSTM Neural Network

# > LSTM stands for long short term memory. It is a model or architecture that extends the memory of recurrent neural networks. Typically, recurrent neural networks have ‘short term memory’ in that they use persistent previous information to be used in the current neural network. Essentially, the previous information is used in the present task. That means we do not have a list of all of the previous information available for the neural node.
# > LSTM introduces long-term memory into recurrent neural networks. It mitigates the vanishing gradient problem, which is where the neural network stops learning because the updates to the various weights within a given neural network become smaller and smaller. It does this by using a series of ‘gates’. These are contained in memory blocks which are connected through layers, like this:
#
# ![](https://hub.packtpub.com/wp-content/uploads/2018/04/LSTM-696x494.png)
#
# > LSTM work
# There are three types of gates within a unit:
# Input Gate: Scales input to cell (write)
# Output Gate: Scales output to cell (read)
# Forget Gate: Scales old cell value (reset)
# Each gate is like a switch that controls the read/write, thus incorporating the long-term memory function into the model.
#
# For more detail:
# <br>
# [What is LSTM?](https://hub.packtpub.com/what-is-lstm/)
# <br>
# [What is LSTM? - Quora](https://www.quora.com/What-is-LSTM)
# <br>
# [Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory)

# ## Prophet

# > Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.
#
# [Facebook's Prophet Web Page](https://facebook.github.io/prophet/)<br>
# [Forecasting at Scale](https://peerj.com/preprints/3190.pdf)
#

# # FORECAST

# ## Read Dataset

df = pd.read_csv('monthly-beer-production-in-austr.csv')

df.head()

df.info()

df.Month = pd.to_datetime(df.Month)

df = df.set_index("Month")
df.head()

df.index.freq = 'MS'

ax = df['Monthly beer production'].plot(figsize = (16,5), title = "Monthly Beer Production")
ax.set(xlabel='Dates', ylabel='Total Production');

# When we look at plot we can sey there is a seasonality in data. That's why we will use SARIMA (Seasonal ARIMA) instead of ARIMA.
#
# > Seasonal ARIMA, is an extension of ARIMA that explicitly supports univariate time series data with a seasonal component.
# > It adds three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality.
#
# > There are four seasonal elements that are not part of ARIMA that must be configured; they are:<br>
# **P:** Seasonal autoregressive order.<br>
# **D:** Seasonal difference order.<br>
# **Q:** Seasonal moving average order.<br>
# **m:** The number of time steps for a single seasonal period.<br>

a = seasonal_decompose(df["Monthly beer production"], model = "add")
a.plot();

import matplotlib.pyplot as plt
plt.figure(figsize = (16,7))
a.seasonal.plot();

# ## ARIMA Forecast

# Let's run auto_arima() function to get best p,d,q,P,D,Q values

auto_arima(df['Monthly beer production'], seasonal=True, m=12,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4).summary()

# As we can see best arima model chosen by auto_arima() is SARIMAX(2, 1, 1)x(4, 0, 3, 12)

# Let's split the data into train and test set

train_data = df[:len(df)-12]
test_data = df[len(df)-12:]

arima_model = SARIMAX(train_data['Monthly beer production'], order = (2,1,1), seasonal_order = (4,0,3,12))
arima_result = arima_model.fit()
arima_result.summary()

arima_pred = arima_result.predict(start = len(train_data), end = len(df)-1, typ="levels").rename("ARIMA Predictions")
arima_pred

test_data['Monthly beer production'].plot(figsize = (16,5), legend=True)
arima_pred.plot(legend = True);

# +
arima_rmse_error = rmse(test_data['Monthly beer production'], arima_pred)
arima_mse_error = arima_rmse_error**2
mean_value = df['Monthly beer production'].mean()

print(f'MSE Error: {arima_mse_error}\nRMSE Error: {arima_rmse_error}\nMean: {mean_value}')
# -

test_data['ARIMA_Predictions'] = arima_pred

# ## LSTM Forecast

# First we'll scale our train and test data with MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train_data)
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

# Before creating LSTM model we should create a Time Series Generator object.

# +
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

lstm_model.summary()
# -

lstm_model.fit_generator(generator,epochs=20)

losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_lstm)),losses_lstm);

# +
lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)
# -

# As you know we scaled our data that's why we have to inverse it to see true predictions.

lstm_predictions_scaled

lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

lstm_predictions

test_data['LSTM_Predictions'] = lstm_predictions

test_data

test_data['Monthly beer production'].plot(figsize = (16,5), legend=True)
test_data['LSTM_Predictions'].plot(legend = True);

# +
lstm_rmse_error = rmse(test_data['Monthly beer production'], test_data["LSTM_Predictions"])
lstm_mse_error = lstm_rmse_error**2
mean_value = df['Monthly beer production'].mean()

print(f'MSE Error: {lstm_mse_error}\nRMSE Error: {lstm_rmse_error}\nMean: {mean_value}')
# -

# ## Prophet Forecast

df.info()

df_pr = df.copy()
df_pr = df.reset_index()

df_pr.columns = ['ds','y'] # To use prophet column names should be like that

train_data_pr = df_pr.iloc[:len(df)-12]
test_data_pr = df_pr.iloc[len(df)-12:]

from fbprophet import Prophet

m = Prophet()
m.fit(train_data_pr)
future = m.make_future_dataframe(periods=12,freq='MS')
prophet_pred = m.predict(future)

prophet_pred.tail()

prophet_pred = pd.DataFrame({"Date" : prophet_pred[-12:]['ds'], "Pred" : prophet_pred[-12:]["yhat"]})

prophet_pred = prophet_pred.set_index("Date")

prophet_pred.index.freq = "MS"

prophet_pred

test_data["Prophet_Predictions"] = prophet_pred['Pred'].values

import seaborn as sns

plt.figure(figsize=(16,5))
ax = sns.lineplot(x= test_data.index, y=test_data["Monthly beer production"])
sns.lineplot(x=test_data.index, y = test_data["Prophet_Predictions"]);

# +
prophet_rmse_error = rmse(test_data['Monthly beer production'], test_data["Prophet_Predictions"])
prophet_mse_error = prophet_rmse_error**2
mean_value = df['Monthly beer production'].mean()

print(f'MSE Error: {prophet_mse_error}\nRMSE Error: {prophet_rmse_error}\nMean: {mean_value}')
# -

rmse_errors = [arima_rmse_error, lstm_rmse_error, prophet_rmse_error]
mse_errors = [arima_mse_error, lstm_mse_error, prophet_mse_error]
errors = pd.DataFrame({"Models" : ["ARIMA", "LSTM", "Prophet"],"RMSE Errors" : rmse_errors, "MSE Errors" : mse_errors})

plt.figure(figsize=(16,9))
plt.plot_date(test_data.index, test_data["Monthly beer production"], linestyle="-")
plt.plot_date(test_data.index, test_data["ARIMA_Predictions"], linestyle="-.")
plt.plot_date(test_data.index, test_data["LSTM_Predictions"], linestyle="--")
plt.plot_date(test_data.index, test_data["Prophet_Predictions"], linestyle=":")
plt.legend()
plt.show()

print(f"Mean: {test_data['Monthly beer production'].mean()}")
errors

test_data

# Don't forget they are just quick and basic predictions so you can improve these models with tuning and according to your data and business knowledge.
#
# <br>
#
# Thanks!
