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

# # CO2 time series

# Reading list:  
# - [How to Develop a Skillful Machine Learning Time Series Forecasting Model](https://machinelearningmastery.com/how-to-develop-a-skilful-time-series-forecasting-model/)  
# - [11 Classical Time Series Forecasting Methods in Python (Cheat Sheet)](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/)

# As is best practice, start by importing the libraries you will need at the top 
# of your notebook:
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import statsmodels.tsa.stattools as ts
plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")

def run_sequence_plot(x, y, title='', xlabel="time", ylabel="series"):
    plt.plot(x, y, 'b-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3);

def wordwrap(text, width=70):
    for i in ['# ' + i for i in textwrap.wrap(text, width)]:
        print(i)

def plots(tmsr, lags=None):
    layout = (1, 3)
    raw  = plt.subplot2grid(layout, (0, 0))
    acf  = plt.subplot2grid(layout, (0, 1))
    pacf = plt.subplot2grid(layout, (0, 2))
    sns.set(style="ticks", rc={"lines.linewidth": 0.7}) #<----
    # https://stackoverflow.com/questions/45540886/reduce-line-width-of-seaborn-timeseries-plot
    tmsr.plot(ax=raw)
    smt.graphics.plot_acf(tmsr, lags=lags, ax=acf)
    smt.graphics.plot_pacf(tmsr, lags=lags, ax=pacf)
    sns.despine()
    plt.tight_layout()        

# Reference: [A Guide to Time Series Forecasting with ARIMA in Python 3](https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3)
# > - [The Statsmodels Datasets Package: a List](https://www.statsmodels.org/devel/datasets/index.html)

# ## Structured Data

# We’ll be working with a dataset called “Atmospheric CO2 from Continuous Air 
# Samples at Mauna Loa Observatory, Hawaii, U.S.A.,” which collected CO2 samples
# from March 1958 to December 2001. 
data = sm.datasets.co2.load_pandas()

data.names

y = data.data

# Data Preprocessing  
# Let’s preprocess our data a little bit before moving forward. Weekly data can
# be tricky to work with since it’s a briefer amount of time, so let’s use 
# monthly averages instead. We’ll make the conversion with the resample 
# function. For simplicity, we can also use the fillna() function to ensure 
# that we have no missing values in our time series.

# The 'MS' string groups the data in buckets by start of the month
y = y['co2'].resample('MS').mean()

# Reference: [What's the difference between .isna() and .isnull()](https://datascience.stackexchange.com/questions/37878/difference-between-isna-and-isnull-in-pandas#answer-37879)

all(y.notna())

# The term bfill means that we use the value before filling in missing values
y.fillna(method ='bfill', inplace = True)

y.plot(figsize=(10, 5));

# Parameter Selection for the ARIMA Time Series Model  
# We will use a “grid search” to iteratively explore different combinations of 
# parameters. For each combination of parameters, we fit a new seasonal ARIMA 
# model with the SARIMAX() function from the statsmodels module and assess its
# overall quality. Once we have explored the entire landscape of parameters, 
# our optimal set of parameters will be the one that yields the best performance
# for our criteria of interest.
#
# In Statistics and Machine Learning, this process is known as grid search 
# (or hyperparameter optimization) for model selection.

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]
# [itertools.product](https://www.hackerrank.com/challenges/itertools-product/problem)
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# When evaluating and comparing statistical models fitted with different 
# parameters, each can be ranked against one another based on how well it fits
# the data or its ability to accurately predict future data points. We will use 
# the AIC (Akaike Information Criterion) value, which is conveniently returned 
# with ARIMA models fitted using statsmodels. The AIC measures how well a model 
# fits the data while taking into account the overall complexity of the model. 
# A model that fits the data very well while using lots of features will be 
# assigned a larger AIC score than a model that uses fewer features to achieve 
# the same goodness-of-fit. Therefore, we are interested in finding the model 
# that yields the __lowest__ AIC value.

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, 
                                                 param_seasonal, 
                                                 results.aic))
        except:
            continue

# Because some parameter combinations may lead to numerical misspecifications, 
# we explicitly disabled warning messages in order to avoid an overload of 
# warning messages. These misspecifications can also lead to errors and throw 
# an exception, so we make sure to catch these exceptions and ignore the 
# parameter combinations that cause these issues.

# The output of our code suggests that SARIMAX(1, 1, 1)x(1, 1, 1, 12) yields 
# the lowest AIC value of 277.78. We should therefore consider this to be 
# optimal option out of all the models we have considered.

# The output of our code suggests that SARIMAX(1, 1, 1)x(1, 1, 1, 12)
# yields the lowest AIC value of 277.78. We should therefore consider
# this to be optimal option out of all the models we have considered.           

# Model Estimation

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.tight_layout()

print(results.summary()) # With print(), the result shows in txt mode.

# ## Unstructured Data

# Load Data
# Note that the indices of date is not compatible with Pandas yet.
co2 = pd.read_csv('./Data/co2-ppm-mauna-loa-19651980.csv', 
                  header = 0,
                  names = ['idx', 'co2'],
                  skipfooter = 2)
co2.info()

co2 = co2.drop('idx', axis=1)
# recast co2 column to float
co2['co2'] = pd.to_numeric(co2['co2'])

co2.head()

co2.index = np.arange(1, len(co2)+1) 

# To reset indices starting from 1, `reset_index()` is not the answer.

co2.head()

# ### set index in datetime format

# set index
index = pd.date_range('1/1/1965', periods=192, freq='M')
index

co2.index = pd.to_datetime(index)
co2['co2'].head()

co2.plot(figsize=(10,4)); # Or else plt.plot(co2.index, co2.co2); plt.grid()

# +
# co2 = xbackup
# -

# resampling with the modification to annual frequency
co2['co2'].resample('A').mean().head()

# resample to monthly and check missing values
co2 = co2['co2'].resample('M').mean()

co2.isna().sum()

# decompose data into trend, seasonal, and residual
plt.style.use('fivethirtyeight')
decomposition = sm.tsa.seasonal_decompose(co2, model='additive')
fig = decomposition.plot();

# build model
co2sar = sm.tsa.statespace.SARIMAX(co2, order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit()
print(co2sar.summary())

# check diagnostics
plt.rcParams['figure.figsize'] = [15, 12] 
co2sar.plot_diagnostics() # Or else put figsize=(15, 12) inside
plt.tight_layout()

# create predictions and confidence intervals
pred = co2sar.get_prediction(start=pd.to_datetime('1979-4-30'), dynamic=False) # we use as many true values as possible to predict
pred_ci = pred.conf_int()

# +
# plot predictions
plt.rcParams['figure.figsize'] = [10, 6] 
ax = co2.plot(label='Observed CO2 Levels')
pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.8) # this is using all available info

ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.1)

ax.set_xlabel('Year')
ax.set_ylabel('CO2')
plt.legend();
# -

# compute mean square error
fcast = pred.predicted_mean
true = co2['1979-4-30':]
mse = ((fcast - true) ** 2).mean()
print('MSE of our forecasts is {}'.format(round(mse, 3)))

# Now we set the argument `dynamic` to be True. It leads to larger prediction variance. See for the illustration below.

# dynamic forecast
fcast = co2sar.get_prediction(start=pd.to_datetime('1979-4-30'), dynamic=True, full_results=True)
fcast_ci = fcast.conf_int()

fcast_ci.head()

# +
# plot predictions
ax = co2['1979':].plot(label='Observed CO2 Levels')
fcast.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(fcast_ci.index, fcast_ci.iloc[:, 0],
                fcast_ci.iloc[:, 1], color='k', alpha=.25)

ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1980-11-30'), co2.index[-1],
                 alpha=.1, zorder=-1)

ax.set_xlabel('Year')
ax.set_ylabel('CO2')

plt.legend(loc="upper left");

# +
# compute mean square error
fcast_avg = fcast.predicted_mean
true = co2['1979-4-30':]

mse = ((fcast_avg - true) ** 2).mean()
print('MSE is {}'.format(round(mse, 3)))

# notice it's much higher

# +
# forecast next 100 months and get confidence interval
pred_uc = co2sar.get_forecast(steps=100)

pred_ci = pred_uc.conf_int()

# +
# plot forecast
ax = co2[:].plot(label='Observed CO2 Levels')
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Year')
ax.set_ylabel('CO2')

plt.legend(loc="upper left");

# + [markdown] toc-hr-collapsed=true toc-nb-collapsed=true
# # Temperature time series
# -

# load data and convert to datetime
monthly_temp = pd.read_csv('./Data/mean-monthly-temperature-1907-19.csv', 
                           skipfooter=2, 
                           infer_datetime_format=True, 
                           header=0, 
                           index_col=0, 
                           names=['month', 'temp'])

monthly_temp.info()

monthly_temp.head()

# Now we want to make the indices of the series in datetime format.

monthly_temp.index = pd.to_datetime(monthly_temp.index)

monthly_temp.info() 

monthly_temp.describe()

# resample to annual and plot each
plt.rcParams['figure.figsize'] = [14, 4]
annual_temp = monthly_temp.resample('A').mean()
monthly_temp.plot(grid=True)
annual_temp.plot(grid=True);

# plot both on same figure
plt.plot(monthly_temp)
plt.plot(annual_temp)
plt.grid();

# violinplot of months to determine variance and range
sns.violinplot(x=monthly_temp.index.month, y=monthly_temp.temp)
plt.grid();

# Are these datasets stationary? We can look at a few things per the list above, including a visual check (there seems to be a small upward trend in the annual, too hard to tell for monthly), a standard deviation check on various differences (smallest one is usually most stationary), and the formal Dickey-Fuller test.

# check montly deviations for various diffs
print(monthly_temp.temp.std())
print(monthly_temp.temp.diff().std())
print(monthly_temp.temp.diff().diff().std()) # theoretically lowest, but > 1 is close enough
print(monthly_temp.temp.diff().diff().diff().std())

# check annual deviations for various diffs
print(annual_temp.temp.std()) # looks stationary
print(annual_temp.temp.diff().std())
print(annual_temp.temp.diff().diff().std())
print(annual_temp.temp.diff().diff().diff().std())

# define Dickey-Fuller Test (DFT) function
import statsmodels.tsa.stattools as ts
def dftest(timeseries):
    dftest = ts.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic','p-value','Lags Used','Observations Used'])
    
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.grid()
    plt.show(block=False)


# run DFT on monthly
dftest(monthly_temp.temp)
# p-value allows us to reject a unit root: data is stationary

# helper plot for monthly temps
plots(monthly_temp, lags=75);
# open Duke guide for visual
# we note a 12-period cycle (yearly) with suspension bridge design, so must use SARIMA

# ## Box-Jenkins Method
# Reference:  
# - Duke Rule of ARIMA model identification [link](https://people.duke.edu/~rnau/arimrule.htm)

# +
import statsmodels.api as sm

# fit SARIMA monthly based on helper plots
sar = sm.tsa.statespace.SARIMAX(monthly_temp.temp, 
                                order=(0,0,0), 
                                seasonal_order=(0,1,0,12), 
                                trend='c').fit()
sar.summary()
# -

# plot resids
plots(sar.resid, lags=40);

sar = sm.tsa.statespace.SARIMAX(monthly_temp.temp, 
                                order=(0,0,0), 
                                seasonal_order=(0,1,0,12), 
                                trend='c').fit()
sar.summary()

# Thought process:
# 010010 is overdiff by AIC and negative ACR, but 000010 is a big underdiff with better AIC we pick 000010,12 and Trend='c' per rule 4/5l
#
# Now look at seasonal. Notice negative ACR spike at 12: per rule 13, we add a SMA term and we see a big drop to 4284 AIC looks like ACR looks good at seasonal lags, so we move back to ARIMA portion.
#
# Rule 6 says we're a bit underdiff, so we add AR=3 based on PACF: 4261 AIC.
#

sar = sm.tsa.statespace.SARIMAX(monthly_temp.temp, 
                                order=(1,0,0), 
                                seasonal_order=(0,1,0,12), 
                                trend='c').fit()
plots(sar.resid, lags=40);

plt.rcParams['figure.figsize'] = [13, 4] 
sar = sm.tsa.statespace.SARIMAX(monthly_temp.temp, 
                                order=(1,0,0), 
                                seasonal_order=(0,1,1,12), 
                                trend='c').fit()
plots(sar.resid, lags=40);

plt.rcParams['figure.figsize'] = [8, 6] 
sar.plot_diagnostics();
plt.tight_layout() # <---

plt.rcParams['figure.figsize'] = [13, 4] 
sar = sm.tsa.statespace.SARIMAX(monthly_temp.temp, 
                                order=(1,1,1), 
                                seasonal_order=(0,1,1,12), 
                                trend='c').fit()
plots(sar.resid, lags=40);

plt.rcParams['figure.figsize'] = [8,6] 
sar.plot_diagnostics();
plt.tight_layout() # <---

monthly_temp['forecast'] = sar.predict(start = 750, end= 790, dynamic=False)  
monthly_temp[730:][['temp', 'forecast']].plot();

# Our primary concern is to ensure that the residuals of our model are
# uncorrelated and normally distributed with zero-mean. If the seasonal
# ARIMA model does not satisfy these properties, it is a good indication
# that it can be further improved.

# In the top right plot, we see that the red KDE line follows closely
# with the N(0,1) line (where N(0,1)) is the standard notation for a
# normal distribution with mean 0 and standard deviation of 1). This is
# a good indication that the residuals are normally distributed. 
#
# The qq-plot on the bottom left shows that the ordered distribution of
# residuals (blue dots) follows the linear trend of the samples taken
# from a standard normal distribution with N(0, 1). Again, this is a
# strong indication that the residuals are normally distributed. 
#
# The residuals over time (top left plot) don’t display any obvious
# seasonality and appear to be white noise. This is confirmed by the
# autocorrelation (i.e. correlogram) plot on the bottom right, which
# shows that the time series residuals have low correlation with lagged
# versions of itself.

# Validating Forecasts  
# The get_prediction() and conf_int() attributes allow us to obtain the
# values and associated confidence intervals for forecasts of the time
# series.

pred = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
pred_ci = pred.conf_int()

# The dynamic=False argument ensures that we produce one-step ahead
# forecasts, meaning that forecasts at each point are generated using
# the full history up to that point.

ax = y['1990':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()

# We will use the MSE (Mean Squared Error), which summarizes the average
# error of our forecasts. For each predicted value, we compute its
# distance to the true value and square the result. The results need to
# be squared so that positive/negative differences do not cancel each
# other out when we compute the overall mean.

y_forecasted = pred.predicted_mean
y_truth = y['1998-01-01':]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# However, a better representation of our true predictive power can be
# obtained using dynamic forecasts. In this case, we only use
# information from the time series up to a certain point, and after
# that, forecasts are generated using values from previous forecasted
# time points.

# In the code chunk below, we specify to start computing the dynamic forecasts 
# and confidence intervals from January 1998 onwards.

pred_dynamic = results.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

ax = y['1990':].plot(label='observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1998-01-01'), y.index[-1],
                 alpha=.1, zorder=-1) # shaded area in transparent blue
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()

# Plotting the observed and forecasted values of the time series, we see
# that the overall forecasts are accurate even when using dynamic
# forecasts. All forecasted values (red line) match pretty closely to
# the ground truth (blue line), and are well within the confidence
# intervals of our forecast.

# Extract the predicted and true values of our time series
y_forecasted = pred_dynamic.predicted_mean
y_truth = y['1998-01-01':]
# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

# Producing and Visualizing Forecasts

# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=500)
# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

ax = y.plot(label='observed', figsize=(20, 15))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()

# As we forecast further out into the future, it is natural for us to
# become less confident in our values. This is reflected by the
# confidence intervals generated by our model, which grow larger as we
# move further out into the future.

plt.rcParams['figure.figsize'] = [14, 4]
plots(y, lags=30)

def dftest(timeseries):
    dftest = ts.adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic','p-value','Lags Used','Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.grid()
    plt.show(block=False)

dftest(y)

autores = sm.tsa.arma_order_select_ic(y, 
                                      ic=['aic', 'bic'], 
                                      trend='c', 
                                      max_ar=4, 
                                      max_ma=4, 
                                      fit_kw=dict(method='css-mle'))

print('AIC', autores.aic_min_order) # will use this as inputs for annual
print('BIC', autores.bic_min_order)

# # Kalman Filter Approach
# The Kalman filter is a set of mathematical equations that provides an
# efficient computational (recursive) means to estimate the state of a
# process, in a way that minimizes the mean of the squared error.

# Reference
# - First, you should review some baiscs with a great example, so see this post:  
# [How a Kalman Filter works, in pictures](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
# - Seocond, move onto a short tutorial with more rigorous treatment in this paf:  
# [An Introduction to the Kalman Filter, pages 11-15](https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf)
# Also, look at Figure 1-1, Tables 1-1 and 1-2 on page 5.
# - Then, look at the webpage for an implemenation just mentioned in the pdf file    
# [SciPy Cookbook](https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html)  
# Now, we replicate the example in the SciPy Cookbook below.

plt.rcParams['figure.figsize'] = (10, 8)
# intial parameters
n_iter = 50
sz = (n_iter,) # size of array
x = -0.37727 # truth value (typo in example at top of p. 13 calls this z)
z = np.random.normal(x,0.1,size=sz) # observations (normal about x, sigma=0.1)
Q = 1e-5 # process variance
R = 0.1**2 # estimate of measurement variance, change to see effect

# xhat : $[\hat{x}_k]$  
# xhatminus : $[\hat{x}^{-}_k]$

# allocate space for arrays
xhat=np.zeros(sz)      # a posteri estimate of x
P=np.zeros(sz)         # a posteri error estimate
xhatminus=np.zeros(sz) # a priori estimate of x
Pminus=np.zeros(sz)    # a priori error estimate
K=np.zeros(sz)         # gain or blending factor
# intial guesses
xhat[0] = 0.0
P[0] = 1.0

# **Note:** Choosing $P_0 = 0$ would cause the filter to initially and always 
# believe $\hat{x}_k=0$.

# (See page 12) In this exame, time update equations are reduced to:  
# $\begin{gather*}
# \hat{x}_{k}^{-}=\hat{x}_{k-1},\\
# P_{k}^{-}=P_{k-1}+Q.
# \end{gather*}$  
# Measurement update equations are reduced to:  
# $\begin{gather*}
# K_{k}=P_{k}^{-}(P_{k}^{-}+R)^{-1},\\
# \hat{x}_{k}=\hat{x}_{k}^{-}+K_{k}(z_{k}-\hat{x}_{k}^{-}),\\
# P_{k}=(1-K_{k})P_{k}^{-}.
# \end{gather*}$

for k in range(1,n_iter):
    # time update
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1]+Q

    # measurement update
    K[k] = Pminus[k]/( Pminus[k]+R )
    xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
    P[k] = (1-K[k])*Pminus[k]

plt.figure(1)
plt.plot(z,'m+',label='noisy measurements')
plt.plot(xhat,'b-',label='a posteri estimate')
plt.axhline(x,color='g',label='truth value')
plt.legend()
plt.title('Estimate vs. iteration step', fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('Voltage')
plt.show()

plt.figure(2)
valid_iter = range(1,n_iter) # Pminus not valid at step 0
plt.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')
plt.title('Estimated $\it{\mathbf{a \ priori}}$ error vs. iteration step', 
          fontweight='bold')
plt.xlabel('Iteration')
plt.ylabel('$(Voltage)^2$')
plt.setp(plt.gca(),'ylim',[0,.01])
plt.show()

# # GCP Environment

# Connecting to Jupyter Lab [link](https://cloud.google.com/ai-platform/deep-learning-vm/docs/jupyter)


