import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller,pacf,acf
from statsmodels.tools.eval_measures import rmse, aic

### ARIMA
# load data
dir = 'https://raw.githubusercontent.com/baixuanzhao3/thesis/main/Housing%20market%20data.csv'
df1 = pd.read_csv(dir, index_col = 'Date', parse_dates=True)
df1.head(5)

# rename columns
column_names =  df1.columns.to_list()
print(column_names)
new_names = ['Price_Index','FHS_Q','FHS_A','TA','TS','TTU','TV','UNE%','CPI','Monthly_HIBOR','M3','HSI-close','HSI-volume']
df1.columns = new_names

# plot of target variable against time
plt.plot(df1.Price_Index)
plt.xlabel("Time")
plt.ylabel("Private Domestic(Price Index)")
plt.title("Private Domestic(Price Index) Over Time")
plt.savefig("initial_plot.png", dpi=250)
plt.show()

# Only time series of house price market is needed
df = df1['Price_Index'].to_frame()

# find order of differentiation d
# adf test
result = adfuller(df)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


result1 = adfuller(df.Price_Index.diff().dropna())
print('ADF Statistic: %f' % result1[0])
print('p-value: %f' % result1[1])

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Original Series
fig, axes = plt.subplots(2, 1, sharex=True)
axes[0].plot(df.Price_Index); axes[0].set_title('Original Series')

# 1st Differencing
axes[1].plot(df.Price_Index.diff()); axes[1].set_title('1st Order Differencing')


plt.show()

# right order of differencing is 1

# find order of lag p in AR model
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.Price_Index.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.Price_Index.diff().dropna(), ax=axes[1])

plt.show()


critical_line = 0.22*np.ones((80,))
pacf_,conf = pacf(df.Price_Index.diff().dropna(), nlags=80, method='ywadjusted', alpha=0.05)
plt.title('pacf and critical pacf')
plt.plot(pacf_,label='pacf')
plt.plot(critical_line,label='critical value')
plt.legend()

# lag above critical value: 4 So p=4
#find the order of the MA term (q)
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig1, axes1 = plt.subplots(1, 2, sharex=True)
axes1[0].plot(df.Price_Index.diff()); axes1[0].set_title('1st Differencing')
axes1[1].set(ylim=(0,5))
plot_acf(df.Price_Index.diff().dropna(), ax=axes1[1])

acf_,conf = acf(df.Price_Index.diff().dropna(), nlags=80,alpha=0.05)
plt.title('acf and critical acf')
plt.plot(acf_,label='acf')
plt.plot(critical_line,label='critical value')
plt.legend()

# q=4

# Building a model using all in-sample lagged values,
# model get trained till the last value to make prediction of the next
from statsmodels.tsa.arima.model import ARIMA

# 1,1,2 ARIMA Model
model = ARIMA(df.Price_Index[0:-680], order=(4,1,4))
model_fit = model.fit()
print(model_fit.summary())

predicted_value = model_fit.predict(2).to_frame().iloc[0,0]

plt.title('ARIMA fitted line for training set')
plt.plot(df.Price_Index[0:-680],label='training set')
plt.plot(predicted_value,label='ARIMA fitted line')
plt.legend()
plt.show()


# mutistep Out-of-Time Cross validation
nobs = 680
nobs1 = 340
train, test_all = df.Price_Index[0:-nobs], df.Price_Index[-nobs:]
val,test = test_all[0:-nobs1],test_all[-nobs1:]
model1 = ARIMA(train, order=(5,1,5))
model1_fit = model1.fit()



fc= model1_fit.forecast(680)
Index = test_all.index
fc.index = Index

# single step out-of-time cross validation

predictions = list()
history_arr = np.concatenate((train,val),axis=0)
history = [x for x in history_arr]
for i in range(len(test)):
    testy = test.iloc[i]
    new_model = ARIMA(history, order=(5,1,5))
    new_model.initialize_approximate_diffuse()
    new_model_fit = new_model.fit()
    y_hat = new_model_fit.forecast(1)[0]
    predictions.append(y_hat)
    history.append(y_hat)
    print('>expected=%.1f, predicted=%.1f' % (testy, y_hat))





plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training Price Index')
plt.plot(test_all[-nobs1:], label='actual Price Index ')
plt.plot(fc[-nobs1:], label='forecasted Price Index')

plt.title('Forecasted vs Actual Price Index')
plt.legend(loc='upper left', fontsize=8)
plt.show()

plt.plot(test_all[-nobs1:], label='actual Price Index ')
plt.plot(fc[-nobs1:], label='forecasted Price Index')

plt.title('Forecasted vs Actual Price Index from 2018-07-2020-01')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# with uncertainty 
fcu = model1_fit.get_forecast(680)

yhat = fcu.predicted_mean
yhat_conf_int = fcu.conf_int(alpha=0.05)
yhat.index = Index
yhat_conf_int.index = Index


plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training Price Index')
plt.plot(yhat_conf_int[-nobs1:], label='prediction interval bond')
plt.plot(yhat[-nobs1:], label='forecasted mean Price Index')
plt.plot(test_all[-nobs1:],label = 'actual Price Index')
plt.title('Forecasted vs Actual Price Index')
plt.legend(loc='upper left', fontsize=8)
plt.show()





plt.plot(yhat_conf_int[-nobs1:], label='prediction interval bond')
plt.plot(yhat[-nobs1:], label='forecasted mean Price Index')
plt.plot(test_all[-nobs1:],label = 'actual Price Index')
plt.title('Forecasted vs Actual Price Index from 2018-07-2020-01')
plt.legend(loc='upper left', fontsize=8)
plt.show()




test_arr = test.values
pred_arr = np.array(predictions)
test_f = pd.Series(test_arr,index=test.index)
pred_f = pd.Series(predictions,index=test.index)

plt.plot(pred_f, label='forecasted Price Index')
plt.plot(test_f, label='actual Price Index ')


plt.title('Forecasted vs Actual Price Index from 2018-07-2020-01')
plt.legend(loc='upper left', fontsize=8)
plt.show()

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'corr':corr, 'minmax':minmax})

def adjust1(val, length= 6): return str(val).ljust(length)
print('Forecast Accuracy of: Price Index')
accuracy_prod = forecast_accuracy(pred_arr, test_arr)
for k, v in accuracy_prod.items():
    print(adjust1(k), ': ', round(v,4))

