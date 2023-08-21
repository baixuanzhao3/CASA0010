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

predicted_value = model_fit.predict(2).to_frame()

plt.title('VAR fitted line for training set')
plt.plot(df.Price_Index[0:-680],label='training set')
plt.plot(predicted_value,label='VAR fitted line')
plt.legend()
plt.show()


# Out-of-Time Cross validation
nobs = 680
nobs1 = 340
train, test_all = df.Price_Index[0:-nobs], df.Price_Index[-nobs:]
val,test = test_all[0:-nobs1],test_all[-nobs1:]
model1 = ARIMA(train, order=(5,1,5))
model1_fit = model1.fit()

sqrt1 = np.sqrt(sum(test**2)/len(test))
# multistep forecast 
fc= model1_fit.forecast(680)
Index = test_all.index
fc.index = Index
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training Price Index')
plt.plot(test_all[0:-nobs1], label='actual Price Index ')
plt.plot(fc[0:-nobs1], label='forecasted Price Index')

plt.title('Forecasted vs Actual Price Index')
plt.legend(loc='upper left', fontsize=8)
plt.show()

plt.plot(test_all[-nobs1:], label='actual Price Index ')
plt.plot(fc[-nobs1:], label='forecasted Price Index')

plt.title('Forecasted vs Actual Price Index from 2018-07-2020-01')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# single step forecast
predictions = []
history = train.values.tolist()

ARIMA(history, order=(4,1,4))
for i in range(len(test_all)):
    model2 = ARIMA(history, order=(4,1,4))
    model2_fit = model2.fit()
    fc1 = model2_fit.forecast(1).tolist()[0]
    predictions.append(fc1)
    history.append(test_all[i])

plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training Price Index')
plt.plot(test_all[-nobs1:], label='actual Price Index ')
plt.plot(fc[-nobs1:], label='forecasted Price Index')

plt.title('Forecasted vs Actual Price Index')
plt.legend(loc='upper left', fontsize=8)
plt.show()

Pr = pd.Series(predictions,index=test_all.index)

plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training Price Index')
#plt.title('Forecasted vs Actual Price Index, one step forecaast')
plt.plot(test_all[0:-nobs1],label='atual Price Index')
plt.plot(Pr[0:-nobs1],label='forecasted price index')

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

### Var
# load data
dir = 'https://raw.githubusercontent.com/baixuanzhao3/thesis/main/Housing%20market%20data.csv'
df = pd.read_csv(dir, index_col = 'Date', parse_dates=True)
df.head(5)

# rename columns
column_names =  df.columns.to_list()
print(column_names)
new_names = ['Price_Index','FHS_Q','FHS_A','TA','TS','TTU','TV','UNE%','CPI','Monthly_HIBOR','M3','HSI-close','HSI-volume']
df.columns = new_names

# plot of target variable against time
plt.plot(df.Price_Index)
plt.xlabel("Time")
plt.ylabel("Private Domestic(Price Index)")
plt.title("Private Domestic(Price Index) Over Time")
plt.savefig("initial_plot.png", dpi=250)
plt.show()

df_feature = df.drop(['Price_Index'],axis=1)
fig, axes = plt.subplots(nrows=4, ncols=3, dpi=120, figsize=(10,6))
fig.suptitle('feature time series')
for i, ax in enumerate(axes.flatten()):
    data = df_feature[df_feature.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(df_feature.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()

from statsmodels.tsa.stattools import grangercausalitytests
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

grangers_causation_matrix(df, variables = df.columns) 

from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,1)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df)
out1 = coint_johansen(df,-1,1).cvt
nobs = 680
nobs1 =340
df_train, df_test_all = df[0:-nobs], df[-nobs:]
df_val,df_test = df_test_all[0:-nobs1],df_test_all[-nobs1:]


# Check size
print(df_test.shape)  

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")  


for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


df_differenced = df_train.diff().dropna()

for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

model = VAR(df_differenced)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

model_fitted = model.fit(6)
model_fitted.summary()

from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

def adjust1(val, length= 6): return str(val).ljust(length)
for col, val in zip(df.columns, out):
    print(adjust1(col), ':', round(val, 2))

lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = df_differenced.values[-lag_order:]
forecast_input
# validation/test
fc = model_fitted.forecast(y=forecast_input, steps=680)
df_forecast = pd.DataFrame(fc, index=df_test_all.index, columns=df.columns + '_1d')
def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:        
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
    return df_fc

df_results = invert_transformation(df_train, df_forecast, second_diff=False)

history_arr1 = pd.concat((df_train,df_val),axis=0)

history_arr2 = df_train
prediction_var = []
for i in range(len(df_val)):
    df_differenced1 = history_arr2.diff().dropna()
    model11 = VAR(df_differenced1)
    model_fitted1 = model11.fit(6)
    forecast_input1 = df_differenced1.values[-lag_order:]
    fc1 = model_fitted1.forecast(y=forecast_input1, steps=1)
    df_forecast1 = pd.DataFrame(fc1, columns=df.columns + '_1d')


    df_results1 = invert_transformation(history_arr2, df_forecast1, second_diff=False)
    prediction_var.append(df_results1.iloc[0,13])
    history_arr2  = pd.concat((history_arr2,df_val.iloc[:i+1,:]),axis=0)

forc = pd.Series(prediction_var,index=df_val.index)
real = pd.Series(df_val.iloc[:,0].values,index=df_val.index)




# plot predicted house price index and test values
plt.title('Predicted vs. Actual House Price Index from 2018-07 to 2020-01')
plt.plot(df_results.Price_Index_forecast[0:nobs1],label='forecasted mean')
plt.plot(df_test_all.Price_Index[0:nobs1],label='actual')
#plt.plot(yhat_conf_int[-nobs1:],label='prediction interval bond')
plt.xlabel('Date')
plt.ylabel('House Price Index')
plt.legend()

# plot predicted house price index and all actual values
plt.title('Predicted vs. Actual House Price Index')
plt.plot(df_results.Price_Index_forecast[-nobs1:],label='forecast')
plt.plot(df_test_all.Price_Index[-nobs1:],label='actual')
#plt.plot(yhat_conf_int[-nobs1:],label='prediction interval bond')
#plt.plot(df_train.Price_Index,label='training set')

plt.xlabel('Date')
plt.ylabel('House Price Index')
plt.legend()

plt.title('Predicted vs. Actual House Price Index')
plt.plot(forc,label='forecast')
plt.plot(real,label='actual')
#plt.plot(yhat_conf_int[-nobs1:],label='prediction interval bond')
#plt.plot(df_train.Price_Index,label='training set')

plt.xlabel('Date')
plt.ylabel('House Price Index')
plt.legend()





# with uncertainty



from statsmodels.tsa.stattools import acf
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

print('Forecast Accuracy of: Price Index')
accuracy_prod = forecast_accuracy(forc, real)
for k, v in accuracy_prod.items():
    print(adjust1(k), ': ', round(v,4))


