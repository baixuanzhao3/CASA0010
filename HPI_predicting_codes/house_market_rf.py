import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    df = pd.DataFrame(data)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    agg = pd.concat(cols, axis=1)
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]
# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
    train = np.asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    model = RandomForestRegressor(n_estimators=1000)
    model.fit(trainX, trainy)
    yhat = model.predict([testX])
    return yhat[0]

from sklearn.metrics import mean_absolute_error
def walk_forward_validation(data, n_test):
    predictions = list()
    train, test = train_test_split(data, n_test)
    history = [x for x in train]
    for i in range(len(test)):
        testX, testy = test[i, :-1], test[i, -1]
        yhat = random_forest_forecast(history, testX)
        predictions.append(yhat)
        history.append(test[i])
        print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
    error = mean_absolute_error(test[:, -1], predictions)
    return error, test[:, -1], predictions

dir = 'https://raw.githubusercontent.com/baixuanzhao3/thesis/main/Housing%20market%20data.csv'
df1 = pd.read_csv(dir, index_col = 'Date', parse_dates=True)
df1.head(5)

# rename columns
column_names =  df1.columns.to_list()
print(column_names)
new_names = ['Price_Index','FHS_Q','FHS_A','TA','TS','TTU','TV','UNE%','CPI','Monthly_HIBOR','M3','HSI-close','HSI-volume']
df1.columns = new_names

values = df1.Price_Index.values



data = series_to_supervised(values, n_in=6)

e,expected,predictions = walk_forward_validation(data, 680)

nobs = 680
nobs1 = 340
df_train = df1.Price_Index[0:-nobs]
df_test_all = df1.Price_Index[-nobs:]
df_val ,df_test = df_test_all[0:-nobs1], df_test_all[-nobs1:]

df_forecast = pd.DataFrame(predictions, index=df_test_all.index, columns=['Price_Index'])
df_val = pd.DataFrame(expected, index=df_test_all.index, columns=['Price_Index'])[0:nobs1]
df_test = pd.DataFrame(expected, index=df_test_all.index, columns=['Price_Index'])[-nobs1:]
plt.title('Predicted vs. Actual Price Index  2018-07-2020-01')
plt.plot(df_test, label='Actual Price Index')
plt.plot(df_forecast[-nobs1:], label='Predicted Price Index')
#plt.plot(df_train,label = 'training set')
plt.legend()
plt.show()

df_f1 = df_forecast[0:-nobs1]
df_f2 = df_forecast[-nobs1:]

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE

    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse})

def adjust1(val, length= 6): return str(val).ljust(length)
print('Forecast Accuracy of: Price Index')
accuracy_prod = forecast_accuracy(df_f2, df_test)
for k, v in accuracy_prod.items():
    print(adjust1(k), ': ', round(v,4))