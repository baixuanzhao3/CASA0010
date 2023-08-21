import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
import pandas as pd
dir = 'https://raw.githubusercontent.com/baixuanzhao3/thesis/main/Housing%20market%20data.csv'
df1 = pd.read_csv(dir, index_col = 'Date', parse_dates=True)
df1.head(5)

# rename columns
column_names =  df1.columns.to_list()
print(column_names)
new_names = ['Price_Index','FHS_Q','FHS_A','TA','TS','TTU','TV','UNE%','CPI','Monthly_HIBOR','M3','HSI-close','HSI-volume']
df1.columns = new_names

nobs = 680
nobs1 =340
values = df1.Price_Index.values
time = df1.index.values
long_predict = 680
def svm_timeseries_prediction(c_parameter,gamma_paramenter):
    X_data = time
    Y_data = values
    print(len(X_data))
    # 整个数据的长度
    long = len(X_data)
    # how many previous data X_data to predict next data
    X_long = 1
    error = []
    svr_rbf = SVR(kernel='rbf', C=c_parameter, gamma=gamma_paramenter)
    # svr_rbf = SVR(kernel='rbf', C=1e5, gamma=1e1)
    # svr_rbf = SVR(kernel='linear',C=1e5)
    # svr_rbf = SVR(kernel='poly',C=1e2, degree=1)
    X = []
    Y = []
    for k in range(len(X_data) - X_long - 1):
        t = k + X_long
        X.append(Y_data[k:t])
        Y.append(Y_data[t + 1])
    y_rbf = svr_rbf.fit(X[:-long_predict], Y[:-long_predict]).predict(X[:])
    for e in range(len(y_rbf)):
        error.append(Y_data[X_long + 1 + e] - y_rbf[e])
    return X_data,Y_data,X_data[X_long+1:],y_rbf,error
 
 
X_data,Y_data,X_prediction,y_prediction,error = svm_timeseries_prediction(50,1e-7)
figure = plt.figure()
tick_plot = figure.add_subplot(2, 1, 1)
tick_plot.plot(X_data, Y_data, label='Actual', color='green', linestyle='-')
tick_plot.axvline(x=X_data[-long_predict], alpha=0.2, color='gray')
# tick_plot.plot(X_data[:-X_long-1], y_rbf, label='data', color='red', linestyle='--')
tick_plot.plot(X_prediction[-nobs1:], y_prediction[-nobs1:], label='Predicted', color='red', linestyle='--')
tick_plot.legend()
tick_plot.set_xlabel('Date')
tick_plot.set_ylabel('Price Index')
tick_plot.set_title('Predicted vs. Actual Price Index 2018-07-2020-01,c=50, gamma=1e-7')
tick_plot = figure.add_subplot(2, 1, 2)
tick_plot.plot(X_prediction,error)
tick_plot.set_xlabel('Date')
tick_plot.set_ylabel('Price Index Estimation Error')

plt.show()

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
accuracy_prod = forecast_accuracy(y_prediction[-nobs1:], Y_data[-nobs1:])
for k, v in accuracy_prod.items():
    print(adjust1(k), ': ', round(v,4))