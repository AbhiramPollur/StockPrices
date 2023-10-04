#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, Adamax

import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score, mean_squared_error, confusion_matrix

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima
import statsmodels.api as sm


# In[3]:


nifty_data = pd.read_csv(r"C:\Users\Abhiram P\Desktop\WissenRC\NIFTY50_all.csv")
meta_data = pd.read_csv(r"C:\Users\Abhiram P\Desktop\WissenRC\stock_metadata.csv")


# In[4]:


nifty_data.head(10)


# In[5]:


nifty_data.shape


# In[6]:


df1 = nifty_data.dropna()
df1['Timestamp'] = pd.to_datetime(df1['Date']).astype('int64')//10**9
df1


# In[7]:


label = LabelEncoder()
encoder = OneHotEncoder()


# In[8]:


df1_dummy = df1
df1_dummy['Symbol'] = label.fit_transform(df1_dummy['Symbol'])
df1_dummy['Series'] = label.fit_transform(df1_dummy['Series'])
df1_dummy


# In[9]:


x_dummy1_open = df1_dummy[['Symbol', 'Series', 'Prev Close', 'High', 'Low', 'Last', 'Close', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble', 'Timestamp']]
y_dummy1_open = df1_dummy[['Open']]


# In[10]:


scaler = MinMaxScaler()


# In[11]:


xtrain_dummy1_open, xtest_dummy1_open, ytrain_dummy1_open, ytest_dummy1_open = train_test_split(x_dummy1_open, y_dummy1_open, test_size=0.2, random_state=42)


# In[12]:


xtrain_dummy1_open = scaler.fit_transform(xtrain_dummy1_open)
xtest_dummy1_open = scaler.transform(xtest_dummy1_open)

xtrain_dummy1_open = xtrain_dummy1_open.reshape(xtrain_dummy1_open.shape[0], 1, xtrain_dummy1_open.shape[1])
xtest_dummy1_open = xtest_dummy1_open.reshape(xtest_dummy1_open.shape[0], 1, xtest_dummy1_open.shape[1])


# In[13]:


model1 = Sequential()

model1.add(LSTM(units=50, input_shape=(xtrain_dummy1_open.shape[1], xtrain_dummy1_open.shape[2]), return_sequences=True))
model1.add(Dropout(0.2)) 
model1.add(LSTM(units=50, return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(units=50, return_sequences=True))
model1.add(Dropout(0.2))
model1.add(LSTM(units=50))
model1.add(Dropout(0.2))
model1.add(Dense(units=1))
model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

model1.summary()

model1.compile(optimizer=Adam(learning_rate=(0.001)), loss='mean_squared_error')




model1.fit(xtrain_dummy1_open, ytrain_dummy1_open, epochs=100, validation_split=0.2, batch_size=32)




from keras.layers import RNN, Conv1D




def prepare_sequences(df1, n_steps):
    xdf1, ydf1 = [], []
    for i in range(len(df1) - n_steps):
        xdf1.append(df1[i:i+n_steps])
        ydf1.append(df1[i+n_steps])
    return np.array(xdf1), np.array(ydf1)

n_steps=10


# In[18]:


x_train_seq, y_train_seq = prepare_sequences(ytrain_dummy1_open.values, n_steps)
x_test_seq, y_test_seq = prepare_sequences(ytest_dummy1_open.values, n_steps)


# In[19]:


model2 = Sequential()

model2.add(LSTM(units=64, input_shape=(x_train_seq.shape[1], 1), return_sequences=True))
model2.add(LSTM(units=32, return_sequences=True))
model2.add(Dense(units=1))


# In[20]:


model2.compile(optimizer='adam', loss='mean_squared_error')


# In[21]:


hist2 = model2.fit(x_train_seq, y_train_seq, epochs=50, batch_size=32, validation_split=0.2)


# In[22]:


df = df1


# In[23]:


import pandas as pd
import numpy as np
from pmdarima import auto_arima
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


time_series_column = 'Close'
y = df[time_series_column].values

scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

model = auto_arima(y_scaled, seasonal=True, stepwise=True, suppress_warnings=True,
                   error_action="ignore", max_order=None, trace=True)

best_order = model.get_params()['order']

print(f"Best ARIMA Order: {best_order}")

final_model = sm.tsa.ARIMA(y_scaled, order=best_order)
final_model_fit = final_model.fit()

forecast_periods = 10

forecast = final_model_fit.forecast(steps=forecast_periods)
forecast = forecast.reshape(-1, 1)
forecast = scaler.inverse_transform(forecast)

print("Forecasts:", forecast)


# In[24]:


df1.tail()


# In[25]:



tail_50_data = df.tail(50)

actual_open_prices = tail_50_data['Open'].values

forecast_periods = 50
forecast = final_model_fit.forecast(steps=forecast_periods)
forecast = forecast.reshape(-1, 1)
forecast = scaler.inverse_transform(forecast)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(actual_open_prices, forecast)
mse = mean_squared_error(actual_open_prices, forecast)
rmse = np.sqrt(mse)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(actual_open_prices, forecast)

print("Forecasts:", forecast)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")



# In[26]:


nifty_data


# In[28]:


df2 = nifty_data


# In[29]:


df2.isna().sum()


# In[30]:


df2['Trades'] = df2['Trades'].fillna(nifty_data['Trades'].mean())
df2['Deliverable Volume'] = nifty_data['Trades'].fillna(nifty_data['Deliverable Volume'].mean())
df2['%Deliverble'] = nifty_data['%Deliverble'].fillna(nifty_data['%Deliverble'].mean())
df2


# In[31]:


time_series_column = 'Close'
y = df2[time_series_column].values

scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

model = auto_arima(y_scaled, seasonal=True, stepwise=True, suppress_warnings=True,
                   error_action="ignore", max_order=None, trace=True)

best_order = model.get_params()['order']

print(f"Best ARIMA Order: {best_order}")

final_model = sm.tsa.ARIMA(y_scaled, order=best_order)
final_model_fit = final_model.fit()

forecast_periods = 10

forecast = final_model_fit.forecast(steps=forecast_periods)
forecast = forecast.reshape(-1, 1)
forecast = scaler.inverse_transform(forecast)

print("Forecasts:", forecast)

tail_50_data = df2.tail(50)

actual_open_prices = tail_50_data['Open'].values

unique_companies = df2['Symbol'].unique()

company_datasets = {}

for company in unique_companies:
    company_df = df2[df2['Symbol'] == company]
    
    company_datasets[company] = company_df


df2['Symbol'].unique()


mundraport = company_datasets['MUNDRAPORT']
adaniports = company_datasets['ADANIPORTS']
asianpaint = company_datasets['ASIANPAINT']
utibank = company_datasets['UTIBANK']
axisbank = company_datasets['AXISBANK']
bajajauto = company_datasets['BAJAJ-AUTO']
bajajfinsv = company_datasets['BAJAJFINSV']
bajautofin = company_datasets['BAJAUTOFIN']
bajfinance = company_datasets['BAJFINANCE']
bharti = company_datasets['BHARTI']
bhartiartl = company_datasets['BHARTIARTL']
bpcl = company_datasets['BPCL']
britannia = company_datasets['BRITANNIA']
cipla = company_datasets['CIPLA']
coalindia = company_datasets['COALINDIA']
drreddy = company_datasets['DRREDDY']
eichermot = company_datasets['EICHERMOT']
gail = company_datasets['GAIL']
grasim = company_datasets['GRASIM']
hcltech = company_datasets['HCLTECH']
hdfc = company_datasets['HDFC']
hdfcbank = company_datasets['HDFCBANK']
herohonda = company_datasets['HEROHONDA']
heromotoco = company_datasets['HEROMOTOCO']
hindalco = company_datasets['HINDALCO']
hindelever = company_datasets['HINDLEVER']
hindunilvr = company_datasets['HINDUNILVR']
icicibank = company_datasets['ICICIBANK']
indusindbk = company_datasets['INDUSINDBK']
infosystch = company_datasets['INFOSYSTCH']
infy = company_datasets['INFY']
ioc = company_datasets['IOC']
itc = company_datasets['ITC']
jswstl = company_datasets['JSWSTL']
kotakmah = company_datasets['KOTAKMAH']
kotakbank = company_datasets['KOTAKBANK']
lt = company_datasets['LT']
mandm = company_datasets['M&M']
maruti = company_datasets['MARUTI']
nestleind = company_datasets['NESTLEIND']
ntpc = company_datasets['NTPC']
ongc = company_datasets['ONGC']
powergrid = company_datasets['POWERGRID']
reliance = company_datasets['RELIANCE']
sbin = company_datasets['SBIN']
shreecem = company_datasets['SHREECEM']
sunpharma = company_datasets['SUNPHARMA']
telco = company_datasets['TELCO']
tatamotors = company_datasets['TATAMOTORS']
tisco = company_datasets['TISCO']
tatasteel = company_datasets['TATASTEEL']
tcs = company_datasets['TCS']
techm = company_datasets['TECHM']
titan = company_datasets['TITAN']
ultracemco = company_datasets['ULTRACEMCO']
uniphos = company_datasets['UNIPHOS']
upl = company_datasets['UPL']
sesagoa = company_datasets['SESAGOA']
sslt = company_datasets['SSLT']
vedl = company_datasets['VEDL']
wipro = company_datasets['WIPRO']
zeetele = company_datasets['ZEETELE']
zeel = company_datasets['ZEEL']


scaler = StandardScaler()

forecast_period = 10
forecast_periods = 50

adaniports_y_close = adaniports['Close'].values
adaniports_y_open = adaniports['Open'].values
adaniports_y_high = adaniports['High'].values
adaniports_y_low = adaniports['Low'].values

adaniports_y_close_scaled = scaler.fit_transform(adaniports_y_close.reshape(-1, 1))
adaniports_y_open_scaled = scaler.fit_transform(adaniports_y_open.reshape(-1, 1))
adaniports_y_high_scaled = scaler.fit_transform(adaniports_y_high.reshape(-1, 1))
adaniports_y_low_scaled = scaler.fit_transform(adaniports_y_low.reshape(-1, 1))

adaniports_close_model = auto_arima(
    adaniports_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

adaniports_open_model = auto_arima(
    adaniports_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

adaniports_high_model = auto_arima(
    adaniports_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

adaniports_low_model = auto_arima(
    adaniports_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

adaniports_close_best_order = adaniports_close_model.get_params()['order']
adaniports_open_best_order = adaniports_open_model.get_params()['order']
adaniports_high_best_order = adaniports_high_model.get_params()['order']
adaniports_low_best_order = adaniports_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {adaniports_close_best_order}")
print(f"Best ARIMA Order for Open: {adaniports_open_best_order}")
print(f"Best ARIMA Order for High: {adaniports_high_best_order}")
print(f"Best ARIMA Order for Low: {adaniports_low_best_order}")

adaniports_close_final_model = sm.tsa.ARIMA(
    adaniports_y_close_scaled,
    order=adaniports_close_best_order
)
adaniports_close_final_model = adaniports_close_final_model.fit()
adaniports_close_forecast = adaniports_close_final_model.forecast(steps=forecast_period)
adaniports_close_forecast = adaniports_close_forecast.reshape(-1, 1)
adaniports_close_forecast = scaler.inverse_transform(adaniports_close_forecast)

adaniports_open_final_model = sm.tsa.ARIMA(
    adaniports_y_open_scaled,
    order=adaniports_open_best_order
)
adaniports_open_final_model = adaniports_open_final_model.fit()
adaniports_open_forecast = adaniports_open_final_model.forecast(steps=forecast_period)
adaniports_open_forecast = adaniports_open_forecast.reshape(-1, 1)
adaniports_open_forecast = scaler.inverse_transform(adaniports_open_forecast)

adaniports_high_final_model = sm.tsa.ARIMA(
    adaniports_y_high_scaled,
    order=adaniports_high_best_order
)
adaniports_high_final_model = adaniports_high_final_model.fit()
adaniports_high_forecast = adaniports_high_final_model.forecast(steps=forecast_period)
adaniports_high_forecast = adaniports_high_forecast.reshape(-1, 1)
adaniports_high_forecast = scaler.inverse_transform(adaniports_high_forecast)

adaniports_low_final_model = sm.tsa.ARIMA(
    adaniports_y_low_scaled,
    order=adaniports_low_best_order
)
adaniports_low_final_model = adaniports_low_final_model.fit()
adaniports_low_forecast = adaniports_low_final_model.forecast(steps=forecast_period)
adaniports_low_forecast = adaniports_low_forecast.reshape(-1, 1)
adaniports_low_forecast = scaler.inverse_transform(adaniports_low_forecast)

print("Close Forecasts:", adaniports_close_forecast)
print("Open Forecasts:", adaniports_open_forecast)
print("High Forecasts:", adaniports_high_forecast)
print("Low Forecasts:", adaniports_low_forecast)


# In[140]:


adaniports_tail_50_data = adaniports.tail(forecast_periods)

adaniports_actual_close_prices = adaniports_tail_50_data['Close'].values
adaniports_actual_open_prices = adaniports_tail_50_data['Open'].values
adaniports_actual_high_prices = adaniports_tail_50_data['High'].values
adaniports_actual_low_prices = adaniports_tail_50_data['Low'].values

adaniports_forecast_close = adaniports_close_final_model.forecast(steps=forecast_periods)
adaniports_forecast_close = adaniports_forecast_close.reshape(-1, 1)
adaniports_forecast_close = scaler.inverse_transform(adaniports_forecast_close)

adaniports_forecast_open = adaniports_open_final_model.forecast(steps=forecast_periods)
adaniports_forecast_open = adaniports_forecast_open.reshape(-1, 1)
adaniports_forecast_open = scaler.inverse_transform(adaniports_forecast_open)

adaniports_forecast_high = adaniports_high_final_model.forecast(steps=forecast_periods)
adaniports_forecast_high = adaniports_forecast_high.reshape(-1, 1)
adaniports_forecast_high = scaler.inverse_transform(adaniports_forecast_high)

adaniports_forecast_low = adaniports_low_final_model.forecast(steps=forecast_periods)
adaniports_forecast_low = adaniports_forecast_low.reshape(-1, 1)
adaniports_forecast_low = scaler.inverse_transform(adaniports_forecast_low)

adaniports_close_mae = mean_absolute_error(adaniports_actual_close_prices, adaniports_forecast_close)
adaniports_close_mse = mean_squared_error(adaniports_actual_close_prices, adaniports_forecast_close)
adaniports_close_rmse = np.sqrt(adaniports_close_mse)

adaniports_open_mae = mean_absolute_error(adaniports_actual_open_prices, adaniports_forecast_open)
adaniports_open_mse = mean_squared_error(adaniports_actual_open_prices, adaniports_forecast_open)
adaniports_open_rmse = np.sqrt(adaniports_open_mse)

adaniports_high_mae = mean_absolute_error(adaniports_actual_high_prices, adaniports_forecast_high)
adaniports_high_mse = mean_squared_error(adaniports_actual_high_prices, adaniports_forecast_high)
adaniports_high_rmse = np.sqrt(adaniports_high_mse)

adaniports_low_mae = mean_absolute_error(adaniports_actual_low_prices, adaniports_forecast_low)
adaniports_low_mse = mean_squared_error(adaniports_actual_low_prices, adaniports_forecast_low)
adaniports_low_rmse = np.sqrt(adaniports_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

adaniports_close_mape = mean_absolute_percentage_error(adaniports_actual_close_prices, adaniports_forecast_close)
adaniports_open_mape = mean_absolute_percentage_error(adaniports_actual_open_prices, adaniports_forecast_open)
adaniports_high_mape = mean_absolute_percentage_error(adaniports_actual_high_prices, adaniports_forecast_high)
adaniports_low_mape = mean_absolute_percentage_error(adaniports_actual_low_prices, adaniports_forecast_low)


print("Close Forecasts:", adaniports_forecast_close)
print(f"Close Mean Absolute Error (MAE): {adaniports_close_mae}")
print(f"Close Mean Squared Error (MSE): {adaniports_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {adaniports_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {adaniports_close_mape}%")

print("Open Forecasts:", adaniports_forecast_open)
print(f"Open Mean Absolute Error (MAE): {adaniports_open_mae}")
print(f"Open Mean Squared Error (MSE): {adaniports_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {adaniports_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {adaniports_open_mape}%")

print("High Forecasts:", adaniports_forecast_high)
print(f"High Mean Absolute Error (MAE): {adaniports_high_mae}")
print(f"High Mean Squared Error (MSE): {adaniports_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {adaniports_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {adaniports_high_mape}%")

print("Low Forecasts:", adaniports_forecast_low)
print(f"Low Mean Absolute Error (MAE): {adaniports_low_mae}")
print(f"Low Mean Squared Error (MSE): {adaniports_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {adaniports_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {adaniports_low_mape}%")


# In[141]:


asianpaint_y_close = asianpaint['Close'].values
asianpaint_y_open = asianpaint['Open'].values
asianpaint_y_high = asianpaint['High'].values
asianpaint_y_low = asianpaint['Low'].values

asianpaint_y_close_scaled = scaler.fit_transform(asianpaint_y_close.reshape(-1, 1))
asianpaint_y_open_scaled = scaler.fit_transform(asianpaint_y_open.reshape(-1, 1))
asianpaint_y_high_scaled = scaler.fit_transform(asianpaint_y_high.reshape(-1, 1))
asianpaint_y_low_scaled = scaler.fit_transform(asianpaint_y_low.reshape(-1, 1))

asianapaint_close_model = auto_arima(
    asianpaint_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

asianpaint_open_model = auto_arima(
    asianpaint_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

asianpaint_high_model = auto_arima(
    asianpaint_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

asianpaint_low_model = auto_arima(
    asianpaint_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

asianpaint_close_best_order = asianpaint_close_model.get_params()['order']
asianpaint_open_best_order = asianpaint_open_model.get_params()['order']
asianpaint_high_best_order = asianpaint_high_model.get_params()['order']
asianpaint_low_best_order = asianpaint_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {asianpaint_close_best_order}")
print(f"Best ARIMA Order for Open: {asianpaint_open_best_order}")
print(f"Best ARIMA Order for High: {asianpaint_high_best_order}")
print(f"Best ARIMA Order for Low: {asianpaint_low_best_order}")

asianpaint_close_final_model = sm.tsa.ARIMA(
    asianpaint_y_close_scaled,
    order=asianpaint_close_best_order
)
asianpaint_close_final_model = asianpaint_close_final_model.fit()
asianpaint_close_forecast = asianpaint_close_final_model.forecast(steps=forecast_period)
asianpaint_close_forecast = asianpaint_close_forecast.reshape(-1, 1)
asianpaint_close_forecast = scaler.inverse_transform(asianpaint_close_forecast)

asianpaint_open_final_model = sm.tsa.ARIMA(
    asianpaint_y_open_scaled,
    order=asianpaint_open_best_order
)
asianpaint_open_final_model = asianpaint_open_final_model.fit()
asianpaint_open_forecast = asianpaint_open_final_model.forecast(steps=forecast_period)
asianpaint_open_forecast = asianpaint_open_forecast.reshape(-1, 1)
asianpaint_open_forecast = scaler.inverse_transform(asianpaint_open_forecast)

asianpaint_high_final_model = sm.tsa.ARIMA(
    asianpaint_y_high_scaled,
    order=asianpaint_high_best_order
)
asianpaint_high_final_model = asianpaint_high_final_model.fit()
asianpaint_high_forecast = asianpaint_high_final_model.forecast(steps=forecast_period)
asianpaint_high_forecast = asianpaint_high_forecast.reshape(-1, 1)
asianpaint_high_forecast = scaler.inverse_transform(asianpaint_high_forecast)

asianpaint_low_final_model = sm.tsa.ARIMA(
    asianpaint_y_low_scaled,
    order=asianpaint_low_best_order
)
asianpaint_low_final_model = asianpaint_low_final_model.fit()
asianpaint_low_forecast = asianpaint_low_final_model.forecast(steps=forecast_period)
asianpaint_low_forecast = asianpaint_low_forecast.reshape(-1, 1)
asianpaint_low_forecast = scaler.inverse_transform(asianpaint_low_forecast)

print("Close Forecasts:", asianpaint_close_forecast)
print("Open Forecasts:", asianpaint_open_forecast)
print("High Forecasts:", asianpaint_high_forecast)
print("Low Forecasts:", asianpaint_low_forecast)


# In[142]:


asianpaint_tail_50_data = asianpaint.tail(forecast_periods)

asianpaint_actual_close_prices = asianpaint_tail_50_data['Close'].values
asianpaint_actual_open_prices = asianpaint_tail_50_data['Open'].values
asianpaint_actual_high_prices = asianpaint_tail_50_data['High'].values
asianpaint_actual_low_prices = asianpaint_tail_50_data['Low'].values

asianpaint_forecast_close = asianpaint_close_final_model.forecast(steps=forecast_periods)
asianpaint_forecast_close = asianpaint_forecast_close.reshape(-1, 1)
asianpaint_forecast_close = scaler.inverse_transform(asianpaint_forecast_close)

asianpaint_forecast_open = asianpaint_open_final_model.forecast(steps=forecast_periods)
asianpaint_forecast_open = asianpaint_forecast_open.reshape(-1, 1)
asianpaint_forecast_open = scaler.inverse_transform(asianpaint_forecast_open)

asianpaint_forecast_high = asianpaint_high_final_model.forecast(steps=forecast_periods)
asianpaint_forecast_high = asianpaint_forecast_high.reshape(-1, 1)
asianpaint_forecast_high = scaler.inverse_transform(asianpaint_forecast_high)

asianpaint_forecast_low = asianpaint_low_final_model.forecast(steps=forecast_periods)
asianpaint_forecast_low = asianpaint_forecast_low.reshape(-1, 1)
asianpaint_forecast_low = scaler.inverse_transform(asianpaint_forecast_low)

asianpaint_close_mae = mean_absolute_error(asianpaint_actual_close_prices, asianpaint_forecast_close)
asianpaint_close_mse = mean_squared_error(asianpaint_actual_close_prices, asianpaint_forecast_close)
asianpaint_close_rmse = np.sqrt(asianpaint_close_mse)

asianpaint_open_mae = mean_absolute_error(asianpaint_actual_open_prices, asianpaint_forecast_open)
asianpaint_open_mse = mean_squared_error(asianpaint_actual_open_prices, asianpaint_forecast_open)
asianpaint_open_rmse = np.sqrt(asianpaint_open_mse)

asianpaint_high_mae = mean_absolute_error(asianpaint_actual_high_prices, asianpaint_forecast_high)
asianpaint_high_mse = mean_squared_error(asianpaint_actual_high_prices, asianpaint_forecast_high)
asianpaint_high_rmse = np.sqrt(asianpaint_high_mse)

asianpaint_low_mae = mean_absolute_error(asianpaint_actual_low_prices, asianpaint_forecast_low)
asianpaint_low_mse = mean_squared_error(asianpaint_actual_low_prices, asianpaint_forecast_low)
asianpaint_low_rmse = np.sqrt(asianpaint_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

asianpaint_close_mape = mean_absolute_percentage_error(asianpaint_actual_close_prices, asianpaint_forecast_close)
asianpaint_open_mape = mean_absolute_percentage_error(asianpaint_actual_open_prices, asianpaint_forecast_open)
asianpaint_high_mape = mean_absolute_percentage_error(asianpaint_actual_high_prices, asianpaint_forecast_high)
asianpaint_low_mape = mean_absolute_percentage_error(asianpaint_actual_low_prices, asianpaint_forecast_low)


print("Close Forecasts:", asianpaint_forecast_close)
print(f"Close Mean Absolute Error (MAE): {asianpaint_close_mae}")
print(f"Close Mean Squared Error (MSE): {asianpaint_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {asianpaint_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {asianpaint_close_mape}%")

print("Open Forecasts:", asianpaint_forecast_open)
print(f"Open Mean Absolute Error (MAE): {asianpaint_open_mae}")
print(f"Open Mean Squared Error (MSE): {asianpaint_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {asianpaint_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {asianpaint_open_mape}%")

print("High Forecasts:", asianpaint_forecast_high)
print(f"High Mean Absolute Error (MAE): {asianpaint_high_mae}")
print(f"High Mean Squared Error (MSE): {asianpaint_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {asianpaint_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {asianpaint_high_mape}%")

print("Low Forecasts:", asianpaint_forecast_low)
print(f"Low Mean Absolute Error (MAE): {asianpaint_low_mae}")
print(f"Low Mean Squared Error (MSE): {asianpaint_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {asianpaint_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {asianpaint_low_mape}%")


# In[143]:


utibank_y_close = utibank['Close'].values
utibank_y_open = utibank['Open'].values
utibank_y_high = utibank['High'].values
utibank_y_low = utibank['Low'].values

utibank_y_close_scaled = scaler.fit_transform(utibank_y_close.reshape(-1, 1))
utibank_y_open_scaled = scaler.fit_transform(utibank_y_open.reshape(-1, 1))
utibank_y_high_scaled = scaler.fit_transform(utibank_y_high.reshape(-1, 1))
utibank_y_low_scaled = scaler.fit_transform(utibank_y_low.reshape(-1, 1))

utibank_close_model = auto_arima(
    utibank_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

utibank_open_model = auto_arima(
    utibank_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

utibank_high_model = auto_arima(
    utibank_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

utibank_low_model = auto_arima(
    utibank_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

utibank_close_best_order = utibank_close_model.get_params()['order']
utibank_open_best_order = utibank_open_model.get_params()['order']
utibank_high_best_order = utibank_high_model.get_params()['order']
utibank_low_best_order = utibank_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {utibank_close_best_order}")
print(f"Best ARIMA Order for Open: {utibank_open_best_order}")
print(f"Best ARIMA Order for High: {utibank_high_best_order}")
print(f"Best ARIMA Order for Low: {utibank_low_best_order}")

utibank_close_final_model = sm.tsa.ARIMA(
    utibank_y_close_scaled,
    order=utibank_close_best_order
)
utibank_close_final_model = utibank_close_final_model.fit()
utibank_close_forecast = utibank_close_final_model.forecast(steps=forecast_period)
utibank_close_forecast = utibank_close_forecast.reshape(-1, 1)
utibank_close_forecast = scaler.inverse_transform(utibank_close_forecast)

utibank_open_final_model = sm.tsa.ARIMA(
    utibank_y_open_scaled,
    order=utibank_open_best_order
)
utibank_open_final_model = utibank_open_final_model.fit()
utibank_open_forecast = utibank_open_final_model.forecast(steps=forecast_period)
utibank_open_forecast = utibank_open_forecast.reshape(-1, 1)
utibank_open_forecast = scaler.inverse_transform(utibank_open_forecast)

utibank_high_final_model = sm.tsa.ARIMA(
    utibank_y_high_scaled,
    order=utibank_high_best_order
)
utibank_high_final_model = utibank_high_final_model.fit()
utibank_high_forecast = utibank_high_final_model.forecast(steps=forecast_period)
utibank_high_forecast = utibank_high_forecast.reshape(-1, 1)
utibank_high_forecast = scaler.inverse_transform(utibank_high_forecast)

utibank_low_final_model = sm.tsa.ARIMA(
    utibank_y_low_scaled,
    order=utibank_low_best_order
)
utibank_low_final_model = utibank_low_final_model.fit()
utibank_low_forecast = utibank_low_final_model.forecast(steps=forecast_period)
utibank_low_forecast = utibank_low_forecast.reshape(-1, 1)
utibank_low_forecast = scaler.inverse_transform(utibank_low_forecast)

print("Close Forecasts:", utibank_close_forecast)
print("Open Forecasts:", utibank_open_forecast)
print("High Forecasts:", utibank_high_forecast)
print("Low Forecasts:", utibank_low_forecast)


# In[144]:


utibank_tail_50_data = utibank.tail(forecast_periods)

utibank_actual_close_prices = utibank_tail_50_data['Close'].values
utibank_actual_open_prices = utibank_tail_50_data['Open'].values
utibank_actual_high_prices = utibank_tail_50_data['High'].values
utibank_actual_low_prices = utibank_tail_50_data['Low'].values

utibank_forecast_close = utibank_close_final_model.forecast(steps=forecast_periods)
utibank_forecast_close = utibank_forecast_close.reshape(-1, 1)
utibank_forecast_close = scaler.inverse_transform(utibank_forecast_close)

utibank_forecast_open = utibank_open_final_model.forecast(steps=forecast_periods)
utibank_forecast_open = utibank_forecast_open.reshape(-1, 1)
utibank_forecast_open = scaler.inverse_transform(utibank_forecast_open)

utibank_forecast_high = utibank_high_final_model.forecast(steps=forecast_periods)
utibank_forecast_high = utibank_forecast_high.reshape(-1, 1)
utibank_forecast_high = scaler.inverse_transform(utibank_forecast_high)

utibank_forecast_low = utibank_low_final_model.forecast(steps=forecast_periods)
utibank_forecast_low = utibank_forecast_low.reshape(-1, 1)
utibank_forecast_low = scaler.inverse_transform(utibank_forecast_low)

utibank_close_mae = mean_absolute_error(utibank_actual_close_prices, utibank_forecast_close)
utibank_close_mse = mean_squared_error(utibank_actual_close_prices, utibank_forecast_close)
utibank_close_rmse = np.sqrt(utibank_close_mse)

utibank_open_mae = mean_absolute_error(utibank_actual_open_prices, utibank_forecast_open)
utibank_open_mse = mean_squared_error(utibank_actual_open_prices, utibank_forecast_open)
utibank_open_rmse = np.sqrt(utibank_open_mse)

utibank_high_mae = mean_absolute_error(utibank_actual_high_prices, utibank_forecast_high)
utibank_high_mse = mean_squared_error(utibank_actual_high_prices, utibank_forecast_high)
utibank_high_rmse = np.sqrt(utibank_high_mse)

utibank_low_mae = mean_absolute_error(utibank_actual_low_prices, utibank_forecast_low)
utibank_low_mse = mean_squared_error(utibank_actual_low_prices, utibank_forecast_low)
utibank_low_rmse = np.sqrt(utibank_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

utibank_close_mape = mean_absolute_percentage_error(utibank_actual_close_prices, utibank_forecast_close)
utibank_open_mape = mean_absolute_percentage_error(utibank_actual_open_prices, utibank_forecast_open)
utibank_high_mape = mean_absolute_percentage_error(utibank_actual_high_prices, utibank_forecast_high)
utibank_low_mape = mean_absolute_percentage_error(utibank_actual_low_prices, utibank_forecast_low)


print("Close Forecasts:", utibank_forecast_close)
print(f"Close Mean Absolute Error (MAE): {utibank_close_mae}")
print(f"Close Mean Squared Error (MSE): {utibank_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {utibank_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {utibank_close_mape}%")

print("Open Forecasts:", utibank_forecast_open)
print(f"Open Mean Absolute Error (MAE): {utibank_open_mae}")
print(f"Open Mean Squared Error (MSE): {utibank_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {utibank_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {utibank_open_mape}%")

print("High Forecasts:", utibank_forecast_high)
print(f"High Mean Absolute Error (MAE): {utibank_high_mae}")
print(f"High Mean Squared Error (MSE): {utibank_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {utibank_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {utibank_high_mape}%")

print("Low Forecasts:", utibank_forecast_low)
print(f"Low Mean Absolute Error (MAE): {utibank_low_mae}")
print(f"Low Mean Squared Error (MSE): {utibank_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {utibank_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {utibank_low_mape}%")


# In[145]:


axisbank_y_close = axisbank['Close'].values
axisbank_y_open = axisbank['Open'].values
axisbank_y_high = axisbank['High'].values
axisbank_y_low = axisbank['Low'].values

axisbank_y_close_scaled = scaler.fit_transform(axisbank_y_close.reshape(-1, 1))
axisbank_y_open_scaled = scaler.fit_transform(axisbank_y_open.reshape(-1, 1))
axisbank_y_high_scaled = scaler.fit_transform(axisbank_y_high.reshape(-1, 1))
axisbank_y_low_scaled = scaler.fit_transform(axisbank_y_low.reshape(-1, 1))

axisbank_close_model = auto_arima(
    axisbank_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

axisbank_open_model = auto_arima(
    axisbank_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

axisbank_high_model = auto_arima(
    axisbank_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

axisbank_low_model = auto_arima(
    axisbank_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

axisbank_close_best_order = axisbank_close_model.get_params()['order']
axisbank_open_best_order = axisbank_open_model.get_params()['order']
axisbank_high_best_order = axisbank_high_model.get_params()['order']
axisbank_low_best_order = axisbank_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {axisbank_close_best_order}")
print(f"Best ARIMA Order for Open: {axisbank_open_best_order}")
print(f"Best ARIMA Order for High: {axisbank_high_best_order}")
print(f"Best ARIMA Order for Low: {axisbank_low_best_order}")

axisbank_close_final_model = sm.tsa.ARIMA(
    axisbank_y_close_scaled,
    order=axisbank_close_best_order
)
axisbank_close_final_model = axisbank_close_final_model.fit()
axisbank_close_forecast = axisbank_close_final_model.forecast(steps=forecast_period)
axisbank_close_forecast = axisbank_close_forecast.reshape(-1, 1)
axisbank_close_forecast = scaler.inverse_transform(axisbank_close_forecast)

axisbank_open_final_model = sm.tsa.ARIMA(
    axisbank_y_open_scaled,
    order=axisbank_open_best_order
)
axisbank_open_final_model = axisbank_open_final_model.fit()
axisbank_open_forecast = axisbank_open_final_model.forecast(steps=forecast_period)
axisbank_open_forecast = axisbank_open_forecast.reshape(-1, 1)
axisbank_open_forecast = scaler.inverse_transform(axisbank_open_forecast)

axisbank_high_final_model = sm.tsa.ARIMA(
    axisbank_y_high_scaled,
    order=axisbank_high_best_order
)
axisbank_high_final_model = axisbank_high_final_model.fit()
axisbank_high_forecast = axisbank_high_final_model.forecast(steps=forecast_period)
axisbank_high_forecast = axisbank_high_forecast.reshape(-1, 1)
axisbank_high_forecast = scaler.inverse_transform(axisbank_high_forecast)

axisbank_low_final_model = sm.tsa.ARIMA(
    axisbank_y_low_scaled,
    order=axisbank_low_best_order
)
axisbank_low_final_model = axisbank_low_final_model.fit()
axisbank_low_forecast = axisbank_low_final_model.forecast(steps=forecast_period)
axisbank_low_forecast = axisbank_low_forecast.reshape(-1, 1)
axisbank_low_forecast = scaler.inverse_transform(axisbank_low_forecast)

print("Close Forecasts:", axisbank_close_forecast)
print("Open Forecasts:", axisbank_open_forecast)
print("High Forecasts:", axisbank_high_forecast)
print("Low Forecasts:", axisbank_low_forecast)


# In[146]:


axisbank_tail_50_data = axisbank.tail(forecast_periods)

axisbank_actual_close_prices = axisbank_tail_50_data['Close'].values
axisbank_actual_open_prices = axisbank_tail_50_data['Open'].values
axisbank_actual_high_prices = axisbank_tail_50_data['High'].values
axisbank_actual_low_prices = axisbank_tail_50_data['Low'].values

axisbank_forecast_close = axisbank_close_final_model.forecast(steps=forecast_periods)
axisbank_forecast_close = axisbank_forecast_close.reshape(-1, 1)
axisbank_forecast_close = scaler.inverse_transform(axisbank_forecast_close)

axisbank_forecast_open = axisbank_open_final_model.forecast(steps=forecast_periods)
axisbank_forecast_open = axisbank_forecast_open.reshape(-1, 1)
axisbank_forecast_open = scaler.inverse_transform(axisbank_forecast_open)

axisbank_forecast_high = axisbank_high_final_model.forecast(steps=forecast_periods)
axisbank_forecast_high = axisbank_forecast_high.reshape(-1, 1)
axisbank_forecast_high = scaler.inverse_transform(axisbank_forecast_high)

axisbank_forecast_low = axisbank_low_final_model.forecast(steps=forecast_periods)
axisbank_forecast_low = axisbank_forecast_low.reshape(-1, 1)
axisbank_forecast_low = scaler.inverse_transform(axisbank_forecast_low)

axisbank_close_mae = mean_absolute_error(axisbank_actual_close_prices, axisbank_forecast_close)
axisbank_close_mse = mean_squared_error(axisbank_actual_close_prices, axisbank_forecast_close)
axisbank_close_rmse = np.sqrt(axisbank_close_mse)

axisbank_open_mae = mean_absolute_error(axisbank_actual_open_prices, axisbank_forecast_open)
axisbank_open_mse = mean_squared_error(axisbank_actual_open_prices, axisbank_forecast_open)
axisbank_open_rmse = np.sqrt(axisbank_open_mse)

axisbank_high_mae = mean_absolute_error(axisbank_actual_high_prices, axisbank_forecast_high)
axisbank_high_mse = mean_squared_error(axisbank_actual_high_prices, axisbank_forecast_high)
axisbank_high_rmse = np.sqrt(axisbank_high_mse)

axisbank_low_mae = mean_absolute_error(axisbank_actual_low_prices, axisbank_forecast_low)
axisbank_low_mse = mean_squared_error(axisbank_actual_low_prices, axisbank_forecast_low)
axisbank_low_rmse = np.sqrt(axisbank_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

axisbank_close_mape = mean_absolute_percentage_error(axisbank_actual_close_prices, axisbank_forecast_close)
axisbank_open_mape = mean_absolute_percentage_error(axisbank_actual_open_prices, axisbank_forecast_open)
axisbank_high_mape = mean_absolute_percentage_error(axisbank_actual_high_prices, axisbank_forecast_high)
axisbank_low_mape = mean_absolute_percentage_error(axisbank_actual_low_prices, axisbank_forecast_low)


print("Close Forecasts:", axisbank_forecast_close)
print(f"Close Mean Absolute Error (MAE): {axisbank_close_mae}")
print(f"Close Mean Squared Error (MSE): {axisbank_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {axisbank_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {axisbank_close_mape}%")

print("Open Forecasts:", axisbank_forecast_open)
print(f"Open Mean Absolute Error (MAE): {axisbank_open_mae}")
print(f"Open Mean Squared Error (MSE): {axisbank_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {axisbank_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {axisbank_open_mape}%")

print("High Forecasts:", axisbank_forecast_high)
print(f"High Mean Absolute Error (MAE): {axisbank_high_mae}")
print(f"High Mean Squared Error (MSE): {axisbank_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {axisbank_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {axisbank_high_mape}%")

print("Low Forecasts:", axisbank_forecast_low)
print(f"Low Mean Absolute Error (MAE): {axisbank_low_mae}")
print(f"Low Mean Squared Error (MSE): {axisbank_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {axisbank_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {axisbank_low_mape}%")


# In[147]:


bajajauto_y_close = bajajauto['Close'].values
bajajauto_y_open = bajajauto['Open'].values
bajajauto_y_high = bajajauto['High'].values
bajajauto_y_low = bajajauto['Low'].values

bajajauto_y_close_scaled = scaler.fit_transform(bajajauto_y_close.reshape(-1, 1))
bajajauto_y_open_scaled = scaler.fit_transform(bajajauto_y_open.reshape(-1, 1))
bajajauto_y_high_scaled = scaler.fit_transform(bajajauto_y_high.reshape(-1, 1))
bajajauto_y_low_scaled = scaler.fit_transform(bajajauto_y_low.reshape(-1, 1))

bajajauto_close_model = auto_arima(
    bajajauto_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajajauto_open_model = auto_arima(
    bajajauto_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajajauto_high_model = auto_arima(
    bajajauto_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajajauto_low_model = auto_arima(
    bajajauto_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajajauto_close_best_order = bajajauto_close_model.get_params()['order']
bajajauto_open_best_order = bajajauto_open_model.get_params()['order']
bajajauto_high_best_order = bajajauto_high_model.get_params()['order']
bajajauto_low_best_order = bajajauto_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {bajajauto_close_best_order}")
print(f"Best ARIMA Order for Open: {bajajauto_open_best_order}")
print(f"Best ARIMA Order for High: {bajajauto_high_best_order}")
print(f"Best ARIMA Order for Low: {bajajauto_low_best_order}")

bajajauto_close_final_model = sm.tsa.ARIMA(
    bajajauto_y_close_scaled,
    order=bajajauto_close_best_order
)
bajajauto_close_final_model = bajajauto_close_final_model.fit()
bajajauto_close_forecast = bajajauto_close_final_model.forecast(steps=forecast_period)
bajajauto_close_forecast = bajajauto_close_forecast.reshape(-1, 1)
bajajauto_close_forecast = scaler.inverse_transform(bajajauto_close_forecast)

bajajauto_open_final_model = sm.tsa.ARIMA(
    bajajauto_y_open_scaled,
    order=bajajauto_open_best_order
)
bajajauto_open_final_model = bajajauto_open_final_model.fit()
bajajauto_open_forecast = bajajauto_open_final_model.forecast(steps=forecast_period)
bajajauto_open_forecast = bajajauto_open_forecast.reshape(-1, 1)
bajajauto_open_forecast = scaler.inverse_transform(bajajauto_open_forecast)

bajajauto_high_final_model = sm.tsa.ARIMA(
    bajajauto_y_high_scaled,
    order=bajajauto_high_best_order
)
bajajauto_high_final_model = bajajauto_high_final_model.fit()
bajajauto_high_forecast = bajajauto_high_final_model.forecast(steps=forecast_period)
bajajauto_high_forecast = bajajauto_high_forecast.reshape(-1, 1)
bajajauto_high_forecast = scaler.inverse_transform(bajajauto_high_forecast)

bajajauto_low_final_model = sm.tsa.ARIMA(
    bajajauto_y_low_scaled,
    order=bajajauto_low_best_order
)
bajajauto_low_final_model = bajajauto_low_final_model.fit()
bajajauto_low_forecast = bajajauto_low_final_model.forecast(steps=forecast_period)
bajajauto_low_forecast = bajajauto_low_forecast.reshape(-1, 1)
bajajauto_low_forecast = scaler.inverse_transform(bajajauto_low_forecast)

print("Close Forecasts:", bajajauto_close_forecast)
print("Open Forecasts:", bajajauto_open_forecast)
print("High Forecasts:", bajajauto_high_forecast)
print("Low Forecasts:", bajajauto_low_forecast)


# In[148]:


bajajauto_tail_50_data = bajajauto.tail(forecast_periods)

bajajauto_actual_close_prices = bajajauto_tail_50_data['Close'].values
bajajauto_actual_open_prices = bajajauto_tail_50_data['Open'].values
bajajauto_actual_high_prices = bajajauto_tail_50_data['High'].values
bajajauto_actual_low_prices = bajajauto_tail_50_data['Low'].values

bajajauto_forecast_close = bajajauto_close_final_model.forecast(steps=forecast_periods)
bajajauto_forecast_close = bajajauto_forecast_close.reshape(-1, 1)
bajajauto_forecast_close = scaler.inverse_transform(bajajauto_forecast_close)

bajajauto_forecast_open = bajajauto_open_final_model.forecast(steps=forecast_periods)
bajajauto_forecast_open = bajajauto_forecast_open.reshape(-1, 1)
bajajauto_forecast_open = scaler.inverse_transform(bajajauto_forecast_open)

bajajauto_forecast_high = bajajauto_high_final_model.forecast(steps=forecast_periods)
bajajauto_forecast_high = bajajauto_forecast_high.reshape(-1, 1)
bajajauto_forecast_high = scaler.inverse_transform(bajajauto_forecast_high)

bajajauto_forecast_low = bajajauto_low_final_model.forecast(steps=forecast_periods)
bajajauto_forecast_low = bajajauto_forecast_low.reshape(-1, 1)
bajajauto_forecast_low = scaler.inverse_transform(bajajauto_forecast_low)

bajajauto_close_mae = mean_absolute_error(bajajauto_actual_close_prices, bajajauto_forecast_close)
bajajauto_close_mse = mean_squared_error(bajajauto_actual_close_prices, bajajauto_forecast_close)
bajajauto_close_rmse = np.sqrt(bajajauto_close_mse)

bajajauto_open_mae = mean_absolute_error(bajajauto_actual_open_prices, bajajauto_forecast_open)
bajajauto_open_mse = mean_squared_error(bajajauto_actual_open_prices, bajajauto_forecast_open)
bajajauto_open_rmse = np.sqrt(bajajauto_open_mse)

bajajauto_high_mae = mean_absolute_error(bajajauto_actual_high_prices, bajajauto_forecast_high)
bajajauto_high_mse = mean_squared_error(bajajauto_actual_high_prices, bajajauto_forecast_high)
bajajauto_high_rmse = np.sqrt(bajajauto_high_mse)

bajajauto_low_mae = mean_absolute_error(bajajauto_actual_low_prices, bajajauto_forecast_low)
bajajauto_low_mse = mean_squared_error(bajajauto_actual_low_prices, bajajauto_forecast_low)
bajajauto_low_rmse = np.sqrt(bajajauto_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

bajajauto_close_mape = mean_absolute_percentage_error(bajajauto_actual_close_prices, bajajauto_forecast_close)
bajajauto_open_mape = mean_absolute_percentage_error(bajajauto_actual_open_prices, bajajauto_forecast_open)
bajajauto_high_mape = mean_absolute_percentage_error(bajajauto_actual_high_prices, bajajauto_forecast_high)
bajajauto_low_mape = mean_absolute_percentage_error(bajajauto_actual_low_prices, bajajauto_forecast_low)


print("Close Forecasts:", bajajauto_forecast_close)
print(f"Close Mean Absolute Error (MAE): {bajajauto_close_mae}")
print(f"Close Mean Squared Error (MSE): {bajajauto_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {bajajauto_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {bajajauto_close_mape}%")

print("Open Forecasts:", bajajauto_forecast_open)
print(f"Open Mean Absolute Error (MAE): {bajajauto_open_mae}")
print(f"Open Mean Squared Error (MSE): {bajajauto_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {bajajauto_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {bajajauto_open_mape}%")

print("High Forecasts:", bajajauto_forecast_high)
print(f"High Mean Absolute Error (MAE): {bajajauto_high_mae}")
print(f"High Mean Squared Error (MSE): {bajajauto_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {bajajauto_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {bajajauto_high_mape}%")

print("Low Forecasts:", bajajauto_forecast_low)
print(f"Low Mean Absolute Error (MAE): {bajajauto_low_mae}")
print(f"Low Mean Squared Error (MSE): {bajajauto_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {bajajauto_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {bajajauto_low_mape}%")


# In[149]:


bajajfinsv_y_close = bajajfinsv['Close'].values
bajajfinsv_y_open = bajajfinsv['Open'].values
bajajfinsv_y_high = bajajfinsv['High'].values
bajajfinsv_y_low = bajajfinsv['Low'].values

bajajfinsv_y_close_scaled = scaler.fit_transform(bajajfinsv_y_close.reshape(-1, 1))
bajajfinsv_y_open_scaled = scaler.fit_transform(bajajfinsv_y_open.reshape(-1, 1))
bajajfinsv_y_high_scaled = scaler.fit_transform(bajajfinsv_y_high.reshape(-1, 1))
bajajfinsv_y_low_scaled = scaler.fit_transform(bajajfinsv_y_low.reshape(-1, 1))

bajajfinsv_close_model = auto_arima(
    bajajfinsv_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajajfinsv_open_model = auto_arima(
    bajajfinsv_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajajfinsv_high_model = auto_arima(
    bajajfinsv_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajajfinsv_low_model = auto_arima(
    bajajfinsv_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajajfinsv_close_best_order = bajajfinsv_close_model.get_params()['order']
bajajfinsv_open_best_order = bajajfinsv_open_model.get_params()['order']
bajajfinsv_high_best_order = bajajfinsv_high_model.get_params()['order']
bajajfinsv_low_best_order = bajajfinsv_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {bajajfinsv_close_best_order}")
print(f"Best ARIMA Order for Open: {bajajfinsv_open_best_order}")
print(f"Best ARIMA Order for High: {bajajfinsv_high_best_order}")
print(f"Best ARIMA Order for Low: {bajajfinsv_low_best_order}")

bajajfinsv_close_final_model = sm.tsa.ARIMA(
    bajajfinsv_y_close_scaled,
    order=bajajfinsv_close_best_order
)
bajajfinsv_close_final_model = bajajfinsv_close_final_model.fit()
bajajfinsv_close_forecast = bajajfinsv_close_final_model.forecast(steps=forecast_period)
bajajfinsv_close_forecast = bajajfinsv_close_forecast.reshape(-1, 1)
bajajfinsv_close_forecast = scaler.inverse_transform(bajajfinsv_close_forecast)

bajajfinsv_open_final_model = sm.tsa.ARIMA(
    bajajfinsv_y_open_scaled,
    order=bajajfinsv_open_best_order
)
bajajfinsv_open_final_model = bajajfinsv_open_final_model.fit()
bajajfinsv_open_forecast = bajajfinsv_open_final_model.forecast(steps=forecast_period)
bajajfinsv_open_forecast = bajajfinsv_open_forecast.reshape(-1, 1)
bajajfinsv_open_forecast = scaler.inverse_transform(bajajfinsv_open_forecast)

bajajfinsv_high_final_model = sm.tsa.ARIMA(
    bajajfinsv_y_high_scaled,
    order=bajajfinsv_high_best_order
)
bajajfinsv_high_final_model = bajajfinsv_high_final_model.fit()
bajajfinsv_high_forecast = bajajfinsv_high_final_model.forecast(steps=forecast_period)
bajajfinsv_high_forecast = bajajfinsv_high_forecast.reshape(-1, 1)
bajajfinsv_high_forecast = scaler.inverse_transform(bajajfinsv_high_forecast)

bajajfinsv_low_final_model = sm.tsa.ARIMA(
    bajajfinsv_y_low_scaled,
    order=bajajfinsv_low_best_order
)
bajajfinsv_low_final_model = bajajfinsv_low_final_model.fit()
bajajfinsv_low_forecast = bajajfinsv_low_final_model.forecast(steps=forecast_period)
bajajfinsv_low_forecast = bajajfinsv_low_forecast.reshape(-1, 1)
bajajfinsv_low_forecast = scaler.inverse_transform(bajajfinsv_low_forecast)

print("Close Forecasts:", bajajfinsv_close_forecast)
print("Open Forecasts:", bajajfinsv_open_forecast)
print("High Forecasts:", bajajfinsv_high_forecast)
print("Low Forecasts:", bajajfinsv_low_forecast)


# In[150]:


bajajfinsv_tail_50_data = bajajfinsv.tail(forecast_periods)

bajajfinsv_actual_close_prices = bajajfinsv_tail_50_data['Close'].values
bajajfinsv_actual_open_prices = bajajfinsv_tail_50_data['Open'].values
bajajfinsv_actual_high_prices = bajajfinsv_tail_50_data['High'].values
bajajfinsv_actual_low_prices = bajajfinsv_tail_50_data['Low'].values

bajajfinsv_forecast_close = bajajfinsv_close_final_model.forecast(steps=forecast_periods)
bajajfinsv_forecast_close = bajajfinsv_forecast_close.reshape(-1, 1)
bajajfinsv_forecast_close = scaler.inverse_transform(bajajfinsv_forecast_close)

bajajfinsv_forecast_open = bajajfinsv_open_final_model.forecast(steps=forecast_periods)
bajajfinsv_forecast_open = bajajfinsv_forecast_open.reshape(-1, 1)
bajajfinsv_forecast_open = scaler.inverse_transform(bajajfinsv_forecast_open)

bajajfinsv_forecast_high = bajajfinsv_high_final_model.forecast(steps=forecast_periods)
bajajfinsv_forecast_high = bajajfinsv_forecast_high.reshape(-1, 1)
bajajfinsv_forecast_high = scaler.inverse_transform(bajajfinsv_forecast_high)

bajajfinsv_forecast_low = bajajfinsv_low_final_model.forecast(steps=forecast_periods)
bajajfinsv_forecast_low = bajajfinsv_forecast_low.reshape(-1, 1)
bajajfinsv_forecast_low = scaler.inverse_transform(bajajfinsv_forecast_low)

bajajfinsv_close_mae = mean_absolute_error(bajajfinsv_actual_close_prices, bajajfinsv_forecast_close)
bajajfinsv_close_mse = mean_squared_error(bajajfinsv_actual_close_prices, bajajfinsv_forecast_close)
bajajfinsv_close_rmse = np.sqrt(bajajfinsv_close_mse)

bajajfinsv_open_mae = mean_absolute_error(bajajfinsv_actual_open_prices, bajajfinsv_forecast_open)
bajajfinsv_open_mse = mean_squared_error(bajajfinsv_actual_open_prices, bajajfinsv_forecast_open)
bajajfinsv_open_rmse = np.sqrt(bajajfinsv_open_mse)

bajajfinsv_high_mae = mean_absolute_error(bajajfinsv_actual_high_prices, bajajfinsv_forecast_high)
bajajfinsv_high_mse = mean_squared_error(bajajfinsv_actual_high_prices, bajajfinsv_forecast_high)
bajajfinsv_high_rmse = np.sqrt(bajajfinsv_high_mse)

bajajfinsv_low_mae = mean_absolute_error(bajajfinsv_actual_low_prices, bajajfinsv_forecast_low)
bajajfinsv_low_mse = mean_squared_error(bajajfinsv_actual_low_prices, bajajfinsv_forecast_low)
bajajfinsv_low_rmse = np.sqrt(bajajfinsv_low_mse)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

bajajfinsv_close_mape = mean_absolute_percentage_error(bajajfinsv_actual_close_prices, bajajfinsv_forecast_close)
bajajfinsv_open_mape = mean_absolute_percentage_error(bajajfinsv_actual_open_prices, bajajfinsv_forecast_open)
bajajfinsv_high_mape = mean_absolute_percentage_error(bajajfinsv_actual_high_prices, bajajfinsv_forecast_high)
bajajfinsv_low_mape = mean_absolute_percentage_error(bajajfinsv_actual_low_prices, bajajfinsv_forecast_low)

print("Close Forecasts:", bajajfinsv_forecast_close)
print(f"Close Mean Absolute Error (MAE): {bajajfinsv_close_mae}")
print(f"Close Mean Squared Error (MSE): {bajajfinsv_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {bajajfinsv_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {bajajfinsv_close_mape}%")

print("Open Forecasts:", bajajfinsv_forecast_open)
print(f"Open Mean Absolute Error (MAE): {bajajfinsv_open_mae}")
print(f"Open Mean Squared Error (MSE): {bajajfinsv_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {bajajfinsv_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {bajajfinsv_open_mape}%")

print("High Forecasts:", bajajfinsv_forecast_high)
print(f"High Mean Absolute Error (MAE): {bajajfinsv_high_mae}")
print(f"High Mean Squared Error (MSE): {bajajfinsv_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {bajajfinsv_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {bajajfinsv_high_mape}%")

print("Low Forecasts:", bajajfinsv_forecast_low)
print(f"Low Mean Absolute Error (MAE): {bajajfinsv_low_mae}")
print(f"Low Mean Squared Error (MSE): {bajajfinsv_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {bajajfinsv_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {bajajfinsv_low_mape}%")


# In[151]:


bajautofin_y_close = bajautofin['Close'].values
bajautofin_y_open = bajautofin['Open'].values
bajautofin_y_high = bajautofin['High'].values
bajautofin_y_low = bajautofin['Low'].values

bajautofin_y_close_scaled = scaler.fit_transform(bajautofin_y_close.reshape(-1, 1))
bajautofin_y_open_scaled = scaler.fit_transform(bajautofin_y_open.reshape(-1, 1))
bajautofin_y_high_scaled = scaler.fit_transform(bajautofin_y_high.reshape(-1, 1))
bajautofin_y_low_scaled = scaler.fit_transform(bajautofin_y_low.reshape(-1, 1))

bajautofin_close_model = auto_arima(
    bajautofin_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajautofin_open_model = auto_arima(
    bajautofin_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajautofin_high_model = auto_arima(
    bajautofin_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajautofin_low_model = auto_arima(
    bajautofin_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajautofin_close_best_order = bajautofin_close_model.get_params()['order']
bajautofin_open_best_order = bajautofin_open_model.get_params()['order']
bajautofin_high_best_order = bajautofin_high_model.get_params()['order']
bajautofin_low_best_order = bajautofin_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {bajautofin_close_best_order}")
print(f"Best ARIMA Order for Open: {bajautofin_open_best_order}")
print(f"Best ARIMA Order for High: {bajautofin_high_best_order}")
print(f"Best ARIMA Order for Low: {bajautofin_low_best_order}")

bajautofin_close_final_model = sm.tsa.ARIMA(
    bajautofin_y_close_scaled,
    order=bajautofin_close_best_order
)
bajautofin_close_final_model = bajautofin_close_final_model.fit()
bajautofin_close_forecast = bajautofin_close_final_model.forecast(steps=forecast_period)
bajautofin_close_forecast = bajautofin_close_forecast.reshape(-1, 1)
bajautofin_close_forecast = scaler.inverse_transform(bajautofin_close_forecast)

bajautofin_open_final_model = sm.tsa.ARIMA(
    bajautofin_y_open_scaled,
    order=bajautofin_open_best_order
)
bajautofin_open_final_model = bajautofin_open_final_model.fit()
bajautofin_open_forecast = bajautofin_open_final_model.forecast(steps=forecast_period)
bajautofin_open_forecast = bajautofin_open_forecast.reshape(-1, 1)
bajautofin_open_forecast = scaler.inverse_transform(bajautofin_open_forecast)

bajautofin_high_final_model = sm.tsa.ARIMA(
    bajautofin_y_high_scaled,
    order=bajautofin_high_best_order
)
bajautofin_high_final_model = bajautofin_high_final_model.fit()
bajautofin_high_forecast = bajautofin_high_final_model.forecast(steps=forecast_period)
bajautofin_high_forecast = bajautofin_high_forecast.reshape(-1, 1)
bajautofin_high_forecast = scaler.inverse_transform(bajautofin_high_forecast)

bajautofin_low_final_model = sm.tsa.ARIMA(
    bajautofin_y_low_scaled,
    order=bajautofin_low_best_order
)
bajautofin_low_final_model = bajautofin_low_final_model.fit()
bajautofin_low_forecast = bajautofin_low_final_model.forecast(steps=forecast_period)
bajautofin_low_forecast = bajautofin_low_forecast.reshape(-1, 1)
bajautofin_low_forecast = scaler.inverse_transform(bajautofin_low_forecast)

print("Close Forecasts:", bajautofin_close_forecast)
print("Open Forecasts:", bajautofin_open_forecast)
print("High Forecasts:", bajautofin_high_forecast)
print("Low Forecasts:", bajautofin_low_forecast)


# In[152]:


bajautofin_tail_50_data = bajautofin.tail(forecast_periods)

bajautofin_actual_close_prices = bajautofin_tail_50_data['Close'].values
bajautofin_actual_open_prices = bajautofin_tail_50_data['Open'].values
bajautofin_actual_high_prices = bajautofin_tail_50_data['High'].values
bajautofin_actual_low_prices = bajautofin_tail_50_data['Low'].values

bajautofin_forecast_close = bajautofin_close_final_model.forecast(steps=forecast_periods)
bajautofin_forecast_close = bajautofin_forecast_close.reshape(-1, 1)
bajautofin_forecast_close = scaler.inverse_transform(bajautofin_forecast_close)

bajautofin_forecast_open = bajautofin_open_final_model.forecast(steps=forecast_periods)
bajautofin_forecast_open = bajautofin_forecast_open.reshape(-1, 1)
bajautofin_forecast_open = scaler.inverse_transform(bajautofin_forecast_open)

bajautofin_forecast_high = bajautofin_high_final_model.forecast(steps=forecast_periods)
bajautofin_forecast_high = bajautofin_forecast_high.reshape(-1, 1)
bajautofin_forecast_high = scaler.inverse_transform(bajautofin_forecast_high)

bajautofin_forecast_low = bajautofin_low_final_model.forecast(steps=forecast_periods)
bajautofin_forecast_low = bajautofin_forecast_low.reshape(-1, 1)
bajautofin_forecast_low = scaler.inverse_transform(bajautofin_forecast_low)

bajautofin_close_mae = mean_absolute_error(bajautofin_actual_close_prices, bajautofin_forecast_close)
bajautofin_close_mse = mean_squared_error(bajautofin_actual_close_prices, bajautofin_forecast_close)
bajautofin_close_rmse = np.sqrt(bajautofin_close_mse)

bajautofin_open_mae = mean_absolute_error(bajautofin_actual_open_prices, bajautofin_forecast_open)
bajautofin_open_mse = mean_squared_error(bajautofin_actual_open_prices, bajautofin_forecast_open)
bajautofin_open_rmse = np.sqrt(bajautofin_open_mse)

bajautofin_high_mae = mean_absolute_error(bajautofin_actual_high_prices, bajautofin_forecast_high)
bajautofin_high_mse = mean_squared_error(bajautofin_actual_high_prices, bajautofin_forecast_high)
bajautofin_high_rmse = np.sqrt(bajautofin_high_mse)

bajautofin_low_mae = mean_absolute_error(bajautofin_actual_low_prices, bajautofin_forecast_low)
bajautofin_low_mse = mean_squared_error(bajautofin_actual_low_prices, bajautofin_forecast_low)
bajautofin_low_rmse = np.sqrt(bajautofin_low_mse)

bajautofin_close_mape = mean_absolute_percentage_error(bajautofin_actual_close_prices, bajautofin_forecast_close)
bajautofin_open_mape = mean_absolute_percentage_error(bajautofin_actual_open_prices, bajautofin_forecast_open)
bajautofin_high_mape = mean_absolute_percentage_error(bajautofin_actual_high_prices, bajautofin_forecast_high)
bajautofin_low_mape = mean_absolute_percentage_error(bajautofin_actual_low_prices, bajautofin_forecast_low)

print("Close Forecasts:", bajautofin_forecast_close)
print(f"Close Mean Absolute Error (MAE): {bajautofin_close_mae}")
print(f"Close Mean Squared Error (MSE): {bajautofin_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {bajautofin_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {bajautofin_close_mape}%")

print("Open Forecasts:", bajautofin_forecast_open)
print(f"Open Mean Absolute Error (MAE): {bajautofin_open_mae}")
print(f"Open Mean Squared Error (MSE): {bajautofin_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {bajautofin_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {bajautofin_open_mape}%")

print("High Forecasts:", bajautofin_forecast_high)
print(f"High Mean Absolute Error (MAE): {bajautofin_high_mae}")
print(f"High Mean Squared Error (MSE): {bajautofin_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {bajautofin_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {bajautofin_high_mape}%")

print("Low Forecasts:", bajautofin_forecast_low)
print(f"Low Mean Absolute Error (MAE): {bajautofin_low_mae}")
print(f"Low Mean Squared Error (MSE): {bajautofin_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {bajautofin_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {bajautofin_low_mape}%")


# In[153]:


bajfinance_y_close = bajfinance['Close'].values
bajfinance_y_open = bajfinance['Open'].values
bajfinance_y_high = bajfinance['High'].values
bajfinance_y_low = bajfinance['Low'].values

bajfinance_y_close_scaled = scaler.fit_transform(bajfinance_y_close.reshape(-1, 1))
bajfinance_y_open_scaled = scaler.fit_transform(bajfinance_y_open.reshape(-1, 1))
bajfinance_y_high_scaled = scaler.fit_transform(bajfinance_y_high.reshape(-1, 1))
bajfinance_y_low_scaled = scaler.fit_transform(bajfinance_y_low.reshape(-1, 1))

bajfinance_close_model = auto_arima(
    bajfinance_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajfinance_open_model = auto_arima(
    bajfinance_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajfinance_high_model = auto_arima(
    bajfinance_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajfinance_low_model = auto_arima(
    bajfinance_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bajfinance_close_best_order = bajfinance_close_model.get_params()['order']
bajfinance_open_best_order = bajfinance_open_model.get_params()['order']
bajfinance_high_best_order = bajfinance_high_model.get_params()['order']
bajfinance_low_best_order = bajfinance_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {bajfinance_close_best_order}")
print(f"Best ARIMA Order for Open: {bajfinance_open_best_order}")
print(f"Best ARIMA Order for High: {bajfinance_high_best_order}")
print(f"Best ARIMA Order for Low: {bajfinance_low_best_order}")

bajfinance_close_final_model = sm.tsa.ARIMA(
    bajfinance_y_close_scaled,
    order=bajfinance_close_best_order
)
bajfinance_close_final_model = bajfinance_close_final_model.fit()
bajfinance_close_forecast = bajfinance_close_final_model.forecast(steps=forecast_period)
bajfinance_close_forecast = bajfinance_close_forecast.reshape(-1, 1)
bajfinance_close_forecast = scaler.inverse_transform(bajfinance_close_forecast)

bajfinance_open_final_model = sm.tsa.ARIMA(
    bajfinance_y_open_scaled,
    order=bajfinance_open_best_order
)
bajfinance_open_final_model = bajfinance_open_final_model.fit()
bajfinance_open_forecast = bajfinance_open_final_model.forecast(steps=forecast_period)
bajfinance_open_forecast = bajfinance_open_forecast.reshape(-1, 1)
bajfinance_open_forecast = scaler.inverse_transform(bajfinance_open_forecast)

bajfinance_high_final_model = sm.tsa.ARIMA(
    bajfinance_y_high_scaled,
    order=bajfinance_high_best_order
)
bajfinance_high_final_model = bajfinance_high_final_model.fit()
bajfinance_high_forecast = bajfinance_high_final_model.forecast(steps=forecast_period)
bajfinance_high_forecast = bajfinance_high_forecast.reshape(-1, 1)
bajfinance_high_forecast = scaler.inverse_transform(bajfinance_high_forecast)

bajfinance_low_final_model = sm.tsa.ARIMA(
    bajfinance_y_low_scaled,
    order=bajfinance_low_best_order
)
bajfinance_low_final_model = bajfinance_low_final_model.fit()
bajfinance_low_forecast = bajfinance_low_final_model.forecast(steps=forecast_period)
bajfinance_low_forecast = bajfinance_low_forecast.reshape(-1, 1)
bajfinance_low_forecast = scaler.inverse_transform(bajfinance_low_forecast)

print("Close Forecasts:", bajfinance_close_forecast)
print("Open Forecasts:", bajfinance_open_forecast)
print("High Forecasts:", bajfinance_high_forecast)
print("Low Forecasts:", bajfinance_low_forecast)


# In[154]:


bajfinance_tail_50_data = bajfinance.tail(forecast_periods)

bajfinance_actual_close_prices = bajfinance_tail_50_data['Close'].values
bajfinance_actual_open_prices = bajfinance_tail_50_data['Open'].values
bajfinance_actual_high_prices = bajfinance_tail_50_data['High'].values
bajfinance_actual_low_prices = bajfinance_tail_50_data['Low'].values

bajfinance_forecast_close = bajfinance_close_final_model.forecast(steps=forecast_periods)
bajfinance_forecast_close = bajfinance_forecast_close.reshape(-1, 1)
bajfinance_forecast_close = scaler.inverse_transform(bajfinance_forecast_close)

bajfinance_forecast_open = bajfinance_open_final_model.forecast(steps=forecast_periods)
bajfinance_forecast_open = bajfinance_forecast_open.reshape(-1, 1)
bajfinance_forecast_open = scaler.inverse_transform(bajfinance_forecast_open)

bajfinance_forecast_high = bajfinance_high_final_model.forecast(steps=forecast_periods)
bajfinance_forecast_high = bajfinance_forecast_high.reshape(-1, 1)
bajfinance_forecast_high = scaler.inverse_transform(bajfinance_forecast_high)

bajfinance_forecast_low = bajfinance_low_final_model.forecast(steps=forecast_periods)
bajfinance_forecast_low = bajfinance_forecast_low.reshape(-1, 1)
bajfinance_forecast_low = scaler.inverse_transform(bajfinance_forecast_low)

bajfinance_close_mae = mean_absolute_error(bajfinance_actual_close_prices, bajfinance_forecast_close)
bajfinance_close_mse = mean_squared_error(bajfinance_actual_close_prices, bajfinance_forecast_close)
bajfinance_close_rmse = np.sqrt(bajfinance_close_mse)

bajfinance_open_mae = mean_absolute_error(bajfinance_actual_open_prices, bajfinance_forecast_open)
bajfinance_open_mse = mean_squared_error(bajfinance_actual_open_prices, bajfinance_forecast_open)
bajfinance_open_rmse = np.sqrt(bajfinance_open_mse)

bajfinance_high_mae = mean_absolute_error(bajfinance_actual_high_prices, bajfinance_forecast_high)
bajfinance_high_mse = mean_squared_error(bajfinance_actual_high_prices, bajfinance_forecast_high)
bajfinance_high_rmse = np.sqrt(bajfinance_high_mse)

bajfinance_low_mae = mean_absolute_error(bajfinance_actual_low_prices, bajfinance_forecast_low)
bajfinance_low_mse = mean_squared_error(bajfinance_actual_low_prices, bajfinance_forecast_low)
bajfinance_low_rmse = np.sqrt(bajfinance_low_mse)

bajfinance_close_mape = mean_absolute_percentage_error(bajfinance_actual_close_prices, bajfinance_forecast_close)
bajfinance_open_mape = mean_absolute_percentage_error(bajfinance_actual_open_prices, bajfinance_forecast_open)
bajfinance_high_mape = mean_absolute_percentage_error(bajfinance_actual_high_prices, bajfinance_forecast_high)
bajfinance_low_mape = mean_absolute_percentage_error(bajfinance_actual_low_prices, bajfinance_forecast_low)

print("Close Forecasts:", bajfinance_forecast_close)
print(f"Close Mean Absolute Error (MAE): {bajfinance_close_mae}")
print(f"Close Mean Squared Error (MSE): {bajfinance_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {bajfinance_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {bajfinance_close_mape}%")

print("Open Forecasts:", bajfinance_forecast_open)
print(f"Open Mean Absolute Error (MAE): {bajfinance_open_mae}")
print(f"Open Mean Squared Error (MSE): {bajfinance_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {bajfinance_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {bajfinance_open_mape}%")

print("High Forecasts:", bajfinance_forecast_high)
print(f"High Mean Absolute Error (MAE): {bajfinance_high_mae}")
print(f"High Mean Squared Error (MSE): {bajfinance_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {bajfinance_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {bajfinance_high_mape}%")

print("Low Forecasts:", bajfinance_forecast_low)
print(f"Low Mean Absolute Error (MAE): {bajfinance_low_mae}")
print(f"Low Mean Squared Error (MSE): {bajfinance_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {bajfinance_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {bajfinance_low_mape}%")


# In[155]:


bharti_y_close = bharti['Close'].values
bharti_y_open = bharti['Open'].values
bharti_y_high = bharti['High'].values
bharti_y_low = bharti['Low'].values

bharti_y_close_scaled = scaler.fit_transform(bharti_y_close.reshape(-1, 1))
bharti_y_open_scaled = scaler.fit_transform(bharti_y_open.reshape(-1, 1))
bharti_y_high_scaled = scaler.fit_transform(bharti_y_high.reshape(-1, 1))
bharti_y_low_scaled = scaler.fit_transform(bharti_y_low.reshape(-1, 1))

bharti_close_model = auto_arima(
    bharti_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bharti_open_model = auto_arima(
    bharti_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bharti_high_model = auto_arima(
    bharti_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bharti_low_model = auto_arima(
    bharti_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bharti_close_best_order = bharti_close_model.get_params()['order']
bharti_open_best_order = bharti_open_model.get_params()['order']
bharti_high_best_order = bharti_high_model.get_params()['order']
bharti_low_best_order = bharti_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {bharti_close_best_order}")
print(f"Best ARIMA Order for Open: {bharti_open_best_order}")
print(f"Best ARIMA Order for High: {bharti_high_best_order}")
print(f"Best ARIMA Order for Low: {bharti_low_best_order}")

bharti_close_final_model = sm.tsa.ARIMA(
    bharti_y_close_scaled,
    order=bharti_close_best_order
)
bharti_close_final_model = bharti_close_final_model.fit()
bharti_close_forecast = bharti_close_final_model.forecast(steps=forecast_period)
bharti_close_forecast = bharti_close_forecast.reshape(-1, 1)
bharti_close_forecast = scaler.inverse_transform(bharti_close_forecast)

bharti_open_final_model = sm.tsa.ARIMA(
    bharti_y_open_scaled,
    order=bharti_open_best_order
)
bharti_open_final_model = bharti_open_final_model.fit()
bharti_open_forecast = bharti_open_final_model.forecast(steps=forecast_period)
bharti_open_forecast = bharti_open_forecast.reshape(-1, 1)
bharti_open_forecast = scaler.inverse_transform(bharti_open_forecast)

bharti_high_final_model = sm.tsa.ARIMA(
    bharti_y_high_scaled,
    order=bharti_high_best_order
)
bharti_high_final_model = bharti_high_final_model.fit()
bharti_high_forecast = bharti_high_final_model.forecast(steps=forecast_period)
bharti_high_forecast = bharti_high_forecast.reshape(-1, 1)
bharti_high_forecast = scaler.inverse_transform(bharti_high_forecast)

bharti_low_final_model = sm.tsa.ARIMA(
    bharti_y_low_scaled,
    order=bharti_low_best_order
)
bharti_low_final_model = bharti_low_final_model.fit()
bharti_low_forecast = bharti_low_final_model.forecast(steps=forecast_period)
bharti_low_forecast = bharti_low_forecast.reshape(-1, 1)
bharti_low_forecast = scaler.inverse_transform(bharti_low_forecast)

print("Close Forecasts:", bharti_close_forecast)
print("Open Forecasts:", bharti_open_forecast)
print("High Forecasts:", bharti_high_forecast)
print("Low Forecasts:", bharti_low_forecast)


# In[156]:


bharti_tail_50_data = bharti.tail(forecast_periods)

bharti_actual_close_prices = bharti_tail_50_data['Close'].values
bharti_actual_open_prices = bharti_tail_50_data['Open'].values
bharti_actual_high_prices = bharti_tail_50_data['High'].values
bharti_actual_low_prices = bharti_tail_50_data['Low'].values

bharti_forecast_close = bharti_close_final_model.forecast(steps=forecast_periods)
bharti_forecast_close = bharti_forecast_close.reshape(-1, 1)
bharti_forecast_close = scaler.inverse_transform(bharti_forecast_close)

bharti_forecast_open = bharti_open_final_model.forecast(steps=forecast_periods)
bharti_forecast_open = bharti_forecast_open.reshape(-1, 1)
bharti_forecast_open = scaler.inverse_transform(bharti_forecast_open)

bharti_forecast_high = bharti_high_final_model.forecast(steps=forecast_periods)
bharti_forecast_high = bharti_forecast_high.reshape(-1, 1)
bharti_forecast_high = scaler.inverse_transform(bharti_forecast_high)

bharti_forecast_low = bharti_low_final_model.forecast(steps=forecast_periods)
bharti_forecast_low = bharti_forecast_low.reshape(-1, 1)
bharti_forecast_low = scaler.inverse_transform(bharti_forecast_low)

bharti_close_mae = mean_absolute_error(bharti_actual_close_prices, bharti_forecast_close)
bharti_close_mse = mean_squared_error(bharti_actual_close_prices, bharti_forecast_close)
bharti_close_rmse = np.sqrt(bharti_close_mse)

bharti_open_mae = mean_absolute_error(bharti_actual_open_prices, bharti_forecast_open)
bharti_open_mse = mean_squared_error(bharti_actual_open_prices, bharti_forecast_open)
bharti_open_rmse = np.sqrt(bharti_open_mse)

bharti_high_mae = mean_absolute_error(bharti_actual_high_prices, bharti_forecast_high)
bharti_high_mse = mean_squared_error(bharti_actual_high_prices, bharti_forecast_high)
bharti_high_rmse = np.sqrt(bharti_high_mse)

bharti_low_mae = mean_absolute_error(bharti_actual_low_prices, bharti_forecast_low)
bharti_low_mse = mean_squared_error(bharti_actual_low_prices, bharti_forecast_low)
bharti_low_rmse = np.sqrt(bharti_low_mse)

bharti_close_mape = mean_absolute_percentage_error(bharti_actual_close_prices, bharti_forecast_close)
bharti_open_mape = mean_absolute_percentage_error(bharti_actual_open_prices, bharti_forecast_open)
bharti_high_mape = mean_absolute_percentage_error(bharti_actual_high_prices, bharti_forecast_high)
bharti_low_mape = mean_absolute_percentage_error(bharti_actual_low_prices, bharti_forecast_low)

print("Close Forecasts:", bharti_forecast_close)
print(f"Close Mean Absolute Error (MAE): {bharti_close_mae}")
print(f"Close Mean Squared Error (MSE): {bharti_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {bharti_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {bharti_close_mape}%")

print("Open Forecasts:", bharti_forecast_open)
print(f"Open Mean Absolute Error (MAE): {bharti_open_mae}")
print(f"Open Mean Squared Error (MSE): {bharti_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {bharti_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {bharti_open_mape}%")

print("High Forecasts:", bharti_forecast_high)
print(f"High Mean Absolute Error (MAE): {bharti_high_mae}")
print(f"High Mean Squared Error (MSE): {bharti_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {bharti_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {bharti_high_mape}%")

print("Low Forecasts:", bharti_forecast_low)
print(f"Low Mean Absolute Error (MAE): {bharti_low_mae}")
print(f"Low Mean Squared Error (MSE): {bharti_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {bharti_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {bharti_low_mape}%")


# In[157]:


bhartiartl_y_close = bhartiartl['Close'].values
bhartiartl_y_open = bhartiartl['Open'].values
bhartiartl_y_high = bhartiartl['High'].values
bhartiartl_y_low = bhartiartl['Low'].values

bhartiartl_y_close_scaled = scaler.fit_transform(bhartiartl_y_close.reshape(-1, 1))
bhartiartl_y_open_scaled = scaler.fit_transform(bhartiartl_y_open.reshape(-1, 1))
bhartiartl_y_high_scaled = scaler.fit_transform(bhartiartl_y_high.reshape(-1, 1))
bhartiartl_y_low_scaled = scaler.fit_transform(bhartiartl_y_low.reshape(-1, 1))

bhartiartl_close_model = auto_arima(
    bhartiartl_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bhartiartl_open_model = auto_arima(
    bhartiartl_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bhartiartl_high_model = auto_arima(
    bhartiartl_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bhartiartl_low_model = auto_arima(
    bhartiartl_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bhartiartl_close_best_order = bhartiartl_close_model.get_params()['order']
bhartiartl_open_best_order = bhartiartl_open_model.get_params()['order']
bhartiartl_high_best_order = bhartiartl_high_model.get_params()['order']
bhartiartl_low_best_order = bhartiartl_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {bhartiartl_close_best_order}")
print(f"Best ARIMA Order for Open: {bhartiartl_open_best_order}")
print(f"Best ARIMA Order for High: {bhartiartl_high_best_order}")
print(f"Best ARIMA Order for Low: {bhartiartl_low_best_order}")

bhartiartl_close_final_model = sm.tsa.ARIMA(
    bhartiartl_y_close_scaled,
    order=bhartiartl_close_best_order
)
bhartiartl_close_final_model = bhartiartl_close_final_model.fit()
bhartiartl_close_forecast = bhartiartl_close_final_model.forecast(steps=forecast_period)
bhartiartl_close_forecast = bhartiartl_close_forecast.reshape(-1, 1)
bhartiartl_close_forecast = scaler.inverse_transform(bhartiartl_close_forecast)

bhartiartl_open_final_model = sm.tsa.ARIMA(
    bhartiartl_y_open_scaled,
    order=bhartiartl_open_best_order
)
bhartiartl_open_final_model = bhartiartl_open_final_model.fit()
bhartiartl_open_forecast = bhartiartl_open_final_model.forecast(steps=forecast_period)
bhartiartl_open_forecast = bhartiartl_open_forecast.reshape(-1, 1)
bhartiartl_open_forecast = scaler.inverse_transform(bhartiartl_open_forecast)

bhartiartl_high_final_model = sm.tsa.ARIMA(
    bhartiartl_y_high_scaled,
    order=bhartiartl_high_best_order
)
bhartiartl_high_final_model = bhartiartl_high_final_model.fit()
bhartiartl_high_forecast = bhartiartl_high_final_model.forecast(steps=forecast_period)
bhartiartl_high_forecast = bhartiartl_high_forecast.reshape(-1, 1)
bhartiartl_high_forecast = scaler.inverse_transform(bhartiartl_high_forecast)

bhartiartl_low_final_model = sm.tsa.ARIMA(
    bhartiartl_y_low_scaled,
    order=bhartiartl_low_best_order
)
bhartiartl_low_final_model = bhartiartl_low_final_model.fit()
bhartiartl_low_forecast = bhartiartl_low_final_model.forecast(steps=forecast_period)
bhartiartl_low_forecast = bhartiartl_low_forecast.reshape(-1, 1)
bhartiartl_low_forecast = scaler.inverse_transform(bhartiartl_low_forecast)

print("Close Forecasts:", bhartiartl_close_forecast)
print("Open Forecasts:", bhartiartl_open_forecast)
print("High Forecasts:", bhartiartl_high_forecast)
print("Low Forecasts:", bhartiartl_low_forecast)


# In[158]:


bhartiartl_tail_50_data = bhartiartl.tail(forecast_periods)

bhartiartl_actual_close_prices = bhartiartl_tail_50_data['Close'].values
bhartiartl_actual_open_prices = bhartiartl_tail_50_data['Open'].values
bhartiartl_actual_high_prices = bhartiartl_tail_50_data['High'].values
bhartiartl_actual_low_prices = bhartiartl_tail_50_data['Low'].values

bhartiartl_forecast_close = bhartiartl_close_final_model.forecast(steps=forecast_periods)
bhartiartl_forecast_close = bhartiartl_forecast_close.reshape(-1, 1)
bhartiartl_forecast_close = scaler.inverse_transform(bhartiartl_forecast_close)

bhartiartl_forecast_open = bhartiartl_open_final_model.forecast(steps=forecast_periods)
bhartiartl_forecast_open = bhartiartl_forecast_open.reshape(-1, 1)
bhartiartl_forecast_open = scaler.inverse_transform(bhartiartl_forecast_open)

bhartiartl_forecast_high = bhartiartl_high_final_model.forecast(steps=forecast_periods)
bhartiartl_forecast_high = bhartiartl_forecast_high.reshape(-1, 1)
bhartiartl_forecast_high = scaler.inverse_transform(bhartiartl_forecast_high)

bhartiartl_forecast_low = bhartiartl_low_final_model.forecast(steps=forecast_periods)
bhartiartl_forecast_low = bhartiartl_forecast_low.reshape(-1, 1)
bhartiartl_forecast_low = scaler.inverse_transform(bhartiartl_forecast_low)

bhartiartl_close_mae = mean_absolute_error(bhartiartl_actual_close_prices, bhartiartl_forecast_close)
bhartiartl_close_mse = mean_squared_error(bhartiartl_actual_close_prices, bhartiartl_forecast_close)
bhartiartl_close_rmse = np.sqrt(bhartiartl_close_mse)

bhartiartl_open_mae = mean_absolute_error(bhartiartl_actual_open_prices, bhartiartl_forecast_open)
bhartiartl_open_mse = mean_squared_error(bhartiartl_actual_open_prices, bhartiartl_forecast_open)
bhartiartl_open_rmse = np.sqrt(bhartiartl_open_mse)

bhartiartl_high_mae = mean_absolute_error(bhartiartl_actual_high_prices, bhartiartl_forecast_high)
bhartiartl_high_mse = mean_squared_error(bhartiartl_actual_high_prices, bhartiartl_forecast_high)
bhartiartl_high_rmse = np.sqrt(bhartiartl_high_mse)

bhartiartl_low_mae = mean_absolute_error(bhartiartl_actual_low_prices, bhartiartl_forecast_low)
bhartiartl_low_mse = mean_squared_error(bhartiartl_actual_low_prices, bhartiartl_forecast_low)
bhartiartl_low_rmse = np.sqrt(bhartiartl_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

bhartiartl_close_mape = mean_absolute_percentage_error(bhartiartl_actual_close_prices, bhartiartl_forecast_close)
bhartiartl_open_mape = mean_absolute_percentage_error(bhartiartl_actual_open_prices, bhartiartl_forecast_open)
bhartiartl_high_mape = mean_absolute_percentage_error(bhartiartl_actual_high_prices, bhartiartl_forecast_high)
bhartiartl_low_mape = mean_absolute_percentage_error(bhartiartl_actual_low_prices, bhartiartl_forecast_low)


print("Close Forecasts:", bhartiartl_forecast_close)
print(f"Close Mean Absolute Error (MAE): {bhartiartl_close_mae}")
print(f"Close Mean Squared Error (MSE): {bhartiartl_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {bhartiartl_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {bhartiartl_close_mape}%")

print("Open Forecasts:", bhartiartl_forecast_open)
print(f"Open Mean Absolute Error (MAE): {bhartiartl_open_mae}")
print(f"Open Mean Squared Error (MSE): {bhartiartl_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {bhartiartl_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {bhartiartl_open_mape}%")

print("High Forecasts:", bhartiartl_forecast_high)
print(f"High Mean Absolute Error (MAE): {bhartiartl_high_mae}")
print(f"High Mean Squared Error (MSE): {bhartiartl_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {bhartiartl_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {bhartiartl_high_mape}%")

print("Low Forecasts:", bhartiartl_forecast_low)
print(f"Low Mean Absolute Error (MAE): {bhartiartl_low_mae}")
print(f"Low Mean Squared Error (MSE): {bhartiartl_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {bhartiartl_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {bhartiartl_low_mape}%")


# In[159]:


bpcl_y_close = bpcl['Close'].values
bpcl_y_open = bpcl['Open'].values
bpcl_y_high = bpcl['High'].values
bpcl_y_low = bpcl['Low'].values

bpcl_y_close_scaled = scaler.fit_transform(bpcl_y_close.reshape(-1, 1))
bpcl_y_open_scaled = scaler.fit_transform(bpcl_y_open.reshape(-1, 1))
bpcl_y_high_scaled = scaler.fit_transform(bpcl_y_high.reshape(-1, 1))
bpcl_y_low_scaled = scaler.fit_transform(bpcl_y_low.reshape(-1, 1))

bpcl_close_model = auto_arima(
    bpcl_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bpcl_open_model = auto_arima(
    bpcl_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bpcl_high_model = auto_arima(
    bpcl_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bpcl_low_model = auto_arima(
    bpcl_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bpcl_close_best_order = bpcl_close_model.get_params()['order']
bpcl_open_best_order = bpcl_open_model.get_params()['order']
bpcl_high_best_order = bpcl_high_model.get_params()['order']
bpcl_low_best_order = bpcl_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {bpcl_close_best_order}")
print(f"Best ARIMA Order for Open: {bpcl_open_best_order}")
print(f"Best ARIMA Order for High: {bpcl_high_best_order}")
print(f"Best ARIMA Order for Low: {bpcl_low_best_order}")

bpcl_close_final_model = sm.tsa.ARIMA(
    bpcl_y_close_scaled,
    order=bpcl_close_best_order
)
bpcl_close_final_model = bpcl_close_final_model.fit()
bpcl_close_forecast = bpcl_close_final_model.forecast(steps=forecast_period)
bpcl_close_forecast = bpcl_close_forecast.reshape(-1, 1)
bpcl_close_forecast = scaler.inverse_transform(bpcl_close_forecast)

bpcl_open_final_model = sm.tsa.ARIMA(
    bpcl_y_open_scaled,
    order=bpcl_open_best_order
)
bpcl_open_final_model = bpcl_open_final_model.fit()
bpcl_open_forecast = bpcl_open_final_model.forecast(steps=forecast_period)
bpcl_open_forecast = bpcl_open_forecast.reshape(-1, 1)
bpcl_open_forecast = scaler.inverse_transform(bpcl_open_forecast)

bpcl_high_final_model = sm.tsa.ARIMA(
    bpcl_y_high_scaled,
    order=bpcl_high_best_order
)
bpcl_high_final_model = bpcl_high_final_model.fit()
bpcl_high_forecast = bpcl_high_final_model.forecast(steps=forecast_period)
bpcl_high_forecast = bpcl_high_forecast.reshape(-1, 1)
bpcl_high_forecast = scaler.inverse_transform(bpcl_high_forecast)

bpcl_low_final_model = sm.tsa.ARIMA(
    bpcl_y_low_scaled,
    order=bpcl_low_best_order
)
bpcl_low_final_model = bpcl_low_final_model.fit()
bpcl_low_forecast = bpcl_low_final_model.forecast(steps=forecast_period)
bpcl_low_forecast = bpcl_low_forecast.reshape(-1, 1)
bpcl_low_forecast = scaler.inverse_transform(bpcl_low_forecast)

print("Close Forecasts:", bpcl_close_forecast)
print("Open Forecasts:", bpcl_open_forecast)
print("High Forecasts:", bpcl_high_forecast)
print("Low Forecasts:", bpcl_low_forecast)


# In[160]:


bpcl_tail_50_data = bpcl.tail(forecast_periods)

bpcl_actual_close_prices = bpcl_tail_50_data['Close'].values
bpcl_actual_open_prices = bpcl_tail_50_data['Open'].values
bpcl_actual_high_prices = bpcl_tail_50_data['High'].values
bpcl_actual_low_prices = bpcl_tail_50_data['Low'].values

bpcl_forecast_close = bpcl_close_final_model.forecast(steps=forecast_periods)
bpcl_forecast_close = bpcl_forecast_close.reshape(-1, 1)
bpcl_forecast_close = scaler.inverse_transform(bpcl_forecast_close)

bpcl_forecast_open = bpcl_open_final_model.forecast(steps=forecast_periods)
bpcl_forecast_open = bpcl_forecast_open.reshape(-1, 1)
bpcl_forecast_open = scaler.inverse_transform(bpcl_forecast_open)

bpcl_forecast_high = bpcl_high_final_model.forecast(steps=forecast_periods)
bpcl_forecast_high = bpcl_forecast_high.reshape(-1, 1)
bpcl_forecast_high = scaler.inverse_transform(bpcl_forecast_high)

bpcl_forecast_low = bpcl_low_final_model.forecast(steps=forecast_periods)
bpcl_forecast_low = bpcl_forecast_low.reshape(-1, 1)
bpcl_forecast_low = scaler.inverse_transform(bpcl_forecast_low)

bpcl_close_mae = mean_absolute_error(bpcl_actual_close_prices, bpcl_forecast_close)
bpcl_close_mse = mean_squared_error(bpcl_actual_close_prices, bpcl_forecast_close)
bpcl_close_rmse = np.sqrt(bpcl_close_mse)

bpcl_open_mae = mean_absolute_error(bpcl_actual_open_prices, bpcl_forecast_open)
bpcl_open_mse = mean_squared_error(bpcl_actual_open_prices, bpcl_forecast_open)
bpcl_open_rmse = np.sqrt(bpcl_open_mse)

bpcl_high_mae = mean_absolute_error(bpcl_actual_high_prices, bpcl_forecast_high)
bpcl_high_mse = mean_squared_error(bpcl_actual_high_prices, bpcl_forecast_high)
bpcl_high_rmse = np.sqrt(bpcl_high_mse)

bpcl_low_mae = mean_absolute_error(bpcl_actual_low_prices, bpcl_forecast_low)
bpcl_low_mse = mean_squared_error(bpcl_actual_low_prices, bpcl_forecast_low)
bpcl_low_rmse = np.sqrt(bpcl_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

bpcl_close_mape = mean_absolute_percentage_error(bpcl_actual_close_prices, bpcl_forecast_close)
bpcl_open_mape = mean_absolute_percentage_error(bpcl_actual_open_prices, bpcl_forecast_open)
bpcl_high_mape = mean_absolute_percentage_error(bpcl_actual_high_prices, bpcl_forecast_high)
bpcl_low_mape = mean_absolute_percentage_error(bpcl_actual_low_prices, bpcl_forecast_low)


print("Close Forecasts:", bpcl_forecast_close)
print(f"Close Mean Absolute Error (MAE): {bpcl_close_mae}")
print(f"Close Mean Squared Error (MSE): {bpcl_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {bpcl_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {bpcl_close_mape}%")

print("Open Forecasts:", bpcl_forecast_open)
print(f"Open Mean Absolute Error (MAE): {bpcl_open_mae}")
print(f"Open Mean Squared Error (MSE): {bpcl_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {bpcl_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {bpcl_open_mape}%")

print("High Forecasts:", bpcl_forecast_high)
print(f"High Mean Absolute Error (MAE): {bpcl_high_mae}")
print(f"High Mean Squared Error (MSE): {bpcl_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {bpcl_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {bpcl_high_mape}%")

print("Low Forecasts:", bpcl_forecast_low)
print(f"Low Mean Absolute Error (MAE): {bpcl_low_mae}")
print(f"Low Mean Squared Error (MSE): {bpcl_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {bpcl_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {bpcl_low_mape}%")


# In[161]:


bpcl_y_close = bpcl['Close'].values
bpcl_y_open = bpcl['Open'].values
bpcl_y_high = bpcl['High'].values
bpcl_y_low = bpcl['Low'].values

bpcl_y_close_scaled = scaler.fit_transform(bpcl_y_close.reshape(-1, 1))
bpcl_y_open_scaled = scaler.fit_transform(bpcl_y_open.reshape(-1, 1))
bpcl_y_high_scaled = scaler.fit_transform(bpcl_y_high.reshape(-1, 1))
bpcl_y_low_scaled = scaler.fit_transform(bpcl_y_low.reshape(-1, 1))

bpcl_close_model = auto_arima(
    bpcl_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bpcl_open_model = auto_arima(
    bpcl_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bpcl_high_model = auto_arima(
    bpcl_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bpcl_low_model = auto_arima(
    bpcl_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

bpcl_close_best_order = bpcl_close_model.get_params()['order']
bpcl_open_best_order = bpcl_open_model.get_params()['order']
bpcl_high_best_order = bpcl_high_model.get_params()['order']
bpcl_low_best_order = bpcl_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {bpcl_close_best_order}")
print(f"Best ARIMA Order for Open: {bpcl_open_best_order}")
print(f"Best ARIMA Order for High: {bpcl_high_best_order}")
print(f"Best ARIMA Order for Low: {bpcl_low_best_order}")

bpcl_close_final_model = sm.tsa.ARIMA(
    bpcl_y_close_scaled,
    order=bpcl_close_best_order
)
bpcl_close_final_model = bpcl_close_final_model.fit()
bpcl_close_forecast = bpcl_close_final_model.forecast(steps=forecast_period)
bpcl_close_forecast = bpcl_close_forecast.reshape(-1, 1)
bpcl_close_forecast = scaler.inverse_transform(bpcl_close_forecast)

bpcl_open_final_model = sm.tsa.ARIMA(
    bpcl_y_open_scaled,
    order=bpcl_open_best_order
)
bpcl_open_final_model = bpcl_open_final_model.fit()
bpcl_open_forecast = bpcl_open_final_model.forecast(steps=forecast_period)
bpcl_open_forecast = bpcl_open_forecast.reshape(-1, 1)
bpcl_open_forecast = scaler.inverse_transform(bpcl_open_forecast)

bpcl_high_final_model = sm.tsa.ARIMA(
    bpcl_y_high_scaled,
    order=bpcl_high_best_order
)
bpcl_high_final_model = bpcl_high_final_model.fit()
bpcl_high_forecast = bpcl_high_final_model.forecast(steps=forecast_period)
bpcl_high_forecast = bpcl_high_forecast.reshape(-1, 1)
bpcl_high_forecast = scaler.inverse_transform(bpcl_high_forecast)

bpcl_low_final_model = sm.tsa.ARIMA(
    bpcl_y_low_scaled,
    order=bpcl_low_best_order
)
bpcl_low_final_model = bpcl_low_final_model.fit()
bpcl_low_forecast = bpcl_low_final_model.forecast(steps=forecast_period)
bpcl_low_forecast = bpcl_low_forecast.reshape(-1, 1)
bpcl_low_forecast = scaler.inverse_transform(bpcl_low_forecast)

print("Close Forecasts:", bpcl_close_forecast)
print("Open Forecasts:", bpcl_open_forecast)
print("High Forecasts:", bpcl_high_forecast)
print("Low Forecasts:", bpcl_low_forecast)


# In[162]:


bpcl_tail_50_data = bpcl.tail(forecast_periods)

bpcl_actual_close_prices = bpcl_tail_50_data['Close'].values
bpcl_actual_open_prices = bpcl_tail_50_data['Open'].values
bpcl_actual_high_prices = bpcl_tail_50_data['High'].values
bpcl_actual_low_prices = bpcl_tail_50_data['Low'].values

bpcl_forecast_close = bpcl_close_final_model.forecast(steps=forecast_periods)
bpcl_forecast_close = bpcl_forecast_close.reshape(-1, 1)
bpcl_forecast_close = scaler.inverse_transform(bpcl_forecast_close)

bpcl_forecast_open = bpcl_open_final_model.forecast(steps=forecast_periods)
bpcl_forecast_open = bpcl_forecast_open.reshape(-1, 1)
bpcl_forecast_open = scaler.inverse_transform(bpcl_forecast_open)

bpcl_forecast_high = bpcl_high_final_model.forecast(steps=forecast_periods)
bpcl_forecast_high = bpcl_forecast_high.reshape(-1, 1)
bpcl_forecast_high = scaler.inverse_transform(bpcl_forecast_high)

bpcl_forecast_low = bpcl_low_final_model.forecast(steps=forecast_periods)
bpcl_forecast_low = bpcl_forecast_low.reshape(-1, 1)
bpcl_forecast_low = scaler.inverse_transform(bpcl_forecast_low)

bpcl_close_mae = mean_absolute_error(bpcl_actual_close_prices, bpcl_forecast_close)
bpcl_close_mse = mean_squared_error(bpcl_actual_close_prices, bpcl_forecast_close)
bpcl_close_rmse = np.sqrt(bpcl_close_mse)

bpcl_open_mae = mean_absolute_error(bpcl_actual_open_prices, bpcl_forecast_open)
bpcl_open_mse = mean_squared_error(bpcl_actual_open_prices, bpcl_forecast_open)
bpcl_open_rmse = np.sqrt(bpcl_open_mse)

bpcl_high_mae = mean_absolute_error(bpcl_actual_high_prices, bpcl_forecast_high)
bpcl_high_mse = mean_squared_error(bpcl_actual_high_prices, bpcl_forecast_high)
bpcl_high_rmse = np.sqrt(bpcl_high_mse)

bpcl_low_mae = mean_absolute_error(bpcl_actual_low_prices, bpcl_forecast_low)
bpcl_low_mse = mean_squared_error(bpcl_actual_low_prices, bpcl_forecast_low)
bpcl_low_rmse = np.sqrt(bpcl_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

bpcl_close_mape = mean_absolute_percentage_error(bpcl_actual_close_prices, bpcl_forecast_close)
bpcl_open_mape = mean_absolute_percentage_error(bpcl_actual_open_prices, bpcl_forecast_open)
bpcl_high_mape = mean_absolute_percentage_error(bpcl_actual_high_prices, bpcl_forecast_high)
bpcl_low_mape = mean_absolute_percentage_error(bpcl_actual_low_prices, bpcl_forecast_low)


print("Close Forecasts:", bpcl_forecast_close)
print(f"Close Mean Absolute Error (MAE): {bpcl_close_mae}")
print(f"Close Mean Squared Error (MSE): {bpcl_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {bpcl_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {bpcl_close_mape}%")

print("Open Forecasts:", bpcl_forecast_open)
print(f"Open Mean Absolute Error (MAE): {bpcl_open_mae}")
print(f"Open Mean Squared Error (MSE): {bpcl_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {bpcl_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {bpcl_open_mape}%")

print("High Forecasts:", bpcl_forecast_high)
print(f"High Mean Absolute Error (MAE): {bpcl_high_mae}")
print(f"High Mean Squared Error (MSE): {bpcl_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {bpcl_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {bpcl_high_mape}%")

print("Low Forecasts:", bpcl_forecast_low)
print(f"Low Mean Absolute Error (MAE): {bpcl_low_mae}")
print(f"Low Mean Squared Error (MSE): {bpcl_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {bpcl_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {bpcl_low_mape}%")


# In[163]:


cipla_y_close = cipla['Close'].values
cipla_y_open = cipla['Open'].values
cipla_y_high = cipla['High'].values
cipla_y_low = cipla['Low'].values

cipla_y_close_scaled = scaler.fit_transform(cipla_y_close.reshape(-1, 1))
cipla_y_open_scaled = scaler.fit_transform(cipla_y_open.reshape(-1, 1))
cipla_y_high_scaled = scaler.fit_transform(cipla_y_high.reshape(-1, 1))
cipla_y_low_scaled = scaler.fit_transform(cipla_y_low.reshape(-1, 1))

cipla_close_model = auto_arima(
    cipla_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

cipla_open_model = auto_arima(
    cipla_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

cipla_high_model = auto_arima(
    cipla_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

cipla_low_model = auto_arima(
    cipla_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

cipla_close_best_order = cipla_close_model.get_params()['order']
cipla_open_best_order = cipla_open_model.get_params()['order']
cipla_high_best_order = cipla_high_model.get_params()['order']
cipla_low_best_order = cipla_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {cipla_close_best_order}")
print(f"Best ARIMA Order for Open: {cipla_open_best_order}")
print(f"Best ARIMA Order for High: {cipla_high_best_order}")
print(f"Best ARIMA Order for Low: {cipla_low_best_order}")

cipla_close_final_model = sm.tsa.ARIMA(
    cipla_y_close_scaled,
    order=cipla_close_best_order
)
cipla_close_final_model = cipla_close_final_model.fit()
cipla_close_forecast = cipla_close_final_model.forecast(steps=forecast_period)
cipla_close_forecast = cipla_close_forecast.reshape(-1, 1)
cipla_close_forecast = scaler.inverse_transform(cipla_close_forecast)

cipla_open_final_model = sm.tsa.ARIMA(
    cipla_y_open_scaled,
    order=cipla_open_best_order
)
cipla_open_final_model = cipla_open_final_model.fit()
cipla_open_forecast = cipla_open_final_model.forecast(steps=forecast_period)
cipla_open_forecast = cipla_open_forecast.reshape(-1, 1)
cipla_open_forecast = scaler.inverse_transform(cipla_open_forecast)

cipla_high_final_model = sm.tsa.ARIMA(
    cipla_y_high_scaled,
    order=cipla_high_best_order
)
cipla_high_final_model = cipla_high_final_model.fit()
cipla_high_forecast = cipla_high_final_model.forecast(steps=forecast_period)
cipla_high_forecast = cipla_high_forecast.reshape(-1, 1)
cipla_high_forecast = scaler.inverse_transform(cipla_high_forecast)

cipla_low_final_model = sm.tsa.ARIMA(
    cipla_y_low_scaled,
    order=cipla_low_best_order
)
cipla_low_final_model = cipla_low_final_model.fit()
cipla_low_forecast = cipla_low_final_model.forecast(steps=forecast_period)
cipla_low_forecast = cipla_low_forecast.reshape(-1, 1)
cipla_low_forecast = scaler.inverse_transform(cipla_low_forecast)

print("Close Forecasts:", cipla_close_forecast)
print("Open Forecasts:", cipla_open_forecast)
print("High Forecasts:", cipla_high_forecast)
print("Low Forecasts:", cipla_low_forecast)


# In[164]:


cipla_tail_50_data = cipla.tail(forecast_periods)

cipla_actual_close_prices = cipla_tail_50_data['Close'].values
cipla_actual_open_prices = cipla_tail_50_data['Open'].values
cipla_actual_high_prices = cipla_tail_50_data['High'].values
cipla_actual_low_prices = cipla_tail_50_data['Low'].values

cipla_forecast_close = cipla_close_final_model.forecast(steps=forecast_periods)
cipla_forecast_close = cipla_forecast_close.reshape(-1, 1)
cipla_forecast_close = scaler.inverse_transform(cipla_forecast_close)

cipla_forecast_open = cipla_open_final_model.forecast(steps=forecast_periods)
cipla_forecast_open = cipla_forecast_open.reshape(-1, 1)
cipla_forecast_open = scaler.inverse_transform(cipla_forecast_open)

cipla_forecast_high = cipla_high_final_model.forecast(steps=forecast_periods)
cipla_forecast_high = cipla_forecast_high.reshape(-1, 1)
cipla_forecast_high = scaler.inverse_transform(cipla_forecast_high)

cipla_forecast_low = cipla_low_final_model.forecast(steps=forecast_periods)
cipla_forecast_low = cipla_forecast_low.reshape(-1, 1)
cipla_forecast_low = scaler.inverse_transform(cipla_forecast_low)

cipla_close_mae = mean_absolute_error(cipla_actual_close_prices, cipla_forecast_close)
cipla_close_mse = mean_squared_error(cipla_actual_close_prices, cipla_forecast_close)
cipla_close_rmse = np.sqrt(cipla_close_mse)

cipla_open_mae = mean_absolute_error(cipla_actual_open_prices, cipla_forecast_open)
cipla_open_mse = mean_squared_error(cipla_actual_open_prices, cipla_forecast_open)
cipla_open_rmse = np.sqrt(cipla_open_mse)

cipla_high_mae = mean_absolute_error(cipla_actual_high_prices, cipla_forecast_high)
cipla_high_mse = mean_squared_error(cipla_actual_high_prices, cipla_forecast_high)
cipla_high_rmse = np.sqrt(cipla_high_mse)

cipla_low_mae = mean_absolute_error(cipla_actual_low_prices, cipla_forecast_low)
cipla_low_mse = mean_squared_error(cipla_actual_low_prices, cipla_forecast_low)
cipla_low_rmse = np.sqrt(cipla_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

cipla_close_mape = mean_absolute_percentage_error(cipla_actual_close_prices, cipla_forecast_close)
cipla_open_mape = mean_absolute_percentage_error(cipla_actual_open_prices, cipla_forecast_open)
cipla_high_mape = mean_absolute_percentage_error(cipla_actual_high_prices, cipla_forecast_high)
cipla_low_mape = mean_absolute_percentage_error(cipla_actual_low_prices, cipla_forecast_low)


print("Close Forecasts:", cipla_forecast_close)
print(f"Close Mean Absolute Error (MAE): {cipla_close_mae}")
print(f"Close Mean Squared Error (MSE): {cipla_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {cipla_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {cipla_close_mape}%")

print("Open Forecasts:", cipla_forecast_open)
print(f"Open Mean Absolute Error (MAE): {cipla_open_mae}")
print(f"Open Mean Squared Error (MSE): {cipla_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {cipla_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {cipla_open_mape}%")

print("High Forecasts:", cipla_forecast_high)
print(f"High Mean Absolute Error (MAE): {cipla_high_mae}")
print(f"High Mean Squared Error (MSE): {cipla_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {cipla_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {cipla_high_mape}%")

print("Low Forecasts:", cipla_forecast_low)
print(f"Low Mean Absolute Error (MAE): {cipla_low_mae}")
print(f"Low Mean Squared Error (MSE): {cipla_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {cipla_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {cipla_low_mape}")


# In[165]:


coalindia_y_close = coalindia['Close'].values
coalindia_y_open = coalindia['Open'].values
coalindia_y_high = coalindia['High'].values
coalindia_y_low = coalindia['Low'].values

coalindia_y_close_scaled = scaler.fit_transform(coalindia_y_close.reshape(-1, 1))
coalindia_y_open_scaled = scaler.fit_transform(coalindia_y_open.reshape(-1, 1))
coalindia_y_high_scaled = scaler.fit_transform(coalindia_y_high.reshape(-1, 1))
coalindia_y_low_scaled = scaler.fit_transform(coalindia_y_low.reshape(-1, 1))

coalindia_close_model = auto_arima(
    coalindia_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

coalindia_open_model = auto_arima(
    coalindia_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

coalindia_high_model = auto_arima(
    coalindia_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

coalindia_low_model = auto_arima(
    coalindia_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

coalindia_close_best_order = coalindia_close_model.get_params()['order']
coalindia_open_best_order = coalindia_open_model.get_params()['order']
coalindia_high_best_order = coalindia_high_model.get_params()['order']
coalindia_low_best_order = coalindia_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {coalindia_close_best_order}")
print(f"Best ARIMA Order for Open: {coalindia_open_best_order}")
print(f"Best ARIMA Order for High: {coalindia_high_best_order}")
print(f"Best ARIMA Order for Low: {coalindia_low_best_order}")

coalindia_close_final_model = sm.tsa.ARIMA(
    coalindia_y_close_scaled,
    order=coalindia_close_best_order
)
coalindia_close_final_model = coalindia_close_final_model.fit()
coalindia_close_forecast = coalindia_close_final_model.forecast(steps=forecast_period)
coalindia_close_forecast = coalindia_close_forecast.reshape(-1, 1)
coalindia_close_forecast = scaler.inverse_transform(coalindia_close_forecast)

coalindia_open_final_model = sm.tsa.ARIMA(
    coalindia_y_open_scaled,
    order=coalindia_open_best_order
)
coalindia_open_final_model = coalindia_open_final_model.fit()
coalindia_open_forecast = coalindia_open_final_model.forecast(steps=forecast_period)
coalindia_open_forecast = coalindia_open_forecast.reshape(-1, 1)
coalindia_open_forecast = scaler.inverse_transform(coalindia_open_forecast)

coalindia_high_final_model = sm.tsa.ARIMA(
    coalindia_y_high_scaled,
    order=coalindia_high_best_order
)
coalindia_high_final_model = coalindia_high_final_model.fit()
coalindia_high_forecast = coalindia_high_final_model.forecast(steps=forecast_period)
coalindia_high_forecast = coalindia_high_forecast.reshape(-1, 1)
coalindia_high_forecast = scaler.inverse_transform(coalindia_high_forecast)

coalindia_low_final_model = sm.tsa.ARIMA(
    coalindia_y_low_scaled,
    order=coalindia_low_best_order
)
coalindia_low_final_model = coalindia_low_final_model.fit()
coalindia_low_forecast = coalindia_low_final_model.forecast(steps=forecast_period)
coalindia_low_forecast = coalindia_low_forecast.reshape(-1, 1)
coalindia_low_forecast = scaler.inverse_transform(coalindia_low_forecast)

print("Close Forecasts:", coalindia_close_forecast)
print("Open Forecasts:", coalindia_open_forecast)
print("High Forecasts:", coalindia_high_forecast)
print("Low Forecasts:", coalindia_low_forecast)


# In[166]:


coalindia_tail_50_data = coalindia.tail(forecast_periods)

coalindia_actual_close_prices = coalindia_tail_50_data['Close'].values
coalindia_actual_open_prices = coalindia_tail_50_data['Open'].values
coalindia_actual_high_prices = coalindia_tail_50_data['High'].values
coalindia_actual_low_prices = coalindia_tail_50_data['Low'].values

coalindia_forecast_close = coalindia_close_final_model.forecast(steps=forecast_periods)
coalindia_forecast_close = coalindia_forecast_close.reshape(-1, 1)
coalindia_forecast_close = scaler.inverse_transform(coalindia_forecast_close)

coalindia_forecast_open = coalindia_open_final_model.forecast(steps=forecast_periods)
coalindia_forecast_open = coalindia_forecast_open.reshape(-1, 1)
coalindia_forecast_open = scaler.inverse_transform(coalindia_forecast_open)

coalindia_forecast_high = coalindia_high_final_model.forecast(steps=forecast_periods)
coalindia_forecast_high = coalindia_forecast_high.reshape(-1, 1)
coalindia_forecast_high = scaler.inverse_transform(coalindia_forecast_high)

coalindia_forecast_low = coalindia_low_final_model.forecast(steps=forecast_periods)
coalindia_forecast_low = coalindia_forecast_low.reshape(-1, 1)
coalindia_forecast_low = scaler.inverse_transform(coalindia_forecast_low)

coalindia_close_mae = mean_absolute_error(coalindia_actual_close_prices, coalindia_forecast_close)
coalindia_close_mse = mean_squared_error(coalindia_actual_close_prices, coalindia_forecast_close)
coalindia_close_rmse = np.sqrt(coalindia_close_mse)

coalindia_open_mae = mean_absolute_error(coalindia_actual_open_prices, coalindia_forecast_open)
coalindia_open_mse = mean_squared_error(coalindia_actual_open_prices, coalindia_forecast_open)
coalindia_open_rmse = np.sqrt(coalindia_open_mse)

coalindia_high_mae = mean_absolute_error(coalindia_actual_high_prices, coalindia_forecast_high)
coalindia_high_mse = mean_squared_error(coalindia_actual_high_prices, coalindia_forecast_high)
coalindia_high_rmse = np.sqrt(coalindia_high_mse)

coalindia_low_mae = mean_absolute_error(coalindia_actual_low_prices, coalindia_forecast_low)
coalindia_low_mse = mean_squared_error(coalindia_actual_low_prices, coalindia_forecast_low)
coalindia_low_rmse = np.sqrt(coalindia_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

coalindia_close_mape = mean_absolute_percentage_error(coalindia_actual_close_prices, coalindia_forecast_close)
coalindia_open_mape = mean_absolute_percentage_error(coalindia_actual_open_prices, coalindia_forecast_open)
coalindia_high_mape = mean_absolute_percentage_error(coalindia_actual_high_prices, coalindia_forecast_high)
coalindia_low_mape = mean_absolute_percentage_error(coalindia_actual_low_prices, coalindia_forecast_low)


print("Close Forecasts:", coalindia_forecast_close)
print(f"Close Mean Absolute Error (MAE): {coalindia_close_mae}")
print(f"Close Mean Squared Error (MSE): {coalindia_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {coalindia_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {coalindia_close_mape}%")

print("Open Forecasts:", coalindia_forecast_open)
print(f"Open Mean Absolute Error (MAE): {coalindia_open_mae}")
print(f"Open Mean Squared Error (MSE): {coalindia_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {coalindia_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {coalindia_open_mape}%")

print("High Forecasts:", coalindia_forecast_high)
print(f"High Mean Absolute Error (MAE): {coalindia_high_mae}")
print(f"High Mean Squared Error (MSE): {coalindia_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {coalindia_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {coalindia_high_mape}%")

print("Low Forecasts:", coalindia_forecast_low)
print(f"Low Mean Absolute Error (MAE): {coalindia_low_mae}")
print(f"Low Mean Squared Error (MSE): {coalindia_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {coalindia_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {coalindia_low_mape}")


# In[167]:


drreddy_y_close = drreddy['Close'].values
drreddy_y_open = drreddy['Open'].values
drreddy_y_high = drreddy['High'].values
drreddy_y_low = drreddy['Low'].values

drreddy_y_close_scaled = scaler.fit_transform(drreddy_y_close.reshape(-1, 1))
drreddy_y_open_scaled = scaler.fit_transform(drreddy_y_open.reshape(-1, 1))
drreddy_y_high_scaled = scaler.fit_transform(drreddy_y_high.reshape(-1, 1))
drreddy_y_low_scaled = scaler.fit_transform(drreddy_y_low.reshape(-1, 1))

drreddy_close_model = auto_arima(
    drreddy_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

drreddy_open_model = auto_arima(
    drreddy_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

drreddy_high_model = auto_arima(
    drreddy_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

drreddy_low_model = auto_arima(
    drreddy_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

drreddy_close_best_order = drreddy_close_model.get_params()['order']
drreddy_open_best_order = drreddy_open_model.get_params()['order']
drreddy_high_best_order = drreddy_high_model.get_params()['order']
drreddy_low_best_order = drreddy_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {drreddy_close_best_order}")
print(f"Best ARIMA Order for Open: {drreddy_open_best_order}")
print(f"Best ARIMA Order for High: {drreddy_high_best_order}")
print(f"Best ARIMA Order for Low: {drreddy_low_best_order}")

drreddy_close_final_model = sm.tsa.ARIMA(
    drreddy_y_close_scaled,
    order=drreddy_close_best_order
)
drreddy_close_final_model = drreddy_close_final_model.fit()
drreddy_close_forecast = drreddy_close_final_model.forecast(steps=forecast_period)
drreddy_close_forecast = drreddy_close_forecast.reshape(-1, 1)
drreddy_close_forecast = scaler.inverse_transform(drreddy_close_forecast)

drreddy_open_final_model = sm.tsa.ARIMA(
    drreddy_y_open_scaled,
    order=drreddy_open_best_order
)
drreddy_open_final_model = drreddy_open_final_model.fit()
drreddy_open_forecast = drreddy_open_final_model.forecast(steps=forecast_period)
drreddy_open_forecast = drreddy_open_forecast.reshape(-1, 1)
drreddy_open_forecast = scaler.inverse_transform(drreddy_open_forecast)

drreddy_high_final_model = sm.tsa.ARIMA(
    drreddy_y_high_scaled,
    order=drreddy_high_best_order
)
drreddy_high_final_model = drreddy_high_final_model.fit()
drreddy_high_forecast = drreddy_high_final_model.forecast(steps=forecast_period)
drreddy_high_forecast = drreddy_high_forecast.reshape(-1, 1)
drreddy_high_forecast = scaler.inverse_transform(drreddy_high_forecast)

drreddy_low_final_model = sm.tsa.ARIMA(
    drreddy_y_low_scaled,
    order=drreddy_low_best_order
)
drreddy_low_final_model = drreddy_low_final_model.fit()
drreddy_low_forecast = drreddy_low_final_model.forecast(steps=forecast_period)
drreddy_low_forecast = drreddy_low_forecast.reshape(-1, 1)
drreddy_low_forecast = scaler.inverse_transform(drreddy_low_forecast)

print("Close Forecasts:", drreddy_close_forecast)
print("Open Forecasts:", drreddy_open_forecast)
print("High Forecasts:", drreddy_high_forecast)
print("Low Forecasts:", drreddy_low_forecast)


# In[168]:


drreddy_tail_50_data = drreddy.tail(forecast_periods)

drreddy_actual_close_prices = drreddy_tail_50_data['Close'].values
drreddy_actual_open_prices = drreddy_tail_50_data['Open'].values
drreddy_actual_high_prices = drreddy_tail_50_data['High'].values
drreddy_actual_low_prices = drreddy_tail_50_data['Low'].values

drreddy_forecast_close = drreddy_close_final_model.forecast(steps=forecast_periods)
drreddy_forecast_close = drreddy_forecast_close.reshape(-1, 1)
drreddy_forecast_close = scaler.inverse_transform(drreddy_forecast_close)

drreddy_forecast_open = drreddy_open_final_model.forecast(steps=forecast_periods)
drreddy_forecast_open = drreddy_forecast_open.reshape(-1, 1)
drreddy_forecast_open = scaler.inverse_transform(drreddy_forecast_open)

drreddy_forecast_high = drreddy_high_final_model.forecast(steps=forecast_periods)
drreddy_forecast_high = drreddy_forecast_high.reshape(-1, 1)
drreddy_forecast_high = scaler.inverse_transform(drreddy_forecast_high)

drreddy_forecast_low = drreddy_low_final_model.forecast(steps=forecast_periods)
drreddy_forecast_low = drreddy_forecast_low.reshape(-1, 1)
drreddy_forecast_low = scaler.inverse_transform(drreddy_forecast_low)

drreddy_close_mae = mean_absolute_error(drreddy_actual_close_prices, drreddy_forecast_close)
drreddy_close_mse = mean_squared_error(drreddy_actual_close_prices, drreddy_forecast_close)
drreddy_close_rmse = np.sqrt(drreddy_close_mse)

drreddy_open_mae = mean_absolute_error(drreddy_actual_open_prices, drreddy_forecast_open)
drreddy_open_mse = mean_squared_error(drreddy_actual_open_prices, drreddy_forecast_open)
drreddy_open_rmse = np.sqrt(drreddy_open_mse)

drreddy_high_mae = mean_absolute_error(drreddy_actual_high_prices, drreddy_forecast_high)
drreddy_high_mse = mean_squared_error(drreddy_actual_high_prices, drreddy_forecast_high)
drreddy_high_rmse = np.sqrt(drreddy_high_mse)

drreddy_low_mae = mean_absolute_error(drreddy_actual_low_prices, drreddy_forecast_low)
drreddy_low_mse = mean_squared_error(drreddy_actual_low_prices, drreddy_forecast_low)
drreddy_low_rmse = np.sqrt(drreddy_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

drreddy_close_mape = mean_absolute_percentage_error(drreddy_actual_close_prices, drreddy_forecast_close)
drreddy_open_mape = mean_absolute_percentage_error(drreddy_actual_open_prices, drreddy_forecast_open)
drreddy_high_mape = mean_absolute_percentage_error(drreddy_actual_high_prices, drreddy_forecast_high)
drreddy_low_mape = mean_absolute_percentage_error(drreddy_actual_low_prices, drreddy_forecast_low)


print("Close Forecasts:", drreddy_forecast_close)
print(f"Close Mean Absolute Error (MAE): {drreddy_close_mae}")
print(f"Close Mean Squared Error (MSE): {drreddy_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {drreddy_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {drreddy_close_mape}%")

print("Open Forecasts:", drreddy_forecast_open)
print(f"Open Mean Absolute Error (MAE): {drreddy_open_mae}")
print(f"Open Mean Squared Error (MSE): {drreddy_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {drreddy_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {drreddy_open_mape}%")

print("High Forecasts:", drreddy_forecast_high)
print(f"High Mean Absolute Error (MAE): {drreddy_high_mae}")
print(f"High Mean Squared Error (MSE): {drreddy_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {drreddy_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {drreddy_high_mape}%")

print("Low Forecasts:", drreddy_forecast_low)
print(f"Low Mean Absolute Error (MAE): {drreddy_low_mae}")
print(f"Low Mean Squared Error (MSE): {drreddy_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {drreddy_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {drreddy_low_mape}")


# In[169]:


eichermot_y_close = eichermot['Close'].values
eichermot_y_open = eichermot['Open'].values
eichermot_y_high = eichermot['High'].values
eichermot_y_low = eichermot['Low'].values

eichermot_y_close_scaled = scaler.fit_transform(eichermot_y_close.reshape(-1, 1))
eichermot_y_open_scaled = scaler.fit_transform(eichermot_y_open.reshape(-1, 1))
eichermot_y_high_scaled = scaler.fit_transform(eichermot_y_high.reshape(-1, 1))
eichermot_y_low_scaled = scaler.fit_transform(eichermot_y_low.reshape(-1, 1))

eichermot_close_model = auto_arima(
    eichermot_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

eichermot_open_model = auto_arima(
    eichermot_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

eichermot_high_model = auto_arima(
    eichermot_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

eichermot_low_model = auto_arima(
    eichermot_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

eichermot_close_best_order = eichermot_close_model.get_params()['order']
eichermot_open_best_order = eichermot_open_model.get_params()['order']
eichermot_high_best_order = eichermot_high_model.get_params()['order']
eichermot_low_best_order = eichermot_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {eichermot_close_best_order}")
print(f"Best ARIMA Order for Open: {eichermot_open_best_order}")
print(f"Best ARIMA Order for High: {eichermot_high_best_order}")
print(f"Best ARIMA Order for Low: {eichermot_low_best_order}")

eichermot_close_final_model = sm.tsa.ARIMA(
    eichermot_y_close_scaled,
    order=eichermot_close_best_order
)
eichermot_close_final_model = eichermot_close_final_model.fit()
eichermot_close_forecast = eichermot_close_final_model.forecast(steps=forecast_period)
eichermot_close_forecast = eichermot_close_forecast.reshape(-1, 1)
eichermot_close_forecast = scaler.inverse_transform(eichermot_close_forecast)

eichermot_open_final_model = sm.tsa.ARIMA(
    eichermot_y_open_scaled,
    order=eichermot_open_best_order
)
eichermot_open_final_model = eichermot_open_final_model.fit()
eichermot_open_forecast = eichermot_open_final_model.forecast(steps=forecast_period)
eichermot_open_forecast = eichermot_open_forecast.reshape(-1, 1)
eichermot_open_forecast = scaler.inverse_transform(eichermot_open_forecast)

eichermot_high_final_model = sm.tsa.ARIMA(
    eichermot_y_high_scaled,
    order=eichermot_high_best_order
)
eichermot_high_final_model = eichermot_high_final_model.fit()
eichermot_high_forecast = eichermot_high_final_model.forecast(steps=forecast_period)
eichermot_high_forecast = eichermot_high_forecast.reshape(-1, 1)
eichermot_high_forecast = scaler.inverse_transform(eichermot_high_forecast)

eichermot_low_final_model = sm.tsa.ARIMA(
    eichermot_y_low_scaled,
    order=eichermot_low_best_order
)
eichermot_low_final_model = eichermot_low_final_model.fit()
eichermot_low_forecast = eichermot_low_final_model.forecast(steps=forecast_period)
eichermot_low_forecast = eichermot_low_forecast.reshape(-1, 1)
eichermot_low_forecast = scaler.inverse_transform(eichermot_low_forecast)

print("Close Forecasts:", eichermot_close_forecast)
print("Open Forecasts:", eichermot_open_forecast)
print("High Forecasts:", eichermot_high_forecast)
print("Low Forecasts:", eichermot_low_forecast)


# In[170]:


eichermot_tail_50_data = eichermot.tail(forecast_periods)

eichermot_actual_close_prices = eichermot_tail_50_data['Close'].values
eichermot_actual_open_prices = eichermot_tail_50_data['Open'].values
eichermot_actual_high_prices = eichermot_tail_50_data['High'].values
eichermot_actual_low_prices = eichermot_tail_50_data['Low'].values

eichermot_forecast_close = eichermot_close_final_model.forecast(steps=forecast_periods)
eichermot_forecast_close = eichermot_forecast_close.reshape(-1, 1)
eichermot_forecast_close = scaler.inverse_transform(eichermot_forecast_close)

eichermot_forecast_open = eichermot_open_final_model.forecast(steps=forecast_periods)
eichermot_forecast_open = eichermot_forecast_open.reshape(-1, 1)
eichermot_forecast_open = scaler.inverse_transform(eichermot_forecast_open)

eichermot_forecast_high = eichermot_high_final_model.forecast(steps=forecast_periods)
eichermot_forecast_high = eichermot_forecast_high.reshape(-1, 1)
eichermot_forecast_high = scaler.inverse_transform(eichermot_forecast_high)

eichermot_forecast_low = eichermot_low_final_model.forecast(steps=forecast_periods)
eichermot_forecast_low = eichermot_forecast_low.reshape(-1, 1)
eichermot_forecast_low = scaler.inverse_transform(eichermot_forecast_low)

eichermot_close_mae = mean_absolute_error(eichermot_actual_close_prices, eichermot_forecast_close)
eichermot_close_mse = mean_squared_error(eichermot_actual_close_prices, eichermot_forecast_close)
eichermot_close_rmse = np.sqrt(eichermot_close_mse)

eichermot_open_mae = mean_absolute_error(eichermot_actual_open_prices, eichermot_forecast_open)
eichermot_open_mse = mean_squared_error(eichermot_actual_open_prices, eichermot_forecast_open)
eichermot_open_rmse = np.sqrt(eichermot_open_mse)

eichermot_high_mae = mean_absolute_error(eichermot_actual_high_prices, eichermot_forecast_high)
eichermot_high_mse = mean_squared_error(eichermot_actual_high_prices, eichermot_forecast_high)
eichermot_high_rmse = np.sqrt(eichermot_high_mse)

eichermot_low_mae = mean_absolute_error(eichermot_actual_low_prices, eichermot_forecast_low)
eichermot_low_mse = mean_squared_error(eichermot_actual_low_prices, eichermot_forecast_low)
eichermot_low_rmse = np.sqrt(eichermot_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

eichermot_close_mape = mean_absolute_percentage_error(eichermot_actual_close_prices, eichermot_forecast_close)
eichermot_open_mape = mean_absolute_percentage_error(eichermot_actual_open_prices, eichermot_forecast_open)
eichermot_high_mape = mean_absolute_percentage_error(eichermot_actual_high_prices, eichermot_forecast_high)
eichermot_low_mape = mean_absolute_percentage_error(eichermot_actual_low_prices, eichermot_forecast_low)


print("Close Forecasts:", eichermot_forecast_close)
print(f"Close Mean Absolute Error (MAE): {eichermot_close_mae}")
print(f"Close Mean Squared Error (MSE): {eichermot_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {eichermot_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {eichermot_close_mape}%")

print("Open Forecasts:", eichermot_forecast_open)
print(f"Open Mean Absolute Error (MAE): {eichermot_open_mae}")
print(f"Open Mean Squared Error (MSE): {eichermot_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {eichermot_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {eichermot_open_mape}%")

print("High Forecasts:", eichermot_forecast_high)
print(f"High Mean Absolute Error (MAE): {eichermot_high_mae}")
print(f"High Mean Squared Error (MSE): {eichermot_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {eichermot_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {eichermot_high_mape}%")

print("Low Forecasts:", eichermot_forecast_low)
print(f"Low Mean Absolute Error (MAE): {eichermot_low_mae}")
print(f"Low Mean Squared Error (MSE): {eichermot_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {eichermot_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {eichermot_low_mape}")


# In[171]:


gail_y_close = gail['Close'].values
gail_y_open = gail['Open'].values
gail_y_high = gail['High'].values
gail_y_low = gail['Low'].values

gail_y_close_scaled = scaler.fit_transform(gail_y_close.reshape(-1, 1))
gail_y_open_scaled = scaler.fit_transform(gail_y_open.reshape(-1, 1))
gail_y_high_scaled = scaler.fit_transform(gail_y_high.reshape(-1, 1))
gail_y_low_scaled = scaler.fit_transform(gail_y_low.reshape(-1, 1))

gail_close_model = auto_arima(
    gail_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

gail_open_model = auto_arima(
    gail_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

gail_high_model = auto_arima(
    gail_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

gail_low_model = auto_arima(
    gail_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

gail_close_best_order = gail_close_model.get_params()['order']
gail_open_best_order = gail_open_model.get_params()['order']
gail_high_best_order = gail_high_model.get_params()['order']
gail_low_best_order = gail_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {gail_close_best_order}")
print(f"Best ARIMA Order for Open: {gail_open_best_order}")
print(f"Best ARIMA Order for High: {gail_high_best_order}")
print(f"Best ARIMA Order for Low: {gail_low_best_order}")

gail_close_final_model = sm.tsa.ARIMA(
    gail_y_close_scaled,
    order=gail_close_best_order
)
gail_close_final_model = gail_close_final_model.fit()
gail_close_forecast = gail_close_final_model.forecast(steps=forecast_period)
gail_close_forecast = gail_close_forecast.reshape(-1, 1)
gail_close_forecast = scaler.inverse_transform(gail_close_forecast)

gail_open_final_model = sm.tsa.ARIMA(
    gail_y_open_scaled,
    order=gail_open_best_order
)
gail_open_final_model = gail_open_final_model.fit()
gail_open_forecast = gail_open_final_model.forecast(steps=forecast_period)
gail_open_forecast = gail_open_forecast.reshape(-1, 1)
gail_open_forecast = scaler.inverse_transform(gail_open_forecast)

gail_high_final_model = sm.tsa.ARIMA(
    gail_y_high_scaled,
    order=gail_high_best_order
)
gail_high_final_model = gail_high_final_model.fit()
gail_high_forecast = gail_high_final_model.forecast(steps=forecast_period)
gail_high_forecast = gail_high_forecast.reshape(-1, 1)
gail_high_forecast = scaler.inverse_transform(gail_high_forecast)

gail_low_final_model = sm.tsa.ARIMA(
    gail_y_low_scaled,
    order=gail_low_best_order
)
gail_low_final_model = gail_low_final_model.fit()
gail_low_forecast = gail_low_final_model.forecast(steps=forecast_period)
gail_low_forecast = gail_low_forecast.reshape(-1, 1)
gail_low_forecast = scaler.inverse_transform(gail_low_forecast)

print("Close Forecasts:", gail_close_forecast)
print("Open Forecasts:", gail_open_forecast)
print("High Forecasts:", gail_high_forecast)
print("Low Forecasts:", gail_low_forecast)


# In[172]:


gail_tail_50_data = gail.tail(forecast_periods)

gail_actual_close_prices = gail_tail_50_data['Close'].values
gail_actual_open_prices = gail_tail_50_data['Open'].values
gail_actual_high_prices = gail_tail_50_data['High'].values
gail_actual_low_prices = gail_tail_50_data['Low'].values

gail_forecast_close = gail_close_final_model.forecast(steps=forecast_periods)
gail_forecast_close = gail_forecast_close.reshape(-1, 1)
gail_forecast_close = scaler.inverse_transform(gail_forecast_close)

gail_forecast_open = gail_open_final_model.forecast(steps=forecast_periods)
gail_forecast_open = gail_forecast_open.reshape(-1, 1)
gail_forecast_open = scaler.inverse_transform(gail_forecast_open)

gail_forecast_high = gail_high_final_model.forecast(steps=forecast_periods)
gail_forecast_high = gail_forecast_high.reshape(-1, 1)
gail_forecast_high = scaler.inverse_transform(gail_forecast_high)

gail_forecast_low = gail_low_final_model.forecast(steps=forecast_periods)
gail_forecast_low = gail_forecast_low.reshape(-1, 1)
gail_forecast_low = scaler.inverse_transform(gail_forecast_low)

gail_close_mae = mean_absolute_error(gail_actual_close_prices, gail_forecast_close)
gail_close_mse = mean_squared_error(gail_actual_close_prices, gail_forecast_close)
gail_close_rmse = np.sqrt(gail_close_mse)

gail_open_mae = mean_absolute_error(gail_actual_open_prices, gail_forecast_open)
gail_open_mse = mean_squared_error(gail_actual_open_prices, gail_forecast_open)
gail_open_rmse = np.sqrt(gail_open_mse)

gail_high_mae = mean_absolute_error(gail_actual_high_prices, gail_forecast_high)
gail_high_mse = mean_squared_error(gail_actual_high_prices, gail_forecast_high)
gail_high_rmse = np.sqrt(gail_high_mse)

gail_low_mae = mean_absolute_error(gail_actual_low_prices, gail_forecast_low)
gail_low_mse = mean_squared_error(gail_actual_low_prices, gail_forecast_low)
gail_low_rmse = np.sqrt(gail_low_mse)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

gail_close_mape = mean_absolute_percentage_error(gail_actual_close_prices, gail_forecast_close)
gail_open_mape = mean_absolute_percentage_error(gail_actual_open_prices, gail_forecast_open)
gail_high_mape = mean_absolute_percentage_error(gail_actual_high_prices, gail_forecast_high)
gail_low_mape = mean_absolute_percentage_error(gail_actual_low_prices, gail_forecast_low)


print("Close Forecasts:", gail_forecast_close)
print(f"Close Mean Absolute Error (MAE): {gail_close_mae}")
print(f"Close Mean Squared Error (MSE): {gail_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {gail_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {gail_close_mape}%")

print("Open Forecasts:", gail_forecast_open)
print(f"Open Mean Absolute Error (MAE): {gail_open_mae}")
print(f"Open Mean Squared Error (MSE): {gail_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {gail_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {gail_open_mape}%")

print("High Forecasts:", gail_forecast_high)
print(f"High Mean Absolute Error (MAE): {gail_high_mae}")
print(f"High Mean Squared Error (MSE): {gail_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {gail_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {gail_high_mape}%")

print("Low Forecasts:", gail_forecast_low)
print(f"Low Mean Absolute Error (MAE): {gail_low_mae}")
print(f"Low Mean Squared Error (MSE): {gail_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {gail_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {gail_low_mape}")


# In[173]:


grasim_y_close = grasim['Close'].values
grasim_y_open = grasim['Open'].values
grasim_y_high = grasim['High'].values
grasim_y_low = grasim['Low'].values

grasim_y_close_scaled = scaler.fit_transform(grasim_y_close.reshape(-1, 1))
grasim_y_open_scaled = scaler.fit_transform(grasim_y_open.reshape(-1, 1))
grasim_y_high_scaled = scaler.fit_transform(grasim_y_high.reshape(-1, 1))
grasim_y_low_scaled = scaler.fit_transform(grasim_y_low.reshape(-1, 1))

grasim_close_model = auto_arima(
    grasim_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

grasim_open_model = auto_arima(
    grasim_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

grasim_high_model = auto_arima(
    grasim_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

grasim_low_model = auto_arima(
    grasim_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

grasim_close_best_order = grasim_close_model.get_params()['order']
grasim_open_best_order = grasim_open_model.get_params()['order']
grasim_high_best_order = grasim_high_model.get_params()['order']
grasim_low_best_order = grasim_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {grasim_close_best_order}")
print(f"Best ARIMA Order for Open: {grasim_open_best_order}")
print(f"Best ARIMA Order for High: {grasim_high_best_order}")
print(f"Best ARIMA Order for Low: {grasim_low_best_order}")

grasim_close_final_model = sm.tsa.ARIMA(
    grasim_y_close_scaled,
    order=grasim_close_best_order
)
grasim_close_final_model = grasim_close_final_model.fit()
grasim_close_forecast = grasim_close_final_model.forecast(steps=forecast_period)
grasim_close_forecast = grasim_close_forecast.reshape(-1, 1)
grasim_close_forecast = scaler.inverse_transform(grasim_close_forecast)

grasim_open_final_model = sm.tsa.ARIMA(
    grasim_y_open_scaled,
    order=grasim_open_best_order
)
grasim_open_final_model = grasim_open_final_model.fit()
grasim_open_forecast = grasim_open_final_model.forecast(steps=forecast_period)
grasim_open_forecast = grasim_open_forecast.reshape(-1, 1)
grasim_open_forecast = scaler.inverse_transform(grasim_open_forecast)

grasim_high_final_model = sm.tsa.ARIMA(
    grasim_y_high_scaled,
    order=grasim_high_best_order
)
grasim_high_final_model = grasim_high_final_model.fit()
grasim_high_forecast = grasim_high_final_model.forecast(steps=forecast_period)
grasim_high_forecast = grasim_high_forecast.reshape(-1, 1)
grasim_high_forecast = scaler.inverse_transform(grasim_high_forecast)

grasim_low_final_model = sm.tsa.ARIMA(
    grasim_y_low_scaled,
    order=grasim_low_best_order
)
grasim_low_final_model = grasim_low_final_model.fit()
grasim_low_forecast = grasim_low_final_model.forecast(steps=forecast_period)
grasim_low_forecast = grasim_low_forecast.reshape(-1, 1)
grasim_low_forecast = scaler.inverse_transform(grasim_low_forecast)

print("Close Forecasts:", grasim_close_forecast)
print("Open Forecasts:", grasim_open_forecast)
print("High Forecasts:", grasim_high_forecast)
print("Low Forecasts:", grasim_low_forecast)


# In[174]:


grasim_tail_50_data = grasim.tail(forecast_periods)

grasim_actual_close_prices = grasim_tail_50_data['Close'].values
grasim_actual_open_prices = grasim_tail_50_data['Open'].values
grasim_actual_high_prices = grasim_tail_50_data['High'].values
grasim_actual_low_prices = grasim_tail_50_data['Low'].values

grasim_forecast_close = grasim_close_final_model.forecast(steps=forecast_periods)
grasim_forecast_close = grasim_forecast_close.reshape(-1, 1)
grasim_forecast_close = scaler.inverse_transform(grasim_forecast_close)

grasim_forecast_open = grasim_open_final_model.forecast(steps=forecast_periods)
grasim_forecast_open = grasim_forecast_open.reshape(-1, 1)
grasim_forecast_open = scaler.inverse_transform(grasim_forecast_open)

grasim_forecast_high = grasim_high_final_model.forecast(steps=forecast_periods)
grasim_forecast_high = grasim_forecast_high.reshape(-1, 1)
grasim_forecast_high = scaler.inverse_transform(grasim_forecast_high)

grasim_forecast_low = grasim_low_final_model.forecast(steps=forecast_periods)
grasim_forecast_low = grasim_forecast_low.reshape(-1, 1)
grasim_forecast_low = scaler.inverse_transform(grasim_forecast_low)

grasim_close_mae = mean_absolute_error(grasim_actual_close_prices, grasim_forecast_close)
grasim_close_mse = mean_squared_error(grasim_actual_close_prices, grasim_forecast_close)
grasim_close_rmse = np.sqrt(grasim_close_mse)

grasim_open_mae = mean_absolute_error(grasim_actual_open_prices, grasim_forecast_open)
grasim_open_mse = mean_squared_error(grasim_actual_open_prices, grasim_forecast_open)
grasim_open_rmse = np.sqrt(grasim_open_mse)

grasim_high_mae = mean_absolute_error(grasim_actual_high_prices, grasim_forecast_high)
grasim_high_mse = mean_squared_error(grasim_actual_high_prices, grasim_forecast_high)
grasim_high_rmse = np.sqrt(grasim_high_mse)

grasim_low_mae = mean_absolute_error(grasim_actual_low_prices, grasim_forecast_low)
grasim_low_mse = mean_squared_error(grasim_actual_low_prices, grasim_forecast_low)
grasim_low_rmse = np.sqrt(grasim_low_mse)

grasim_close_mape = mean_absolute_percentage_error(grasim_actual_close_prices, grasim_forecast_close)
grasim_open_mape = mean_absolute_percentage_error(grasim_actual_open_prices, grasim_forecast_open)
grasim_high_mape = mean_absolute_percentage_error(grasim_actual_high_prices, grasim_forecast_high)
grasim_low_mape = mean_absolute_percentage_error(grasim_actual_low_prices, grasim_forecast_low)

print("Close Forecasts:", grasim_forecast_close)
print(f"Close Mean Absolute Error (MAE): {grasim_close_mae}")
print(f"Close Mean Squared Error (MSE): {grasim_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {grasim_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {grasim_close_mape}%")

print("Open Forecasts:", grasim_forecast_open)
print(f"Open Mean Absolute Error (MAE): {grasim_open_mae}")
print(f"Open Mean Squared Error (MSE): {grasim_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {grasim_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {grasim_open_mape}%")

print("High Forecasts:", grasim_forecast_high)
print(f"High Mean Absolute Error (MAE): {grasim_high_mae}")
print(f"High Mean Squared Error (MSE): {grasim_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {grasim_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {grasim_high_mape}%")

print("Low Forecasts:", grasim_forecast_low)
print(f"Low Mean Absolute Error (MAE): {grasim_low_mae}")
print(f"Low Mean Squared Error (MSE): {grasim_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {grasim_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {grasim_low_mape}")


# In[175]:


hcltech_y_close = hcltech['Close'].values
hcltech_y_open = hcltech['Open'].values
hcltech_y_high = hcltech['High'].values
hcltech_y_low = hcltech['Low'].values

hcltech_y_close_scaled = scaler.fit_transform(hcltech_y_close.reshape(-1, 1))
hcltech_y_open_scaled = scaler.fit_transform(hcltech_y_open.reshape(-1, 1))
hcltech_y_high_scaled = scaler.fit_transform(hcltech_y_high.reshape(-1, 1))
hcltech_y_low_scaled = scaler.fit_transform(hcltech_y_low.reshape(-1, 1))

hcltech_close_model = auto_arima(
    hcltech_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hcltech_open_model = auto_arima(
    hcltech_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hcltech_high_model = auto_arima(
    hcltech_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hcltech_low_model = auto_arima(
    hcltech_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hcltech_close_best_order = hcltech_close_model.get_params()['order']
hcltech_open_best_order = hcltech_open_model.get_params()['order']
hcltech_high_best_order = hcltech_high_model.get_params()['order']
hcltech_low_best_order = hcltech_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {hcltech_close_best_order}")
print(f"Best ARIMA Order for Open: {hcltech_open_best_order}")
print(f"Best ARIMA Order for High: {hcltech_high_best_order}")
print(f"Best ARIMA Order for Low: {hcltech_low_best_order}")

hcltech_close_final_model = sm.tsa.ARIMA(
    hcltech_y_close_scaled,
    order=hcltech_close_best_order
)
hcltech_close_final_model = hcltech_close_final_model.fit()
hcltech_close_forecast = hcltech_close_final_model.forecast(steps=forecast_period)
hcltech_close_forecast = hcltech_close_forecast.reshape(-1, 1)
hcltech_close_forecast = scaler.inverse_transform(hcltech_close_forecast)

hcltech_open_final_model = sm.tsa.ARIMA(
    hcltech_y_open_scaled,
    order=hcltech_open_best_order
)
hcltech_open_final_model = hcltech_open_final_model.fit()
hcltech_open_forecast = hcltech_open_final_model.forecast(steps=forecast_period)
hcltech_open_forecast = hcltech_open_forecast.reshape(-1, 1)
hcltech_open_forecast = scaler.inverse_transform(hcltech_open_forecast)

hcltech_high_final_model = sm.tsa.ARIMA(
    hcltech_y_high_scaled,
    order=hcltech_high_best_order
)
hcltech_high_final_model = hcltech_high_final_model.fit()
hcltech_high_forecast = hcltech_high_final_model.forecast(steps=forecast_period)
hcltech_high_forecast = hcltech_high_forecast.reshape(-1, 1)
hcltech_high_forecast = scaler.inverse_transform(hcltech_high_forecast)

hcltech_low_final_model = sm.tsa.ARIMA(
    hcltech_y_low_scaled,
    order=hcltech_low_best_order
)
hcltech_low_final_model = hcltech_low_final_model.fit()
hcltech_low_forecast = hcltech_low_final_model.forecast(steps=forecast_period)
hcltech_low_forecast = hcltech_low_forecast.reshape(-1, 1)
hcltech_low_forecast = scaler.inverse_transform(hcltech_low_forecast)

print("Close Forecasts:", hcltech_close_forecast)
print("Open Forecasts:", hcltech_open_forecast)
print("High Forecasts:", hcltech_high_forecast)
print("Low Forecasts:", hcltech_low_forecast)


# In[176]:


hcltech_tail_50_data = hcltech.tail(forecast_periods)

hcltech_actual_close_prices = hcltech_tail_50_data['Close'].values
hcltech_actual_open_prices = hcltech_tail_50_data['Open'].values
hcltech_actual_high_prices = hcltech_tail_50_data['High'].values
hcltech_actual_low_prices = hcltech_tail_50_data['Low'].values

hcltech_forecast_close = hcltech_close_final_model.forecast(steps=forecast_periods)
hcltech_forecast_close = hcltech_forecast_close.reshape(-1, 1)
hcltech_forecast_close = scaler.inverse_transform(hcltech_forecast_close)

hcltech_forecast_open = hcltech_open_final_model.forecast(steps=forecast_periods)
hcltech_forecast_open = hcltech_forecast_open.reshape(-1, 1)
hcltech_forecast_open = scaler.inverse_transform(hcltech_forecast_open)

hcltech_forecast_high = hcltech_high_final_model.forecast(steps=forecast_periods)
hcltech_forecast_high = hcltech_forecast_high.reshape(-1, 1)
hcltech_forecast_high = scaler.inverse_transform(hcltech_forecast_high)

hcltech_forecast_low = hcltech_low_final_model.forecast(steps=forecast_periods)
hcltech_forecast_low = hcltech_forecast_low.reshape(-1, 1)
hcltech_forecast_low = scaler.inverse_transform(hcltech_forecast_low)

hcltech_close_mae = mean_absolute_error(hcltech_actual_close_prices, hcltech_forecast_close)
hcltech_close_mse = mean_squared_error(hcltech_actual_close_prices, hcltech_forecast_close)
hcltech_close_rmse = np.sqrt(hcltech_close_mse)

hcltech_open_mae = mean_absolute_error(hcltech_actual_open_prices, hcltech_forecast_open)
hcltech_open_mse = mean_squared_error(hcltech_actual_open_prices, hcltech_forecast_open)
hcltech_open_rmse = np.sqrt(hcltech_open_mse)

hcltech_high_mae = mean_absolute_error(hcltech_actual_high_prices, hcltech_forecast_high)
hcltech_high_mse = mean_squared_error(hcltech_actual_high_prices, hcltech_forecast_high)
hcltech_high_rmse = np.sqrt(hcltech_high_mse)

hcltech_low_mae = mean_absolute_error(hcltech_actual_low_prices, hcltech_forecast_low)
hcltech_low_mse = mean_squared_error(hcltech_actual_low_prices, hcltech_forecast_low)
hcltech_low_rmse = np.sqrt(hcltech_low_mse)

hcltech_close_mape = mean_absolute_percentage_error(hcltech_actual_close_prices, hcltech_forecast_close)
hcltech_open_mape = mean_absolute_percentage_error(hcltech_actual_open_prices, hcltech_forecast_open)
hcltech_high_mape = mean_absolute_percentage_error(hcltech_actual_high_prices, hcltech_forecast_high)
hcltech_low_mape = mean_absolute_percentage_error(hcltech_actual_low_prices, hcltech_forecast_low)

print("Close Forecasts:", hcltech_forecast_close)
print(f"Close Mean Absolute Error (MAE): {hcltech_close_mae}")
print(f"Close Mean Squared Error (MSE): {hcltech_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {hcltech_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {hcltech_close_mape}%")

print("Open Forecasts:", hcltech_forecast_open)
print(f"Open Mean Absolute Error (MAE): {hcltech_open_mae}")
print(f"Open Mean Squared Error (MSE): {hcltech_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {hcltech_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {hcltech_open_mape}%")

print("High Forecasts:", hcltech_forecast_high)
print(f"High Mean Absolute Error (MAE): {hcltech_high_mae}")
print(f"High Mean Squared Error (MSE): {hcltech_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {hcltech_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {hcltech_high_mape}%")

print("Low Forecasts:", hcltech_forecast_low)
print(f"Low Mean Absolute Error (MAE): {hcltech_low_mae}")
print(f"Low Mean Squared Error (MSE): {hcltech_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {hcltech_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {hcltech_low_mape}")


# In[179]:


hdfc_y_close = hdfc['Close'].values
hdfc_y_open = hdfc['Open'].values
hdfc_y_high = hdfc['High'].values
hdfc_y_low = hdfc['Low'].values

hdfc_y_close_scaled = scaler.fit_transform(hdfc_y_close.reshape(-1, 1))
hdfc_y_open_scaled = scaler.fit_transform(hdfc_y_open.reshape(-1, 1))
hdfc_y_high_scaled = scaler.fit_transform(hdfc_y_high.reshape(-1, 1))
hdfc_y_low_scaled = scaler.fit_transform(hdfc_y_low.reshape(-1, 1))

hdfc_close_model = auto_arima(
    hdfc_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hdfc_open_model = auto_arima(
    hdfc_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hdfc_high_model = auto_arima(
    hdfc_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hdfc_low_model = auto_arima(
    hdfc_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hdfc_close_best_order = hdfc_close_model.get_params()['order']
hdfc_open_best_order = hdfc_open_model.get_params()['order']
hdfc_high_best_order = hdfc_high_model.get_params()['order']
hdfc_low_best_order = hdfc_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {hdfc_close_best_order}")
print(f"Best ARIMA Order for Open: {hdfc_open_best_order}")
print(f"Best ARIMA Order for High: {hdfc_high_best_order}")
print(f"Best ARIMA Order for Low: {hdfc_low_best_order}")

hdfc_close_final_model = sm.tsa.ARIMA(
    hdfc_y_close_scaled,
    order=hdfc_close_best_order
)
hdfc_close_final_model = hdfc_close_final_model.fit()
hdfc_close_forecast = hdfc_close_final_model.forecast(steps=forecast_period)
hdfc_close_forecast = hdfc_close_forecast.reshape(-1, 1)
hdfc_close_forecast = scaler.inverse_transform(hdfc_close_forecast)

hdfc_open_final_model = sm.tsa.ARIMA(
    hdfc_y_open_scaled,
    order=hdfc_open_best_order
)
hdfc_open_final_model = hdfc_open_final_model.fit()
hdfc_open_forecast = hdfc_open_final_model.forecast(steps=forecast_period)
hdfc_open_forecast = hdfc_open_forecast.reshape(-1, 1)
hdfc_open_forecast = scaler.inverse_transform(hdfc_open_forecast)

hdfc_high_final_model = sm.tsa.ARIMA(
    hdfc_y_high_scaled,
    order=hdfc_high_best_order
)
hdfc_high_final_model = hdfc_high_final_model.fit()
hdfc_high_forecast = hdfc_high_final_model.forecast(steps=forecast_period)
hdfc_high_forecast = hdfc_high_forecast.reshape(-1, 1)
hdfc_high_forecast = scaler.inverse_transform(hdfc_high_forecast)

hdfc_low_final_model = sm.tsa.ARIMA(
    hdfc_y_low_scaled,
    order=hdfc_low_best_order
)
hdfc_low_final_model = hdfc_low_final_model.fit()
hdfc_low_forecast = hdfc_low_final_model.forecast(steps=forecast_period)
hdfc_low_forecast = hdfc_low_forecast.reshape(-1, 1)
hdfc_low_forecast = scaler.inverse_transform(hdfc_low_forecast)

print("Close Forecasts:", hdfc_close_forecast)
print("Open Forecasts:", hdfc_open_forecast)
print("High Forecasts:", hdfc_high_forecast)
print("Low Forecasts:", hdfc_low_forecast)


# In[180]:


hdfc_tail_50_data = hdfc.tail(forecast_periods)

hdfc_actual_close_prices = hdfc_tail_50_data['Close'].values
hdfc_actual_open_prices = hdfc_tail_50_data['Open'].values
hdfc_actual_high_prices = hdfc_tail_50_data['High'].values
hdfc_actual_low_prices = hdfc_tail_50_data['Low'].values

hdfc_forecast_close = hdfc_close_final_model.forecast(steps=forecast_periods)
hdfc_forecast_close = hdfc_forecast_close.reshape(-1, 1)
hdfc_forecast_close = scaler.inverse_transform(hdfc_forecast_close)

hdfc_forecast_open = hdfc_open_final_model.forecast(steps=forecast_periods)
hdfc_forecast_open = hdfc_forecast_open.reshape(-1, 1)
hdfc_forecast_open = scaler.inverse_transform(hdfc_forecast_open)

hdfc_forecast_high = hdfc_high_final_model.forecast(steps=forecast_periods)
hdfc_forecast_high = hdfc_forecast_high.reshape(-1, 1)
hdfc_forecast_high = scaler.inverse_transform(hdfc_forecast_high)

hdfc_forecast_low = hdfc_low_final_model.forecast(steps=forecast_periods)
hdfc_forecast_low = hdfc_forecast_low.reshape(-1, 1)
hdfc_forecast_low = scaler.inverse_transform(hdfc_forecast_low)

hdfc_close_mae = mean_absolute_error(hdfc_actual_close_prices, hdfc_forecast_close)
hdfc_close_mse = mean_squared_error(hdfc_actual_close_prices, hdfc_forecast_close)
hdfc_close_rmse = np.sqrt(hdfc_close_mse)

hdfc_open_mae = mean_absolute_error(hdfc_actual_open_prices, hdfc_forecast_open)
hdfc_open_mse = mean_squared_error(hdfc_actual_open_prices, hdfc_forecast_open)
hdfc_open_rmse = np.sqrt(hdfc_open_mse)

hdfc_high_mae = mean_absolute_error(hdfc_actual_high_prices, hdfc_forecast_high)
hdfc_high_mse = mean_squared_error(hdfc_actual_high_prices, hdfc_forecast_high)
hdfc_high_rmse = np.sqrt(hdfc_high_mse)

hdfc_low_mae = mean_absolute_error(hdfc_actual_low_prices, hdfc_forecast_low)
hdfc_low_mse = mean_squared_error(hdfc_actual_low_prices, hdfc_forecast_low)
hdfc_low_rmse = np.sqrt(hdfc_low_mse)

hdfc_close_mape = mean_absolute_percentage_error(hdfc_actual_close_prices, hdfc_forecast_close)
hdfc_open_mape = mean_absolute_percentage_error(hdfc_actual_open_prices, hdfc_forecast_open)
hdfc_high_mape = mean_absolute_percentage_error(hdfc_actual_high_prices, hdfc_forecast_high)
hdfc_low_mape = mean_absolute_percentage_error(hdfc_actual_low_prices, hdfc_forecast_low)

print("Close Forecasts:", hdfc_forecast_close)
print(f"Close Mean Absolute Error (MAE): {hdfc_close_mae}")
print(f"Close Mean Squared Error (MSE): {hdfc_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {hdfc_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {hdfc_close_mape}%")

print("Open Forecasts:", hdfc_forecast_open)
print(f"Open Mean Absolute Error (MAE): {hdfc_open_mae}")
print(f"Open Mean Squared Error (MSE): {hdfc_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {hdfc_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {hdfc_open_mape}%")

print("High Forecasts:", hdfc_forecast_high)
print(f"High Mean Absolute Error (MAE): {hdfc_high_mae}")
print(f"High Mean Squared Error (MSE): {hdfc_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {hdfc_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {hdfc_high_mape}%")

print("Low Forecasts:", hdfc_forecast_low)
print(f"Low Mean Absolute Error (MAE): {hdfc_low_mae}")
print(f"Low Mean Squared Error (MSE): {hdfc_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {hdfc_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {hdfc_low_mape}")


# In[181]:


hdfcbank_y_close = hdfcbank['Close'].values
hdfcbank_y_open = hdfcbank['Open'].values
hdfcbank_y_high = hdfcbank['High'].values
hdfcbank_y_low = hdfcbank['Low'].values

hdfcbank_y_close_scaled = scaler.fit_transform(hdfcbank_y_close.reshape(-1, 1))
hdfcbank_y_open_scaled = scaler.fit_transform(hdfcbank_y_open.reshape(-1, 1))
hdfcbank_y_high_scaled = scaler.fit_transform(hdfcbank_y_high.reshape(-1, 1))
hdfcbank_y_low_scaled = scaler.fit_transform(hdfcbank_y_low.reshape(-1, 1))

hdfcbank_close_model = auto_arima(
    hdfcbank_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hdfcbank_open_model = auto_arima(
    hdfcbank_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hdfcbank_high_model = auto_arima(
    hdfcbank_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hdfcbank_low_model = auto_arima(
    hdfcbank_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hdfcbank_close_best_order = hdfcbank_close_model.get_params()['order']
hdfcbank_open_best_order = hdfcbank_open_model.get_params()['order']
hdfcbank_high_best_order = hdfcbank_high_model.get_params()['order']
hdfcbank_low_best_order = hdfcbank_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {hdfcbank_close_best_order}")
print(f"Best ARIMA Order for Open: {hdfcbank_open_best_order}")
print(f"Best ARIMA Order for High: {hdfcbank_high_best_order}")
print(f"Best ARIMA Order for Low: {hdfcbank_low_best_order}")

hdfcbank_close_final_model = sm.tsa.ARIMA(
    hdfcbank_y_close_scaled,
    order=hdfcbank_close_best_order
)
hdfcbank_close_final_model = hdfcbank_close_final_model.fit()
hdfcbank_close_forecast = hdfcbank_close_final_model.forecast(steps=forecast_period)
hdfcbank_close_forecast = hdfcbank_close_forecast.reshape(-1, 1)
hdfcbank_close_forecast = scaler.inverse_transform(hdfcbank_close_forecast)

hdfcbank_open_final_model = sm.tsa.ARIMA(
    hdfcbank_y_open_scaled,
    order=hdfcbank_open_best_order
)
hdfcbank_open_final_model = hdfcbank_open_final_model.fit()
hdfcbank_open_forecast = hdfcbank_open_final_model.forecast(steps=forecast_period)
hdfcbank_open_forecast = hdfcbank_open_forecast.reshape(-1, 1)
hdfcbank_open_forecast = scaler.inverse_transform(hdfcbank_open_forecast)

hdfcbank_high_final_model = sm.tsa.ARIMA(
    hdfcbank_y_high_scaled,
    order=hdfcbank_high_best_order
)
hdfcbank_high_final_model = hdfcbank_high_final_model.fit()
hdfcbank_high_forecast = hdfcbank_high_final_model.forecast(steps=forecast_period)
hdfcbank_high_forecast = hdfcbank_high_forecast.reshape(-1, 1)
hdfcbank_high_forecast = scaler.inverse_transform(hdfcbank_high_forecast)

hdfcbank_low_final_model = sm.tsa.ARIMA(
    hdfcbank_y_low_scaled,
    order=hdfcbank_low_best_order
)
hdfcbank_low_final_model = hdfcbank_low_final_model.fit()
hdfcbank_low_forecast = hdfcbank_low_final_model.forecast(steps=forecast_period)
hdfcbank_low_forecast = hdfcbank_low_forecast.reshape(-1, 1)
hdfcbank_low_forecast = scaler.inverse_transform(hdfcbank_low_forecast)

print("Close Forecasts:", hdfcbank_close_forecast)
print("Open Forecasts:", hdfcbank_open_forecast)
print("High Forecasts:", hdfcbank_high_forecast)
print("Low Forecasts:", hdfcbank_low_forecast)


# In[182]:


hdfcbank_tail_50_data = hdfcbank.tail(forecast_periods)

hdfcbank_actual_close_prices = hdfcbank_tail_50_data['Close'].values
hdfcbank_actual_open_prices = hdfcbank_tail_50_data['Open'].values
hdfcbank_actual_high_prices = hdfcbank_tail_50_data['High'].values
hdfcbank_actual_low_prices = hdfcbank_tail_50_data['Low'].values

hdfcbank_forecast_close = hdfcbank_close_final_model.forecast(steps=forecast_periods)
hdfcbank_forecast_close = hdfcbank_forecast_close.reshape(-1, 1)
hdfcbank_forecast_close = scaler.inverse_transform(hdfcbank_forecast_close)

hdfcbank_forecast_open = hdfcbank_open_final_model.forecast(steps=forecast_periods)
hdfcbank_forecast_open = hdfcbank_forecast_open.reshape(-1, 1)
hdfcbank_forecast_open = scaler.inverse_transform(hdfcbank_forecast_open)

hdfcbank_forecast_high = hdfcbank_high_final_model.forecast(steps=forecast_periods)
hdfcbank_forecast_high = hdfcbank_forecast_high.reshape(-1, 1)
hdfcbank_forecast_high = scaler.inverse_transform(hdfcbank_forecast_high)

hdfcbank_forecast_low = hdfcbank_low_final_model.forecast(steps=forecast_periods)
hdfcbank_forecast_low = hdfcbank_forecast_low.reshape(-1, 1)
hdfcbank_forecast_low = scaler.inverse_transform(hdfcbank_forecast_low)

hdfcbank_close_mae = mean_absolute_error(hdfcbank_actual_close_prices, hdfcbank_forecast_close)
hdfcbank_close_mse = mean_squared_error(hdfcbank_actual_close_prices, hdfcbank_forecast_close)
hdfcbank_close_rmse = np.sqrt(hdfcbank_close_mse)

hdfcbank_open_mae = mean_absolute_error(hdfcbank_actual_open_prices, hdfcbank_forecast_open)
hdfcbank_open_mse = mean_squared_error(hdfcbank_actual_open_prices, hdfcbank_forecast_open)
hdfcbank_open_rmse = np.sqrt(hdfcbank_open_mse)

hdfcbank_high_mae = mean_absolute_error(hdfcbank_actual_high_prices, hdfcbank_forecast_high)
hdfcbank_high_mse = mean_squared_error(hdfcbank_actual_high_prices, hdfcbank_forecast_high)
hdfcbank_high_rmse = np.sqrt(hdfcbank_high_mse)

hdfcbank_low_mae = mean_absolute_error(hdfcbank_actual_low_prices, hdfcbank_forecast_low)
hdfcbank_low_mse = mean_squared_error(hdfcbank_actual_low_prices, hdfcbank_forecast_low)
hdfcbank_low_rmse = np.sqrt(hdfcbank_low_mse)

hdfcbank_close_mape = mean_absolute_percentage_error(hdfcbank_actual_close_prices, hdfcbank_forecast_close)
hdfcbank_open_mape = mean_absolute_percentage_error(hdfcbank_actual_open_prices, hdfcbank_forecast_open)
hdfcbank_high_mape = mean_absolute_percentage_error(hdfcbank_actual_high_prices, hdfcbank_forecast_high)
hdfcbank_low_mape = mean_absolute_percentage_error(hdfcbank_actual_low_prices, hdfcbank_forecast_low)

print("Close Forecasts:", hdfcbank_forecast_close)
print(f"Close Mean Absolute Error (MAE): {hdfcbank_close_mae}")
print(f"Close Mean Squared Error (MSE): {hdfcbank_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {hdfcbank_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {hdfcbank_close_mape}%")

print("Open Forecasts:", hdfcbank_forecast_open)
print(f"Open Mean Absolute Error (MAE): {hdfcbank_open_mae}")
print(f"Open Mean Squared Error (MSE): {hdfcbank_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {hdfcbank_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {hdfcbank_open_mape}%")

print("High Forecasts:", hdfcbank_forecast_high)
print(f"High Mean Absolute Error (MAE): {hdfcbank_high_mae}")
print(f"High Mean Squared Error (MSE): {hdfcbank_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {hdfcbank_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {hdfcbank_high_mape}%")

print("Low Forecasts:", hdfcbank_forecast_low)
print(f"Low Mean Absolute Error (MAE): {hdfcbank_low_mae}")
print(f"Low Mean Squared Error (MSE): {hdfcbank_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {hdfcbank_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {hdfcbank_low_mape}")


# In[183]:


herohonda_y_close = herohonda['Close'].values
herohonda_y_open = herohonda['Open'].values
herohonda_y_high = herohonda['High'].values
herohonda_y_low = herohonda['Low'].values

herohonda_y_close_scaled = scaler.fit_transform(herohonda_y_close.reshape(-1, 1))
herohonda_y_open_scaled = scaler.fit_transform(herohonda_y_open.reshape(-1, 1))
herohonda_y_high_scaled = scaler.fit_transform(herohonda_y_high.reshape(-1, 1))
herohonda_y_low_scaled = scaler.fit_transform(herohonda_y_low.reshape(-1, 1))

herohonda_close_model = auto_arima(
    herohonda_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

herohonda_open_model = auto_arima(
    herohonda_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

herohonda_high_model = auto_arima(
    herohonda_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

herohonda_low_model = auto_arima(
    herohonda_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

herohonda_close_best_order = herohonda_close_model.get_params()['order']
herohonda_open_best_order = herohonda_open_model.get_params()['order']
herohonda_high_best_order = herohonda_high_model.get_params()['order']
herohonda_low_best_order = herohonda_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {herohonda_close_best_order}")
print(f"Best ARIMA Order for Open: {herohonda_open_best_order}")
print(f"Best ARIMA Order for High: {herohonda_high_best_order}")
print(f"Best ARIMA Order for Low: {herohonda_low_best_order}")

herohonda_close_final_model = sm.tsa.ARIMA(
    herohonda_y_close_scaled,
    order=herohonda_close_best_order
)
herohonda_close_final_model = herohonda_close_final_model.fit()
herohonda_close_forecast = herohonda_close_final_model.forecast(steps=forecast_period)
herohonda_close_forecast = herohonda_close_forecast.reshape(-1, 1)
herohonda_close_forecast = scaler.inverse_transform(herohonda_close_forecast)

herohonda_open_final_model = sm.tsa.ARIMA(
    herohonda_y_open_scaled,
    order=herohonda_open_best_order
)
herohonda_open_final_model = herohonda_open_final_model.fit()
herohonda_open_forecast = herohonda_open_final_model.forecast(steps=forecast_period)
herohonda_open_forecast = herohonda_open_forecast.reshape(-1, 1)
herohonda_open_forecast = scaler.inverse_transform(herohonda_open_forecast)

herohonda_high_final_model = sm.tsa.ARIMA(
    herohonda_y_high_scaled,
    order=herohonda_high_best_order
)
herohonda_high_final_model = herohonda_high_final_model.fit()
herohonda_high_forecast = herohonda_high_final_model.forecast(steps=forecast_period)
herohonda_high_forecast = herohonda_high_forecast.reshape(-1, 1)
herohonda_high_forecast = scaler.inverse_transform(herohonda_high_forecast)

herohonda_low_final_model = sm.tsa.ARIMA(
    herohonda_y_low_scaled,
    order=herohonda_low_best_order
)
herohonda_low_final_model = herohonda_low_final_model.fit()
herohonda_low_forecast = herohonda_low_final_model.forecast(steps=forecast_period)
herohonda_low_forecast = herohonda_low_forecast.reshape(-1, 1)
herohonda_low_forecast = scaler.inverse_transform(herohonda_low_forecast)

print("Close Forecasts:", herohonda_close_forecast)
print("Open Forecasts:", herohonda_open_forecast)
print("High Forecasts:", herohonda_high_forecast)
print("Low Forecasts:", herohonda_low_forecast)


# In[184]:


herohonda_tail_50_data = herohonda.tail(forecast_periods)

herohonda_actual_close_prices = herohonda_tail_50_data['Close'].values
herohonda_actual_open_prices = herohonda_tail_50_data['Open'].values
herohonda_actual_high_prices = herohonda_tail_50_data['High'].values
herohonda_actual_low_prices = herohonda_tail_50_data['Low'].values

herohonda_forecast_close = herohonda_close_final_model.forecast(steps=forecast_periods)
herohonda_forecast_close = herohonda_forecast_close.reshape(-1, 1)
herohonda_forecast_close = scaler.inverse_transform(herohonda_forecast_close)

herohonda_forecast_open = herohonda_open_final_model.forecast(steps=forecast_periods)
herohonda_forecast_open = herohonda_forecast_open.reshape(-1, 1)
herohonda_forecast_open = scaler.inverse_transform(herohonda_forecast_open)

herohonda_forecast_high = herohonda_high_final_model.forecast(steps=forecast_periods)
herohonda_forecast_high = herohonda_forecast_high.reshape(-1, 1)
herohonda_forecast_high = scaler.inverse_transform(herohonda_forecast_high)

herohonda_forecast_low = herohonda_low_final_model.forecast(steps=forecast_periods)
herohonda_forecast_low = herohonda_forecast_low.reshape(-1, 1)
herohonda_forecast_low = scaler.inverse_transform(herohonda_forecast_low)

herohonda_close_mae = mean_absolute_error(herohonda_actual_close_prices, herohonda_forecast_close)
herohonda_close_mse = mean_squared_error(herohonda_actual_close_prices, herohonda_forecast_close)
herohonda_close_rmse = np.sqrt(herohonda_close_mse)

herohonda_open_mae = mean_absolute_error(herohonda_actual_open_prices, herohonda_forecast_open)
herohonda_open_mse = mean_squared_error(herohonda_actual_open_prices, herohonda_forecast_open)
herohonda_open_rmse = np.sqrt(herohonda_open_mse)

herohonda_high_mae = mean_absolute_error(herohonda_actual_high_prices, herohonda_forecast_high)
herohonda_high_mse = mean_squared_error(herohonda_actual_high_prices, herohonda_forecast_high)
herohonda_high_rmse = np.sqrt(herohonda_high_mse)

herohonda_low_mae = mean_absolute_error(herohonda_actual_low_prices, herohonda_forecast_low)
herohonda_low_mse = mean_squared_error(herohonda_actual_low_prices, herohonda_forecast_low)
herohonda_low_rmse = np.sqrt(herohonda_low_mse)

herohonda_close_mape = mean_absolute_percentage_error(herohonda_actual_close_prices, herohonda_forecast_close)
herohonda_open_mape = mean_absolute_percentage_error(herohonda_actual_open_prices, herohonda_forecast_open)
herohonda_high_mape = mean_absolute_percentage_error(herohonda_actual_high_prices, herohonda_forecast_high)
herohonda_low_mape = mean_absolute_percentage_error(herohonda_actual_low_prices, herohonda_forecast_low)

print("Close Forecasts:", herohonda_forecast_close)
print(f"Close Mean Absolute Error (MAE): {herohonda_close_mae}")
print(f"Close Mean Squared Error (MSE): {herohonda_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {herohonda_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {herohonda_close_mape}%")

print("Open Forecasts:", herohonda_forecast_open)
print(f"Open Mean Absolute Error (MAE): {herohonda_open_mae}")
print(f"Open Mean Squared Error (MSE): {herohonda_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {herohonda_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {herohonda_open_mape}%")

print("High Forecasts:", herohonda_forecast_high)
print(f"High Mean Absolute Error (MAE): {herohonda_high_mae}")
print(f"High Mean Squared Error (MSE): {herohonda_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {herohonda_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {herohonda_high_mape}%")

print("Low Forecasts:", herohonda_forecast_low)
print(f"Low Mean Absolute Error (MAE): {herohonda_low_mae}")
print(f"Low Mean Squared Error (MSE): {herohonda_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {herohonda_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {herohonda_low_mape}")


# In[185]:


heromotoco_y_close = heromotoco['Close'].values
heromotoco_y_open = heromotoco['Open'].values
heromotoco_y_high = heromotoco['High'].values
heromotoco_y_low = heromotoco['Low'].values

heromotoco_y_close_scaled = scaler.fit_transform(heromotoco_y_close.reshape(-1, 1))
heromotoco_y_open_scaled = scaler.fit_transform(heromotoco_y_open.reshape(-1, 1))
heromotoco_y_high_scaled = scaler.fit_transform(heromotoco_y_high.reshape(-1, 1))
heromotoco_y_low_scaled = scaler.fit_transform(heromotoco_y_low.reshape(-1, 1))

heromotoco_close_model = auto_arima(
    heromotoco_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

heromotoco_open_model = auto_arima(
    heromotoco_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

heromotoco_high_model = auto_arima(
    heromotoco_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

heromotoco_low_model = auto_arima(
    heromotoco_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

heromotoco_close_best_order = heromotoco_close_model.get_params()['order']
heromotoco_open_best_order = heromotoco_open_model.get_params()['order']
heromotoco_high_best_order = heromotoco_high_model.get_params()['order']
heromotoco_low_best_order = heromotoco_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {heromotoco_close_best_order}")
print(f"Best ARIMA Order for Open: {heromotoco_open_best_order}")
print(f"Best ARIMA Order for High: {heromotoco_high_best_order}")
print(f"Best ARIMA Order for Low: {heromotoco_low_best_order}")

heromotoco_close_final_model = sm.tsa.ARIMA(
    heromotoco_y_close_scaled,
    order=heromotoco_close_best_order
)
heromotoco_close_final_model = heromotoco_close_final_model.fit()
heromotoco_close_forecast = heromotoco_close_final_model.forecast(steps=forecast_period)
heromotoco_close_forecast = heromotoco_close_forecast.reshape(-1, 1)
heromotoco_close_forecast = scaler.inverse_transform(heromotoco_close_forecast)

heromotoco_open_final_model = sm.tsa.ARIMA(
    heromotoco_y_open_scaled,
    order=heromotoco_open_best_order
)
heromotoco_open_final_model = heromotoco_open_final_model.fit()
heromotoco_open_forecast = heromotoco_open_final_model.forecast(steps=forecast_period)
heromotoco_open_forecast = heromotoco_open_forecast.reshape(-1, 1)
heromotoco_open_forecast = scaler.inverse_transform(heromotoco_open_forecast)

heromotoco_high_final_model = sm.tsa.ARIMA(
    heromotoco_y_high_scaled,
    order=heromotoco_high_best_order
)
heromotoco_high_final_model = heromotoco_high_final_model.fit()
heromotoco_high_forecast = heromotoco_high_final_model.forecast(steps=forecast_period)
heromotoco_high_forecast = heromotoco_high_forecast.reshape(-1, 1)
heromotoco_high_forecast = scaler.inverse_transform(heromotoco_high_forecast)

heromotoco_low_final_model = sm.tsa.ARIMA(
    heromotoco_y_low_scaled,
    order=heromotoco_low_best_order
)
heromotoco_low_final_model = heromotoco_low_final_model.fit()
heromotoco_low_forecast = heromotoco_low_final_model.forecast(steps=forecast_period)
heromotoco_low_forecast = heromotoco_low_forecast.reshape(-1, 1)
heromotoco_low_forecast = scaler.inverse_transform(heromotoco_low_forecast)

print("Close Forecasts:", heromotoco_close_forecast)
print("Open Forecasts:", heromotoco_open_forecast)
print("High Forecasts:", heromotoco_high_forecast)
print("Low Forecasts:", heromotoco_low_forecast)


# In[186]:


heromotoco_tail_50_data = heromotoco.tail(forecast_periods)

heromotoco_actual_close_prices = heromotoco_tail_50_data['Close'].values
heromotoco_actual_open_prices = heromotoco_tail_50_data['Open'].values
heromotoco_actual_high_prices = heromotoco_tail_50_data['High'].values
heromotoco_actual_low_prices = heromotoco_tail_50_data['Low'].values

heromotoco_forecast_close = heromotoco_close_final_model.forecast(steps=forecast_periods)
heromotoco_forecast_close = heromotoco_forecast_close.reshape(-1, 1)
heromotoco_forecast_close = scaler.inverse_transform(heromotoco_forecast_close)

heromotoco_forecast_open = heromotoco_open_final_model.forecast(steps=forecast_periods)
heromotoco_forecast_open = heromotoco_forecast_open.reshape(-1, 1)
heromotoco_forecast_open = scaler.inverse_transform(heromotoco_forecast_open)

heromotoco_forecast_high = heromotoco_high_final_model.forecast(steps=forecast_periods)
heromotoco_forecast_high = heromotoco_forecast_high.reshape(-1, 1)
heromotoco_forecast_high = scaler.inverse_transform(heromotoco_forecast_high)

heromotoco_forecast_low = heromotoco_low_final_model.forecast(steps=forecast_periods)
heromotoco_forecast_low = heromotoco_forecast_low.reshape(-1, 1)
heromotoco_forecast_low = scaler.inverse_transform(heromotoco_forecast_low)

heromotoco_close_mae = mean_absolute_error(heromotoco_actual_close_prices, heromotoco_forecast_close)
heromotoco_close_mse = mean_squared_error(heromotoco_actual_close_prices, heromotoco_forecast_close)
heromotoco_close_rmse = np.sqrt(heromotoco_close_mse)

heromotoco_open_mae = mean_absolute_error(heromotoco_actual_open_prices, heromotoco_forecast_open)
heromotoco_open_mse = mean_squared_error(heromotoco_actual_open_prices, heromotoco_forecast_open)
heromotoco_open_rmse = np.sqrt(heromotoco_open_mse)

heromotoco_high_mae = mean_absolute_error(heromotoco_actual_high_prices, heromotoco_forecast_high)
heromotoco_high_mse = mean_squared_error(heromotoco_actual_high_prices, heromotoco_forecast_high)
heromotoco_high_rmse = np.sqrt(heromotoco_high_mse)

heromotoco_low_mae = mean_absolute_error(heromotoco_actual_low_prices, heromotoco_forecast_low)
heromotoco_low_mse = mean_squared_error(heromotoco_actual_low_prices, heromotoco_forecast_low)
heromotoco_low_rmse = np.sqrt(heromotoco_low_mse)

heromotoco_close_mape = mean_absolute_percentage_error(heromotoco_actual_close_prices, heromotoco_forecast_close)
heromotoco_open_mape = mean_absolute_percentage_error(heromotoco_actual_open_prices, heromotoco_forecast_open)
heromotoco_high_mape = mean_absolute_percentage_error(heromotoco_actual_high_prices, heromotoco_forecast_high)
heromotoco_low_mape = mean_absolute_percentage_error(heromotoco_actual_low_prices, heromotoco_forecast_low)

print("Close Forecasts:", heromotoco_forecast_close)
print(f"Close Mean Absolute Error (MAE): {heromotoco_close_mae}")
print(f"Close Mean Squared Error (MSE): {heromotoco_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {heromotoco_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {heromotoco_close_mape}%")

print("Open Forecasts:", heromotoco_forecast_open)
print(f"Open Mean Absolute Error (MAE): {heromotoco_open_mae}")
print(f"Open Mean Squared Error (MSE): {heromotoco_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {heromotoco_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {heromotoco_open_mape}%")

print("High Forecasts:", heromotoco_forecast_high)
print(f"High Mean Absolute Error (MAE): {heromotoco_high_mae}")
print(f"High Mean Squared Error (MSE): {heromotoco_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {heromotoco_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {heromotoco_high_mape}%")

print("Low Forecasts:", heromotoco_forecast_low)
print(f"Low Mean Absolute Error (MAE): {heromotoco_low_mae}")
print(f"Low Mean Squared Error (MSE): {heromotoco_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {heromotoco_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {heromotoco_low_mape}")


# In[187]:


hindalco_y_close = hindalco['Close'].values
hindalco_y_open = hindalco['Open'].values
hindalco_y_high = hindalco['High'].values
hindalco_y_low = hindalco['Low'].values

hindalco_y_close_scaled = scaler.fit_transform(hindalco_y_close.reshape(-1, 1))
hindalco_y_open_scaled = scaler.fit_transform(hindalco_y_open.reshape(-1, 1))
hindalco_y_high_scaled = scaler.fit_transform(hindalco_y_high.reshape(-1, 1))
hindalco_y_low_scaled = scaler.fit_transform(hindalco_y_low.reshape(-1, 1))

hindalco_close_model = auto_arima(
    hindalco_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindalco_open_model = auto_arima(
    hindalco_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindalco_high_model = auto_arima(
    hindalco_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindalco_low_model = auto_arima(
    hindalco_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindalco_close_best_order = hindalco_close_model.get_params()['order']
hindalco_open_best_order = hindalco_open_model.get_params()['order']
hindalco_high_best_order = hindalco_high_model.get_params()['order']
hindalco_low_best_order = hindalco_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {hindalco_close_best_order}")
print(f"Best ARIMA Order for Open: {hindalco_open_best_order}")
print(f"Best ARIMA Order for High: {hindalco_high_best_order}")
print(f"Best ARIMA Order for Low: {hindalco_low_best_order}")

hindalco_close_final_model = sm.tsa.ARIMA(
    hindalco_y_close_scaled,
    order=hindalco_close_best_order
)
hindalco_close_final_model = hindalco_close_final_model.fit()
hindalco_close_forecast = hindalco_close_final_model.forecast(steps=forecast_period)
hindalco_close_forecast = hindalco_close_forecast.reshape(-1, 1)
hindalco_close_forecast = scaler.inverse_transform(hindalco_close_forecast)

hindalco_open_final_model = sm.tsa.ARIMA(
    hindalco_y_open_scaled,
    order=hindalco_open_best_order
)
hindalco_open_final_model = hindalco_open_final_model.fit()
hindalco_open_forecast = hindalco_open_final_model.forecast(steps=forecast_period)
hindalco_open_forecast = hindalco_open_forecast.reshape(-1, 1)
hindalco_open_forecast = scaler.inverse_transform(hindalco_open_forecast)

hindalco_high_final_model = sm.tsa.ARIMA(
    hindalco_y_high_scaled,
    order=hindalco_high_best_order
)
hindalco_high_final_model = hindalco_high_final_model.fit()
hindalco_high_forecast = hindalco_high_final_model.forecast(steps=forecast_period)
hindalco_high_forecast = hindalco_high_forecast.reshape(-1, 1)
hindalco_high_forecast = scaler.inverse_transform(hindalco_high_forecast)

hindalco_low_final_model = sm.tsa.ARIMA(
    hindalco_y_low_scaled,
    order=hindalco_low_best_order
)
hindalco_low_final_model = hindalco_low_final_model.fit()
hindalco_low_forecast = hindalco_low_final_model.forecast(steps=forecast_period)
hindalco_low_forecast = hindalco_low_forecast.reshape(-1, 1)
hindalco_low_forecast = scaler.inverse_transform(hindalco_low_forecast)

print("Close Forecasts:", hindalco_close_forecast)
print("Open Forecasts:", hindalco_open_forecast)
print("High Forecasts:", hindalco_high_forecast)
print("Low Forecasts:", hindalco_low_forecast)


# In[188]:


hindalco_tail_50_data = hindalco.tail(forecast_periods)

hindalco_actual_close_prices = hindalco_tail_50_data['Close'].values
hindalco_actual_open_prices = hindalco_tail_50_data['Open'].values
hindalco_actual_high_prices = hindalco_tail_50_data['High'].values
hindalco_actual_low_prices = hindalco_tail_50_data['Low'].values

hindalco_forecast_close = hindalco_close_final_model.forecast(steps=forecast_periods)
hindalco_forecast_close = hindalco_forecast_close.reshape(-1, 1)
hindalco_forecast_close = scaler.inverse_transform(hindalco_forecast_close)

hindalco_forecast_open = hindalco_open_final_model.forecast(steps=forecast_periods)
hindalco_forecast_open = hindalco_forecast_open.reshape(-1, 1)
hindalco_forecast_open = scaler.inverse_transform(hindalco_forecast_open)

hindalco_forecast_high = hindalco_high_final_model.forecast(steps=forecast_periods)
hindalco_forecast_high = hindalco_forecast_high.reshape(-1, 1)
hindalco_forecast_high = scaler.inverse_transform(hindalco_forecast_high)

hindalco_forecast_low = hindalco_low_final_model.forecast(steps=forecast_periods)
hindalco_forecast_low = hindalco_forecast_low.reshape(-1, 1)
hindalco_forecast_low = scaler.inverse_transform(hindalco_forecast_low)

hindalco_close_mae = mean_absolute_error(hindalco_actual_close_prices, hindalco_forecast_close)
hindalco_close_mse = mean_squared_error(hindalco_actual_close_prices, hindalco_forecast_close)
hindalco_close_rmse = np.sqrt(hindalco_close_mse)

hindalco_open_mae = mean_absolute_error(hindalco_actual_open_prices, hindalco_forecast_open)
hindalco_open_mse = mean_squared_error(hindalco_actual_open_prices, hindalco_forecast_open)
hindalco_open_rmse = np.sqrt(hindalco_open_mse)

hindalco_high_mae = mean_absolute_error(hindalco_actual_high_prices, hindalco_forecast_high)
hindalco_high_mse = mean_squared_error(hindalco_actual_high_prices, hindalco_forecast_high)
hindalco_high_rmse = np.sqrt(hindalco_high_mse)

hindalco_low_mae = mean_absolute_error(hindalco_actual_low_prices, hindalco_forecast_low)
hindalco_low_mse = mean_squared_error(hindalco_actual_low_prices, hindalco_forecast_low)
hindalco_low_rmse = np.sqrt(hindalco_low_mse)

hindalco_close_mape = mean_absolute_percentage_error(hindalco_actual_close_prices, hindalco_forecast_close)
hindalco_open_mape = mean_absolute_percentage_error(hindalco_actual_open_prices, hindalco_forecast_open)
hindalco_high_mape = mean_absolute_percentage_error(hindalco_actual_high_prices, hindalco_forecast_high)
hindalco_low_mape = mean_absolute_percentage_error(hindalco_actual_low_prices, hindalco_forecast_low)

print("Close Forecasts:", hindalco_forecast_close)
print(f"Close Mean Absolute Error (MAE): {hindalco_close_mae}")
print(f"Close Mean Squared Error (MSE): {hindalco_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {hindalco_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {hindalco_close_mape}%")

print("Open Forecasts:", hindalco_forecast_open)
print(f"Open Mean Absolute Error (MAE): {hindalco_open_mae}")
print(f"Open Mean Squared Error (MSE): {hindalco_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {hindalco_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {hindalco_open_mape}%")

print("High Forecasts:", hindalco_forecast_high)
print(f"High Mean Absolute Error (MAE): {hindalco_high_mae}")
print(f"High Mean Squared Error (MSE): {hindalco_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {hindalco_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {hindalco_high_mape}%")

print("Low Forecasts:", hindalco_forecast_low)
print(f"Low Mean Absolute Error (MAE): {hindalco_low_mae}")
print(f"Low Mean Squared Error (MSE): {hindalco_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {hindalco_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {hindalco_low_mape}")


# In[189]:


hindelever_y_close = hindelever['Close'].values
hindelever_y_open = hindelever['Open'].values
hindelever_y_high = hindelever['High'].values
hindelever_y_low = hindelever['Low'].values

hindelever_y_close_scaled = scaler.fit_transform(hindelever_y_close.reshape(-1, 1))
hindelever_y_open_scaled = scaler.fit_transform(hindelever_y_open.reshape(-1, 1))
hindelever_y_high_scaled = scaler.fit_transform(hindelever_y_high.reshape(-1, 1))
hindelever_y_low_scaled = scaler.fit_transform(hindelever_y_low.reshape(-1, 1))

hindelever_close_model = auto_arima(
    hindelever_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindelever_open_model = auto_arima(
    hindelever_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindelever_high_model = auto_arima(
    hindelever_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindelever_low_model = auto_arima(
    hindelever_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindelever_close_best_order = hindelever_close_model.get_params()['order']
hindelever_open_best_order = hindelever_open_model.get_params()['order']
hindelever_high_best_order = hindelever_high_model.get_params()['order']
hindelever_low_best_order = hindelever_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {hindelever_close_best_order}")
print(f"Best ARIMA Order for Open: {hindelever_open_best_order}")
print(f"Best ARIMA Order for High: {hindelever_high_best_order}")
print(f"Best ARIMA Order for Low: {hindelever_low_best_order}")

hindelever_close_final_model = sm.tsa.ARIMA(
    hindelever_y_close_scaled,
    order=hindelever_close_best_order
)
hindelever_close_final_model = hindelever_close_final_model.fit()
hindelever_close_forecast = hindelever_close_final_model.forecast(steps=forecast_period)
hindelever_close_forecast = hindelever_close_forecast.reshape(-1, 1)
hindelever_close_forecast = scaler.inverse_transform(hindelever_close_forecast)

hindelever_open_final_model = sm.tsa.ARIMA(
    hindelever_y_open_scaled,
    order=hindelever_open_best_order
)
hindelever_open_final_model = hindelever_open_final_model.fit()
hindelever_open_forecast = hindelever_open_final_model.forecast(steps=forecast_period)
hindelever_open_forecast = hindelever_open_forecast.reshape(-1, 1)
hindelever_open_forecast = scaler.inverse_transform(hindelever_open_forecast)

hindelever_high_final_model = sm.tsa.ARIMA(
    hindelever_y_high_scaled,
    order=hindelever_high_best_order
)
hindelever_high_final_model = hindelever_high_final_model.fit()
hindelever_high_forecast = hindelever_high_final_model.forecast(steps=forecast_period)
hindelever_high_forecast = hindelever_high_forecast.reshape(-1, 1)
hindelever_high_forecast = scaler.inverse_transform(hindelever_high_forecast)

hindelever_low_final_model = sm.tsa.ARIMA(
    hindelever_y_low_scaled,
    order=hindelever_low_best_order
)
hindelever_low_final_model = hindelever_low_final_model.fit()
hindelever_low_forecast = hindelever_low_final_model.forecast(steps=forecast_period)
hindelever_low_forecast = hindelever_low_forecast.reshape(-1, 1)
hindelever_low_forecast = scaler.inverse_transform(hindelever_low_forecast)

print("Close Forecasts:", hindelever_close_forecast)
print("Open Forecasts:", hindelever_open_forecast)
print("High Forecasts:", hindelever_high_forecast)
print("Low Forecasts:", hindelever_low_forecast)


# In[190]:


hindelever_tail_50_data = hindelever.tail(forecast_periods)

hindelever_actual_close_prices = hindelever_tail_50_data['Close'].values
hindelever_actual_open_prices = hindelever_tail_50_data['Open'].values
hindelever_actual_high_prices = hindelever_tail_50_data['High'].values
hindelever_actual_low_prices = hindelever_tail_50_data['Low'].values

hindelever_forecast_close = hindelever_close_final_model.forecast(steps=forecast_periods)
hindelever_forecast_close = hindelever_forecast_close.reshape(-1, 1)
hindelever_forecast_close = scaler.inverse_transform(hindelever_forecast_close)

hindelever_forecast_open = hindelever_open_final_model.forecast(steps=forecast_periods)
hindelever_forecast_open = hindelever_forecast_open.reshape(-1, 1)
hindelever_forecast_open = scaler.inverse_transform(hindelever_forecast_open)

hindelever_forecast_high = hindelever_high_final_model.forecast(steps=forecast_periods)
hindelever_forecast_high = hindelever_forecast_high.reshape(-1, 1)
hindelever_forecast_high = scaler.inverse_transform(hindelever_forecast_high)

hindelever_forecast_low = hindelever_low_final_model.forecast(steps=forecast_periods)
hindelever_forecast_low = hindelever_forecast_low.reshape(-1, 1)
hindelever_forecast_low = scaler.inverse_transform(hindelever_forecast_low)

hindelever_close_mae = mean_absolute_error(hindelever_actual_close_prices, hindelever_forecast_close)
hindelever_close_mse = mean_squared_error(hindelever_actual_close_prices, hindelever_forecast_close)
hindelever_close_rmse = np.sqrt(hindelever_close_mse)

hindelever_open_mae = mean_absolute_error(hindelever_actual_open_prices, hindelever_forecast_open)
hindelever_open_mse = mean_squared_error(hindelever_actual_open_prices, hindelever_forecast_open)
hindelever_open_rmse = np.sqrt(hindelever_open_mse)

hindelever_high_mae = mean_absolute_error(hindelever_actual_high_prices, hindelever_forecast_high)
hindelever_high_mse = mean_squared_error(hindelever_actual_high_prices, hindelever_forecast_high)
hindelever_high_rmse = np.sqrt(hindelever_high_mse)

hindelever_low_mae = mean_absolute_error(hindelever_actual_low_prices, hindelever_forecast_low)
hindelever_low_mse = mean_squared_error(hindelever_actual_low_prices, hindelever_forecast_low)
hindelever_low_rmse = np.sqrt(hindelever_low_mse)

hindelever_close_mape = mean_absolute_percentage_error(hindelever_actual_close_prices, hindelever_forecast_close)
hindelever_open_mape = mean_absolute_percentage_error(hindelever_actual_open_prices, hindelever_forecast_open)
hindelever_high_mape = mean_absolute_percentage_error(hindelever_actual_high_prices, hindelever_forecast_high)
hindelever_low_mape = mean_absolute_percentage_error(hindelever_actual_low_prices, hindelever_forecast_low)

print("Close Forecasts:", hindelever_forecast_close)
print(f"Close Mean Absolute Error (MAE): {hindelever_close_mae}")
print(f"Close Mean Squared Error (MSE): {hindelever_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {hindelever_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {hindelever_close_mape}%")

print("Open Forecasts:", hindelever_forecast_open)
print(f"Open Mean Absolute Error (MAE): {hindelever_open_mae}")
print(f"Open Mean Squared Error (MSE): {hindelever_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {hindelever_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {hindelever_open_mape}%")

print("High Forecasts:", hindelever_forecast_high)
print(f"High Mean Absolute Error (MAE): {hindelever_high_mae}")
print(f"High Mean Squared Error (MSE): {hindelever_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {hindelever_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {hindelever_high_mape}%")

print("Low Forecasts:", hindelever_forecast_low)
print(f"Low Mean Absolute Error (MAE): {hindelever_low_mae}")
print(f"Low Mean Squared Error (MSE): {hindelever_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {hindelever_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {hindelever_low_mape}")


# icicibank = company_datasets['ICICIBANK']
# indusindbk = company_datasets['INDUSINDBK']
# infosystch = company_datasets['INFOSYSTCH']
# infy = company_datasets['INFY']
# ioc = company_datasets['IOC']
# itc = company_datasets['ITC']
# jswstl = company_datasets['JSWSTL']
# kotakmah = company_datasets['KOTAKMAH']
# kotakbank = company_datasets['KOTAKBANK']
# lt = company_datasets['LT']
# mandm = company_datasets['M&M']
# maruti = company_datasets['MARUTI']
# nestleind = company_datasets['NESTLEIND']
# ntpc = company_datasets['NTPC']
# ongc = company_datasets['ONGC']
# powergrid = company_datasets['POWERGRID']
# reliance = company_datasets['RELIANCE']
# sbin = company_datasets['SBIN']
# shreecem = company_datasets['SHREECEM']
# sunpharma = company_datasets['SUNPHARMA']
# telco = company_datasets['TELCO']
# tatamotors = company_datasets['TATAMOTORS']
# tisco = company_datasets['TISCO']
# tatasteel = company_datasets['TATASTEEL']
# tcs = company_datasets['TCS']
# techm = company_datasets['TECHM']
# titan = company_datasets['TITAN']
# ultracemco = company_datasets['ULTRACEMCO']
# uniphos = company_datasets['UNIPHOS']
# upl = company_datasets['UPL']
# sesagoa = company_datasets['SESAGOA']
# sslt = company_datasets['SSLT']
# vedl = company_datasets['VEDL']
# wipro = company_datasets['WIPRO']
# zeetele = company_datasets['ZEETELE']
# zeel = company_datasets['ZEEL']

# In[191]:


hindunilvr_y_close = hindunilvr['Close'].values
hindunilvr_y_open = hindunilvr['Open'].values
hindunilvr_y_high = hindunilvr['High'].values
hindunilvr_y_low = hindunilvr['Low'].values

hindunilvr_y_close_scaled = scaler.fit_transform(hindunilvr_y_close.reshape(-1, 1))
hindunilvr_y_open_scaled = scaler.fit_transform(hindunilvr_y_open.reshape(-1, 1))
hindunilvr_y_high_scaled = scaler.fit_transform(hindunilvr_y_high.reshape(-1, 1))
hindunilvr_y_low_scaled = scaler.fit_transform(hindunilvr_y_low.reshape(-1, 1))

hindunilvr_close_model = auto_arima(
    hindunilvr_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindunilvr_open_model = auto_arima(
    hindunilvr_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindunilvr_high_model = auto_arima(
    hindunilvr_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindunilvr_low_model = auto_arima(
    hindunilvr_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

hindunilvr_close_best_order = hindunilvr_close_model.get_params()['order']
hindunilvr_open_best_order = hindunilvr_open_model.get_params()['order']
hindunilvr_high_best_order = hindunilvr_high_model.get_params()['order']
hindunilvr_low_best_order = hindunilvr_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {hindunilvr_close_best_order}")
print(f"Best ARIMA Order for Open: {hindunilvr_open_best_order}")
print(f"Best ARIMA Order for High: {hindunilvr_high_best_order}")
print(f"Best ARIMA Order for Low: {hindunilvr_low_best_order}")

hindunilvr_close_final_model = sm.tsa.ARIMA(
    hindunilvr_y_close_scaled,
    order=hindunilvr_close_best_order
)
hindunilvr_close_final_model = hindunilvr_close_final_model.fit()
hindunilvr_close_forecast = hindunilvr_close_final_model.forecast(steps=forecast_period)
hindunilvr_close_forecast = hindunilvr_close_forecast.reshape(-1, 1)
hindunilvr_close_forecast = scaler.inverse_transform(hindunilvr_close_forecast)

hindunilvr_open_final_model = sm.tsa.ARIMA(
    hindunilvr_y_open_scaled,
    order=hindunilvr_open_best_order
)
hindunilvr_open_final_model = hindunilvr_open_final_model.fit()
hindunilvr_open_forecast = hindunilvr_open_final_model.forecast(steps=forecast_period)
hindunilvr_open_forecast = hindunilvr_open_forecast.reshape(-1, 1)
hindunilvr_open_forecast = scaler.inverse_transform(hindunilvr_open_forecast)

hindunilvr_high_final_model = sm.tsa.ARIMA(
    hindunilvr_y_high_scaled,
    order=hindunilvr_high_best_order
)
hindunilvr_high_final_model = hindunilvr_high_final_model.fit()
hindunilvr_high_forecast = hindunilvr_high_final_model.forecast(steps=forecast_period)
hindunilvr_high_forecast = hindunilvr_high_forecast.reshape(-1, 1)
hindunilvr_high_forecast = scaler.inverse_transform(hindunilvr_high_forecast)

hindunilvr_low_final_model = sm.tsa.ARIMA(
    hindunilvr_y_low_scaled,
    order=hindunilvr_low_best_order
)
hindunilvr_low_final_model = hindunilvr_low_final_model.fit()
hindunilvr_low_forecast = hindunilvr_low_final_model.forecast(steps=forecast_period)
hindunilvr_low_forecast = hindunilvr_low_forecast.reshape(-1, 1)
hindunilvr_low_forecast = scaler.inverse_transform(hindunilvr_low_forecast)

print("Close Forecasts:", hindunilvr_close_forecast)
print("Open Forecasts:", hindunilvr_open_forecast)
print("High Forecasts:", hindunilvr_high_forecast)
print("Low Forecasts:", hindunilvr_low_forecast)


# In[192]:


hindunilvr_tail_50_data = hindunilvr.tail(forecast_periods)

hindunilvr_actual_close_prices = hindunilvr_tail_50_data['Close'].values
hindunilvr_actual_open_prices = hindunilvr_tail_50_data['Open'].values
hindunilvr_actual_high_prices = hindunilvr_tail_50_data['High'].values
hindunilvr_actual_low_prices = hindunilvr_tail_50_data['Low'].values

hindunilvr_forecast_close = hindunilvr_close_final_model.forecast(steps=forecast_periods)
hindunilvr_forecast_close = hindunilvr_forecast_close.reshape(-1, 1)
hindunilvr_forecast_close = scaler.inverse_transform(hindunilvr_forecast_close)

hindunilvr_forecast_open = hindunilvr_open_final_model.forecast(steps=forecast_periods)
hindunilvr_forecast_open = hindunilvr_forecast_open.reshape(-1, 1)
hindunilvr_forecast_open = scaler.inverse_transform(hindunilvr_forecast_open)

hindunilvr_forecast_high = hindunilvr_high_final_model.forecast(steps=forecast_periods)
hindunilvr_forecast_high = hindunilvr_forecast_high.reshape(-1, 1)
hindunilvr_forecast_high = scaler.inverse_transform(hindunilvr_forecast_high)

hindunilvr_forecast_low = hindunilvr_low_final_model.forecast(steps=forecast_periods)
hindunilvr_forecast_low = hindunilvr_forecast_low.reshape(-1, 1)
hindunilvr_forecast_low = scaler.inverse_transform(hindunilvr_forecast_low)

hindunilvr_close_mae = mean_absolute_error(hindunilvr_actual_close_prices, hindunilvr_forecast_close)
hindunilvr_close_mse = mean_squared_error(hindunilvr_actual_close_prices, hindunilvr_forecast_close)
hindunilvr_close_rmse = np.sqrt(hindunilvr_close_mse)

hindunilvr_open_mae = mean_absolute_error(hindunilvr_actual_open_prices, hindunilvr_forecast_open)
hindunilvr_open_mse = mean_squared_error(hindunilvr_actual_open_prices, hindunilvr_forecast_open)
hindunilvr_open_rmse = np.sqrt(hindunilvr_open_mse)

hindunilvr_high_mae = mean_absolute_error(hindunilvr_actual_high_prices, hindunilvr_forecast_high)
hindunilvr_high_mse = mean_squared_error(hindunilvr_actual_high_prices, hindunilvr_forecast_high)
hindunilvr_high_rmse = np.sqrt(hindunilvr_high_mse)

hindunilvr_low_mae = mean_absolute_error(hindunilvr_actual_low_prices, hindunilvr_forecast_low)
hindunilvr_low_mse = mean_squared_error(hindunilvr_actual_low_prices, hindunilvr_forecast_low)
hindunilvr_low_rmse = np.sqrt(hindunilvr_low_mse)

hindunilvr_close_mape = mean_absolute_percentage_error(hindunilvr_actual_close_prices, hindunilvr_forecast_close)
hindunilvr_open_mape = mean_absolute_percentage_error(hindunilvr_actual_open_prices, hindunilvr_forecast_open)
hindunilvr_high_mape = mean_absolute_percentage_error(hindunilvr_actual_high_prices, hindunilvr_forecast_high)
hindunilvr_low_mape = mean_absolute_percentage_error(hindunilvr_actual_low_prices, hindunilvr_forecast_low)

print("Close Forecasts:", hindunilvr_forecast_close)
print(f"Close Mean Absolute Error (MAE): {hindunilvr_close_mae}")
print(f"Close Mean Squared Error (MSE): {hindunilvr_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {hindunilvr_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {hindunilvr_close_mape}%")

print("Open Forecasts:", hindunilvr_forecast_open)
print(f"Open Mean Absolute Error (MAE): {hindunilvr_open_mae}")
print(f"Open Mean Squared Error (MSE): {hindunilvr_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {hindunilvr_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {hindunilvr_open_mape}%")

print("High Forecasts:", hindunilvr_forecast_high)
print(f"High Mean Absolute Error (MAE): {hindunilvr_high_mae}")
print(f"High Mean Squared Error (MSE): {hindunilvr_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {hindunilvr_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {hindunilvr_high_mape}%")

print("Low Forecasts:", hindunilvr_forecast_low)
print(f"Low Mean Absolute Error (MAE): {hindunilvr_low_mae}")
print(f"Low Mean Squared Error (MSE): {hindunilvr_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {hindunilvr_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {hindunilvr_low_mape}")


# In[193]:


icicibank_y_close = icicibank['Close'].values
icicibank_y_open = icicibank['Open'].values
icicibank_y_high = icicibank['High'].values
icicibank_y_low = icicibank['Low'].values

icicibank_y_close_scaled = scaler.fit_transform(icicibank_y_close.reshape(-1, 1))
icicibank_y_open_scaled = scaler.fit_transform(icicibank_y_open.reshape(-1, 1))
icicibank_y_high_scaled = scaler.fit_transform(icicibank_y_high.reshape(-1, 1))
icicibank_y_low_scaled = scaler.fit_transform(icicibank_y_low.reshape(-1, 1))

icicibank_close_model = auto_arima(
    icicibank_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

icicibank_open_model = auto_arima(
    icicibank_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

icicibank_high_model = auto_arima(
    icicibank_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

icicibank_low_model = auto_arima(
    icicibank_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

icicibank_close_best_order = icicibank_close_model.get_params()['order']
icicibank_open_best_order = icicibank_open_model.get_params()['order']
icicibank_high_best_order = icicibank_high_model.get_params()['order']
icicibank_low_best_order = icicibank_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {icicibank_close_best_order}")
print(f"Best ARIMA Order for Open: {icicibank_open_best_order}")
print(f"Best ARIMA Order for High: {icicibank_high_best_order}")
print(f"Best ARIMA Order for Low: {icicibank_low_best_order}")

icicibank_close_final_model = sm.tsa.ARIMA(
    icicibank_y_close_scaled,
    order=icicibank_close_best_order
)
icicibank_close_final_model = icicibank_close_final_model.fit()
icicibank_close_forecast = icicibank_close_final_model.forecast(steps=forecast_period)
icicibank_close_forecast = icicibank_close_forecast.reshape(-1, 1)
icicibank_close_forecast = scaler.inverse_transform(icicibank_close_forecast)

icicibank_open_final_model = sm.tsa.ARIMA(
    icicibank_y_open_scaled,
    order=icicibank_open_best_order
)
icicibank_open_final_model = icicibank_open_final_model.fit()
icicibank_open_forecast = icicibank_open_final_model.forecast(steps=forecast_period)
icicibank_open_forecast = icicibank_open_forecast.reshape(-1, 1)
icicibank_open_forecast = scaler.inverse_transform(icicibank_open_forecast)

icicibank_high_final_model = sm.tsa.ARIMA(
    icicibank_y_high_scaled,
    order=icicibank_high_best_order
)
icicibank_high_final_model = icicibank_high_final_model.fit()
icicibank_high_forecast = icicibank_high_final_model.forecast(steps=forecast_period)
icicibank_high_forecast = icicibank_high_forecast.reshape(-1, 1)
icicibank_high_forecast = scaler.inverse_transform(icicibank_high_forecast)

icicibank_low_final_model = sm.tsa.ARIMA(
    icicibank_y_low_scaled,
    order=icicibank_low_best_order
)
icicibank_low_final_model = icicibank_low_final_model.fit()
icicibank_low_forecast = icicibank_low_final_model.forecast(steps=forecast_period)
icicibank_low_forecast = icicibank_low_forecast.reshape(-1, 1)
icicibank_low_forecast = scaler.inverse_transform(icicibank_low_forecast)

print("Close Forecasts:", icicibank_close_forecast)
print("Open Forecasts:", icicibank_open_forecast)
print("High Forecasts:", icicibank_high_forecast)
print("Low Forecasts:", icicibank_low_forecast)


# In[194]:


icicibank_tail_50_data = icicibank.tail(forecast_periods)

icicibank_actual_close_prices = icicibank_tail_50_data['Close'].values
icicibank_actual_open_prices = icicibank_tail_50_data['Open'].values
icicibank_actual_high_prices = icicibank_tail_50_data['High'].values
icicibank_actual_low_prices = icicibank_tail_50_data['Low'].values

icicibank_forecast_close = icicibank_close_final_model.forecast(steps=forecast_periods)
icicibank_forecast_close = icicibank_forecast_close.reshape(-1, 1)
icicibank_forecast_close = scaler.inverse_transform(icicibank_forecast_close)

icicibank_forecast_open = icicibank_open_final_model.forecast(steps=forecast_periods)
icicibank_forecast_open = icicibank_forecast_open.reshape(-1, 1)
icicibank_forecast_open = scaler.inverse_transform(icicibank_forecast_open)

icicibank_forecast_high = icicibank_high_final_model.forecast(steps=forecast_periods)
icicibank_forecast_high = icicibank_forecast_high.reshape(-1, 1)
icicibank_forecast_high = scaler.inverse_transform(icicibank_forecast_high)

icicibank_forecast_low = icicibank_low_final_model.forecast(steps=forecast_periods)
icicibank_forecast_low = icicibank_forecast_low.reshape(-1, 1)
icicibank_forecast_low = scaler.inverse_transform(icicibank_forecast_low)

icicibank_close_mae = mean_absolute_error(icicibank_actual_close_prices, icicibank_forecast_close)
icicibank_close_mse = mean_squared_error(icicibank_actual_close_prices, icicibank_forecast_close)
icicibank_close_rmse = np.sqrt(icicibank_close_mse)

icicibank_open_mae = mean_absolute_error(icicibank_actual_open_prices, icicibank_forecast_open)
icicibank_open_mse = mean_squared_error(icicibank_actual_open_prices, icicibank_forecast_open)
icicibank_open_rmse = np.sqrt(icicibank_open_mse)

icicibank_high_mae = mean_absolute_error(icicibank_actual_high_prices, icicibank_forecast_high)
icicibank_high_mse = mean_squared_error(icicibank_actual_high_prices, icicibank_forecast_high)
icicibank_high_rmse = np.sqrt(icicibank_high_mse)

icicibank_low_mae = mean_absolute_error(icicibank_actual_low_prices, icicibank_forecast_low)
icicibank_low_mse = mean_squared_error(icicibank_actual_low_prices, icicibank_forecast_low)
icicibank_low_rmse = np.sqrt(icicibank_low_mse)

icicibank_close_mape = mean_absolute_percentage_error(icicibank_actual_close_prices, icicibank_forecast_close)
icicibank_open_mape = mean_absolute_percentage_error(icicibank_actual_open_prices, icicibank_forecast_open)
icicibank_high_mape = mean_absolute_percentage_error(icicibank_actual_high_prices, icicibank_forecast_high)
icicibank_low_mape = mean_absolute_percentage_error(icicibank_actual_low_prices, icicibank_forecast_low)

print("Close Forecasts:", icicibank_forecast_close)
print(f"Close Mean Absolute Error (MAE): {icicibank_close_mae}")
print(f"Close Mean Squared Error (MSE): {icicibank_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {icicibank_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {icicibank_close_mape}%")

print("Open Forecasts:", icicibank_forecast_open)
print(f"Open Mean Absolute Error (MAE): {icicibank_open_mae}")
print(f"Open Mean Squared Error (MSE): {icicibank_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {icicibank_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {icicibank_open_mape}%")

print("High Forecasts:", icicibank_forecast_high)
print(f"High Mean Absolute Error (MAE): {icicibank_high_mae}")
print(f"High Mean Squared Error (MSE): {icicibank_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {icicibank_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {icicibank_high_mape}%")

print("Low Forecasts:", icicibank_forecast_low)
print(f"Low Mean Absolute Error (MAE): {icicibank_low_mae}")
print(f"Low Mean Squared Error (MSE): {icicibank_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {icicibank_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {icicibank_low_mape}")


# In[195]:


indusindbk_y_close = indusindbk['Close'].values
indusindbk_y_open = indusindbk['Open'].values
indusindbk_y_high = indusindbk['High'].values
indusindbk_y_low = indusindbk['Low'].values

indusindbk_y_close_scaled = scaler.fit_transform(indusindbk_y_close.reshape(-1, 1))
indusindbk_y_open_scaled = scaler.fit_transform(indusindbk_y_open.reshape(-1, 1))
indusindbk_y_high_scaled = scaler.fit_transform(indusindbk_y_high.reshape(-1, 1))
indusindbk_y_low_scaled = scaler.fit_transform(indusindbk_y_low.reshape(-1, 1))

indusindbk_close_model = auto_arima(
    indusindbk_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

indusindbk_open_model = auto_arima(
    indusindbk_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

indusindbk_high_model = auto_arima(
    indusindbk_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

indusindbk_low_model = auto_arima(
    indusindbk_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

indusindbk_close_best_order = indusindbk_close_model.get_params()['order']
indusindbk_open_best_order = indusindbk_open_model.get_params()['order']
indusindbk_high_best_order = indusindbk_high_model.get_params()['order']
indusindbk_low_best_order = indusindbk_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {indusindbk_close_best_order}")
print(f"Best ARIMA Order for Open: {indusindbk_open_best_order}")
print(f"Best ARIMA Order for High: {indusindbk_high_best_order}")
print(f"Best ARIMA Order for Low: {indusindbk_low_best_order}")

indusindbk_close_final_model = sm.tsa.ARIMA(
    indusindbk_y_close_scaled,
    order=indusindbk_close_best_order
)
indusindbk_close_final_model = indusindbk_close_final_model.fit()
indusindbk_close_forecast = indusindbk_close_final_model.forecast(steps=forecast_period)
indusindbk_close_forecast = indusindbk_close_forecast.reshape(-1, 1)
indusindbk_close_forecast = scaler.inverse_transform(indusindbk_close_forecast)

indusindbk_open_final_model = sm.tsa.ARIMA(
    indusindbk_y_open_scaled,
    order=indusindbk_open_best_order
)
indusindbk_open_final_model = indusindbk_open_final_model.fit()
indusindbk_open_forecast = indusindbk_open_final_model.forecast(steps=forecast_period)
indusindbk_open_forecast = indusindbk_open_forecast.reshape(-1, 1)
indusindbk_open_forecast = scaler.inverse_transform(indusindbk_open_forecast)

indusindbk_high_final_model = sm.tsa.ARIMA(
    indusindbk_y_high_scaled,
    order=indusindbk_high_best_order
)
indusindbk_high_final_model = indusindbk_high_final_model.fit()
indusindbk_high_forecast = indusindbk_high_final_model.forecast(steps=forecast_period)
indusindbk_high_forecast = indusindbk_high_forecast.reshape(-1, 1)
indusindbk_high_forecast = scaler.inverse_transform(indusindbk_high_forecast)

indusindbk_low_final_model = sm.tsa.ARIMA(
    indusindbk_y_low_scaled,
    order=indusindbk_low_best_order
)
indusindbk_low_final_model = indusindbk_low_final_model.fit()
indusindbk_low_forecast = indusindbk_low_final_model.forecast(steps=forecast_period)
indusindbk_low_forecast = indusindbk_low_forecast.reshape(-1, 1)
indusindbk_low_forecast = scaler.inverse_transform(indusindbk_low_forecast)

print("Close Forecasts:", indusindbk_close_forecast)
print("Open Forecasts:", indusindbk_open_forecast)
print("High Forecasts:", indusindbk_high_forecast)
print("Low Forecasts:", indusindbk_low_forecast)


# In[196]:


indusindbk_tail_50_data = indusindbk.tail(forecast_periods)

indusindbk_actual_close_prices = indusindbk_tail_50_data['Close'].values
indusindbk_actual_open_prices = indusindbk_tail_50_data['Open'].values
indusindbk_actual_high_prices = indusindbk_tail_50_data['High'].values
indusindbk_actual_low_prices = indusindbk_tail_50_data['Low'].values

indusindbk_forecast_close = indusindbk_close_final_model.forecast(steps=forecast_periods)
indusindbk_forecast_close = indusindbk_forecast_close.reshape(-1, 1)
indusindbk_forecast_close = scaler.inverse_transform(indusindbk_forecast_close)

indusindbk_forecast_open = indusindbk_open_final_model.forecast(steps=forecast_periods)
indusindbk_forecast_open = indusindbk_forecast_open.reshape(-1, 1)
indusindbk_forecast_open = scaler.inverse_transform(indusindbk_forecast_open)

indusindbk_forecast_high = indusindbk_high_final_model.forecast(steps=forecast_periods)
indusindbk_forecast_high = indusindbk_forecast_high.reshape(-1, 1)
indusindbk_forecast_high = scaler.inverse_transform(indusindbk_forecast_high)

indusindbk_forecast_low = indusindbk_low_final_model.forecast(steps=forecast_periods)
indusindbk_forecast_low = indusindbk_forecast_low.reshape(-1, 1)
indusindbk_forecast_low = scaler.inverse_transform(indusindbk_forecast_low)

indusindbk_close_mae = mean_absolute_error(indusindbk_actual_close_prices, indusindbk_forecast_close)
indusindbk_close_mse = mean_squared_error(indusindbk_actual_close_prices, indusindbk_forecast_close)
indusindbk_close_rmse = np.sqrt(indusindbk_close_mse)

indusindbk_open_mae = mean_absolute_error(indusindbk_actual_open_prices, indusindbk_forecast_open)
indusindbk_open_mse = mean_squared_error(indusindbk_actual_open_prices, indusindbk_forecast_open)
indusindbk_open_rmse = np.sqrt(indusindbk_open_mse)

indusindbk_high_mae = mean_absolute_error(indusindbk_actual_high_prices, indusindbk_forecast_high)
indusindbk_high_mse = mean_squared_error(indusindbk_actual_high_prices, indusindbk_forecast_high)
indusindbk_high_rmse = np.sqrt(indusindbk_high_mse)

indusindbk_low_mae = mean_absolute_error(indusindbk_actual_low_prices, indusindbk_forecast_low)
indusindbk_low_mse = mean_squared_error(indusindbk_actual_low_prices, indusindbk_forecast_low)
indusindbk_low_rmse = np.sqrt(indusindbk_low_mse)

indusindbk_close_mape = mean_absolute_percentage_error(indusindbk_actual_close_prices, indusindbk_forecast_close)
indusindbk_open_mape = mean_absolute_percentage_error(indusindbk_actual_open_prices, indusindbk_forecast_open)
indusindbk_high_mape = mean_absolute_percentage_error(indusindbk_actual_high_prices, indusindbk_forecast_high)
indusindbk_low_mape = mean_absolute_percentage_error(indusindbk_actual_low_prices, indusindbk_forecast_low)

print("Close Forecasts:", indusindbk_forecast_close)
print(f"Close Mean Absolute Error (MAE): {indusindbk_close_mae}")
print(f"Close Mean Squared Error (MSE): {indusindbk_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {indusindbk_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {indusindbk_close_mape}%")

print("Open Forecasts:", indusindbk_forecast_open)
print(f"Open Mean Absolute Error (MAE): {indusindbk_open_mae}")
print(f"Open Mean Squared Error (MSE): {indusindbk_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {indusindbk_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {indusindbk_open_mape}%")

print("High Forecasts:", indusindbk_forecast_high)
print(f"High Mean Absolute Error (MAE): {indusindbk_high_mae}")
print(f"High Mean Squared Error (MSE): {indusindbk_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {indusindbk_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {indusindbk_high_mape}%")

print("Low Forecasts:", indusindbk_forecast_low)
print(f"Low Mean Absolute Error (MAE): {indusindbk_low_mae}")
print(f"Low Mean Squared Error (MSE): {indusindbk_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {indusindbk_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {indusindbk_low_mape}")


# In[197]:


infosystch_y_close = infosystch['Close'].values
infosystch_y_open = infosystch['Open'].values
infosystch_y_high = infosystch['High'].values
infosystch_y_low = infosystch['Low'].values

infosystch_y_close_scaled = scaler.fit_transform(infosystch_y_close.reshape(-1, 1))
infosystch_y_open_scaled = scaler.fit_transform(infosystch_y_open.reshape(-1, 1))
infosystch_y_high_scaled = scaler.fit_transform(infosystch_y_high.reshape(-1, 1))
infosystch_y_low_scaled = scaler.fit_transform(infosystch_y_low.reshape(-1, 1))

infosystch_close_model = auto_arima(
    infosystch_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

infosystch_open_model = auto_arima(
    infosystch_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

infosystch_high_model = auto_arima(
    infosystch_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

infosystch_low_model = auto_arima(
    infosystch_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

infosystch_close_best_order = infosystch_close_model.get_params()['order']
infosystch_open_best_order = infosystch_open_model.get_params()['order']
infosystch_high_best_order = infosystch_high_model.get_params()['order']
infosystch_low_best_order = infosystch_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {infosystch_close_best_order}")
print(f"Best ARIMA Order for Open: {infosystch_open_best_order}")
print(f"Best ARIMA Order for High: {infosystch_high_best_order}")
print(f"Best ARIMA Order for Low: {infosystch_low_best_order}")

infosystch_close_final_model = sm.tsa.ARIMA(
    infosystch_y_close_scaled,
    order=infosystch_close_best_order
)
infosystch_close_final_model = infosystch_close_final_model.fit()
infosystch_close_forecast = infosystch_close_final_model.forecast(steps=forecast_period)
infosystch_close_forecast = infosystch_close_forecast.reshape(-1, 1)
infosystch_close_forecast = scaler.inverse_transform(infosystch_close_forecast)

infosystch_open_final_model = sm.tsa.ARIMA(
    infosystch_y_open_scaled,
    order=infosystch_open_best_order
)
infosystch_open_final_model = infosystch_open_final_model.fit()
infosystch_open_forecast = infosystch_open_final_model.forecast(steps=forecast_period)
infosystch_open_forecast = infosystch_open_forecast.reshape(-1, 1)
infosystch_open_forecast = scaler.inverse_transform(infosystch_open_forecast)

infosystch_high_final_model = sm.tsa.ARIMA(
    infosystch_y_high_scaled,
    order=infosystch_high_best_order
)
infosystch_high_final_model = infosystch_high_final_model.fit()
infosystch_high_forecast = infosystch_high_final_model.forecast(steps=forecast_period)
infosystch_high_forecast = infosystch_high_forecast.reshape(-1, 1)
infosystch_high_forecast = scaler.inverse_transform(infosystch_high_forecast)

infosystch_low_final_model = sm.tsa.ARIMA(
    infosystch_y_low_scaled,
    order=infosystch_low_best_order
)
infosystch_low_final_model = infosystch_low_final_model.fit()
infosystch_low_forecast = infosystch_low_final_model.forecast(steps=forecast_period)
infosystch_low_forecast = infosystch_low_forecast.reshape(-1, 1)
infosystch_low_forecast = scaler.inverse_transform(infosystch_low_forecast)

print("Close Forecasts:", infosystch_close_forecast)
print("Open Forecasts:", infosystch_open_forecast)
print("High Forecasts:", infosystch_high_forecast)
print("Low Forecasts:", infosystch_low_forecast)


# In[198]:


infosystch_tail_50_data = infosystch.tail(forecast_periods)

infosystch_actual_close_prices = infosystch_tail_50_data['Close'].values
infosystch_actual_open_prices = infosystch_tail_50_data['Open'].values
infosystch_actual_high_prices = infosystch_tail_50_data['High'].values
infosystch_actual_low_prices = infosystch_tail_50_data['Low'].values

infosystch_forecast_close = infosystch_close_final_model.forecast(steps=forecast_periods)
infosystch_forecast_close = infosystch_forecast_close.reshape(-1, 1)
infosystch_forecast_close = scaler.inverse_transform(infosystch_forecast_close)

infosystch_forecast_open = infosystch_open_final_model.forecast(steps=forecast_periods)
infosystch_forecast_open = infosystch_forecast_open.reshape(-1, 1)
infosystch_forecast_open = scaler.inverse_transform(infosystch_forecast_open)

infosystch_forecast_high = infosystch_high_final_model.forecast(steps=forecast_periods)
infosystch_forecast_high = infosystch_forecast_high.reshape(-1, 1)
infosystch_forecast_high = scaler.inverse_transform(infosystch_forecast_high)

infosystch_forecast_low = infosystch_low_final_model.forecast(steps=forecast_periods)
infosystch_forecast_low = infosystch_forecast_low.reshape(-1, 1)
infosystch_forecast_low = scaler.inverse_transform(infosystch_forecast_low)

infosystch_close_mae = mean_absolute_error(infosystch_actual_close_prices, infosystch_forecast_close)
infosystch_close_mse = mean_squared_error(infosystch_actual_close_prices, infosystch_forecast_close)
infosystch_close_rmse = np.sqrt(infosystch_close_mse)

infosystch_open_mae = mean_absolute_error(infosystch_actual_open_prices, infosystch_forecast_open)
infosystch_open_mse = mean_squared_error(infosystch_actual_open_prices, infosystch_forecast_open)
infosystch_open_rmse = np.sqrt(infosystch_open_mse)

infosystch_high_mae = mean_absolute_error(infosystch_actual_high_prices, infosystch_forecast_high)
infosystch_high_mse = mean_squared_error(infosystch_actual_high_prices, infosystch_forecast_high)
infosystch_high_rmse = np.sqrt(infosystch_high_mse)

infosystch_low_mae = mean_absolute_error(infosystch_actual_low_prices, infosystch_forecast_low)
infosystch_low_mse = mean_squared_error(infosystch_actual_low_prices, infosystch_forecast_low)
infosystch_low_rmse = np.sqrt(infosystch_low_mse)

infosystch_close_mape = mean_absolute_percentage_error(infosystch_actual_close_prices, infosystch_forecast_close)
infosystch_open_mape = mean_absolute_percentage_error(infosystch_actual_open_prices, infosystch_forecast_open)
infosystch_high_mape = mean_absolute_percentage_error(infosystch_actual_high_prices, infosystch_forecast_high)
infosystch_low_mape = mean_absolute_percentage_error(infosystch_actual_low_prices, infosystch_forecast_low)

print("Close Forecasts:", infosystch_forecast_close)
print(f"Close Mean Absolute Error (MAE): {infosystch_close_mae}")
print(f"Close Mean Squared Error (MSE): {infosystch_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {infosystch_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {infosystch_close_mape}%")

print("Open Forecasts:", infosystch_forecast_open)
print(f"Open Mean Absolute Error (MAE): {infosystch_open_mae}")
print(f"Open Mean Squared Error (MSE): {infosystch_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {infosystch_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {infosystch_open_mape}%")

print("High Forecasts:", infosystch_forecast_high)
print(f"High Mean Absolute Error (MAE): {infosystch_high_mae}")
print(f"High Mean Squared Error (MSE): {infosystch_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {infosystch_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {infosystch_high_mape}%")

print("Low Forecasts:", infosystch_forecast_low)
print(f"Low Mean Absolute Error (MAE): {infosystch_low_mae}")
print(f"Low Mean Squared Error (MSE): {infosystch_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {infosystch_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {infosystch_low_mape}")


# In[199]:


infy_y_close = infy['Close'].values
infy_y_open = infy['Open'].values
infy_y_high = infy['High'].values
infy_y_low = infy['Low'].values

infy_y_close_scaled = scaler.fit_transform(infy_y_close.reshape(-1, 1))
infy_y_open_scaled = scaler.fit_transform(infy_y_open.reshape(-1, 1))
infy_y_high_scaled = scaler.fit_transform(infy_y_high.reshape(-1, 1))
infy_y_low_scaled = scaler.fit_transform(infy_y_low.reshape(-1, 1))

infy_close_model = auto_arima(
    infy_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

infy_open_model = auto_arima(
    infy_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

infy_high_model = auto_arima(
    infy_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

infy_low_model = auto_arima(
    infy_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

infy_close_best_order = infy_close_model.get_params()['order']
infy_open_best_order = infy_open_model.get_params()['order']
infy_high_best_order = infy_high_model.get_params()['order']
infy_low_best_order = infy_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {infy_close_best_order}")
print(f"Best ARIMA Order for Open: {infy_open_best_order}")
print(f"Best ARIMA Order for High: {infy_high_best_order}")
print(f"Best ARIMA Order for Low: {infy_low_best_order}")

infy_close_final_model = sm.tsa.ARIMA(
    infy_y_close_scaled,
    order=infy_close_best_order
)
infy_close_final_model = infy_close_final_model.fit()
infy_close_forecast = infy_close_final_model.forecast(steps=forecast_period)
infy_close_forecast = infy_close_forecast.reshape(-1, 1)
infy_close_forecast = scaler.inverse_transform(infy_close_forecast)

infy_open_final_model = sm.tsa.ARIMA(
    infy_y_open_scaled,
    order=infy_open_best_order
)
infy_open_final_model = infy_open_final_model.fit()
infy_open_forecast = infy_open_final_model.forecast(steps=forecast_period)
infy_open_forecast = infy_open_forecast.reshape(-1, 1)
infy_open_forecast = scaler.inverse_transform(infy_open_forecast)

infy_high_final_model = sm.tsa.ARIMA(
    infy_y_high_scaled,
    order=infy_high_best_order
)
infy_high_final_model = infy_high_final_model.fit()
infy_high_forecast = infy_high_final_model.forecast(steps=forecast_period)
infy_high_forecast = infy_high_forecast.reshape(-1, 1)
infy_high_forecast = scaler.inverse_transform(infy_high_forecast)

infy_low_final_model = sm.tsa.ARIMA(
    infy_y_low_scaled,
    order=infy_low_best_order
)
infy_low_final_model = infy_low_final_model.fit()
infy_low_forecast = infy_low_final_model.forecast(steps=forecast_period)
infy_low_forecast = infy_low_forecast.reshape(-1, 1)
infy_low_forecast = scaler.inverse_transform(infy_low_forecast)

print("Close Forecasts:", infy_close_forecast)
print("Open Forecasts:", infy_open_forecast)
print("High Forecasts:", infy_high_forecast)
print("Low Forecasts:", infy_low_forecast)


# In[200]:


infy_tail_50_data = infy.tail(forecast_periods)

infy_actual_close_prices = infy_tail_50_data['Close'].values
infy_actual_open_prices = infy_tail_50_data['Open'].values
infy_actual_high_prices = infy_tail_50_data['High'].values
infy_actual_low_prices = infy_tail_50_data['Low'].values

infy_forecast_close = infy_close_final_model.forecast(steps=forecast_periods)
infy_forecast_close = infy_forecast_close.reshape(-1, 1)
infy_forecast_close = scaler.inverse_transform(infy_forecast_close)

infy_forecast_open = infy_open_final_model.forecast(steps=forecast_periods)
infy_forecast_open = infy_forecast_open.reshape(-1, 1)
infy_forecast_open = scaler.inverse_transform(infy_forecast_open)

infy_forecast_high = infy_high_final_model.forecast(steps=forecast_periods)
infy_forecast_high = infy_forecast_high.reshape(-1, 1)
infy_forecast_high = scaler.inverse_transform(infy_forecast_high)

infy_forecast_low = infy_low_final_model.forecast(steps=forecast_periods)
infy_forecast_low = infy_forecast_low.reshape(-1, 1)
infy_forecast_low = scaler.inverse_transform(infy_forecast_low)

infy_close_mae = mean_absolute_error(infy_actual_close_prices, infy_forecast_close)
infy_close_mse = mean_squared_error(infy_actual_close_prices, infy_forecast_close)
infy_close_rmse = np.sqrt(infy_close_mse)

infy_open_mae = mean_absolute_error(infy_actual_open_prices, infy_forecast_open)
infy_open_mse = mean_squared_error(infy_actual_open_prices, infy_forecast_open)
infy_open_rmse = np.sqrt(infy_open_mse)

infy_high_mae = mean_absolute_error(infy_actual_high_prices, infy_forecast_high)
infy_high_mse = mean_squared_error(infy_actual_high_prices, infy_forecast_high)
infy_high_rmse = np.sqrt(infy_high_mse)

infy_low_mae = mean_absolute_error(infy_actual_low_prices, infy_forecast_low)
infy_low_mse = mean_squared_error(infy_actual_low_prices, infy_forecast_low)
infy_low_rmse = np.sqrt(infy_low_mse)

infy_close_mape = mean_absolute_percentage_error(infy_actual_close_prices, infy_forecast_close)
infy_open_mape = mean_absolute_percentage_error(infy_actual_open_prices, infy_forecast_open)
infy_high_mape = mean_absolute_percentage_error(infy_actual_high_prices, infy_forecast_high)
infy_low_mape = mean_absolute_percentage_error(infy_actual_low_prices, infy_forecast_low)

print("Close Forecasts:", infy_forecast_close)
print(f"Close Mean Absolute Error (MAE): {infy_close_mae}")
print(f"Close Mean Squared Error (MSE): {infy_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {infy_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {infy_close_mape}%")

print("Open Forecasts:", infy_forecast_open)
print(f"Open Mean Absolute Error (MAE): {infy_open_mae}")
print(f"Open Mean Squared Error (MSE): {infy_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {infy_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {infy_open_mape}%")

print("High Forecasts:", infy_forecast_high)
print(f"High Mean Absolute Error (MAE): {infy_high_mae}")
print(f"High Mean Squared Error (MSE): {infy_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {infy_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {infy_high_mape}%")

print("Low Forecasts:", infy_forecast_low)
print(f"Low Mean Absolute Error (MAE): {infy_low_mae}")
print(f"Low Mean Squared Error (MSE): {infy_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {infy_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {infy_low_mape}")


# In[201]:


ioc_y_close = ioc['Close'].values
ioc_y_open = ioc['Open'].values
ioc_y_high = ioc['High'].values
ioc_y_low = ioc['Low'].values

ioc_y_close_scaled = scaler.fit_transform(ioc_y_close.reshape(-1, 1))
ioc_y_open_scaled = scaler.fit_transform(ioc_y_open.reshape(-1, 1))
ioc_y_high_scaled = scaler.fit_transform(ioc_y_high.reshape(-1, 1))
ioc_y_low_scaled = scaler.fit_transform(ioc_y_low.reshape(-1, 1))

ioc_close_model = auto_arima(
    ioc_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ioc_open_model = auto_arima(
    ioc_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ioc_high_model = auto_arima(
    ioc_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ioc_low_model = auto_arima(
    ioc_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ioc_close_best_order = ioc_close_model.get_params()['order']
ioc_open_best_order = ioc_open_model.get_params()['order']
ioc_high_best_order = ioc_high_model.get_params()['order']
ioc_low_best_order = ioc_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {ioc_close_best_order}")
print(f"Best ARIMA Order for Open: {ioc_open_best_order}")
print(f"Best ARIMA Order for High: {ioc_high_best_order}")
print(f"Best ARIMA Order for Low: {ioc_low_best_order}")

ioc_close_final_model = sm.tsa.ARIMA(
    ioc_y_close_scaled,
    order=ioc_close_best_order
)
ioc_close_final_model = ioc_close_final_model.fit()
ioc_close_forecast = ioc_close_final_model.forecast(steps=forecast_period)
ioc_close_forecast = ioc_close_forecast.reshape(-1, 1)
ioc_close_forecast = scaler.inverse_transform(ioc_close_forecast)

ioc_open_final_model = sm.tsa.ARIMA(
    ioc_y_open_scaled,
    order=ioc_open_best_order
)
ioc_open_final_model = ioc_open_final_model.fit()
ioc_open_forecast = ioc_open_final_model.forecast(steps=forecast_period)
ioc_open_forecast = ioc_open_forecast.reshape(-1, 1)
ioc_open_forecast = scaler.inverse_transform(ioc_open_forecast)

ioc_high_final_model = sm.tsa.ARIMA(
    ioc_y_high_scaled,
    order=ioc_high_best_order
)
ioc_high_final_model = ioc_high_final_model.fit()
ioc_high_forecast = ioc_high_final_model.forecast(steps=forecast_period)
ioc_high_forecast = ioc_high_forecast.reshape(-1, 1)
ioc_high_forecast = scaler.inverse_transform(ioc_high_forecast)

ioc_low_final_model = sm.tsa.ARIMA(
    ioc_y_low_scaled,
    order=ioc_low_best_order
)
ioc_low_final_model = ioc_low_final_model.fit()
ioc_low_forecast = ioc_low_final_model.forecast(steps=forecast_period)
ioc_low_forecast = ioc_low_forecast.reshape(-1, 1)
ioc_low_forecast = scaler.inverse_transform(ioc_low_forecast)

print("Close Forecasts:", ioc_close_forecast)
print("Open Forecasts:", ioc_open_forecast)
print("High Forecasts:", ioc_high_forecast)
print("Low Forecasts:", ioc_low_forecast)


# In[202]:


ioc_tail_50_data = ioc.tail(forecast_periods)

ioc_actual_close_prices = ioc_tail_50_data['Close'].values
ioc_actual_open_prices = ioc_tail_50_data['Open'].values
ioc_actual_high_prices = ioc_tail_50_data['High'].values
ioc_actual_low_prices = ioc_tail_50_data['Low'].values

ioc_forecast_close = ioc_close_final_model.forecast(steps=forecast_periods)
ioc_forecast_close = ioc_forecast_close.reshape(-1, 1)
ioc_forecast_close = scaler.inverse_transform(ioc_forecast_close)

ioc_forecast_open = ioc_open_final_model.forecast(steps=forecast_periods)
ioc_forecast_open = ioc_forecast_open.reshape(-1, 1)
ioc_forecast_open = scaler.inverse_transform(ioc_forecast_open)

ioc_forecast_high = ioc_high_final_model.forecast(steps=forecast_periods)
ioc_forecast_high = ioc_forecast_high.reshape(-1, 1)
ioc_forecast_high = scaler.inverse_transform(ioc_forecast_high)

ioc_forecast_low = ioc_low_final_model.forecast(steps=forecast_periods)
ioc_forecast_low = ioc_forecast_low.reshape(-1, 1)
ioc_forecast_low = scaler.inverse_transform(ioc_forecast_low)

ioc_close_mae = mean_absolute_error(ioc_actual_close_prices, ioc_forecast_close)
ioc_close_mse = mean_squared_error(ioc_actual_close_prices, ioc_forecast_close)
ioc_close_rmse = np.sqrt(ioc_close_mse)

ioc_open_mae = mean_absolute_error(ioc_actual_open_prices, ioc_forecast_open)
ioc_open_mse = mean_squared_error(ioc_actual_open_prices, ioc_forecast_open)
ioc_open_rmse = np.sqrt(ioc_open_mse)

ioc_high_mae = mean_absolute_error(ioc_actual_high_prices, ioc_forecast_high)
ioc_high_mse = mean_squared_error(ioc_actual_high_prices, ioc_forecast_high)
ioc_high_rmse = np.sqrt(ioc_high_mse)

ioc_low_mae = mean_absolute_error(ioc_actual_low_prices, ioc_forecast_low)
ioc_low_mse = mean_squared_error(ioc_actual_low_prices, ioc_forecast_low)
ioc_low_rmse = np.sqrt(ioc_low_mse)

ioc_close_mape = mean_absolute_percentage_error(ioc_actual_close_prices, ioc_forecast_close)
ioc_open_mape = mean_absolute_percentage_error(ioc_actual_open_prices, ioc_forecast_open)
ioc_high_mape = mean_absolute_percentage_error(ioc_actual_high_prices, ioc_forecast_high)
ioc_low_mape = mean_absolute_percentage_error(ioc_actual_low_prices, ioc_forecast_low)

print("Close Forecasts:", ioc_forecast_close)
print(f"Close Mean Absolute Error (MAE): {ioc_close_mae}")
print(f"Close Mean Squared Error (MSE): {ioc_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {ioc_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {ioc_close_mape}%")

print("Open Forecasts:", ioc_forecast_open)
print(f"Open Mean Absolute Error (MAE): {ioc_open_mae}")
print(f"Open Mean Squared Error (MSE): {ioc_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {ioc_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {ioc_open_mape}%")

print("High Forecasts:", ioc_forecast_high)
print(f"High Mean Absolute Error (MAE): {ioc_high_mae}")
print(f"High Mean Squared Error (MSE): {ioc_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {ioc_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {ioc_high_mape}%")

print("Low Forecasts:", ioc_forecast_low)
print(f"Low Mean Absolute Error (MAE): {ioc_low_mae}")
print(f"Low Mean Squared Error (MSE): {ioc_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {ioc_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {ioc_low_mape}")


# In[203]:


itc_y_close = itc['Close'].values
itc_y_open = itc['Open'].values
itc_y_high = itc['High'].values
itc_y_low = itc['Low'].values

itc_y_close_scaled = scaler.fit_transform(itc_y_close.reshape(-1, 1))
itc_y_open_scaled = scaler.fit_transform(itc_y_open.reshape(-1, 1))
itc_y_high_scaled = scaler.fit_transform(itc_y_high.reshape(-1, 1))
itc_y_low_scaled = scaler.fit_transform(itc_y_low.reshape(-1, 1))

itc_close_model = auto_arima(
    itc_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

itc_open_model = auto_arima(
    itc_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

itc_high_model = auto_arima(
    itc_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

itc_low_model = auto_arima(
    itc_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

itc_close_best_order = itc_close_model.get_params()['order']
itc_open_best_order = itc_open_model.get_params()['order']
itc_high_best_order = itc_high_model.get_params()['order']
itc_low_best_order = itc_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {itc_close_best_order}")
print(f"Best ARIMA Order for Open: {itc_open_best_order}")
print(f"Best ARIMA Order for High: {itc_high_best_order}")
print(f"Best ARIMA Order for Low: {itc_low_best_order}")

itc_close_final_model = sm.tsa.ARIMA(
    itc_y_close_scaled,
    order=itc_close_best_order
)
itc_close_final_model = itc_close_final_model.fit()
itc_close_forecast = itc_close_final_model.forecast(steps=forecast_period)
itc_close_forecast = itc_close_forecast.reshape(-1, 1)
itc_close_forecast = scaler.inverse_transform(itc_close_forecast)

itc_open_final_model = sm.tsa.ARIMA(
    itc_y_open_scaled,
    order=itc_open_best_order
)
itc_open_final_model = itc_open_final_model.fit()
itc_open_forecast = itc_open_final_model.forecast(steps=forecast_period)
itc_open_forecast = itc_open_forecast.reshape(-1, 1)
itc_open_forecast = scaler.inverse_transform(itc_open_forecast)

itc_high_final_model = sm.tsa.ARIMA(
    itc_y_high_scaled,
    order=itc_high_best_order
)
itc_high_final_model = itc_high_final_model.fit()
itc_high_forecast = itc_high_final_model.forecast(steps=forecast_period)
itc_high_forecast = itc_high_forecast.reshape(-1, 1)
itc_high_forecast = scaler.inverse_transform(itc_high_forecast)

itc_low_final_model = sm.tsa.ARIMA(
    itc_y_low_scaled,
    order=itc_low_best_order
)
itc_low_final_model = itc_low_final_model.fit()
itc_low_forecast = itc_low_final_model.forecast(steps=forecast_period)
itc_low_forecast = itc_low_forecast.reshape(-1, 1)
itc_low_forecast = scaler.inverse_transform(itc_low_forecast)

print("Close Forecasts:", itc_close_forecast)
print("Open Forecasts:", itc_open_forecast)
print("High Forecasts:", itc_high_forecast)
print("Low Forecasts:", itc_low_forecast)


# In[204]:


itc_tail_50_data = itc.tail(forecast_periods)

itc_actual_close_prices = itc_tail_50_data['Close'].values
itc_actual_open_prices = itc_tail_50_data['Open'].values
itc_actual_high_prices = itc_tail_50_data['High'].values
itc_actual_low_prices = itc_tail_50_data['Low'].values

itc_forecast_close = itc_close_final_model.forecast(steps=forecast_periods)
itc_forecast_close = itc_forecast_close.reshape(-1, 1)
itc_forecast_close = scaler.inverse_transform(itc_forecast_close)

itc_forecast_open = itc_open_final_model.forecast(steps=forecast_periods)
itc_forecast_open = itc_forecast_open.reshape(-1, 1)
itc_forecast_open = scaler.inverse_transform(itc_forecast_open)

itc_forecast_high = itc_high_final_model.forecast(steps=forecast_periods)
itc_forecast_high = itc_forecast_high.reshape(-1, 1)
itc_forecast_high = scaler.inverse_transform(itc_forecast_high)

itc_forecast_low = itc_low_final_model.forecast(steps=forecast_periods)
itc_forecast_low = itc_forecast_low.reshape(-1, 1)
itc_forecast_low = scaler.inverse_transform(itc_forecast_low)

itc_close_mae = mean_absolute_error(itc_actual_close_prices, itc_forecast_close)
itc_close_mse = mean_squared_error(itc_actual_close_prices, itc_forecast_close)
itc_close_rmse = np.sqrt(itc_close_mse)

itc_open_mae = mean_absolute_error(itc_actual_open_prices, itc_forecast_open)
itc_open_mse = mean_squared_error(itc_actual_open_prices, itc_forecast_open)
itc_open_rmse = np.sqrt(itc_open_mse)

itc_high_mae = mean_absolute_error(itc_actual_high_prices, itc_forecast_high)
itc_high_mse = mean_squared_error(itc_actual_high_prices, itc_forecast_high)
itc_high_rmse = np.sqrt(itc_high_mse)

itc_low_mae = mean_absolute_error(itc_actual_low_prices, itc_forecast_low)
itc_low_mse = mean_squared_error(itc_actual_low_prices, itc_forecast_low)
itc_low_rmse = np.sqrt(itc_low_mse)

itc_close_mape = mean_absolute_percentage_error(itc_actual_close_prices, itc_forecast_close)
itc_open_mape = mean_absolute_percentage_error(itc_actual_open_prices, itc_forecast_open)
itc_high_mape = mean_absolute_percentage_error(itc_actual_high_prices, itc_forecast_high)
itc_low_mape = mean_absolute_percentage_error(itc_actual_low_prices, itc_forecast_low)

print("Close Forecasts:", itc_forecast_close)
print(f"Close Mean Absolute Error (MAE): {itc_close_mae}")
print(f"Close Mean Squared Error (MSE): {itc_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {itc_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {itc_close_mape}%")

print("Open Forecasts:", itc_forecast_open)
print(f"Open Mean Absolute Error (MAE): {itc_open_mae}")
print(f"Open Mean Squared Error (MSE): {itc_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {itc_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {itc_open_mape}%")

print("High Forecasts:", itc_forecast_high)
print(f"High Mean Absolute Error (MAE): {itc_high_mae}")
print(f"High Mean Squared Error (MSE): {itc_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {itc_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {itc_high_mape}%")

print("Low Forecasts:", itc_forecast_low)
print(f"Low Mean Absolute Error (MAE): {itc_low_mae}")
print(f"Low Mean Squared Error (MSE): {itc_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {itc_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {itc_low_mape}")


# In[209]:


kotakmah_y_close = kotakmah['Close'].values
kotakmah_y_open = kotakmah['Open'].values
kotakmah_y_high = kotakmah['High'].values
kotakmah_y_low = kotakmah['Low'].values

kotakmah_y_close_scaled = scaler.fit_transform(kotakmah_y_close.reshape(-1, 1))
kotakmah_y_open_scaled = scaler.fit_transform(kotakmah_y_open.reshape(-1, 1))
kotakmah_y_high_scaled = scaler.fit_transform(kotakmah_y_high.reshape(-1, 1))
kotakmah_y_low_scaled = scaler.fit_transform(kotakmah_y_low.reshape(-1, 1))

kotakmah_close_model = auto_arima(
    kotakmah_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

kotakmah_open_model = auto_arima(
    kotakmah_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

kotakmah_high_model = auto_arima(
    kotakmah_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

kotakmah_low_model = auto_arima(
    kotakmah_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

kotakmah_close_best_order = kotakmah_close_model.get_params()['order']
kotakmah_open_best_order = kotakmah_open_model.get_params()['order']
kotakmah_high_best_order = kotakmah_high_model.get_params()['order']
kotakmah_low_best_order = kotakmah_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {kotakmah_close_best_order}")
print(f"Best ARIMA Order for Open: {kotakmah_open_best_order}")
print(f"Best ARIMA Order for High: {kotakmah_high_best_order}")
print(f"Best ARIMA Order for Low: {kotakmah_low_best_order}")

kotakmah_close_final_model = sm.tsa.ARIMA(
    kotakmah_y_close_scaled,
    order=kotakmah_close_best_order
)
kotakmah_close_final_model = kotakmah_close_final_model.fit()
kotakmah_close_forecast = kotakmah_close_final_model.forecast(steps=forecast_period)
kotakmah_close_forecast = kotakmah_close_forecast.reshape(-1, 1)
kotakmah_close_forecast = scaler.inverse_transform(kotakmah_close_forecast)

kotakmah_open_final_model = sm.tsa.ARIMA(
    kotakmah_y_open_scaled,
    order=kotakmah_open_best_order
)
kotakmah_open_final_model = kotakmah_open_final_model.fit()
kotakmah_open_forecast = kotakmah_open_final_model.forecast(steps=forecast_period)
kotakmah_open_forecast = kotakmah_open_forecast.reshape(-1, 1)
kotakmah_open_forecast = scaler.inverse_transform(kotakmah_open_forecast)

kotakmah_high_final_model = sm.tsa.ARIMA(
    kotakmah_y_high_scaled,
    order=kotakmah_high_best_order
)
kotakmah_high_final_model = kotakmah_high_final_model.fit()
kotakmah_high_forecast = kotakmah_high_final_model.forecast(steps=forecast_period)
kotakmah_high_forecast = kotakmah_high_forecast.reshape(-1, 1)
kotakmah_high_forecast = scaler.inverse_transform(kotakmah_high_forecast)

kotakmah_low_final_model = sm.tsa.ARIMA(
    kotakmah_y_low_scaled,
    order=kotakmah_low_best_order
)
kotakmah_low_final_model = kotakmah_low_final_model.fit()
kotakmah_low_forecast = kotakmah_low_final_model.forecast(steps=forecast_period)
kotakmah_low_forecast = kotakmah_low_forecast.reshape(-1, 1)
kotakmah_low_forecast = scaler.inverse_transform(kotakmah_low_forecast)

print("Close Forecasts:", kotakmah_close_forecast)
print("Open Forecasts:", kotakmah_open_forecast)
print("High Forecasts:", kotakmah_high_forecast)
print("Low Forecasts:", kotakmah_low_forecast)


# In[210]:


kotakmah_tail_50_data = kotakmah.tail(forecast_periods)

kotakmah_actual_close_prices = kotakmah_tail_50_data['Close'].values
kotakmah_actual_open_prices = kotakmah_tail_50_data['Open'].values
kotakmah_actual_high_prices = kotakmah_tail_50_data['High'].values
kotakmah_actual_low_prices = kotakmah_tail_50_data['Low'].values

kotakmah_forecast_close = kotakmah_close_final_model.forecast(steps=forecast_periods)
kotakmah_forecast_close = kotakmah_forecast_close.reshape(-1, 1)
kotakmah_forecast_close = scaler.inverse_transform(kotakmah_forecast_close)

kotakmah_forecast_open = kotakmah_open_final_model.forecast(steps=forecast_periods)
kotakmah_forecast_open = kotakmah_forecast_open.reshape(-1, 1)
kotakmah_forecast_open = scaler.inverse_transform(kotakmah_forecast_open)

kotakmah_forecast_high = kotakmah_high_final_model.forecast(steps=forecast_periods)
kotakmah_forecast_high = kotakmah_forecast_high.reshape(-1, 1)
kotakmah_forecast_high = scaler.inverse_transform(kotakmah_forecast_high)

kotakmah_forecast_low = kotakmah_low_final_model.forecast(steps=forecast_periods)
kotakmah_forecast_low = kotakmah_forecast_low.reshape(-1, 1)
kotakmah_forecast_low = scaler.inverse_transform(kotakmah_forecast_low)

kotakmah_close_mae = mean_absolute_error(kotakmah_actual_close_prices, kotakmah_forecast_close)
kotakmah_close_mse = mean_squared_error(kotakmah_actual_close_prices, kotakmah_forecast_close)
kotakmah_close_rmse = np.sqrt(kotakmah_close_mse)

kotakmah_open_mae = mean_absolute_error(kotakmah_actual_open_prices, kotakmah_forecast_open)
kotakmah_open_mse = mean_squared_error(kotakmah_actual_open_prices, kotakmah_forecast_open)
kotakmah_open_rmse = np.sqrt(kotakmah_open_mse)

kotakmah_high_mae = mean_absolute_error(kotakmah_actual_high_prices, kotakmah_forecast_high)
kotakmah_high_mse = mean_squared_error(kotakmah_actual_high_prices, kotakmah_forecast_high)
kotakmah_high_rmse = np.sqrt(kotakmah_high_mse)

kotakmah_low_mae = mean_absolute_error(kotakmah_actual_low_prices, kotakmah_forecast_low)
kotakmah_low_mse = mean_squared_error(kotakmah_actual_low_prices, kotakmah_forecast_low)
kotakmah_low_rmse = np.sqrt(kotakmah_low_mse)

kotakmah_close_mape = mean_absolute_percentage_error(kotakmah_actual_close_prices, kotakmah_forecast_close)
kotakmah_open_mape = mean_absolute_percentage_error(kotakmah_actual_open_prices, kotakmah_forecast_open)
kotakmah_high_mape = mean_absolute_percentage_error(kotakmah_actual_high_prices, kotakmah_forecast_high)
kotakmah_low_mape = mean_absolute_percentage_error(kotakmah_actual_low_prices, kotakmah_forecast_low)

print("Close Forecasts:", kotakmah_forecast_close)
print(f"Close Mean Absolute Error (MAE): {kotakmah_close_mae}")
print(f"Close Mean Squared Error (MSE): {kotakmah_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {kotakmah_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {kotakmah_close_mape}%")

print("Open Forecasts:", kotakmah_forecast_open)
print(f"Open Mean Absolute Error (MAE): {kotakmah_open_mae}")
print(f"Open Mean Squared Error (MSE): {kotakmah_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {kotakmah_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {kotakmah_open_mape}%")

print("High Forecasts:", kotakmah_forecast_high)
print(f"High Mean Absolute Error (MAE): {kotakmah_high_mae}")
print(f"High Mean Squared Error (MSE): {kotakmah_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {kotakmah_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {kotakmah_high_mape}%")

print("Low Forecasts:", kotakmah_forecast_low)
print(f"Low Mean Absolute Error (MAE): {kotakmah_low_mae}")
print(f"Low Mean Squared Error (MSE): {kotakmah_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {kotakmah_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {kotakmah_low_mape}")


#  lt = company_datasets['LT'] 
#  mandm = company_datasets['M&M'] 
#  maruti = company_datasets['MARUTI']
#  nestleind = company_datasets['NESTLEIND']
#  ntpc = company_datasets['NTPC'] 
#  ongc = company_datasets['ONGC'] powergrid = company_datasets['POWERGRID'] reliance = company_datasets['RELIANCE'] sbin = company_datasets['SBIN'] shreecem = company_datasets['SHREECEM'] sunpharma = company_datasets['SUNPHARMA'] telco = company_datasets['TELCO'] tatamotors = company_datasets['TATAMOTORS'] tisco = company_datasets['TISCO'] tatasteel = company_datasets['TATASTEEL'] tcs = company_datasets['TCS'] techm = company_datasets['TECHM'] titan = company_datasets['TITAN'] ultracemco = company_datasets['ULTRACEMCO'] uniphos = company_datasets['UNIPHOS'] upl = company_datasets['UPL'] sesagoa = company_datasets['SESAGOA'] sslt = company_datasets['SSLT'] vedl = company_datasets['VEDL'] wipro = company_datasets['WIPRO'] zeetele = company_datasets['ZEETELE'] zeel = company_datasets['ZEEL']

# In[211]:


kotakbank_y_close = kotakbank['Close'].values
kotakbank_y_open = kotakbank['Open'].values
kotakbank_y_high = kotakbank['High'].values
kotakbank_y_low = kotakbank['Low'].values

kotakbank_y_close_scaled = scaler.fit_transform(kotakbank_y_close.reshape(-1, 1))
kotakbank_y_open_scaled = scaler.fit_transform(kotakbank_y_open.reshape(-1, 1))
kotakbank_y_high_scaled = scaler.fit_transform(kotakbank_y_high.reshape(-1, 1))
kotakbank_y_low_scaled = scaler.fit_transform(kotakbank_y_low.reshape(-1, 1))

kotakbank_close_model = auto_arima(
    kotakbank_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

kotakbank_open_model = auto_arima(
    kotakbank_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

kotakbank_high_model = auto_arima(
    kotakbank_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

kotakbank_low_model = auto_arima(
    kotakbank_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

kotakbank_close_best_order = kotakbank_close_model.get_params()['order']
kotakbank_open_best_order = kotakbank_open_model.get_params()['order']
kotakbank_high_best_order = kotakbank_high_model.get_params()['order']
kotakbank_low_best_order = kotakbank_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {kotakbank_close_best_order}")
print(f"Best ARIMA Order for Open: {kotakbank_open_best_order}")
print(f"Best ARIMA Order for High: {kotakbank_high_best_order}")
print(f"Best ARIMA Order for Low: {kotakbank_low_best_order}")

kotakbank_close_final_model = sm.tsa.ARIMA(
    kotakbank_y_close_scaled,
    order=kotakbank_close_best_order
)
kotakbank_close_final_model = kotakbank_close_final_model.fit()
kotakbank_close_forecast = kotakbank_close_final_model.forecast(steps=forecast_period)
kotakbank_close_forecast = kotakbank_close_forecast.reshape(-1, 1)
kotakbank_close_forecast = scaler.inverse_transform(kotakbank_close_forecast)

kotakbank_open_final_model = sm.tsa.ARIMA(
    kotakbank_y_open_scaled,
    order=kotakbank_open_best_order
)
kotakbank_open_final_model = kotakbank_open_final_model.fit()
kotakbank_open_forecast = kotakbank_open_final_model.forecast(steps=forecast_period)
kotakbank_open_forecast = kotakbank_open_forecast.reshape(-1, 1)
kotakbank_open_forecast = scaler.inverse_transform(kotakbank_open_forecast)

kotakbank_high_final_model = sm.tsa.ARIMA(
    kotakbank_y_high_scaled,
    order=kotakbank_high_best_order
)
kotakbank_high_final_model = kotakbank_high_final_model.fit()
kotakbank_high_forecast = kotakbank_high_final_model.forecast(steps=forecast_period)
kotakbank_high_forecast = kotakbank_high_forecast.reshape(-1, 1)
kotakbank_high_forecast = scaler.inverse_transform(kotakbank_high_forecast)

kotakbank_low_final_model = sm.tsa.ARIMA(
    kotakbank_y_low_scaled,
    order=kotakbank_low_best_order
)
kotakbank_low_final_model = kotakbank_low_final_model.fit()
kotakbank_low_forecast = kotakbank_low_final_model.forecast(steps=forecast_period)
kotakbank_low_forecast = kotakbank_low_forecast.reshape(-1, 1)
kotakbank_low_forecast = scaler.inverse_transform(kotakbank_low_forecast)

print("Close Forecasts:", kotakbank_close_forecast)
print("Open Forecasts:", kotakbank_open_forecast)
print("High Forecasts:", kotakbank_high_forecast)
print("Low Forecasts:", kotakbank_low_forecast)


# In[212]:


kotakbank_tail_50_data = kotakbank.tail(forecast_periods)

kotakbank_actual_close_prices = kotakbank_tail_50_data['Close'].values
kotakbank_actual_open_prices = kotakbank_tail_50_data['Open'].values
kotakbank_actual_high_prices = kotakbank_tail_50_data['High'].values
kotakbank_actual_low_prices = kotakbank_tail_50_data['Low'].values

kotakbank_forecast_close = kotakbank_close_final_model.forecast(steps=forecast_periods)
kotakbank_forecast_close = kotakbank_forecast_close.reshape(-1, 1)
kotakbank_forecast_close = scaler.inverse_transform(kotakbank_forecast_close)

kotakbank_forecast_open = kotakbank_open_final_model.forecast(steps=forecast_periods)
kotakbank_forecast_open = kotakbank_forecast_open.reshape(-1, 1)
kotakbank_forecast_open = scaler.inverse_transform(kotakbank_forecast_open)

kotakbank_forecast_high = kotakbank_high_final_model.forecast(steps=forecast_periods)
kotakbank_forecast_high = kotakbank_forecast_high.reshape(-1, 1)
kotakbank_forecast_high = scaler.inverse_transform(kotakbank_forecast_high)

kotakbank_forecast_low = kotakbank_low_final_model.forecast(steps=forecast_periods)
kotakbank_forecast_low = kotakbank_forecast_low.reshape(-1, 1)
kotakbank_forecast_low = scaler.inverse_transform(kotakbank_forecast_low)

kotakbank_close_mae = mean_absolute_error(kotakbank_actual_close_prices, kotakbank_forecast_close)
kotakbank_close_mse = mean_squared_error(kotakbank_actual_close_prices, kotakbank_forecast_close)
kotakbank_close_rmse = np.sqrt(kotakbank_close_mse)

kotakbank_open_mae = mean_absolute_error(kotakbank_actual_open_prices, kotakbank_forecast_open)
kotakbank_open_mse = mean_squared_error(kotakbank_actual_open_prices, kotakbank_forecast_open)
kotakbank_open_rmse = np.sqrt(kotakbank_open_mse)

kotakbank_high_mae = mean_absolute_error(kotakbank_actual_high_prices, kotakbank_forecast_high)
kotakbank_high_mse = mean_squared_error(kotakbank_actual_high_prices, kotakbank_forecast_high)
kotakbank_high_rmse = np.sqrt(kotakbank_high_mse)

kotakbank_low_mae = mean_absolute_error(kotakbank_actual_low_prices, kotakbank_forecast_low)
kotakbank_low_mse = mean_squared_error(kotakbank_actual_low_prices, kotakbank_forecast_low)
kotakbank_low_rmse = np.sqrt(kotakbank_low_mse)

kotakbank_close_mape = mean_absolute_percentage_error(kotakbank_actual_close_prices, kotakbank_forecast_close)
kotakbank_open_mape = mean_absolute_percentage_error(kotakbank_actual_open_prices, kotakbank_forecast_open)
kotakbank_high_mape = mean_absolute_percentage_error(kotakbank_actual_high_prices, kotakbank_forecast_high)
kotakbank_low_mape = mean_absolute_percentage_error(kotakbank_actual_low_prices, kotakbank_forecast_low)

print("Close Forecasts:", kotakbank_forecast_close)
print(f"Close Mean Absolute Error (MAE): {kotakbank_close_mae}")
print(f"Close Mean Squared Error (MSE): {kotakbank_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {kotakbank_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {kotakbank_close_mape}%")

print("Open Forecasts:", kotakbank_forecast_open)
print(f"Open Mean Absolute Error (MAE): {kotakbank_open_mae}")
print(f"Open Mean Squared Error (MSE): {kotakbank_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {kotakbank_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {kotakbank_open_mape}%")

print("High Forecasts:", kotakbank_forecast_high)
print(f"High Mean Absolute Error (MAE): {kotakbank_high_mae}")
print(f"High Mean Squared Error (MSE): {kotakbank_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {kotakbank_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {kotakbank_high_mape}%")

print("Low Forecasts:", kotakbank_forecast_low)
print(f"Low Mean Absolute Error (MAE): {kotakbank_low_mae}")
print(f"Low Mean Squared Error (MSE): {kotakbank_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {kotakbank_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {kotakbank_low_mape}")


# In[213]:


lt_y_close = lt['Close'].values
lt_y_open = lt['Open'].values
lt_y_high = lt['High'].values
lt_y_low = lt['Low'].values

lt_y_close_scaled = scaler.fit_transform(lt_y_close.reshape(-1, 1))
lt_y_open_scaled = scaler.fit_transform(lt_y_open.reshape(-1, 1))
lt_y_high_scaled = scaler.fit_transform(lt_y_high.reshape(-1, 1))
lt_y_low_scaled = scaler.fit_transform(lt_y_low.reshape(-1, 1))

lt_close_model = auto_arima(
    lt_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

lt_open_model = auto_arima(
    lt_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

lt_high_model = auto_arima(
    lt_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

lt_low_model = auto_arima(
    lt_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

lt_close_best_order = lt_close_model.get_params()['order']
lt_open_best_order = lt_open_model.get_params()['order']
lt_high_best_order = lt_high_model.get_params()['order']
lt_low_best_order = lt_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {lt_close_best_order}")
print(f"Best ARIMA Order for Open: {lt_open_best_order}")
print(f"Best ARIMA Order for High: {lt_high_best_order}")
print(f"Best ARIMA Order for Low: {lt_low_best_order}")

lt_close_final_model = sm.tsa.ARIMA(
    lt_y_close_scaled,
    order=lt_close_best_order
)
lt_close_final_model = lt_close_final_model.fit()
lt_close_forecast = lt_close_final_model.forecast(steps=forecast_period)
lt_close_forecast = lt_close_forecast.reshape(-1, 1)
lt_close_forecast = scaler.inverse_transform(lt_close_forecast)

lt_open_final_model = sm.tsa.ARIMA(
    lt_y_open_scaled,
    order=lt_open_best_order
)
lt_open_final_model = lt_open_final_model.fit()
lt_open_forecast = lt_open_final_model.forecast(steps=forecast_period)
lt_open_forecast = lt_open_forecast.reshape(-1, 1)
lt_open_forecast = scaler.inverse_transform(lt_open_forecast)

lt_high_final_model = sm.tsa.ARIMA(
    lt_y_high_scaled,
    order=lt_high_best_order
)
lt_high_final_model = lt_high_final_model.fit()
lt_high_forecast = lt_high_final_model.forecast(steps=forecast_period)
lt_high_forecast = lt_high_forecast.reshape(-1, 1)
lt_high_forecast = scaler.inverse_transform(lt_high_forecast)

lt_low_final_model = sm.tsa.ARIMA(
    lt_y_low_scaled,
    order=lt_low_best_order
)
lt_low_final_model = lt_low_final_model.fit()
lt_low_forecast = lt_low_final_model.forecast(steps=forecast_period)
lt_low_forecast = lt_low_forecast.reshape(-1, 1)
lt_low_forecast = scaler.inverse_transform(lt_low_forecast)

print("Close Forecasts:", lt_close_forecast)
print("Open Forecasts:", lt_open_forecast)
print("High Forecasts:", lt_high_forecast)
print("Low Forecasts:", lt_low_forecast)


# In[214]:


lt_tail_50_data = lt.tail(forecast_periods)

lt_actual_close_prices = lt_tail_50_data['Close'].values
lt_actual_open_prices = lt_tail_50_data['Open'].values
lt_actual_high_prices = lt_tail_50_data['High'].values
lt_actual_low_prices = lt_tail_50_data['Low'].values

lt_forecast_close = lt_close_final_model.forecast(steps=forecast_periods)
lt_forecast_close = lt_forecast_close.reshape(-1, 1)
lt_forecast_close = scaler.inverse_transform(lt_forecast_close)

lt_forecast_open = lt_open_final_model.forecast(steps=forecast_periods)
lt_forecast_open = lt_forecast_open.reshape(-1, 1)
lt_forecast_open = scaler.inverse_transform(lt_forecast_open)

lt_forecast_high = lt_high_final_model.forecast(steps=forecast_periods)
lt_forecast_high = lt_forecast_high.reshape(-1, 1)
lt_forecast_high = scaler.inverse_transform(lt_forecast_high)

lt_forecast_low = lt_low_final_model.forecast(steps=forecast_periods)
lt_forecast_low = lt_forecast_low.reshape(-1, 1)
lt_forecast_low = scaler.inverse_transform(lt_forecast_low)

lt_close_mae = mean_absolute_error(lt_actual_close_prices, lt_forecast_close)
lt_close_mse = mean_squared_error(lt_actual_close_prices, lt_forecast_close)
lt_close_rmse = np.sqrt(lt_close_mse)

lt_open_mae = mean_absolute_error(lt_actual_open_prices, lt_forecast_open)
lt_open_mse = mean_squared_error(lt_actual_open_prices, lt_forecast_open)
lt_open_rmse = np.sqrt(lt_open_mse)

lt_high_mae = mean_absolute_error(lt_actual_high_prices, lt_forecast_high)
lt_high_mse = mean_squared_error(lt_actual_high_prices, lt_forecast_high)
lt_high_rmse = np.sqrt(lt_high_mse)

lt_low_mae = mean_absolute_error(lt_actual_low_prices, lt_forecast_low)
lt_low_mse = mean_squared_error(lt_actual_low_prices, lt_forecast_low)
lt_low_rmse = np.sqrt(lt_low_mse)

lt_close_mape = mean_absolute_percentage_error(lt_actual_close_prices, lt_forecast_close)
lt_open_mape = mean_absolute_percentage_error(lt_actual_open_prices, lt_forecast_open)
lt_high_mape = mean_absolute_percentage_error(lt_actual_high_prices, lt_forecast_high)
lt_low_mape = mean_absolute_percentage_error(lt_actual_low_prices, lt_forecast_low)

print("Close Forecasts:", lt_forecast_close)
print(f"Close Mean Absolute Error (MAE): {lt_close_mae}")
print(f"Close Mean Squared Error (MSE): {lt_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {lt_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {lt_close_mape}%")

print("Open Forecasts:", lt_forecast_open)
print(f"Open Mean Absolute Error (MAE): {lt_open_mae}")
print(f"Open Mean Squared Error (MSE): {lt_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {lt_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {lt_open_mape}%")

print("High Forecasts:", lt_forecast_high)
print(f"High Mean Absolute Error (MAE): {lt_high_mae}")
print(f"High Mean Squared Error (MSE): {lt_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {lt_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {lt_high_mape}%")

print("Low Forecasts:", lt_forecast_low)
print(f"Low Mean Absolute Error (MAE): {lt_low_mae}")
print(f"Low Mean Squared Error (MSE): {lt_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {lt_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {lt_low_mape}")


# In[215]:


mandm_y_close = mandm['Close'].values
mandm_y_open = mandm['Open'].values
mandm_y_high = mandm['High'].values
mandm_y_low = mandm['Low'].values

mandm_y_close_scaled = scaler.fit_transform(mandm_y_close.reshape(-1, 1))
mandm_y_open_scaled = scaler.fit_transform(mandm_y_open.reshape(-1, 1))
mandm_y_high_scaled = scaler.fit_transform(mandm_y_high.reshape(-1, 1))
mandm_y_low_scaled = scaler.fit_transform(mandm_y_low.reshape(-1, 1))

mandm_close_model = auto_arima(
    mandm_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

mandm_open_model = auto_arima(
    mandm_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

mandm_high_model = auto_arima(
    mandm_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

mandm_low_model = auto_arima(
    mandm_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

mandm_close_best_order = mandm_close_model.get_params()['order']
mandm_open_best_order = mandm_open_model.get_params()['order']
mandm_high_best_order = mandm_high_model.get_params()['order']
mandm_low_best_order = mandm_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {mandm_close_best_order}")
print(f"Best ARIMA Order for Open: {mandm_open_best_order}")
print(f"Best ARIMA Order for High: {mandm_high_best_order}")
print(f"Best ARIMA Order for Low: {mandm_low_best_order}")

mandm_close_final_model = sm.tsa.ARIMA(
    mandm_y_close_scaled,
    order=mandm_close_best_order
)
mandm_close_final_model = mandm_close_final_model.fit()
mandm_close_forecast = mandm_close_final_model.forecast(steps=forecast_period)
mandm_close_forecast = mandm_close_forecast.reshape(-1, 1)
mandm_close_forecast = scaler.inverse_transform(mandm_close_forecast)

mandm_open_final_model = sm.tsa.ARIMA(
    mandm_y_open_scaled,
    order=mandm_open_best_order
)
mandm_open_final_model = mandm_open_final_model.fit()
mandm_open_forecast = mandm_open_final_model.forecast(steps=forecast_period)
mandm_open_forecast = mandm_open_forecast.reshape(-1, 1)
mandm_open_forecast = scaler.inverse_transform(mandm_open_forecast)

mandm_high_final_model = sm.tsa.ARIMA(
    mandm_y_high_scaled,
    order=mandm_high_best_order
)
mandm_high_final_model = mandm_high_final_model.fit()
mandm_high_forecast = mandm_high_final_model.forecast(steps=forecast_period)
mandm_high_forecast = mandm_high_forecast.reshape(-1, 1)
mandm_high_forecast = scaler.inverse_transform(mandm_high_forecast)

mandm_low_final_model = sm.tsa.ARIMA(
    mandm_y_low_scaled,
    order=mandm_low_best_order
)
mandm_low_final_model = mandm_low_final_model.fit()
mandm_low_forecast = mandm_low_final_model.forecast(steps=forecast_period)
mandm_low_forecast = mandm_low_forecast.reshape(-1, 1)
mandm_low_forecast = scaler.inverse_transform(mandm_low_forecast)

print("Close Forecasts:", mandm_close_forecast)
print("Open Forecasts:", mandm_open_forecast)
print("High Forecasts:", mandm_high_forecast)
print("Low Forecasts:", mandm_low_forecast)


# In[216]:


mandm_tail_50_data = mandm.tail(forecast_periods)

mandm_actual_close_prices = mandm_tail_50_data['Close'].values
mandm_actual_open_prices = mandm_tail_50_data['Open'].values
mandm_actual_high_prices = mandm_tail_50_data['High'].values
mandm_actual_low_prices = mandm_tail_50_data['Low'].values

mandm_forecast_close = mandm_close_final_model.forecast(steps=forecast_periods)
mandm_forecast_close = mandm_forecast_close.reshape(-1, 1)
mandm_forecast_close = scaler.inverse_transform(mandm_forecast_close)

mandm_forecast_open = mandm_open_final_model.forecast(steps=forecast_periods)
mandm_forecast_open = mandm_forecast_open.reshape(-1, 1)
mandm_forecast_open = scaler.inverse_transform(mandm_forecast_open)

mandm_forecast_high = mandm_high_final_model.forecast(steps=forecast_periods)
mandm_forecast_high = mandm_forecast_high.reshape(-1, 1)
mandm_forecast_high = scaler.inverse_transform(mandm_forecast_high)

mandm_forecast_low = mandm_low_final_model.forecast(steps=forecast_periods)
mandm_forecast_low = mandm_forecast_low.reshape(-1, 1)
mandm_forecast_low = scaler.inverse_transform(mandm_forecast_low)

mandm_close_mae = mean_absolute_error(mandm_actual_close_prices, mandm_forecast_close)
mandm_close_mse = mean_squared_error(mandm_actual_close_prices, mandm_forecast_close)
mandm_close_rmse = np.sqrt(mandm_close_mse)

mandm_open_mae = mean_absolute_error(mandm_actual_open_prices, mandm_forecast_open)
mandm_open_mse = mean_squared_error(mandm_actual_open_prices, mandm_forecast_open)
mandm_open_rmse = np.sqrt(mandm_open_mse)

mandm_high_mae = mean_absolute_error(mandm_actual_high_prices, mandm_forecast_high)
mandm_high_mse = mean_squared_error(mandm_actual_high_prices, mandm_forecast_high)
mandm_high_rmse = np.sqrt(mandm_high_mse)

mandm_low_mae = mean_absolute_error(mandm_actual_low_prices, mandm_forecast_low)
mandm_low_mse = mean_squared_error(mandm_actual_low_prices, mandm_forecast_low)
mandm_low_rmse = np.sqrt(mandm_low_mse)

mandm_close_mape = mean_absolute_percentage_error(mandm_actual_close_prices, mandm_forecast_close)
mandm_open_mape = mean_absolute_percentage_error(mandm_actual_open_prices, mandm_forecast_open)
mandm_high_mape = mean_absolute_percentage_error(mandm_actual_high_prices, mandm_forecast_high)
mandm_low_mape = mean_absolute_percentage_error(mandm_actual_low_prices, mandm_forecast_low)

print("Close Forecasts:", mandm_forecast_close)
print(f"Close Mean Absolute Error (MAE): {mandm_close_mae}")
print(f"Close Mean Squared Error (MSE): {mandm_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {mandm_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {mandm_close_mape}%")

print("Open Forecasts:", mandm_forecast_open)
print(f"Open Mean Absolute Error (MAE): {mandm_open_mae}")
print(f"Open Mean Squared Error (MSE): {mandm_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {mandm_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {mandm_open_mape}%")

print("High Forecasts:", mandm_forecast_high)
print(f"High Mean Absolute Error (MAE): {mandm_high_mae}")
print(f"High Mean Squared Error (MSE): {mandm_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {mandm_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {mandm_high_mape}%")

print("Low Forecasts:", mandm_forecast_low)
print(f"Low Mean Absolute Error (MAE): {mandm_low_mae}")
print(f"Low Mean Squared Error (MSE): {mandm_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {mandm_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {mandm_low_mape}")


# In[217]:


maruti_y_close = maruti['Close'].values
maruti_y_open = maruti['Open'].values
maruti_y_high = maruti['High'].values
maruti_y_low = maruti['Low'].values

maruti_y_close_scaled = scaler.fit_transform(maruti_y_close.reshape(-1, 1))
maruti_y_open_scaled = scaler.fit_transform(maruti_y_open.reshape(-1, 1))
maruti_y_high_scaled = scaler.fit_transform(maruti_y_high.reshape(-1, 1))
maruti_y_low_scaled = scaler.fit_transform(maruti_y_low.reshape(-1, 1))

maruti_close_model = auto_arima(
    maruti_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

maruti_open_model = auto_arima(
    maruti_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

maruti_high_model = auto_arima(
    maruti_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

maruti_low_model = auto_arima(
    maruti_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

maruti_close_best_order = maruti_close_model.get_params()['order']
maruti_open_best_order = maruti_open_model.get_params()['order']
maruti_high_best_order = maruti_high_model.get_params()['order']
maruti_low_best_order = maruti_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {maruti_close_best_order}")
print(f"Best ARIMA Order for Open: {maruti_open_best_order}")
print(f"Best ARIMA Order for High: {maruti_high_best_order}")
print(f"Best ARIMA Order for Low: {maruti_low_best_order}")

maruti_close_final_model = sm.tsa.ARIMA(
    maruti_y_close_scaled,
    order=maruti_close_best_order
)
maruti_close_final_model = maruti_close_final_model.fit()
maruti_close_forecast = maruti_close_final_model.forecast(steps=forecast_period)
maruti_close_forecast = maruti_close_forecast.reshape(-1, 1)
maruti_close_forecast = scaler.inverse_transform(maruti_close_forecast)

maruti_open_final_model = sm.tsa.ARIMA(
    maruti_y_open_scaled,
    order=maruti_open_best_order
)
maruti_open_final_model = maruti_open_final_model.fit()
maruti_open_forecast = maruti_open_final_model.forecast(steps=forecast_period)
maruti_open_forecast = maruti_open_forecast.reshape(-1, 1)
maruti_open_forecast = scaler.inverse_transform(maruti_open_forecast)

maruti_high_final_model = sm.tsa.ARIMA(
    maruti_y_high_scaled,
    order=maruti_high_best_order
)
maruti_high_final_model = maruti_high_final_model.fit()
maruti_high_forecast = maruti_high_final_model.forecast(steps=forecast_period)
maruti_high_forecast = maruti_high_forecast.reshape(-1, 1)
maruti_high_forecast = scaler.inverse_transform(maruti_high_forecast)

maruti_low_final_model = sm.tsa.ARIMA(
    maruti_y_low_scaled,
    order=maruti_low_best_order
)
maruti_low_final_model = maruti_low_final_model.fit()
maruti_low_forecast = maruti_low_final_model.forecast(steps=forecast_period)
maruti_low_forecast = maruti_low_forecast.reshape(-1, 1)
maruti_low_forecast = scaler.inverse_transform(maruti_low_forecast)

print("Close Forecasts:", maruti_close_forecast)
print("Open Forecasts:", maruti_open_forecast)
print("High Forecasts:", maruti_high_forecast)
print("Low Forecasts:", maruti_low_forecast)


# In[218]:


maruti_tail_50_data = maruti.tail(forecast_periods)

maruti_actual_close_prices = maruti_tail_50_data['Close'].values
maruti_actual_open_prices = maruti_tail_50_data['Open'].values
maruti_actual_high_prices = maruti_tail_50_data['High'].values
maruti_actual_low_prices = maruti_tail_50_data['Low'].values

maruti_forecast_close = maruti_close_final_model.forecast(steps=forecast_periods)
maruti_forecast_close = maruti_forecast_close.reshape(-1, 1)
maruti_forecast_close = scaler.inverse_transform(maruti_forecast_close)

maruti_forecast_open = maruti_open_final_model.forecast(steps=forecast_periods)
maruti_forecast_open = maruti_forecast_open.reshape(-1, 1)
maruti_forecast_open = scaler.inverse_transform(maruti_forecast_open)

maruti_forecast_high = maruti_high_final_model.forecast(steps=forecast_periods)
maruti_forecast_high = maruti_forecast_high.reshape(-1, 1)
maruti_forecast_high = scaler.inverse_transform(maruti_forecast_high)

maruti_forecast_low = maruti_low_final_model.forecast(steps=forecast_periods)
maruti_forecast_low = maruti_forecast_low.reshape(-1, 1)
maruti_forecast_low = scaler.inverse_transform(maruti_forecast_low)

maruti_close_mae = mean_absolute_error(maruti_actual_close_prices, maruti_forecast_close)
maruti_close_mse = mean_squared_error(maruti_actual_close_prices, maruti_forecast_close)
maruti_close_rmse = np.sqrt(maruti_close_mse)

maruti_open_mae = mean_absolute_error(maruti_actual_open_prices, maruti_forecast_open)
maruti_open_mse = mean_squared_error(maruti_actual_open_prices, maruti_forecast_open)
maruti_open_rmse = np.sqrt(maruti_open_mse)

maruti_high_mae = mean_absolute_error(maruti_actual_high_prices, maruti_forecast_high)
maruti_high_mse = mean_squared_error(maruti_actual_high_prices, maruti_forecast_high)
maruti_high_rmse = np.sqrt(maruti_high_mse)

maruti_low_mae = mean_absolute_error(maruti_actual_low_prices, maruti_forecast_low)
maruti_low_mse = mean_squared_error(maruti_actual_low_prices, maruti_forecast_low)
maruti_low_rmse = np.sqrt(maruti_low_mse)

maruti_close_mape = mean_absolute_percentage_error(maruti_actual_close_prices, maruti_forecast_close)
maruti_open_mape = mean_absolute_percentage_error(maruti_actual_open_prices, maruti_forecast_open)
maruti_high_mape = mean_absolute_percentage_error(maruti_actual_high_prices, maruti_forecast_high)
maruti_low_mape = mean_absolute_percentage_error(maruti_actual_low_prices, maruti_forecast_low)

print("Close Forecasts:", maruti_forecast_close)
print(f"Close Mean Absolute Error (MAE): {maruti_close_mae}")
print(f"Close Mean Squared Error (MSE): {maruti_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {maruti_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {maruti_close_mape}%")

print("Open Forecasts:", maruti_forecast_open)
print(f"Open Mean Absolute Error (MAE): {maruti_open_mae}")
print(f"Open Mean Squared Error (MSE): {maruti_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {maruti_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {maruti_open_mape}%")

print("High Forecasts:", maruti_forecast_high)
print(f"High Mean Absolute Error (MAE): {maruti_high_mae}")
print(f"High Mean Squared Error (MSE): {maruti_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {maruti_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {maruti_high_mape}%")

print("Low Forecasts:", maruti_forecast_low)
print(f"Low Mean Absolute Error (MAE): {maruti_low_mae}")
print(f"Low Mean Squared Error (MSE): {maruti_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {maruti_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {maruti_low_mape}")


# In[219]:


nestleind_y_close = nestleind['Close'].values
nestleind_y_open = nestleind['Open'].values
nestleind_y_high = nestleind['High'].values
nestleind_y_low = nestleind['Low'].values

nestleind_y_close_scaled = scaler.fit_transform(nestleind_y_close.reshape(-1, 1))
nestleind_y_open_scaled = scaler.fit_transform(nestleind_y_open.reshape(-1, 1))
nestleind_y_high_scaled = scaler.fit_transform(nestleind_y_high.reshape(-1, 1))
nestleind_y_low_scaled = scaler.fit_transform(nestleind_y_low.reshape(-1, 1))

nestleind_close_model = auto_arima(
    nestleind_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

nestleind_open_model = auto_arima(
    nestleind_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

nestleind_high_model = auto_arima(
    nestleind_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

nestleind_low_model = auto_arima(
    nestleind_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

nestleind_close_best_order = nestleind_close_model.get_params()['order']
nestleind_open_best_order = nestleind_open_model.get_params()['order']
nestleind_high_best_order = nestleind_high_model.get_params()['order']
nestleind_low_best_order = nestleind_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {nestleind_close_best_order}")
print(f"Best ARIMA Order for Open: {nestleind_open_best_order}")
print(f"Best ARIMA Order for High: {nestleind_high_best_order}")
print(f"Best ARIMA Order for Low: {nestleind_low_best_order}")

nestleind_close_final_model = sm.tsa.ARIMA(
    nestleind_y_close_scaled,
    order=nestleind_close_best_order
)
nestleind_close_final_model = nestleind_close_final_model.fit()
nestleind_close_forecast = nestleind_close_final_model.forecast(steps=forecast_period)
nestleind_close_forecast = nestleind_close_forecast.reshape(-1, 1)
nestleind_close_forecast = scaler.inverse_transform(nestleind_close_forecast)

nestleind_open_final_model = sm.tsa.ARIMA(
    nestleind_y_open_scaled,
    order=nestleind_open_best_order
)
nestleind_open_final_model = nestleind_open_final_model.fit()
nestleind_open_forecast = nestleind_open_final_model.forecast(steps=forecast_period)
nestleind_open_forecast = nestleind_open_forecast.reshape(-1, 1)
nestleind_open_forecast = scaler.inverse_transform(nestleind_open_forecast)

nestleind_high_final_model = sm.tsa.ARIMA(
    nestleind_y_high_scaled,
    order=nestleind_high_best_order
)
nestleind_high_final_model = nestleind_high_final_model.fit()
nestleind_high_forecast = nestleind_high_final_model.forecast(steps=forecast_period)
nestleind_high_forecast = nestleind_high_forecast.reshape(-1, 1)
nestleind_high_forecast = scaler.inverse_transform(nestleind_high_forecast)

nestleind_low_final_model = sm.tsa.ARIMA(
    nestleind_y_low_scaled,
    order=nestleind_low_best_order
)
nestleind_low_final_model = nestleind_low_final_model.fit()
nestleind_low_forecast = nestleind_low_final_model.forecast(steps=forecast_period)
nestleind_low_forecast = nestleind_low_forecast.reshape(-1, 1)
nestleind_low_forecast = scaler.inverse_transform(nestleind_low_forecast)

print("Close Forecasts:", nestleind_close_forecast)
print("Open Forecasts:", nestleind_open_forecast)
print("High Forecasts:", nestleind_high_forecast)
print("Low Forecasts:", nestleind_low_forecast)


# In[220]:


nestleind_tail_50_data = nestleind.tail(forecast_periods)

nestleind_actual_close_prices = nestleind_tail_50_data['Close'].values
nestleind_actual_open_prices = nestleind_tail_50_data['Open'].values
nestleind_actual_high_prices = nestleind_tail_50_data['High'].values
nestleind_actual_low_prices = nestleind_tail_50_data['Low'].values

nestleind_forecast_close = nestleind_close_final_model.forecast(steps=forecast_periods)
nestleind_forecast_close = nestleind_forecast_close.reshape(-1, 1)
nestleind_forecast_close = scaler.inverse_transform(nestleind_forecast_close)

nestleind_forecast_open = nestleind_open_final_model.forecast(steps=forecast_periods)
nestleind_forecast_open = nestleind_forecast_open.reshape(-1, 1)
nestleind_forecast_open = scaler.inverse_transform(nestleind_forecast_open)

nestleind_forecast_high = nestleind_high_final_model.forecast(steps=forecast_periods)
nestleind_forecast_high = nestleind_forecast_high.reshape(-1, 1)
nestleind_forecast_high = scaler.inverse_transform(nestleind_forecast_high)

nestleind_forecast_low = nestleind_low_final_model.forecast(steps=forecast_periods)
nestleind_forecast_low = nestleind_forecast_low.reshape(-1, 1)
nestleind_forecast_low = scaler.inverse_transform(nestleind_forecast_low)

nestleind_close_mae = mean_absolute_error(nestleind_actual_close_prices, nestleind_forecast_close)
nestleind_close_mse = mean_squared_error(nestleind_actual_close_prices, nestleind_forecast_close)
nestleind_close_rmse = np.sqrt(nestleind_close_mse)

nestleind_open_mae = mean_absolute_error(nestleind_actual_open_prices, nestleind_forecast_open)
nestleind_open_mse = mean_squared_error(nestleind_actual_open_prices, nestleind_forecast_open)
nestleind_open_rmse = np.sqrt(nestleind_open_mse)

nestleind_high_mae = mean_absolute_error(nestleind_actual_high_prices, nestleind_forecast_high)
nestleind_high_mse = mean_squared_error(nestleind_actual_high_prices, nestleind_forecast_high)
nestleind_high_rmse = np.sqrt(nestleind_high_mse)

nestleind_low_mae = mean_absolute_error(nestleind_actual_low_prices, nestleind_forecast_low)
nestleind_low_mse = mean_squared_error(nestleind_actual_low_prices, nestleind_forecast_low)
nestleind_low_rmse = np.sqrt(nestleind_low_mse)

nestleind_close_mape = mean_absolute_percentage_error(nestleind_actual_close_prices, nestleind_forecast_close)
nestleind_open_mape = mean_absolute_percentage_error(nestleind_actual_open_prices, nestleind_forecast_open)
nestleind_high_mape = mean_absolute_percentage_error(nestleind_actual_high_prices, nestleind_forecast_high)
nestleind_low_mape = mean_absolute_percentage_error(nestleind_actual_low_prices, nestleind_forecast_low)

print("Close Forecasts:", nestleind_forecast_close)
print(f"Close Mean Absolute Error (MAE): {nestleind_close_mae}")
print(f"Close Mean Squared Error (MSE): {nestleind_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {nestleind_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {nestleind_close_mape}%")

print("Open Forecasts:", nestleind_forecast_open)
print(f"Open Mean Absolute Error (MAE): {nestleind_open_mae}")
print(f"Open Mean Squared Error (MSE): {nestleind_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {nestleind_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {nestleind_open_mape}%")

print("High Forecasts:", nestleind_forecast_high)
print(f"High Mean Absolute Error (MAE): {nestleind_high_mae}")
print(f"High Mean Squared Error (MSE): {nestleind_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {nestleind_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {nestleind_high_mape}%")

print("Low Forecasts:", nestleind_forecast_low)
print(f"Low Mean Absolute Error (MAE): {nestleind_low_mae}")
print(f"Low Mean Squared Error (MSE): {nestleind_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {nestleind_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {nestleind_low_mape}")


# In[221]:


ntpc_y_close = ntpc['Close'].values
ntpc_y_open = ntpc['Open'].values
ntpc_y_high = ntpc['High'].values
ntpc_y_low = ntpc['Low'].values

ntpc_y_close_scaled = scaler.fit_transform(ntpc_y_close.reshape(-1, 1))
ntpc_y_open_scaled = scaler.fit_transform(ntpc_y_open.reshape(-1, 1))
ntpc_y_high_scaled = scaler.fit_transform(ntpc_y_high.reshape(-1, 1))
ntpc_y_low_scaled = scaler.fit_transform(ntpc_y_low.reshape(-1, 1))

ntpc_close_model = auto_arima(
    ntpc_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ntpc_open_model = auto_arima(
    ntpc_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ntpc_high_model = auto_arima(
    ntpc_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ntpc_low_model = auto_arima(
    ntpc_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ntpc_close_best_order = ntpc_close_model.get_params()['order']
ntpc_open_best_order = ntpc_open_model.get_params()['order']
ntpc_high_best_order = ntpc_high_model.get_params()['order']
ntpc_low_best_order = ntpc_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {ntpc_close_best_order}")
print(f"Best ARIMA Order for Open: {ntpc_open_best_order}")
print(f"Best ARIMA Order for High: {ntpc_high_best_order}")
print(f"Best ARIMA Order for Low: {ntpc_low_best_order}")

ntpc_close_final_model = sm.tsa.ARIMA(
    ntpc_y_close_scaled,
    order=ntpc_close_best_order
)
ntpc_close_final_model = ntpc_close_final_model.fit()
ntpc_close_forecast = ntpc_close_final_model.forecast(steps=forecast_period)
ntpc_close_forecast = ntpc_close_forecast.reshape(-1, 1)
ntpc_close_forecast = scaler.inverse_transform(ntpc_close_forecast)

ntpc_open_final_model = sm.tsa.ARIMA(
    ntpc_y_open_scaled,
    order=ntpc_open_best_order
)
ntpc_open_final_model = ntpc_open_final_model.fit()
ntpc_open_forecast = ntpc_open_final_model.forecast(steps=forecast_period)
ntpc_open_forecast = ntpc_open_forecast.reshape(-1, 1)
ntpc_open_forecast = scaler.inverse_transform(ntpc_open_forecast)

ntpc_high_final_model = sm.tsa.ARIMA(
    ntpc_y_high_scaled,
    order=ntpc_high_best_order
)
ntpc_high_final_model = ntpc_high_final_model.fit()
ntpc_high_forecast = ntpc_high_final_model.forecast(steps=forecast_period)
ntpc_high_forecast = ntpc_high_forecast.reshape(-1, 1)
ntpc_high_forecast = scaler.inverse_transform(ntpc_high_forecast)

ntpc_low_final_model = sm.tsa.ARIMA(
    ntpc_y_low_scaled,
    order=ntpc_low_best_order
)
ntpc_low_final_model = ntpc_low_final_model.fit()
ntpc_low_forecast = ntpc_low_final_model.forecast(steps=forecast_period)
ntpc_low_forecast = ntpc_low_forecast.reshape(-1, 1)
ntpc_low_forecast = scaler.inverse_transform(ntpc_low_forecast)

print("Close Forecasts:", ntpc_close_forecast)
print("Open Forecasts:", ntpc_open_forecast)
print("High Forecasts:", ntpc_high_forecast)
print("Low Forecasts:", ntpc_low_forecast)


# In[222]:


ntpc_tail_50_data = ntpc.tail(forecast_periods)

ntpc_actual_close_prices = ntpc_tail_50_data['Close'].values
ntpc_actual_open_prices = ntpc_tail_50_data['Open'].values
ntpc_actual_high_prices = ntpc_tail_50_data['High'].values
ntpc_actual_low_prices = ntpc_tail_50_data['Low'].values

ntpc_forecast_close = ntpc_close_final_model.forecast(steps=forecast_periods)
ntpc_forecast_close = ntpc_forecast_close.reshape(-1, 1)
ntpc_forecast_close = scaler.inverse_transform(ntpc_forecast_close)

ntpc_forecast_open = ntpc_open_final_model.forecast(steps=forecast_periods)
ntpc_forecast_open = ntpc_forecast_open.reshape(-1, 1)
ntpc_forecast_open = scaler.inverse_transform(ntpc_forecast_open)

ntpc_forecast_high = ntpc_high_final_model.forecast(steps=forecast_periods)
ntpc_forecast_high = ntpc_forecast_high.reshape(-1, 1)
ntpc_forecast_high = scaler.inverse_transform(ntpc_forecast_high)

ntpc_forecast_low = ntpc_low_final_model.forecast(steps=forecast_periods)
ntpc_forecast_low = ntpc_forecast_low.reshape(-1, 1)
ntpc_forecast_low = scaler.inverse_transform(ntpc_forecast_low)

ntpc_close_mae = mean_absolute_error(ntpc_actual_close_prices, ntpc_forecast_close)
ntpc_close_mse = mean_squared_error(ntpc_actual_close_prices, ntpc_forecast_close)
ntpc_close_rmse = np.sqrt(ntpc_close_mse)

ntpc_open_mae = mean_absolute_error(ntpc_actual_open_prices, ntpc_forecast_open)
ntpc_open_mse = mean_squared_error(ntpc_actual_open_prices, ntpc_forecast_open)
ntpc_open_rmse = np.sqrt(ntpc_open_mse)

ntpc_high_mae = mean_absolute_error(ntpc_actual_high_prices, ntpc_forecast_high)
ntpc_high_mse = mean_squared_error(ntpc_actual_high_prices, ntpc_forecast_high)
ntpc_high_rmse = np.sqrt(ntpc_high_mse)

ntpc_low_mae = mean_absolute_error(ntpc_actual_low_prices, ntpc_forecast_low)
ntpc_low_mse = mean_squared_error(ntpc_actual_low_prices, ntpc_forecast_low)
ntpc_low_rmse = np.sqrt(ntpc_low_mse)

ntpc_close_mape = mean_absolute_percentage_error(ntpc_actual_close_prices, ntpc_forecast_close)
ntpc_open_mape = mean_absolute_percentage_error(ntpc_actual_open_prices, ntpc_forecast_open)
ntpc_high_mape = mean_absolute_percentage_error(ntpc_actual_high_prices, ntpc_forecast_high)
ntpc_low_mape = mean_absolute_percentage_error(ntpc_actual_low_prices, ntpc_forecast_low)

print("Close Forecasts:", ntpc_forecast_close)
print(f"Close Mean Absolute Error (MAE): {ntpc_close_mae}")
print(f"Close Mean Squared Error (MSE): {ntpc_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {ntpc_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {ntpc_close_mape}%")

print("Open Forecasts:", ntpc_forecast_open)
print(f"Open Mean Absolute Error (MAE): {ntpc_open_mae}")
print(f"Open Mean Squared Error (MSE): {ntpc_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {ntpc_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {ntpc_open_mape}%")

print("High Forecasts:", ntpc_forecast_high)
print(f"High Mean Absolute Error (MAE): {ntpc_high_mae}")
print(f"High Mean Squared Error (MSE): {ntpc_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {ntpc_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {ntpc_high_mape}%")

print("Low Forecasts:", ntpc_forecast_low)
print(f"Low Mean Absolute Error (MAE): {ntpc_low_mae}")
print(f"Low Mean Squared Error (MSE): {ntpc_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {ntpc_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {ntpc_low_mape}")


# In[223]:


ongc_y_close = ongc['Close'].values
ongc_y_open = ongc['Open'].values
ongc_y_high = ongc['High'].values
ongc_y_low = ongc['Low'].values

ongc_y_close_scaled = scaler.fit_transform(ongc_y_close.reshape(-1, 1))
ongc_y_open_scaled = scaler.fit_transform(ongc_y_open.reshape(-1, 1))
ongc_y_high_scaled = scaler.fit_transform(ongc_y_high.reshape(-1, 1))
ongc_y_low_scaled = scaler.fit_transform(ongc_y_low.reshape(-1, 1))

ongc_close_model = auto_arima(
    ongc_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ongc_open_model = auto_arima(
    ongc_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ongc_high_model = auto_arima(
    ongc_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ongc_low_model = auto_arima(
    ongc_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ongc_close_best_order = ongc_close_model.get_params()['order']
ongc_open_best_order = ongc_open_model.get_params()['order']
ongc_high_best_order = ongc_high_model.get_params()['order']
ongc_low_best_order = ongc_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {ongc_close_best_order}")
print(f"Best ARIMA Order for Open: {ongc_open_best_order}")
print(f"Best ARIMA Order for High: {ongc_high_best_order}")
print(f"Best ARIMA Order for Low: {ongc_low_best_order}")

ongc_close_final_model = sm.tsa.ARIMA(
    ongc_y_close_scaled,
    order=ongc_close_best_order
)
ongc_close_final_model = ongc_close_final_model.fit()
ongc_close_forecast = ongc_close_final_model.forecast(steps=forecast_period)
ongc_close_forecast = ongc_close_forecast.reshape(-1, 1)
ongc_close_forecast = scaler.inverse_transform(ongc_close_forecast)

ongc_open_final_model = sm.tsa.ARIMA(
    ongc_y_open_scaled,
    order=ongc_open_best_order
)
ongc_open_final_model = ongc_open_final_model.fit()
ongc_open_forecast = ongc_open_final_model.forecast(steps=forecast_period)
ongc_open_forecast = ongc_open_forecast.reshape(-1, 1)
ongc_open_forecast = scaler.inverse_transform(ongc_open_forecast)

ongc_high_final_model = sm.tsa.ARIMA(
    ongc_y_high_scaled,
    order=ongc_high_best_order
)
ongc_high_final_model = ongc_high_final_model.fit()
ongc_high_forecast = ongc_high_final_model.forecast(steps=forecast_period)
ongc_high_forecast = ongc_high_forecast.reshape(-1, 1)
ongc_high_forecast = scaler.inverse_transform(ongc_high_forecast)

ongc_low_final_model = sm.tsa.ARIMA(
    ongc_y_low_scaled,
    order=ongc_low_best_order
)
ongc_low_final_model = ongc_low_final_model.fit()
ongc_low_forecast = ongc_low_final_model.forecast(steps=forecast_period)
ongc_low_forecast = ongc_low_forecast.reshape(-1, 1)
ongc_low_forecast = scaler.inverse_transform(ongc_low_forecast)

print("Close Forecasts:", ongc_close_forecast)
print("Open Forecasts:", ongc_open_forecast)
print("High Forecasts:", ongc_high_forecast)
print("Low Forecasts:", ongc_low_forecast)


# In[224]:


ongc_tail_50_data = ongc.tail(forecast_periods)

ongc_actual_close_prices = ongc_tail_50_data['Close'].values
ongc_actual_open_prices = ongc_tail_50_data['Open'].values
ongc_actual_high_prices = ongc_tail_50_data['High'].values
ongc_actual_low_prices = ongc_tail_50_data['Low'].values

ongc_forecast_close = ongc_close_final_model.forecast(steps=forecast_periods)
ongc_forecast_close = ongc_forecast_close.reshape(-1, 1)
ongc_forecast_close = scaler.inverse_transform(ongc_forecast_close)

ongc_forecast_open = ongc_open_final_model.forecast(steps=forecast_periods)
ongc_forecast_open = ongc_forecast_open.reshape(-1, 1)
ongc_forecast_open = scaler.inverse_transform(ongc_forecast_open)

ongc_forecast_high = ongc_high_final_model.forecast(steps=forecast_periods)
ongc_forecast_high = ongc_forecast_high.reshape(-1, 1)
ongc_forecast_high = scaler.inverse_transform(ongc_forecast_high)

ongc_forecast_low = ongc_low_final_model.forecast(steps=forecast_periods)
ongc_forecast_low = ongc_forecast_low.reshape(-1, 1)
ongc_forecast_low = scaler.inverse_transform(ongc_forecast_low)

ongc_close_mae = mean_absolute_error(ongc_actual_close_prices, ongc_forecast_close)
ongc_close_mse = mean_squared_error(ongc_actual_close_prices, ongc_forecast_close)
ongc_close_rmse = np.sqrt(ongc_close_mse)

ongc_open_mae = mean_absolute_error(ongc_actual_open_prices, ongc_forecast_open)
ongc_open_mse = mean_squared_error(ongc_actual_open_prices, ongc_forecast_open)
ongc_open_rmse = np.sqrt(ongc_open_mse)

ongc_high_mae = mean_absolute_error(ongc_actual_high_prices, ongc_forecast_high)
ongc_high_mse = mean_squared_error(ongc_actual_high_prices, ongc_forecast_high)
ongc_high_rmse = np.sqrt(ongc_high_mse)

ongc_low_mae = mean_absolute_error(ongc_actual_low_prices, ongc_forecast_low)
ongc_low_mse = mean_squared_error(ongc_actual_low_prices, ongc_forecast_low)
ongc_low_rmse = np.sqrt(ongc_low_mse)

ongc_close_mape = mean_absolute_percentage_error(ongc_actual_close_prices, ongc_forecast_close)
ongc_open_mape = mean_absolute_percentage_error(ongc_actual_open_prices, ongc_forecast_open)
ongc_high_mape = mean_absolute_percentage_error(ongc_actual_high_prices, ongc_forecast_high)
ongc_low_mape = mean_absolute_percentage_error(ongc_actual_low_prices, ongc_forecast_low)

print("Close Forecasts:", ongc_forecast_close)
print(f"Close Mean Absolute Error (MAE): {ongc_close_mae}")
print(f"Close Mean Squared Error (MSE): {ongc_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {ongc_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {ongc_close_mape}%")

print("Open Forecasts:", ongc_forecast_open)
print(f"Open Mean Absolute Error (MAE): {ongc_open_mae}")
print(f"Open Mean Squared Error (MSE): {ongc_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {ongc_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {ongc_open_mape}%")

print("High Forecasts:", ongc_forecast_high)
print(f"High Mean Absolute Error (MAE): {ongc_high_mae}")
print(f"High Mean Squared Error (MSE): {ongc_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {ongc_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {ongc_high_mape}%")

print("Low Forecasts:", ongc_forecast_low)
print(f"Low Mean Absolute Error (MAE): {ongc_low_mae}")
print(f"Low Mean Squared Error (MSE): {ongc_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {ongc_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {ongc_low_mape}")


# In[225]:


powergrid_y_close = powergrid['Close'].values
powergrid_y_open = powergrid['Open'].values
powergrid_y_high = powergrid['High'].values
powergrid_y_low = powergrid['Low'].values

powergrid_y_close_scaled = scaler.fit_transform(powergrid_y_close.reshape(-1, 1))
powergrid_y_open_scaled = scaler.fit_transform(powergrid_y_open.reshape(-1, 1))
powergrid_y_high_scaled = scaler.fit_transform(powergrid_y_high.reshape(-1, 1))
powergrid_y_low_scaled = scaler.fit_transform(powergrid_y_low.reshape(-1, 1))

powergrid_close_model = auto_arima(
    powergrid_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

powergrid_open_model = auto_arima(
    powergrid_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

powergrid_high_model = auto_arima(
    powergrid_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

powergrid_low_model = auto_arima(
    powergrid_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

powergrid_close_best_order = powergrid_close_model.get_params()['order']
powergrid_open_best_order = powergrid_open_model.get_params()['order']
powergrid_high_best_order = powergrid_high_model.get_params()['order']
powergrid_low_best_order = powergrid_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {powergrid_close_best_order}")
print(f"Best ARIMA Order for Open: {powergrid_open_best_order}")
print(f"Best ARIMA Order for High: {powergrid_high_best_order}")
print(f"Best ARIMA Order for Low: {powergrid_low_best_order}")

powergrid_close_final_model = sm.tsa.ARIMA(
    powergrid_y_close_scaled,
    order=powergrid_close_best_order
)
powergrid_close_final_model = powergrid_close_final_model.fit()
powergrid_close_forecast = powergrid_close_final_model.forecast(steps=forecast_period)
powergrid_close_forecast = powergrid_close_forecast.reshape(-1, 1)
powergrid_close_forecast = scaler.inverse_transform(powergrid_close_forecast)

powergrid_open_final_model = sm.tsa.ARIMA(
    powergrid_y_open_scaled,
    order=powergrid_open_best_order
)
powergrid_open_final_model = powergrid_open_final_model.fit()
powergrid_open_forecast = powergrid_open_final_model.forecast(steps=forecast_period)
powergrid_open_forecast = powergrid_open_forecast.reshape(-1, 1)
powergrid_open_forecast = scaler.inverse_transform(powergrid_open_forecast)

powergrid_high_final_model = sm.tsa.ARIMA(
    powergrid_y_high_scaled,
    order=powergrid_high_best_order
)
powergrid_high_final_model = powergrid_high_final_model.fit()
powergrid_high_forecast = powergrid_high_final_model.forecast(steps=forecast_period)
powergrid_high_forecast = powergrid_high_forecast.reshape(-1, 1)
powergrid_high_forecast = scaler.inverse_transform(powergrid_high_forecast)

powergrid_low_final_model = sm.tsa.ARIMA(
    powergrid_y_low_scaled,
    order=powergrid_low_best_order
)
powergrid_low_final_model = powergrid_low_final_model.fit()
powergrid_low_forecast = powergrid_low_final_model.forecast(steps=forecast_period)
powergrid_low_forecast = powergrid_low_forecast.reshape(-1, 1)
powergrid_low_forecast = scaler.inverse_transform(powergrid_low_forecast)

print("Close Forecasts:", powergrid_close_forecast)
print("Open Forecasts:", powergrid_open_forecast)
print("High Forecasts:", powergrid_high_forecast)
print("Low Forecasts:", powergrid_low_forecast)


# In[226]:


powergrid_tail_50_data = powergrid.tail(forecast_periods)

powergrid_actual_close_prices = powergrid_tail_50_data['Close'].values
powergrid_actual_open_prices = powergrid_tail_50_data['Open'].values
powergrid_actual_high_prices = powergrid_tail_50_data['High'].values
powergrid_actual_low_prices = powergrid_tail_50_data['Low'].values

powergrid_forecast_close = powergrid_close_final_model.forecast(steps=forecast_periods)
powergrid_forecast_close = powergrid_forecast_close.reshape(-1, 1)
powergrid_forecast_close = scaler.inverse_transform(powergrid_forecast_close)

powergrid_forecast_open = powergrid_open_final_model.forecast(steps=forecast_periods)
powergrid_forecast_open = powergrid_forecast_open.reshape(-1, 1)
powergrid_forecast_open = scaler.inverse_transform(powergrid_forecast_open)

powergrid_forecast_high = powergrid_high_final_model.forecast(steps=forecast_periods)
powergrid_forecast_high = powergrid_forecast_high.reshape(-1, 1)
powergrid_forecast_high = scaler.inverse_transform(powergrid_forecast_high)

powergrid_forecast_low = powergrid_low_final_model.forecast(steps=forecast_periods)
powergrid_forecast_low = powergrid_forecast_low.reshape(-1, 1)
powergrid_forecast_low = scaler.inverse_transform(powergrid_forecast_low)

powergrid_close_mae = mean_absolute_error(powergrid_actual_close_prices, powergrid_forecast_close)
powergrid_close_mse = mean_squared_error(powergrid_actual_close_prices, powergrid_forecast_close)
powergrid_close_rmse = np.sqrt(powergrid_close_mse)

powergrid_open_mae = mean_absolute_error(powergrid_actual_open_prices, powergrid_forecast_open)
powergrid_open_mse = mean_squared_error(powergrid_actual_open_prices, powergrid_forecast_open)
powergrid_open_rmse = np.sqrt(powergrid_open_mse)

powergrid_high_mae = mean_absolute_error(powergrid_actual_high_prices, powergrid_forecast_high)
powergrid_high_mse = mean_squared_error(powergrid_actual_high_prices, powergrid_forecast_high)
powergrid_high_rmse = np.sqrt(powergrid_high_mse)

powergrid_low_mae = mean_absolute_error(powergrid_actual_low_prices, powergrid_forecast_low)
powergrid_low_mse = mean_squared_error(powergrid_actual_low_prices, powergrid_forecast_low)
powergrid_low_rmse = np.sqrt(powergrid_low_mse)

powergrid_close_mape = mean_absolute_percentage_error(powergrid_actual_close_prices, powergrid_forecast_close)
powergrid_open_mape = mean_absolute_percentage_error(powergrid_actual_open_prices, powergrid_forecast_open)
powergrid_high_mape = mean_absolute_percentage_error(powergrid_actual_high_prices, powergrid_forecast_high)
powergrid_low_mape = mean_absolute_percentage_error(powergrid_actual_low_prices, powergrid_forecast_low)

print("Close Forecasts:", powergrid_forecast_close)
print(f"Close Mean Absolute Error (MAE): {powergrid_close_mae}")
print(f"Close Mean Squared Error (MSE): {powergrid_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {powergrid_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {powergrid_close_mape}%")

print("Open Forecasts:", powergrid_forecast_open)
print(f"Open Mean Absolute Error (MAE): {powergrid_open_mae}")
print(f"Open Mean Squared Error (MSE): {powergrid_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {powergrid_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {powergrid_open_mape}%")

print("High Forecasts:", powergrid_forecast_high)
print(f"High Mean Absolute Error (MAE): {powergrid_high_mae}")
print(f"High Mean Squared Error (MSE): {powergrid_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {powergrid_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {powergrid_high_mape}%")

print("Low Forecasts:", powergrid_forecast_low)
print(f"Low Mean Absolute Error (MAE): {powergrid_low_mae}")
print(f"Low Mean Squared Error (MSE): {powergrid_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {powergrid_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {powergrid_low_mape}")


# In[227]:


reliance_y_close = reliance['Close'].values
reliance_y_open = reliance['Open'].values
reliance_y_high = reliance['High'].values
reliance_y_low = reliance['Low'].values

reliance_y_close_scaled = scaler.fit_transform(reliance_y_close.reshape(-1, 1))
reliance_y_open_scaled = scaler.fit_transform(reliance_y_open.reshape(-1, 1))
reliance_y_high_scaled = scaler.fit_transform(reliance_y_high.reshape(-1, 1))
reliance_y_low_scaled = scaler.fit_transform(reliance_y_low.reshape(-1, 1))

reliance_close_model = auto_arima(
    reliance_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

reliance_open_model = auto_arima(
    reliance_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

reliance_high_model = auto_arima(
    reliance_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

reliance_low_model = auto_arima(
    reliance_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

reliance_close_best_order = reliance_close_model.get_params()['order']
reliance_open_best_order = reliance_open_model.get_params()['order']
reliance_high_best_order = reliance_high_model.get_params()['order']
reliance_low_best_order = reliance_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {reliance_close_best_order}")
print(f"Best ARIMA Order for Open: {reliance_open_best_order}")
print(f"Best ARIMA Order for High: {reliance_high_best_order}")
print(f"Best ARIMA Order for Low: {reliance_low_best_order}")

reliance_close_final_model = sm.tsa.ARIMA(
    reliance_y_close_scaled,
    order=reliance_close_best_order
)
reliance_close_final_model = reliance_close_final_model.fit()
reliance_close_forecast = reliance_close_final_model.forecast(steps=forecast_period)
reliance_close_forecast = reliance_close_forecast.reshape(-1, 1)
reliance_close_forecast = scaler.inverse_transform(reliance_close_forecast)

reliance_open_final_model = sm.tsa.ARIMA(
    reliance_y_open_scaled,
    order=reliance_open_best_order
)
reliance_open_final_model = reliance_open_final_model.fit()
reliance_open_forecast = reliance_open_final_model.forecast(steps=forecast_period)
reliance_open_forecast = reliance_open_forecast.reshape(-1, 1)
reliance_open_forecast = scaler.inverse_transform(reliance_open_forecast)

reliance_high_final_model = sm.tsa.ARIMA(
    reliance_y_high_scaled,
    order=reliance_high_best_order
)
reliance_high_final_model = reliance_high_final_model.fit()
reliance_high_forecast = reliance_high_final_model.forecast(steps=forecast_period)
reliance_high_forecast = reliance_high_forecast.reshape(-1, 1)
reliance_high_forecast = scaler.inverse_transform(reliance_high_forecast)

reliance_low_final_model = sm.tsa.ARIMA(
    reliance_y_low_scaled,
    order=reliance_low_best_order
)
reliance_low_final_model = reliance_low_final_model.fit()
reliance_low_forecast = reliance_low_final_model.forecast(steps=forecast_period)
reliance_low_forecast = reliance_low_forecast.reshape(-1, 1)
reliance_low_forecast = scaler.inverse_transform(reliance_low_forecast)

print("Close Forecasts:", reliance_close_forecast)
print("Open Forecasts:", reliance_open_forecast)
print("High Forecasts:", reliance_high_forecast)
print("Low Forecasts:", reliance_low_forecast)


# In[228]:


reliance_tail_50_data = reliance.tail(forecast_periods)

reliance_actual_close_prices = reliance_tail_50_data['Close'].values
reliance_actual_open_prices = reliance_tail_50_data['Open'].values
reliance_actual_high_prices = reliance_tail_50_data['High'].values
reliance_actual_low_prices = reliance_tail_50_data['Low'].values

reliance_forecast_close = reliance_close_final_model.forecast(steps=forecast_periods)
reliance_forecast_close = reliance_forecast_close.reshape(-1, 1)
reliance_forecast_close = scaler.inverse_transform(reliance_forecast_close)

reliance_forecast_open = reliance_open_final_model.forecast(steps=forecast_periods)
reliance_forecast_open = reliance_forecast_open.reshape(-1, 1)
reliance_forecast_open = scaler.inverse_transform(reliance_forecast_open)

reliance_forecast_high = reliance_high_final_model.forecast(steps=forecast_periods)
reliance_forecast_high = reliance_forecast_high.reshape(-1, 1)
reliance_forecast_high = scaler.inverse_transform(reliance_forecast_high)

reliance_forecast_low = reliance_low_final_model.forecast(steps=forecast_periods)
reliance_forecast_low = reliance_forecast_low.reshape(-1, 1)
reliance_forecast_low = scaler.inverse_transform(reliance_forecast_low)

reliance_close_mae = mean_absolute_error(reliance_actual_close_prices, reliance_forecast_close)
reliance_close_mse = mean_squared_error(reliance_actual_close_prices, reliance_forecast_close)
reliance_close_rmse = np.sqrt(reliance_close_mse)

reliance_open_mae = mean_absolute_error(reliance_actual_open_prices, reliance_forecast_open)
reliance_open_mse = mean_squared_error(reliance_actual_open_prices, reliance_forecast_open)
reliance_open_rmse = np.sqrt(reliance_open_mse)

reliance_high_mae = mean_absolute_error(reliance_actual_high_prices, reliance_forecast_high)
reliance_high_mse = mean_squared_error(reliance_actual_high_prices, reliance_forecast_high)
reliance_high_rmse = np.sqrt(reliance_high_mse)

reliance_low_mae = mean_absolute_error(reliance_actual_low_prices, reliance_forecast_low)
reliance_low_mse = mean_squared_error(reliance_actual_low_prices, reliance_forecast_low)
reliance_low_rmse = np.sqrt(reliance_low_mse)

reliance_close_mape = mean_absolute_percentage_error(reliance_actual_close_prices, reliance_forecast_close)
reliance_open_mape = mean_absolute_percentage_error(reliance_actual_open_prices, reliance_forecast_open)
reliance_high_mape = mean_absolute_percentage_error(reliance_actual_high_prices, reliance_forecast_high)
reliance_low_mape = mean_absolute_percentage_error(reliance_actual_low_prices, reliance_forecast_low)

print("Close Forecasts:", reliance_forecast_close)
print(f"Close Mean Absolute Error (MAE): {reliance_close_mae}")
print(f"Close Mean Squared Error (MSE): {reliance_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {reliance_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {reliance_close_mape}%")

print("Open Forecasts:", reliance_forecast_open)
print(f"Open Mean Absolute Error (MAE): {reliance_open_mae}")
print(f"Open Mean Squared Error (MSE): {reliance_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {reliance_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {reliance_open_mape}%")

print("High Forecasts:", reliance_forecast_high)
print(f"High Mean Absolute Error (MAE): {reliance_high_mae}")
print(f"High Mean Squared Error (MSE): {reliance_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {reliance_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {reliance_high_mape}%")

print("Low Forecasts:", reliance_forecast_low)
print(f"Low Mean Absolute Error (MAE): {reliance_low_mae}")
print(f"Low Mean Squared Error (MSE): {reliance_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {reliance_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {reliance_low_mape}")


# In[229]:


sbin_y_close = sbin['Close'].values
sbin_y_open = sbin['Open'].values
sbin_y_high = sbin['High'].values
sbin_y_low = sbin['Low'].values

sbin_y_close_scaled = scaler.fit_transform(sbin_y_close.reshape(-1, 1))
sbin_y_open_scaled = scaler.fit_transform(sbin_y_open.reshape(-1, 1))
sbin_y_high_scaled = scaler.fit_transform(sbin_y_high.reshape(-1, 1))
sbin_y_low_scaled = scaler.fit_transform(sbin_y_low.reshape(-1, 1))

sbin_close_model = auto_arima(
    sbin_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sbin_open_model = auto_arima(
    sbin_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sbin_high_model = auto_arima(
    sbin_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sbin_low_model = auto_arima(
    sbin_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sbin_close_best_order = sbin_close_model.get_params()['order']
sbin_open_best_order = sbin_open_model.get_params()['order']
sbin_high_best_order = sbin_high_model.get_params()['order']
sbin_low_best_order = sbin_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {sbin_close_best_order}")
print(f"Best ARIMA Order for Open: {sbin_open_best_order}")
print(f"Best ARIMA Order for High: {sbin_high_best_order}")
print(f"Best ARIMA Order for Low: {sbin_low_best_order}")

sbin_close_final_model = sm.tsa.ARIMA(
    sbin_y_close_scaled,
    order=sbin_close_best_order
)
sbin_close_final_model = sbin_close_final_model.fit()
sbin_close_forecast = sbin_close_final_model.forecast(steps=forecast_period)
sbin_close_forecast = sbin_close_forecast.reshape(-1, 1)
sbin_close_forecast = scaler.inverse_transform(sbin_close_forecast)

sbin_open_final_model = sm.tsa.ARIMA(
    sbin_y_open_scaled,
    order=sbin_open_best_order
)
sbin_open_final_model = sbin_open_final_model.fit()
sbin_open_forecast = sbin_open_final_model.forecast(steps=forecast_period)
sbin_open_forecast = sbin_open_forecast.reshape(-1, 1)
sbin_open_forecast = scaler.inverse_transform(sbin_open_forecast)

sbin_high_final_model = sm.tsa.ARIMA(
    sbin_y_high_scaled,
    order=sbin_high_best_order
)
sbin_high_final_model = sbin_high_final_model.fit()
sbin_high_forecast = sbin_high_final_model.forecast(steps=forecast_period)
sbin_high_forecast = sbin_high_forecast.reshape(-1, 1)
sbin_high_forecast = scaler.inverse_transform(sbin_high_forecast)

sbin_low_final_model = sm.tsa.ARIMA(
    sbin_y_low_scaled,
    order=sbin_low_best_order
)
sbin_low_final_model = sbin_low_final_model.fit()
sbin_low_forecast = sbin_low_final_model.forecast(steps=forecast_period)
sbin_low_forecast = sbin_low_forecast.reshape(-1, 1)
sbin_low_forecast = scaler.inverse_transform(sbin_low_forecast)

print("Close Forecasts:", sbin_close_forecast)
print("Open Forecasts:", sbin_open_forecast)
print("High Forecasts:", sbin_high_forecast)
print("Low Forecasts:", sbin_low_forecast)


# In[230]:


sbin_tail_50_data = sbin.tail(forecast_periods)

sbin_actual_close_prices = sbin_tail_50_data['Close'].values
sbin_actual_open_prices = sbin_tail_50_data['Open'].values
sbin_actual_high_prices = sbin_tail_50_data['High'].values
sbin_actual_low_prices = sbin_tail_50_data['Low'].values

sbin_forecast_close = sbin_close_final_model.forecast(steps=forecast_periods)
sbin_forecast_close = sbin_forecast_close.reshape(-1, 1)
sbin_forecast_close = scaler.inverse_transform(sbin_forecast_close)

sbin_forecast_open = sbin_open_final_model.forecast(steps=forecast_periods)
sbin_forecast_open = sbin_forecast_open.reshape(-1, 1)
sbin_forecast_open = scaler.inverse_transform(sbin_forecast_open)

sbin_forecast_high = sbin_high_final_model.forecast(steps=forecast_periods)
sbin_forecast_high = sbin_forecast_high.reshape(-1, 1)
sbin_forecast_high = scaler.inverse_transform(sbin_forecast_high)

sbin_forecast_low = sbin_low_final_model.forecast(steps=forecast_periods)
sbin_forecast_low = sbin_forecast_low.reshape(-1, 1)
sbin_forecast_low = scaler.inverse_transform(sbin_forecast_low)

sbin_close_mae = mean_absolute_error(sbin_actual_close_prices, sbin_forecast_close)
sbin_close_mse = mean_squared_error(sbin_actual_close_prices, sbin_forecast_close)
sbin_close_rmse = np.sqrt(sbin_close_mse)

sbin_open_mae = mean_absolute_error(sbin_actual_open_prices, sbin_forecast_open)
sbin_open_mse = mean_squared_error(sbin_actual_open_prices, sbin_forecast_open)
sbin_open_rmse = np.sqrt(sbin_open_mse)

sbin_high_mae = mean_absolute_error(sbin_actual_high_prices, sbin_forecast_high)
sbin_high_mse = mean_squared_error(sbin_actual_high_prices, sbin_forecast_high)
sbin_high_rmse = np.sqrt(sbin_high_mse)

sbin_low_mae = mean_absolute_error(sbin_actual_low_prices, sbin_forecast_low)
sbin_low_mse = mean_squared_error(sbin_actual_low_prices, sbin_forecast_low)
sbin_low_rmse = np.sqrt(sbin_low_mse)

sbin_close_mape = mean_absolute_percentage_error(sbin_actual_close_prices, sbin_forecast_close)
sbin_open_mape = mean_absolute_percentage_error(sbin_actual_open_prices, sbin_forecast_open)
sbin_high_mape = mean_absolute_percentage_error(sbin_actual_high_prices, sbin_forecast_high)
sbin_low_mape = mean_absolute_percentage_error(sbin_actual_low_prices, sbin_forecast_low)

print("Close Forecasts:", sbin_forecast_close)
print(f"Close Mean Absolute Error (MAE): {sbin_close_mae}")
print(f"Close Mean Squared Error (MSE): {sbin_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {sbin_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {sbin_close_mape}%")

print("Open Forecasts:", sbin_forecast_open)
print(f"Open Mean Absolute Error (MAE): {sbin_open_mae}")
print(f"Open Mean Squared Error (MSE): {sbin_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {sbin_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {sbin_open_mape}%")

print("High Forecasts:", sbin_forecast_high)
print(f"High Mean Absolute Error (MAE): {sbin_high_mae}")
print(f"High Mean Squared Error (MSE): {sbin_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {sbin_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {sbin_high_mape}%")

print("Low Forecasts:", sbin_forecast_low)
print(f"Low Mean Absolute Error (MAE): {sbin_low_mae}")
print(f"Low Mean Squared Error (MSE): {sbin_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {sbin_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {sbin_low_mape}")


# In[231]:


shreecem_y_close = shreecem['Close'].values
shreecem_y_open = shreecem['Open'].values
shreecem_y_high = shreecem['High'].values
shreecem_y_low = shreecem['Low'].values

shreecem_y_close_scaled = scaler.fit_transform(shreecem_y_close.reshape(-1, 1))
shreecem_y_open_scaled = scaler.fit_transform(shreecem_y_open.reshape(-1, 1))
shreecem_y_high_scaled = scaler.fit_transform(shreecem_y_high.reshape(-1, 1))
shreecem_y_low_scaled = scaler.fit_transform(shreecem_y_low.reshape(-1, 1))

shreecem_close_model = auto_arima(
    shreecem_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

shreecem_open_model = auto_arima(
    shreecem_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

shreecem_high_model = auto_arima(
    shreecem_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

shreecem_low_model = auto_arima(
    shreecem_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

shreecem_close_best_order = shreecem_close_model.get_params()['order']
shreecem_open_best_order = shreecem_open_model.get_params()['order']
shreecem_high_best_order = shreecem_high_model.get_params()['order']
shreecem_low_best_order = shreecem_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {shreecem_close_best_order}")
print(f"Best ARIMA Order for Open: {shreecem_open_best_order}")
print(f"Best ARIMA Order for High: {shreecem_high_best_order}")
print(f"Best ARIMA Order for Low: {shreecem_low_best_order}")

shreecem_close_final_model = sm.tsa.ARIMA(
    shreecem_y_close_scaled,
    order=shreecem_close_best_order
)
shreecem_close_final_model = shreecem_close_final_model.fit()
shreecem_close_forecast = shreecem_close_final_model.forecast(steps=forecast_period)
shreecem_close_forecast = shreecem_close_forecast.reshape(-1, 1)
shreecem_close_forecast = scaler.inverse_transform(shreecem_close_forecast)

shreecem_open_final_model = sm.tsa.ARIMA(
    shreecem_y_open_scaled,
    order=shreecem_open_best_order
)
shreecem_open_final_model = shreecem_open_final_model.fit()
shreecem_open_forecast = shreecem_open_final_model.forecast(steps=forecast_period)
shreecem_open_forecast = shreecem_open_forecast.reshape(-1, 1)
shreecem_open_forecast = scaler.inverse_transform(shreecem_open_forecast)

shreecem_high_final_model = sm.tsa.ARIMA(
    shreecem_y_high_scaled,
    order=shreecem_high_best_order
)
shreecem_high_final_model = shreecem_high_final_model.fit()
shreecem_high_forecast = shreecem_high_final_model.forecast(steps=forecast_period)
shreecem_high_forecast = shreecem_high_forecast.reshape(-1, 1)
shreecem_high_forecast = scaler.inverse_transform(shreecem_high_forecast)

shreecem_low_final_model = sm.tsa.ARIMA(
    shreecem_y_low_scaled,
    order=shreecem_low_best_order
)
shreecem_low_final_model = shreecem_low_final_model.fit()
shreecem_low_forecast = shreecem_low_final_model.forecast(steps=forecast_period)
shreecem_low_forecast = shreecem_low_forecast.reshape(-1, 1)
shreecem_low_forecast = scaler.inverse_transform(shreecem_low_forecast)

print("Close Forecasts:", shreecem_close_forecast)
print("Open Forecasts:", shreecem_open_forecast)
print("High Forecasts:", shreecem_high_forecast)
print("Low Forecasts:", shreecem_low_forecast)


# In[232]:


shreecem_tail_50_data = shreecem.tail(forecast_periods)

shreecem_actual_close_prices = shreecem_tail_50_data['Close'].values
shreecem_actual_open_prices = shreecem_tail_50_data['Open'].values
shreecem_actual_high_prices = shreecem_tail_50_data['High'].values
shreecem_actual_low_prices = shreecem_tail_50_data['Low'].values

shreecem_forecast_close = shreecem_close_final_model.forecast(steps=forecast_periods)
shreecem_forecast_close = shreecem_forecast_close.reshape(-1, 1)
shreecem_forecast_close = scaler.inverse_transform(shreecem_forecast_close)

shreecem_forecast_open = shreecem_open_final_model.forecast(steps=forecast_periods)
shreecem_forecast_open = shreecem_forecast_open.reshape(-1, 1)
shreecem_forecast_open = scaler.inverse_transform(shreecem_forecast_open)

shreecem_forecast_high = shreecem_high_final_model.forecast(steps=forecast_periods)
shreecem_forecast_high = shreecem_forecast_high.reshape(-1, 1)
shreecem_forecast_high = scaler.inverse_transform(shreecem_forecast_high)

shreecem_forecast_low = shreecem_low_final_model.forecast(steps=forecast_periods)
shreecem_forecast_low = shreecem_forecast_low.reshape(-1, 1)
shreecem_forecast_low = scaler.inverse_transform(shreecem_forecast_low)

shreecem_close_mae = mean_absolute_error(shreecem_actual_close_prices, shreecem_forecast_close)
shreecem_close_mse = mean_squared_error(shreecem_actual_close_prices, shreecem_forecast_close)
shreecem_close_rmse = np.sqrt(shreecem_close_mse)

shreecem_open_mae = mean_absolute_error(shreecem_actual_open_prices, shreecem_forecast_open)
shreecem_open_mse = mean_squared_error(shreecem_actual_open_prices, shreecem_forecast_open)
shreecem_open_rmse = np.sqrt(shreecem_open_mse)

shreecem_high_mae = mean_absolute_error(shreecem_actual_high_prices, shreecem_forecast_high)
shreecem_high_mse = mean_squared_error(shreecem_actual_high_prices, shreecem_forecast_high)
shreecem_high_rmse = np.sqrt(shreecem_high_mse)

shreecem_low_mae = mean_absolute_error(shreecem_actual_low_prices, shreecem_forecast_low)
shreecem_low_mse = mean_squared_error(shreecem_actual_low_prices, shreecem_forecast_low)
shreecem_low_rmse = np.sqrt(shreecem_low_mse)

shreecem_close_mape = mean_absolute_percentage_error(shreecem_actual_close_prices, shreecem_forecast_close)
shreecem_open_mape = mean_absolute_percentage_error(shreecem_actual_open_prices, shreecem_forecast_open)
shreecem_high_mape = mean_absolute_percentage_error(shreecem_actual_high_prices, shreecem_forecast_high)
shreecem_low_mape = mean_absolute_percentage_error(shreecem_actual_low_prices, shreecem_forecast_low)

print("Close Forecasts:", shreecem_forecast_close)
print(f"Close Mean Absolute Error (MAE): {shreecem_close_mae}")
print(f"Close Mean Squared Error (MSE): {shreecem_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {shreecem_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {shreecem_close_mape}%")

print("Open Forecasts:", shreecem_forecast_open)
print(f"Open Mean Absolute Error (MAE): {shreecem_open_mae}")
print(f"Open Mean Squared Error (MSE): {shreecem_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {shreecem_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {shreecem_open_mape}%")

print("High Forecasts:", shreecem_forecast_high)
print(f"High Mean Absolute Error (MAE): {shreecem_high_mae}")
print(f"High Mean Squared Error (MSE): {shreecem_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {shreecem_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {shreecem_high_mape}%")

print("Low Forecasts:", shreecem_forecast_low)
print(f"Low Mean Absolute Error (MAE): {shreecem_low_mae}")
print(f"Low Mean Squared Error (MSE): {shreecem_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {shreecem_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {shreecem_low_mape}")


# In[233]:


sunpharma_y_close = sunpharma['Close'].values
sunpharma_y_open = sunpharma['Open'].values
sunpharma_y_high = sunpharma['High'].values
sunpharma_y_low = sunpharma['Low'].values

sunpharma_y_close_scaled = scaler.fit_transform(sunpharma_y_close.reshape(-1, 1))
sunpharma_y_open_scaled = scaler.fit_transform(sunpharma_y_open.reshape(-1, 1))
sunpharma_y_high_scaled = scaler.fit_transform(sunpharma_y_high.reshape(-1, 1))
sunpharma_y_low_scaled = scaler.fit_transform(sunpharma_y_low.reshape(-1, 1))

sunpharma_close_model = auto_arima(
    sunpharma_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sunpharma_open_model = auto_arima(
    sunpharma_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sunpharma_high_model = auto_arima(
    sunpharma_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sunpharma_low_model = auto_arima(
    sunpharma_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sunpharma_close_best_order = sunpharma_close_model.get_params()['order']
sunpharma_open_best_order = sunpharma_open_model.get_params()['order']
sunpharma_high_best_order = sunpharma_high_model.get_params()['order']
sunpharma_low_best_order = sunpharma_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {sunpharma_close_best_order}")
print(f"Best ARIMA Order for Open: {sunpharma_open_best_order}")
print(f"Best ARIMA Order for High: {sunpharma_high_best_order}")
print(f"Best ARIMA Order for Low: {sunpharma_low_best_order}")

sunpharma_close_final_model = sm.tsa.ARIMA(
    sunpharma_y_close_scaled,
    order=sunpharma_close_best_order
)
sunpharma_close_final_model = sunpharma_close_final_model.fit()
sunpharma_close_forecast = sunpharma_close_final_model.forecast(steps=forecast_period)
sunpharma_close_forecast = sunpharma_close_forecast.reshape(-1, 1)
sunpharma_close_forecast = scaler.inverse_transform(sunpharma_close_forecast)

sunpharma_open_final_model = sm.tsa.ARIMA(
    sunpharma_y_open_scaled,
    order=sunpharma_open_best_order
)
sunpharma_open_final_model = sunpharma_open_final_model.fit()
sunpharma_open_forecast = sunpharma_open_final_model.forecast(steps=forecast_period)
sunpharma_open_forecast = sunpharma_open_forecast.reshape(-1, 1)
sunpharma_open_forecast = scaler.inverse_transform(sunpharma_open_forecast)

sunpharma_high_final_model = sm.tsa.ARIMA(
    sunpharma_y_high_scaled,
    order=sunpharma_high_best_order
)
sunpharma_high_final_model = sunpharma_high_final_model.fit()
sunpharma_high_forecast = sunpharma_high_final_model.forecast(steps=forecast_period)
sunpharma_high_forecast = sunpharma_high_forecast.reshape(-1, 1)
sunpharma_high_forecast = scaler.inverse_transform(sunpharma_high_forecast)

sunpharma_low_final_model = sm.tsa.ARIMA(
    sunpharma_y_low_scaled,
    order=sunpharma_low_best_order
)
sunpharma_low_final_model = sunpharma_low_final_model.fit()
sunpharma_low_forecast = sunpharma_low_final_model.forecast(steps=forecast_period)
sunpharma_low_forecast = sunpharma_low_forecast.reshape(-1, 1)
sunpharma_low_forecast = scaler.inverse_transform(sunpharma_low_forecast)

print("Close Forecasts:", sunpharma_close_forecast)
print("Open Forecasts:", sunpharma_open_forecast)
print("High Forecasts:", sunpharma_high_forecast)
print("Low Forecasts:", sunpharma_low_forecast)


# In[234]:


sunpharma_tail_50_data = sunpharma.tail(forecast_periods)

sunpharma_actual_close_prices = sunpharma_tail_50_data['Close'].values
sunpharma_actual_open_prices = sunpharma_tail_50_data['Open'].values
sunpharma_actual_high_prices = sunpharma_tail_50_data['High'].values
sunpharma_actual_low_prices = sunpharma_tail_50_data['Low'].values

sunpharma_forecast_close = sunpharma_close_final_model.forecast(steps=forecast_periods)
sunpharma_forecast_close = sunpharma_forecast_close.reshape(-1, 1)
sunpharma_forecast_close = scaler.inverse_transform(sunpharma_forecast_close)

sunpharma_forecast_open = sunpharma_open_final_model.forecast(steps=forecast_periods)
sunpharma_forecast_open = sunpharma_forecast_open.reshape(-1, 1)
sunpharma_forecast_open = scaler.inverse_transform(sunpharma_forecast_open)

sunpharma_forecast_high = sunpharma_high_final_model.forecast(steps=forecast_periods)
sunpharma_forecast_high = sunpharma_forecast_high.reshape(-1, 1)
sunpharma_forecast_high = scaler.inverse_transform(sunpharma_forecast_high)

sunpharma_forecast_low = sunpharma_low_final_model.forecast(steps=forecast_periods)
sunpharma_forecast_low = sunpharma_forecast_low.reshape(-1, 1)
sunpharma_forecast_low = scaler.inverse_transform(sunpharma_forecast_low)

sunpharma_close_mae = mean_absolute_error(sunpharma_actual_close_prices, sunpharma_forecast_close)
sunpharma_close_mse = mean_squared_error(sunpharma_actual_close_prices, sunpharma_forecast_close)
sunpharma_close_rmse = np.sqrt(sunpharma_close_mse)

sunpharma_open_mae = mean_absolute_error(sunpharma_actual_open_prices, sunpharma_forecast_open)
sunpharma_open_mse = mean_squared_error(sunpharma_actual_open_prices, sunpharma_forecast_open)
sunpharma_open_rmse = np.sqrt(sunpharma_open_mse)

sunpharma_high_mae = mean_absolute_error(sunpharma_actual_high_prices, sunpharma_forecast_high)
sunpharma_high_mse = mean_squared_error(sunpharma_actual_high_prices, sunpharma_forecast_high)
sunpharma_high_rmse = np.sqrt(sunpharma_high_mse)

sunpharma_low_mae = mean_absolute_error(sunpharma_actual_low_prices, sunpharma_forecast_low)
sunpharma_low_mse = mean_squared_error(sunpharma_actual_low_prices, sunpharma_forecast_low)
sunpharma_low_rmse = np.sqrt(sunpharma_low_mse)

sunpharma_close_mape = mean_absolute_percentage_error(sunpharma_actual_close_prices, sunpharma_forecast_close)
sunpharma_open_mape = mean_absolute_percentage_error(sunpharma_actual_open_prices, sunpharma_forecast_open)
sunpharma_high_mape = mean_absolute_percentage_error(sunpharma_actual_high_prices, sunpharma_forecast_high)
sunpharma_low_mape = mean_absolute_percentage_error(sunpharma_actual_low_prices, sunpharma_forecast_low)

print("Close Forecasts:", sunpharma_forecast_close)
print(f"Close Mean Absolute Error (MAE): {sunpharma_close_mae}")
print(f"Close Mean Squared Error (MSE): {sunpharma_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {sunpharma_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {sunpharma_close_mape}%")

print("Open Forecasts:", sunpharma_forecast_open)
print(f"Open Mean Absolute Error (MAE): {sunpharma_open_mae}")
print(f"Open Mean Squared Error (MSE): {sunpharma_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {sunpharma_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {sunpharma_open_mape}%")

print("High Forecasts:", sunpharma_forecast_high)
print(f"High Mean Absolute Error (MAE): {sunpharma_high_mae}")
print(f"High Mean Squared Error (MSE): {sunpharma_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {sunpharma_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {sunpharma_high_mape}%")

print("Low Forecasts:", sunpharma_forecast_low)
print(f"Low Mean Absolute Error (MAE): {sunpharma_low_mae}")
print(f"Low Mean Squared Error (MSE): {sunpharma_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {sunpharma_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {sunpharma_low_mape}")


# In[235]:


telco_y_close = telco['Close'].values
telco_y_open = telco['Open'].values
telco_y_high = telco['High'].values
telco_y_low = telco['Low'].values

telco_y_close_scaled = scaler.fit_transform(telco_y_close.reshape(-1, 1))
telco_y_open_scaled = scaler.fit_transform(telco_y_open.reshape(-1, 1))
telco_y_high_scaled = scaler.fit_transform(telco_y_high.reshape(-1, 1))
telco_y_low_scaled = scaler.fit_transform(telco_y_low.reshape(-1, 1))

telco_close_model = auto_arima(
    telco_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

telco_open_model = auto_arima(
    telco_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

telco_high_model = auto_arima(
    telco_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

telco_low_model = auto_arima(
    telco_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

telco_close_best_order = telco_close_model.get_params()['order']
telco_open_best_order = telco_open_model.get_params()['order']
telco_high_best_order = telco_high_model.get_params()['order']
telco_low_best_order = telco_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {telco_close_best_order}")
print(f"Best ARIMA Order for Open: {telco_open_best_order}")
print(f"Best ARIMA Order for High: {telco_high_best_order}")
print(f"Best ARIMA Order for Low: {telco_low_best_order}")

telco_close_final_model = sm.tsa.ARIMA(
    telco_y_close_scaled,
    order=telco_close_best_order
)
telco_close_final_model = telco_close_final_model.fit()
telco_close_forecast = telco_close_final_model.forecast(steps=forecast_period)
telco_close_forecast = telco_close_forecast.reshape(-1, 1)
telco_close_forecast = scaler.inverse_transform(telco_close_forecast)

telco_open_final_model = sm.tsa.ARIMA(
    telco_y_open_scaled,
    order=telco_open_best_order
)
telco_open_final_model = telco_open_final_model.fit()
telco_open_forecast = telco_open_final_model.forecast(steps=forecast_period)
telco_open_forecast = telco_open_forecast.reshape(-1, 1)
telco_open_forecast = scaler.inverse_transform(telco_open_forecast)

telco_high_final_model = sm.tsa.ARIMA(
    telco_y_high_scaled,
    order=telco_high_best_order
)
telco_high_final_model = telco_high_final_model.fit()
telco_high_forecast = telco_high_final_model.forecast(steps=forecast_period)
telco_high_forecast = telco_high_forecast.reshape(-1, 1)
telco_high_forecast = scaler.inverse_transform(telco_high_forecast)

telco_low_final_model = sm.tsa.ARIMA(
    telco_y_low_scaled,
    order=telco_low_best_order
)
telco_low_final_model = telco_low_final_model.fit()
telco_low_forecast = telco_low_final_model.forecast(steps=forecast_period)
telco_low_forecast = telco_low_forecast.reshape(-1, 1)
telco_low_forecast = scaler.inverse_transform(telco_low_forecast)

print("Close Forecasts:", telco_close_forecast)
print("Open Forecasts:", telco_open_forecast)
print("High Forecasts:", telco_high_forecast)
print("Low Forecasts:", telco_low_forecast)


# In[236]:


telco_tail_50_data = telco.tail(forecast_periods)

telco_actual_close_prices = telco_tail_50_data['Close'].values
telco_actual_open_prices = telco_tail_50_data['Open'].values
telco_actual_high_prices = telco_tail_50_data['High'].values
telco_actual_low_prices = telco_tail_50_data['Low'].values

telco_forecast_close = telco_close_final_model.forecast(steps=forecast_periods)
telco_forecast_close = telco_forecast_close.reshape(-1, 1)
telco_forecast_close = scaler.inverse_transform(telco_forecast_close)

telco_forecast_open = telco_open_final_model.forecast(steps=forecast_periods)
telco_forecast_open = telco_forecast_open.reshape(-1, 1)
telco_forecast_open = scaler.inverse_transform(telco_forecast_open)

telco_forecast_high = telco_high_final_model.forecast(steps=forecast_periods)
telco_forecast_high = telco_forecast_high.reshape(-1, 1)
telco_forecast_high = scaler.inverse_transform(telco_forecast_high)

telco_forecast_low = telco_low_final_model.forecast(steps=forecast_periods)
telco_forecast_low = telco_forecast_low.reshape(-1, 1)
telco_forecast_low = scaler.inverse_transform(telco_forecast_low)

telco_close_mae = mean_absolute_error(telco_actual_close_prices, telco_forecast_close)
telco_close_mse = mean_squared_error(telco_actual_close_prices, telco_forecast_close)
telco_close_rmse = np.sqrt(telco_close_mse)

telco_open_mae = mean_absolute_error(telco_actual_open_prices, telco_forecast_open)
telco_open_mse = mean_squared_error(telco_actual_open_prices, telco_forecast_open)
telco_open_rmse = np.sqrt(telco_open_mse)

telco_high_mae = mean_absolute_error(telco_actual_high_prices, telco_forecast_high)
telco_high_mse = mean_squared_error(telco_actual_high_prices, telco_forecast_high)
telco_high_rmse = np.sqrt(telco_high_mse)

telco_low_mae = mean_absolute_error(telco_actual_low_prices, telco_forecast_low)
telco_low_mse = mean_squared_error(telco_actual_low_prices, telco_forecast_low)
telco_low_rmse = np.sqrt(telco_low_mse)

telco_close_mape = mean_absolute_percentage_error(telco_actual_close_prices, telco_forecast_close)
telco_open_mape = mean_absolute_percentage_error(telco_actual_open_prices, telco_forecast_open)
telco_high_mape = mean_absolute_percentage_error(telco_actual_high_prices, telco_forecast_high)
telco_low_mape = mean_absolute_percentage_error(telco_actual_low_prices, telco_forecast_low)

print("Close Forecasts:", telco_forecast_close)
print(f"Close Mean Absolute Error (MAE): {telco_close_mae}")
print(f"Close Mean Squared Error (MSE): {telco_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {telco_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {telco_close_mape}%")

print("Open Forecasts:", telco_forecast_open)
print(f"Open Mean Absolute Error (MAE): {telco_open_mae}")
print(f"Open Mean Squared Error (MSE): {telco_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {telco_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {telco_open_mape}%")

print("High Forecasts:", telco_forecast_high)
print(f"High Mean Absolute Error (MAE): {telco_high_mae}")
print(f"High Mean Squared Error (MSE): {telco_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {telco_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {telco_high_mape}%")

print("Low Forecasts:", telco_forecast_low)
print(f"Low Mean Absolute Error (MAE): {telco_low_mae}")
print(f"Low Mean Squared Error (MSE): {telco_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {telco_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {telco_low_mape}")


# tatasteel = company_datasets['TATASTEEL'] tcs = company_datasets['TCS'] techm = company_datasets['TECHM'] titan = company_datasets['TITAN'] ultracemco = company_datasets['ULTRACEMCO'] uniphos = company_datasets['UNIPHOS'] upl = company_datasets['UPL'] sesagoa = company_datasets['SESAGOA'] sslt = company_datasets['SSLT'] vedl = company_datasets['VEDL'] wipro = company_datasets['WIPRO'] zeetele = company_datasets['ZEETELE'] zeel = company_datasets['ZEEL']

# In[237]:


tatamotors_y_close = tatamotors['Close'].values
tatamotors_y_open = tatamotors['Open'].values
tatamotors_y_high = tatamotors['High'].values
tatamotors_y_low = tatamotors['Low'].values

tatamotors_y_close_scaled = scaler.fit_transform(tatamotors_y_close.reshape(-1, 1))
tatamotors_y_open_scaled = scaler.fit_transform(tatamotors_y_open.reshape(-1, 1))
tatamotors_y_high_scaled = scaler.fit_transform(tatamotors_y_high.reshape(-1, 1))
tatamotors_y_low_scaled = scaler.fit_transform(tatamotors_y_low.reshape(-1, 1))

tatamotors_close_model = auto_arima(
    tatamotors_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tatamotors_open_model = auto_arima(
    tatamotors_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tatamotors_high_model = auto_arima(
    tatamotors_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tatamotors_low_model = auto_arima(
    tatamotors_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tatamotors_close_best_order = tatamotors_close_model.get_params()['order']
tatamotors_open_best_order = tatamotors_open_model.get_params()['order']
tatamotors_high_best_order = tatamotors_high_model.get_params()['order']
tatamotors_low_best_order = tatamotors_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {tatamotors_close_best_order}")
print(f"Best ARIMA Order for Open: {tatamotors_open_best_order}")
print(f"Best ARIMA Order for High: {tatamotors_high_best_order}")
print(f"Best ARIMA Order for Low: {tatamotors_low_best_order}")

tatamotors_close_final_model = sm.tsa.ARIMA(
    tatamotors_y_close_scaled,
    order=tatamotors_close_best_order
)
tatamotors_close_final_model = tatamotors_close_final_model.fit()
tatamotors_close_forecast = tatamotors_close_final_model.forecast(steps=forecast_period)
tatamotors_close_forecast = tatamotors_close_forecast.reshape(-1, 1)
tatamotors_close_forecast = scaler.inverse_transform(tatamotors_close_forecast)

tatamotors_open_final_model = sm.tsa.ARIMA(
    tatamotors_y_open_scaled,
    order=tatamotors_open_best_order
)
tatamotors_open_final_model = tatamotors_open_final_model.fit()
tatamotors_open_forecast = tatamotors_open_final_model.forecast(steps=forecast_period)
tatamotors_open_forecast = tatamotors_open_forecast.reshape(-1, 1)
tatamotors_open_forecast = scaler.inverse_transform(tatamotors_open_forecast)

tatamotors_high_final_model = sm.tsa.ARIMA(
    tatamotors_y_high_scaled,
    order=tatamotors_high_best_order
)
tatamotors_high_final_model = tatamotors_high_final_model.fit()
tatamotors_high_forecast = tatamotors_high_final_model.forecast(steps=forecast_period)
tatamotors_high_forecast = tatamotors_high_forecast.reshape(-1, 1)
tatamotors_high_forecast = scaler.inverse_transform(tatamotors_high_forecast)

tatamotors_low_final_model = sm.tsa.ARIMA(
    tatamotors_y_low_scaled,
    order=tatamotors_low_best_order
)
tatamotors_low_final_model = tatamotors_low_final_model.fit()
tatamotors_low_forecast = tatamotors_low_final_model.forecast(steps=forecast_period)
tatamotors_low_forecast = tatamotors_low_forecast.reshape(-1, 1)
tatamotors_low_forecast = scaler.inverse_transform(tatamotors_low_forecast)

print("Close Forecasts:", tatamotors_close_forecast)
print("Open Forecasts:", tatamotors_open_forecast)
print("High Forecasts:", tatamotors_high_forecast)
print("Low Forecasts:", tatamotors_low_forecast)


# In[238]:


tatamotors_tail_50_data = tatamotors.tail(forecast_periods)

tatamotors_actual_close_prices = tatamotors_tail_50_data['Close'].values
tatamotors_actual_open_prices = tatamotors_tail_50_data['Open'].values
tatamotors_actual_high_prices = tatamotors_tail_50_data['High'].values
tatamotors_actual_low_prices = tatamotors_tail_50_data['Low'].values

tatamotors_forecast_close = tatamotors_close_final_model.forecast(steps=forecast_periods)
tatamotors_forecast_close = tatamotors_forecast_close.reshape(-1, 1)
tatamotors_forecast_close = scaler.inverse_transform(tatamotors_forecast_close)

tatamotors_forecast_open = tatamotors_open_final_model.forecast(steps=forecast_periods)
tatamotors_forecast_open = tatamotors_forecast_open.reshape(-1, 1)
tatamotors_forecast_open = scaler.inverse_transform(tatamotors_forecast_open)

tatamotors_forecast_high = tatamotors_high_final_model.forecast(steps=forecast_periods)
tatamotors_forecast_high = tatamotors_forecast_high.reshape(-1, 1)
tatamotors_forecast_high = scaler.inverse_transform(tatamotors_forecast_high)

tatamotors_forecast_low = tatamotors_low_final_model.forecast(steps=forecast_periods)
tatamotors_forecast_low = tatamotors_forecast_low.reshape(-1, 1)
tatamotors_forecast_low = scaler.inverse_transform(tatamotors_forecast_low)

tatamotors_close_mae = mean_absolute_error(tatamotors_actual_close_prices, tatamotors_forecast_close)
tatamotors_close_mse = mean_squared_error(tatamotors_actual_close_prices, tatamotors_forecast_close)
tatamotors_close_rmse = np.sqrt(tatamotors_close_mse)

tatamotors_open_mae = mean_absolute_error(tatamotors_actual_open_prices, tatamotors_forecast_open)
tatamotors_open_mse = mean_squared_error(tatamotors_actual_open_prices, tatamotors_forecast_open)
tatamotors_open_rmse = np.sqrt(tatamotors_open_mse)

tatamotors_high_mae = mean_absolute_error(tatamotors_actual_high_prices, tatamotors_forecast_high)
tatamotors_high_mse = mean_squared_error(tatamotors_actual_high_prices, tatamotors_forecast_high)
tatamotors_high_rmse = np.sqrt(tatamotors_high_mse)

tatamotors_low_mae = mean_absolute_error(tatamotors_actual_low_prices, tatamotors_forecast_low)
tatamotors_low_mse = mean_squared_error(tatamotors_actual_low_prices, tatamotors_forecast_low)
tatamotors_low_rmse = np.sqrt(tatamotors_low_mse)

tatamotors_close_mape = mean_absolute_percentage_error(tatamotors_actual_close_prices, tatamotors_forecast_close)
tatamotors_open_mape = mean_absolute_percentage_error(tatamotors_actual_open_prices, tatamotors_forecast_open)
tatamotors_high_mape = mean_absolute_percentage_error(tatamotors_actual_high_prices, tatamotors_forecast_high)
tatamotors_low_mape = mean_absolute_percentage_error(tatamotors_actual_low_prices, tatamotors_forecast_low)

print("Close Forecasts:", tatamotors_forecast_close)
print(f"Close Mean Absolute Error (MAE): {tatamotors_close_mae}")
print(f"Close Mean Squared Error (MSE): {tatamotors_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {tatamotors_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {tatamotors_close_mape}%")

print("Open Forecasts:", tatamotors_forecast_open)
print(f"Open Mean Absolute Error (MAE): {tatamotors_open_mae}")
print(f"Open Mean Squared Error (MSE): {tatamotors_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {tatamotors_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {tatamotors_open_mape}%")

print("High Forecasts:", tatamotors_forecast_high)
print(f"High Mean Absolute Error (MAE): {tatamotors_high_mae}")
print(f"High Mean Squared Error (MSE): {tatamotors_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {tatamotors_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {tatamotors_high_mape}%")

print("Low Forecasts:", tatamotors_forecast_low)
print(f"Low Mean Absolute Error (MAE): {tatamotors_low_mae}")
print(f"Low Mean Squared Error (MSE): {tatamotors_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {tatamotors_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {tatamotors_low_mape}")


# In[239]:


tisco_y_close = tisco['Close'].values
tisco_y_open = tisco['Open'].values
tisco_y_high = tisco['High'].values
tisco_y_low = tisco['Low'].values

tisco_y_close_scaled = scaler.fit_transform(tisco_y_close.reshape(-1, 1))
tisco_y_open_scaled = scaler.fit_transform(tisco_y_open.reshape(-1, 1))
tisco_y_high_scaled = scaler.fit_transform(tisco_y_high.reshape(-1, 1))
tisco_y_low_scaled = scaler.fit_transform(tisco_y_low.reshape(-1, 1))

tisco_close_model = auto_arima(
    tisco_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tisco_open_model = auto_arima(
    tisco_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tisco_high_model = auto_arima(
    tisco_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tisco_low_model = auto_arima(
    tisco_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tisco_close_best_order = tisco_close_model.get_params()['order']
tisco_open_best_order = tisco_open_model.get_params()['order']
tisco_high_best_order = tisco_high_model.get_params()['order']
tisco_low_best_order = tisco_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {tisco_close_best_order}")
print(f"Best ARIMA Order for Open: {tisco_open_best_order}")
print(f"Best ARIMA Order for High: {tisco_high_best_order}")
print(f"Best ARIMA Order for Low: {tisco_low_best_order}")

tisco_close_final_model = sm.tsa.ARIMA(
    tisco_y_close_scaled,
    order=tisco_close_best_order
)
tisco_close_final_model = tisco_close_final_model.fit()
tisco_close_forecast = tisco_close_final_model.forecast(steps=forecast_period)
tisco_close_forecast = tisco_close_forecast.reshape(-1, 1)
tisco_close_forecast = scaler.inverse_transform(tisco_close_forecast)

tisco_open_final_model = sm.tsa.ARIMA(
    tisco_y_open_scaled,
    order=tisco_open_best_order
)
tisco_open_final_model = tisco_open_final_model.fit()
tisco_open_forecast = tisco_open_final_model.forecast(steps=forecast_period)
tisco_open_forecast = tisco_open_forecast.reshape(-1, 1)
tisco_open_forecast = scaler.inverse_transform(tisco_open_forecast)

tisco_high_final_model = sm.tsa.ARIMA(
    tisco_y_high_scaled,
    order=tisco_high_best_order
)
tisco_high_final_model = tisco_high_final_model.fit()
tisco_high_forecast = tisco_high_final_model.forecast(steps=forecast_period)
tisco_high_forecast = tisco_high_forecast.reshape(-1, 1)
tisco_high_forecast = scaler.inverse_transform(tisco_high_forecast)

tisco_low_final_model = sm.tsa.ARIMA(
    tisco_y_low_scaled,
    order=tisco_low_best_order
)
tisco_low_final_model = tisco_low_final_model.fit()
tisco_low_forecast = tisco_low_final_model.forecast(steps=forecast_period)
tisco_low_forecast = tisco_low_forecast.reshape(-1, 1)
tisco_low_forecast = scaler.inverse_transform(tisco_low_forecast)

print("Close Forecasts:", tisco_close_forecast)
print("Open Forecasts:", tisco_open_forecast)
print("High Forecasts:", tisco_high_forecast)
print("Low Forecasts:", tisco_low_forecast)


# In[240]:


tisco_tail_50_data = tisco.tail(forecast_periods)

tisco_actual_close_prices = tisco_tail_50_data['Close'].values
tisco_actual_open_prices = tisco_tail_50_data['Open'].values
tisco_actual_high_prices = tisco_tail_50_data['High'].values
tisco_actual_low_prices = tisco_tail_50_data['Low'].values

tisco_forecast_close = tisco_close_final_model.forecast(steps=forecast_periods)
tisco_forecast_close = tisco_forecast_close.reshape(-1, 1)
tisco_forecast_close = scaler.inverse_transform(tisco_forecast_close)

tisco_forecast_open = tisco_open_final_model.forecast(steps=forecast_periods)
tisco_forecast_open = tisco_forecast_open.reshape(-1, 1)
tisco_forecast_open = scaler.inverse_transform(tisco_forecast_open)

tisco_forecast_high = tisco_high_final_model.forecast(steps=forecast_periods)
tisco_forecast_high = tisco_forecast_high.reshape(-1, 1)
tisco_forecast_high = scaler.inverse_transform(tisco_forecast_high)

tisco_forecast_low = tisco_low_final_model.forecast(steps=forecast_periods)
tisco_forecast_low = tisco_forecast_low.reshape(-1, 1)
tisco_forecast_low = scaler.inverse_transform(tisco_forecast_low)

tisco_close_mae = mean_absolute_error(tisco_actual_close_prices, tisco_forecast_close)
tisco_close_mse = mean_squared_error(tisco_actual_close_prices, tisco_forecast_close)
tisco_close_rmse = np.sqrt(tisco_close_mse)

tisco_open_mae = mean_absolute_error(tisco_actual_open_prices, tisco_forecast_open)
tisco_open_mse = mean_squared_error(tisco_actual_open_prices, tisco_forecast_open)
tisco_open_rmse = np.sqrt(tisco_open_mse)

tisco_high_mae = mean_absolute_error(tisco_actual_high_prices, tisco_forecast_high)
tisco_high_mse = mean_squared_error(tisco_actual_high_prices, tisco_forecast_high)
tisco_high_rmse = np.sqrt(tisco_high_mse)

tisco_low_mae = mean_absolute_error(tisco_actual_low_prices, tisco_forecast_low)
tisco_low_mse = mean_squared_error(tisco_actual_low_prices, tisco_forecast_low)
tisco_low_rmse = np.sqrt(tisco_low_mse)

tisco_close_mape = mean_absolute_percentage_error(tisco_actual_close_prices, tisco_forecast_close)
tisco_open_mape = mean_absolute_percentage_error(tisco_actual_open_prices, tisco_forecast_open)
tisco_high_mape = mean_absolute_percentage_error(tisco_actual_high_prices, tisco_forecast_high)
tisco_low_mape = mean_absolute_percentage_error(tisco_actual_low_prices, tisco_forecast_low)

print("Close Forecasts:", tisco_forecast_close)
print(f"Close Mean Absolute Error (MAE): {tisco_close_mae}")
print(f"Close Mean Squared Error (MSE): {tisco_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {tisco_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {tisco_close_mape}%")

print("Open Forecasts:", tisco_forecast_open)
print(f"Open Mean Absolute Error (MAE): {tisco_open_mae}")
print(f"Open Mean Squared Error (MSE): {tisco_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {tisco_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {tisco_open_mape}%")

print("High Forecasts:", tisco_forecast_high)
print(f"High Mean Absolute Error (MAE): {tisco_high_mae}")
print(f"High Mean Squared Error (MSE): {tisco_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {tisco_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {tisco_high_mape}%")

print("Low Forecasts:", tisco_forecast_low)
print(f"Low Mean Absolute Error (MAE): {tisco_low_mae}")
print(f"Low Mean Squared Error (MSE): {tisco_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {tisco_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {tisco_low_mape}")


# In[241]:


tatasteel_y_close = tatasteel['Close'].values
tatasteel_y_open = tatasteel['Open'].values
tatasteel_y_high = tatasteel['High'].values
tatasteel_y_low = tatasteel['Low'].values

tatasteel_y_close_scaled = scaler.fit_transform(tatasteel_y_close.reshape(-1, 1))
tatasteel_y_open_scaled = scaler.fit_transform(tatasteel_y_open.reshape(-1, 1))
tatasteel_y_high_scaled = scaler.fit_transform(tatasteel_y_high.reshape(-1, 1))
tatasteel_y_low_scaled = scaler.fit_transform(tatasteel_y_low.reshape(-1, 1))

tatasteel_close_model = auto_arima(
    tatasteel_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tatasteel_open_model = auto_arima(
    tatasteel_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tatasteel_high_model = auto_arima(
    tatasteel_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tatasteel_low_model = auto_arima(
    tatasteel_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tatasteel_close_best_order = tatasteel_close_model.get_params()['order']
tatasteel_open_best_order = tatasteel_open_model.get_params()['order']
tatasteel_high_best_order = tatasteel_high_model.get_params()['order']
tatasteel_low_best_order = tatasteel_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {tatasteel_close_best_order}")
print(f"Best ARIMA Order for Open: {tatasteel_open_best_order}")
print(f"Best ARIMA Order for High: {tatasteel_high_best_order}")
print(f"Best ARIMA Order for Low: {tatasteel_low_best_order}")

tatasteel_close_final_model = sm.tsa.ARIMA(
    tatasteel_y_close_scaled,
    order=tatasteel_close_best_order
)
tatasteel_close_final_model = tatasteel_close_final_model.fit()
tatasteel_close_forecast = tatasteel_close_final_model.forecast(steps=forecast_period)
tatasteel_close_forecast = tatasteel_close_forecast.reshape(-1, 1)
tatasteel_close_forecast = scaler.inverse_transform(tatasteel_close_forecast)

tatasteel_open_final_model = sm.tsa.ARIMA(
    tatasteel_y_open_scaled,
    order=tatasteel_open_best_order
)
tatasteel_open_final_model = tatasteel_open_final_model.fit()
tatasteel_open_forecast = tatasteel_open_final_model.forecast(steps=forecast_period)
tatasteel_open_forecast = tatasteel_open_forecast.reshape(-1, 1)
tatasteel_open_forecast = scaler.inverse_transform(tatasteel_open_forecast)

tatasteel_high_final_model = sm.tsa.ARIMA(
    tatasteel_y_high_scaled,
    order=tatasteel_high_best_order
)
tatasteel_high_final_model = tatasteel_high_final_model.fit()
tatasteel_high_forecast = tatasteel_high_final_model.forecast(steps=forecast_period)
tatasteel_high_forecast = tatasteel_high_forecast.reshape(-1, 1)
tatasteel_high_forecast = scaler.inverse_transform(tatasteel_high_forecast)

tatasteel_low_final_model = sm.tsa.ARIMA(
    tatasteel_y_low_scaled,
    order=tatasteel_low_best_order
)
tatasteel_low_final_model = tatasteel_low_final_model.fit()
tatasteel_low_forecast = tatasteel_low_final_model.forecast(steps=forecast_period)
tatasteel_low_forecast = tatasteel_low_forecast.reshape(-1, 1)
tatasteel_low_forecast = scaler.inverse_transform(tatasteel_low_forecast)

print("Close Forecasts:", tatasteel_close_forecast)
print("Open Forecasts:", tatasteel_open_forecast)
print("High Forecasts:", tatasteel_high_forecast)
print("Low Forecasts:", tatasteel_low_forecast)


# In[242]:


tatasteel_tail_50_data = tatasteel.tail(forecast_periods)

tatasteel_actual_close_prices = tatasteel_tail_50_data['Close'].values
tatasteel_actual_open_prices = tatasteel_tail_50_data['Open'].values
tatasteel_actual_high_prices = tatasteel_tail_50_data['High'].values
tatasteel_actual_low_prices = tatasteel_tail_50_data['Low'].values

tatasteel_forecast_close = tatasteel_close_final_model.forecast(steps=forecast_periods)
tatasteel_forecast_close = tatasteel_forecast_close.reshape(-1, 1)
tatasteel_forecast_close = scaler.inverse_transform(tatasteel_forecast_close)

tatasteel_forecast_open = tatasteel_open_final_model.forecast(steps=forecast_periods)
tatasteel_forecast_open = tatasteel_forecast_open.reshape(-1, 1)
tatasteel_forecast_open = scaler.inverse_transform(tatasteel_forecast_open)

tatasteel_forecast_high = tatasteel_high_final_model.forecast(steps=forecast_periods)
tatasteel_forecast_high = tatasteel_forecast_high.reshape(-1, 1)
tatasteel_forecast_high = scaler.inverse_transform(tatasteel_forecast_high)

tatasteel_forecast_low = tatasteel_low_final_model.forecast(steps=forecast_periods)
tatasteel_forecast_low = tatasteel_forecast_low.reshape(-1, 1)
tatasteel_forecast_low = scaler.inverse_transform(tatasteel_forecast_low)

tatasteel_close_mae = mean_absolute_error(tatasteel_actual_close_prices, tatasteel_forecast_close)
tatasteel_close_mse = mean_squared_error(tatasteel_actual_close_prices, tatasteel_forecast_close)
tatasteel_close_rmse = np.sqrt(tatasteel_close_mse)

tatasteel_open_mae = mean_absolute_error(tatasteel_actual_open_prices, tatasteel_forecast_open)
tatasteel_open_mse = mean_squared_error(tatasteel_actual_open_prices, tatasteel_forecast_open)
tatasteel_open_rmse = np.sqrt(tatasteel_open_mse)

tatasteel_high_mae = mean_absolute_error(tatasteel_actual_high_prices, tatasteel_forecast_high)
tatasteel_high_mse = mean_squared_error(tatasteel_actual_high_prices, tatasteel_forecast_high)
tatasteel_high_rmse = np.sqrt(tatasteel_high_mse)

tatasteel_low_mae = mean_absolute_error(tatasteel_actual_low_prices, tatasteel_forecast_low)
tatasteel_low_mse = mean_squared_error(tatasteel_actual_low_prices, tatasteel_forecast_low)
tatasteel_low_rmse = np.sqrt(tatasteel_low_mse)

tatasteel_close_mape = mean_absolute_percentage_error(tatasteel_actual_close_prices, tatasteel_forecast_close)
tatasteel_open_mape = mean_absolute_percentage_error(tatasteel_actual_open_prices, tatasteel_forecast_open)
tatasteel_high_mape = mean_absolute_percentage_error(tatasteel_actual_high_prices, tatasteel_forecast_high)
tatasteel_low_mape = mean_absolute_percentage_error(tatasteel_actual_low_prices, tatasteel_forecast_low)

print("Close Forecasts:", tatasteel_forecast_close)
print(f"Close Mean Absolute Error (MAE): {tatasteel_close_mae}")
print(f"Close Mean Squared Error (MSE): {tatasteel_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {tatasteel_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {tatasteel_close_mape}%")

print("Open Forecasts:", tatasteel_forecast_open)
print(f"Open Mean Absolute Error (MAE): {tatasteel_open_mae}")
print(f"Open Mean Squared Error (MSE): {tatasteel_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {tatasteel_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {tatasteel_open_mape}%")

print("High Forecasts:", tatasteel_forecast_high)
print(f"High Mean Absolute Error (MAE): {tatasteel_high_mae}")
print(f"High Mean Squared Error (MSE): {tatasteel_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {tatasteel_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {tatasteel_high_mape}%")

print("Low Forecasts:", tatasteel_forecast_low)
print(f"Low Mean Absolute Error (MAE): {tatasteel_low_mae}")
print(f"Low Mean Squared Error (MSE): {tatasteel_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {tatasteel_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {tatasteel_low_mape}")


# In[243]:


tcs_y_close = tcs['Close'].values
tcs_y_open = tcs['Open'].values
tcs_y_high = tcs['High'].values
tcs_y_low = tcs['Low'].values

tcs_y_close_scaled = scaler.fit_transform(tcs_y_close.reshape(-1, 1))
tcs_y_open_scaled = scaler.fit_transform(tcs_y_open.reshape(-1, 1))
tcs_y_high_scaled = scaler.fit_transform(tcs_y_high.reshape(-1, 1))
tcs_y_low_scaled = scaler.fit_transform(tcs_y_low.reshape(-1, 1))

tcs_close_model = auto_arima(
    tcs_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tcs_open_model = auto_arima(
    tcs_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tcs_high_model = auto_arima(
    tcs_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tcs_low_model = auto_arima(
    tcs_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

tcs_close_best_order = tcs_close_model.get_params()['order']
tcs_open_best_order = tcs_open_model.get_params()['order']
tcs_high_best_order = tcs_high_model.get_params()['order']
tcs_low_best_order = tcs_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {tcs_close_best_order}")
print(f"Best ARIMA Order for Open: {tcs_open_best_order}")
print(f"Best ARIMA Order for High: {tcs_high_best_order}")
print(f"Best ARIMA Order for Low: {tcs_low_best_order}")

tcs_close_final_model = sm.tsa.ARIMA(
    tcs_y_close_scaled,
    order=tcs_close_best_order
)
tcs_close_final_model = tcs_close_final_model.fit()
tcs_close_forecast = tcs_close_final_model.forecast(steps=forecast_period)
tcs_close_forecast = tcs_close_forecast.reshape(-1, 1)
tcs_close_forecast = scaler.inverse_transform(tcs_close_forecast)

tcs_open_final_model = sm.tsa.ARIMA(
    tcs_y_open_scaled,
    order=tcs_open_best_order
)
tcs_open_final_model = tcs_open_final_model.fit()
tcs_open_forecast = tcs_open_final_model.forecast(steps=forecast_period)
tcs_open_forecast = tcs_open_forecast.reshape(-1, 1)
tcs_open_forecast = scaler.inverse_transform(tcs_open_forecast)

tcs_high_final_model = sm.tsa.ARIMA(
    tcs_y_high_scaled,
    order=tcs_high_best_order
)
tcs_high_final_model = tcs_high_final_model.fit()
tcs_high_forecast = tcs_high_final_model.forecast(steps=forecast_period)
tcs_high_forecast = tcs_high_forecast.reshape(-1, 1)
tcs_high_forecast = scaler.inverse_transform(tcs_high_forecast)

tcs_low_final_model = sm.tsa.ARIMA(
    tcs_y_low_scaled,
    order=tcs_low_best_order
)
tcs_low_final_model = tcs_low_final_model.fit()
tcs_low_forecast = tcs_low_final_model.forecast(steps=forecast_period)
tcs_low_forecast = tcs_low_forecast.reshape(-1, 1)
tcs_low_forecast = scaler.inverse_transform(tcs_low_forecast)

print("Close Forecasts:", tcs_close_forecast)
print("Open Forecasts:", tcs_open_forecast)
print("High Forecasts:", tcs_high_forecast)
print("Low Forecasts:", tcs_low_forecast)


# In[244]:


tcs_tail_50_data = tcs.tail(forecast_periods)

tcs_actual_close_prices = tcs_tail_50_data['Close'].values
tcs_actual_open_prices = tcs_tail_50_data['Open'].values
tcs_actual_high_prices = tcs_tail_50_data['High'].values
tcs_actual_low_prices = tcs_tail_50_data['Low'].values

tcs_forecast_close = tcs_close_final_model.forecast(steps=forecast_periods)
tcs_forecast_close = tcs_forecast_close.reshape(-1, 1)
tcs_forecast_close = scaler.inverse_transform(tcs_forecast_close)

tcs_forecast_open = tcs_open_final_model.forecast(steps=forecast_periods)
tcs_forecast_open = tcs_forecast_open.reshape(-1, 1)
tcs_forecast_open = scaler.inverse_transform(tcs_forecast_open)

tcs_forecast_high = tcs_high_final_model.forecast(steps=forecast_periods)
tcs_forecast_high = tcs_forecast_high.reshape(-1, 1)
tcs_forecast_high = scaler.inverse_transform(tcs_forecast_high)

tcs_forecast_low = tcs_low_final_model.forecast(steps=forecast_periods)
tcs_forecast_low = tcs_forecast_low.reshape(-1, 1)
tcs_forecast_low = scaler.inverse_transform(tcs_forecast_low)

tcs_close_mae = mean_absolute_error(tcs_actual_close_prices, tcs_forecast_close)
tcs_close_mse = mean_squared_error(tcs_actual_close_prices, tcs_forecast_close)
tcs_close_rmse = np.sqrt(tcs_close_mse)

tcs_open_mae = mean_absolute_error(tcs_actual_open_prices, tcs_forecast_open)
tcs_open_mse = mean_squared_error(tcs_actual_open_prices, tcs_forecast_open)
tcs_open_rmse = np.sqrt(tcs_open_mse)

tcs_high_mae = mean_absolute_error(tcs_actual_high_prices, tcs_forecast_high)
tcs_high_mse = mean_squared_error(tcs_actual_high_prices, tcs_forecast_high)
tcs_high_rmse = np.sqrt(tcs_high_mse)

tcs_low_mae = mean_absolute_error(tcs_actual_low_prices, tcs_forecast_low)
tcs_low_mse = mean_squared_error(tcs_actual_low_prices, tcs_forecast_low)
tcs_low_rmse = np.sqrt(tcs_low_mse)

tcs_close_mape = mean_absolute_percentage_error(tcs_actual_close_prices, tcs_forecast_close)
tcs_open_mape = mean_absolute_percentage_error(tcs_actual_open_prices, tcs_forecast_open)
tcs_high_mape = mean_absolute_percentage_error(tcs_actual_high_prices, tcs_forecast_high)
tcs_low_mape = mean_absolute_percentage_error(tcs_actual_low_prices, tcs_forecast_low)

print("Close Forecasts:", tcs_forecast_close)
print(f"Close Mean Absolute Error (MAE): {tcs_close_mae}")
print(f"Close Mean Squared Error (MSE): {tcs_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {tcs_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {tcs_close_mape}%")

print("Open Forecasts:", tcs_forecast_open)
print(f"Open Mean Absolute Error (MAE): {tcs_open_mae}")
print(f"Open Mean Squared Error (MSE): {tcs_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {tcs_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {tcs_open_mape}%")

print("High Forecasts:", tcs_forecast_high)
print(f"High Mean Absolute Error (MAE): {tcs_high_mae}")
print(f"High Mean Squared Error (MSE): {tcs_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {tcs_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {tcs_high_mape}%")

print("Low Forecasts:", tcs_forecast_low)
print(f"Low Mean Absolute Error (MAE): {tcs_low_mae}")
print(f"Low Mean Squared Error (MSE): {tcs_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {tcs_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {tcs_low_mape}")


# In[245]:


techm_y_close = techm['Close'].values
techm_y_open = techm['Open'].values
techm_y_high = techm['High'].values
techm_y_low = techm['Low'].values

techm_y_close_scaled = scaler.fit_transform(techm_y_close.reshape(-1, 1))
techm_y_open_scaled = scaler.fit_transform(techm_y_open.reshape(-1, 1))
techm_y_high_scaled = scaler.fit_transform(techm_y_high.reshape(-1, 1))
techm_y_low_scaled = scaler.fit_transform(techm_y_low.reshape(-1, 1))

techm_close_model = auto_arima(
    techm_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

techm_open_model = auto_arima(
    techm_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

techm_high_model = auto_arima(
    techm_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

techm_low_model = auto_arima(
    techm_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

techm_close_best_order = techm_close_model.get_params()['order']
techm_open_best_order = techm_open_model.get_params()['order']
techm_high_best_order = techm_high_model.get_params()['order']
techm_low_best_order = techm_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {techm_close_best_order}")
print(f"Best ARIMA Order for Open: {techm_open_best_order}")
print(f"Best ARIMA Order for High: {techm_high_best_order}")
print(f"Best ARIMA Order for Low: {techm_low_best_order}")

techm_close_final_model = sm.tsa.ARIMA(
    techm_y_close_scaled,
    order=techm_close_best_order
)
techm_close_final_model = techm_close_final_model.fit()
techm_close_forecast = techm_close_final_model.forecast(steps=forecast_period)
techm_close_forecast = techm_close_forecast.reshape(-1, 1)
techm_close_forecast = scaler.inverse_transform(techm_close_forecast)

techm_open_final_model = sm.tsa.ARIMA(
    techm_y_open_scaled,
    order=techm_open_best_order
)
techm_open_final_model = techm_open_final_model.fit()
techm_open_forecast = techm_open_final_model.forecast(steps=forecast_period)
techm_open_forecast = techm_open_forecast.reshape(-1, 1)
techm_open_forecast = scaler.inverse_transform(techm_open_forecast)

techm_high_final_model = sm.tsa.ARIMA(
    techm_y_high_scaled,
    order=techm_high_best_order
)
techm_high_final_model = techm_high_final_model.fit()
techm_high_forecast = techm_high_final_model.forecast(steps=forecast_period)
techm_high_forecast = techm_high_forecast.reshape(-1, 1)
techm_high_forecast = scaler.inverse_transform(techm_high_forecast)

techm_low_final_model = sm.tsa.ARIMA(
    techm_y_low_scaled,
    order=techm_low_best_order
)
techm_low_final_model = techm_low_final_model.fit()
techm_low_forecast = techm_low_final_model.forecast(steps=forecast_period)
techm_low_forecast = techm_low_forecast.reshape(-1, 1)
techm_low_forecast = scaler.inverse_transform(techm_low_forecast)

print("Close Forecasts:", techm_close_forecast)
print("Open Forecasts:", techm_open_forecast)
print("High Forecasts:", techm_high_forecast)
print("Low Forecasts:", techm_low_forecast)


# In[246]:


techm_tail_50_data = techm.tail(forecast_periods)

techm_actual_close_prices = techm_tail_50_data['Close'].values
techm_actual_open_prices = techm_tail_50_data['Open'].values
techm_actual_high_prices = techm_tail_50_data['High'].values
techm_actual_low_prices = techm_tail_50_data['Low'].values

techm_forecast_close = techm_close_final_model.forecast(steps=forecast_periods)
techm_forecast_close = techm_forecast_close.reshape(-1, 1)
techm_forecast_close = scaler.inverse_transform(techm_forecast_close)

techm_forecast_open = techm_open_final_model.forecast(steps=forecast_periods)
techm_forecast_open = techm_forecast_open.reshape(-1, 1)
techm_forecast_open = scaler.inverse_transform(techm_forecast_open)

techm_forecast_high = techm_high_final_model.forecast(steps=forecast_periods)
techm_forecast_high = techm_forecast_high.reshape(-1, 1)
techm_forecast_high = scaler.inverse_transform(techm_forecast_high)

techm_forecast_low = techm_low_final_model.forecast(steps=forecast_periods)
techm_forecast_low = techm_forecast_low.reshape(-1, 1)
techm_forecast_low = scaler.inverse_transform(techm_forecast_low)

techm_close_mae = mean_absolute_error(techm_actual_close_prices, techm_forecast_close)
techm_close_mse = mean_squared_error(techm_actual_close_prices, techm_forecast_close)
techm_close_rmse = np.sqrt(techm_close_mse)

techm_open_mae = mean_absolute_error(techm_actual_open_prices, techm_forecast_open)
techm_open_mse = mean_squared_error(techm_actual_open_prices, techm_forecast_open)
techm_open_rmse = np.sqrt(techm_open_mse)

techm_high_mae = mean_absolute_error(techm_actual_high_prices, techm_forecast_high)
techm_high_mse = mean_squared_error(techm_actual_high_prices, techm_forecast_high)
techm_high_rmse = np.sqrt(techm_high_mse)

techm_low_mae = mean_absolute_error(techm_actual_low_prices, techm_forecast_low)
techm_low_mse = mean_squared_error(techm_actual_low_prices, techm_forecast_low)
techm_low_rmse = np.sqrt(techm_low_mse)

techm_close_mape = mean_absolute_percentage_error(techm_actual_close_prices, techm_forecast_close)
techm_open_mape = mean_absolute_percentage_error(techm_actual_open_prices, techm_forecast_open)
techm_high_mape = mean_absolute_percentage_error(techm_actual_high_prices, techm_forecast_high)
techm_low_mape = mean_absolute_percentage_error(techm_actual_low_prices, techm_forecast_low)

print("Close Forecasts:", techm_forecast_close)
print(f"Close Mean Absolute Error (MAE): {techm_close_mae}")
print(f"Close Mean Squared Error (MSE): {techm_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {techm_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {techm_close_mape}%")

print("Open Forecasts:", techm_forecast_open)
print(f"Open Mean Absolute Error (MAE): {techm_open_mae}")
print(f"Open Mean Squared Error (MSE): {techm_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {techm_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {techm_open_mape}%")

print("High Forecasts:", techm_forecast_high)
print(f"High Mean Absolute Error (MAE): {techm_high_mae}")
print(f"High Mean Squared Error (MSE): {techm_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {techm_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {techm_high_mape}%")

print("Low Forecasts:", techm_forecast_low)
print(f"Low Mean Absolute Error (MAE): {techm_low_mae}")
print(f"Low Mean Squared Error (MSE): {techm_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {techm_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {techm_low_mape}")


# In[247]:


titan_y_close = titan['Close'].values
titan_y_open = titan['Open'].values
titan_y_high = titan['High'].values
titan_y_low = titan['Low'].values

titan_y_close_scaled = scaler.fit_transform(titan_y_close.reshape(-1, 1))
titan_y_open_scaled = scaler.fit_transform(titan_y_open.reshape(-1, 1))
titan_y_high_scaled = scaler.fit_transform(titan_y_high.reshape(-1, 1))
titan_y_low_scaled = scaler.fit_transform(titan_y_low.reshape(-1, 1))

titan_close_model = auto_arima(
    titan_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

titan_open_model = auto_arima(
    titan_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

titan_high_model = auto_arima(
    titan_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

titan_low_model = auto_arima(
    titan_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

titan_close_best_order = titan_close_model.get_params()['order']
titan_open_best_order = titan_open_model.get_params()['order']
titan_high_best_order = titan_high_model.get_params()['order']
titan_low_best_order = titan_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {titan_close_best_order}")
print(f"Best ARIMA Order for Open: {titan_open_best_order}")
print(f"Best ARIMA Order for High: {titan_high_best_order}")
print(f"Best ARIMA Order for Low: {titan_low_best_order}")

titan_close_final_model = sm.tsa.ARIMA(
    titan_y_close_scaled,
    order=titan_close_best_order
)
titan_close_final_model = titan_close_final_model.fit()
titan_close_forecast = titan_close_final_model.forecast(steps=forecast_period)
titan_close_forecast = titan_close_forecast.reshape(-1, 1)
titan_close_forecast = scaler.inverse_transform(titan_close_forecast)

titan_open_final_model = sm.tsa.ARIMA(
    titan_y_open_scaled,
    order=titan_open_best_order
)
titan_open_final_model = titan_open_final_model.fit()
titan_open_forecast = titan_open_final_model.forecast(steps=forecast_period)
titan_open_forecast = titan_open_forecast.reshape(-1, 1)
titan_open_forecast = scaler.inverse_transform(titan_open_forecast)

titan_high_final_model = sm.tsa.ARIMA(
    titan_y_high_scaled,
    order=titan_high_best_order
)
titan_high_final_model = titan_high_final_model.fit()
titan_high_forecast = titan_high_final_model.forecast(steps=forecast_period)
titan_high_forecast = titan_high_forecast.reshape(-1, 1)
titan_high_forecast = scaler.inverse_transform(titan_high_forecast)

titan_low_final_model = sm.tsa.ARIMA(
    titan_y_low_scaled,
    order=titan_low_best_order
)
titan_low_final_model = titan_low_final_model.fit()
titan_low_forecast = titan_low_final_model.forecast(steps=forecast_period)
titan_low_forecast = titan_low_forecast.reshape(-1, 1)
titan_low_forecast = scaler.inverse_transform(titan_low_forecast)

print("Close Forecasts:", titan_close_forecast)
print("Open Forecasts:", titan_open_forecast)
print("High Forecasts:", titan_high_forecast)
print("Low Forecasts:", titan_low_forecast)


# In[248]:


titan_tail_50_data = titan.tail(forecast_periods)

titan_actual_close_prices = titan_tail_50_data['Close'].values
titan_actual_open_prices = titan_tail_50_data['Open'].values
titan_actual_high_prices = titan_tail_50_data['High'].values
titan_actual_low_prices = titan_tail_50_data['Low'].values

titan_forecast_close = titan_close_final_model.forecast(steps=forecast_periods)
titan_forecast_close = titan_forecast_close.reshape(-1, 1)
titan_forecast_close = scaler.inverse_transform(titan_forecast_close)

titan_forecast_open = titan_open_final_model.forecast(steps=forecast_periods)
titan_forecast_open = titan_forecast_open.reshape(-1, 1)
titan_forecast_open = scaler.inverse_transform(titan_forecast_open)

titan_forecast_high = titan_high_final_model.forecast(steps=forecast_periods)
titan_forecast_high = titan_forecast_high.reshape(-1, 1)
titan_forecast_high = scaler.inverse_transform(titan_forecast_high)

titan_forecast_low = titan_low_final_model.forecast(steps=forecast_periods)
titan_forecast_low = titan_forecast_low.reshape(-1, 1)
titan_forecast_low = scaler.inverse_transform(titan_forecast_low)

titan_close_mae = mean_absolute_error(titan_actual_close_prices, titan_forecast_close)
titan_close_mse = mean_squared_error(titan_actual_close_prices, titan_forecast_close)
titan_close_rmse = np.sqrt(titan_close_mse)

titan_open_mae = mean_absolute_error(titan_actual_open_prices, titan_forecast_open)
titan_open_mse = mean_squared_error(titan_actual_open_prices, titan_forecast_open)
titan_open_rmse = np.sqrt(titan_open_mse)

titan_high_mae = mean_absolute_error(titan_actual_high_prices, titan_forecast_high)
titan_high_mse = mean_squared_error(titan_actual_high_prices, titan_forecast_high)
titan_high_rmse = np.sqrt(titan_high_mse)

titan_low_mae = mean_absolute_error(titan_actual_low_prices, titan_forecast_low)
titan_low_mse = mean_squared_error(titan_actual_low_prices, titan_forecast_low)
titan_low_rmse = np.sqrt(titan_low_mse)

titan_close_mape = mean_absolute_percentage_error(titan_actual_close_prices, titan_forecast_close)
titan_open_mape = mean_absolute_percentage_error(titan_actual_open_prices, titan_forecast_open)
titan_high_mape = mean_absolute_percentage_error(titan_actual_high_prices, titan_forecast_high)
titan_low_mape = mean_absolute_percentage_error(titan_actual_low_prices, titan_forecast_low)

print("Close Forecasts:", titan_forecast_close)
print(f"Close Mean Absolute Error (MAE): {titan_close_mae}")
print(f"Close Mean Squared Error (MSE): {titan_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {titan_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {titan_close_mape}%")

print("Open Forecasts:", titan_forecast_open)
print(f"Open Mean Absolute Error (MAE): {titan_open_mae}")
print(f"Open Mean Squared Error (MSE): {titan_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {titan_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {titan_open_mape}%")

print("High Forecasts:", titan_forecast_high)
print(f"High Mean Absolute Error (MAE): {titan_high_mae}")
print(f"High Mean Squared Error (MSE): {titan_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {titan_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {titan_high_mape}%")

print("Low Forecasts:", titan_forecast_low)
print(f"Low Mean Absolute Error (MAE): {titan_low_mae}")
print(f"Low Mean Squared Error (MSE): {titan_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {titan_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {titan_low_mape}")


# In[249]:


ultracemco_y_close = ultracemco['Close'].values
ultracemco_y_open = ultracemco['Open'].values
ultracemco_y_high = ultracemco['High'].values
ultracemco_y_low = ultracemco['Low'].values

ultracemco_y_close_scaled = scaler.fit_transform(ultracemco_y_close.reshape(-1, 1))
ultracemco_y_open_scaled = scaler.fit_transform(ultracemco_y_open.reshape(-1, 1))
ultracemco_y_high_scaled = scaler.fit_transform(ultracemco_y_high.reshape(-1, 1))
ultracemco_y_low_scaled = scaler.fit_transform(ultracemco_y_low.reshape(-1, 1))

ultracemco_close_model = auto_arima(
    ultracemco_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ultracemco_open_model = auto_arima(
    ultracemco_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ultracemco_high_model = auto_arima(
    ultracemco_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ultracemco_low_model = auto_arima(
    ultracemco_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

ultracemco_close_best_order = ultracemco_close_model.get_params()['order']
ultracemco_open_best_order = ultracemco_open_model.get_params()['order']
ultracemco_high_best_order = ultracemco_high_model.get_params()['order']
ultracemco_low_best_order = ultracemco_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {ultracemco_close_best_order}")
print(f"Best ARIMA Order for Open: {ultracemco_open_best_order}")
print(f"Best ARIMA Order for High: {ultracemco_high_best_order}")
print(f"Best ARIMA Order for Low: {ultracemco_low_best_order}")

ultracemco_close_final_model = sm.tsa.ARIMA(
    ultracemco_y_close_scaled,
    order=ultracemco_close_best_order
)
ultracemco_close_final_model = ultracemco_close_final_model.fit()
ultracemco_close_forecast = ultracemco_close_final_model.forecast(steps=forecast_period)
ultracemco_close_forecast = ultracemco_close_forecast.reshape(-1, 1)
ultracemco_close_forecast = scaler.inverse_transform(ultracemco_close_forecast)

ultracemco_open_final_model = sm.tsa.ARIMA(
    ultracemco_y_open_scaled,
    order=ultracemco_open_best_order
)
ultracemco_open_final_model = ultracemco_open_final_model.fit()
ultracemco_open_forecast = ultracemco_open_final_model.forecast(steps=forecast_period)
ultracemco_open_forecast = ultracemco_open_forecast.reshape(-1, 1)
ultracemco_open_forecast = scaler.inverse_transform(ultracemco_open_forecast)

ultracemco_high_final_model = sm.tsa.ARIMA(
    ultracemco_y_high_scaled,
    order=ultracemco_high_best_order
)
ultracemco_high_final_model = ultracemco_high_final_model.fit()
ultracemco_high_forecast = ultracemco_high_final_model.forecast(steps=forecast_period)
ultracemco_high_forecast = ultracemco_high_forecast.reshape(-1, 1)
ultracemco_high_forecast = scaler.inverse_transform(ultracemco_high_forecast)

ultracemco_low_final_model = sm.tsa.ARIMA(
    ultracemco_y_low_scaled,
    order=ultracemco_low_best_order
)
ultracemco_low_final_model = ultracemco_low_final_model.fit()
ultracemco_low_forecast = ultracemco_low_final_model.forecast(steps=forecast_period)
ultracemco_low_forecast = ultracemco_low_forecast.reshape(-1, 1)
ultracemco_low_forecast = scaler.inverse_transform(ultracemco_low_forecast)

print("Close Forecasts:", ultracemco_close_forecast)
print("Open Forecasts:", ultracemco_open_forecast)
print("High Forecasts:", ultracemco_high_forecast)
print("Low Forecasts:", ultracemco_low_forecast)


# In[250]:


ultracemco_tail_50_data = ultracemco.tail(forecast_periods)

ultracemco_actual_close_prices = ultracemco_tail_50_data['Close'].values
ultracemco_actual_open_prices = ultracemco_tail_50_data['Open'].values
ultracemco_actual_high_prices = ultracemco_tail_50_data['High'].values
ultracemco_actual_low_prices = ultracemco_tail_50_data['Low'].values

ultracemco_forecast_close = ultracemco_close_final_model.forecast(steps=forecast_periods)
ultracemco_forecast_close = ultracemco_forecast_close.reshape(-1, 1)
ultracemco_forecast_close = scaler.inverse_transform(ultracemco_forecast_close)

ultracemco_forecast_open = ultracemco_open_final_model.forecast(steps=forecast_periods)
ultracemco_forecast_open = ultracemco_forecast_open.reshape(-1, 1)
ultracemco_forecast_open = scaler.inverse_transform(ultracemco_forecast_open)

ultracemco_forecast_high = ultracemco_high_final_model.forecast(steps=forecast_periods)
ultracemco_forecast_high = ultracemco_forecast_high.reshape(-1, 1)
ultracemco_forecast_high = scaler.inverse_transform(ultracemco_forecast_high)

ultracemco_forecast_low = ultracemco_low_final_model.forecast(steps=forecast_periods)
ultracemco_forecast_low = ultracemco_forecast_low.reshape(-1, 1)
ultracemco_forecast_low = scaler.inverse_transform(ultracemco_forecast_low)

ultracemco_close_mae = mean_absolute_error(ultracemco_actual_close_prices, ultracemco_forecast_close)
ultracemco_close_mse = mean_squared_error(ultracemco_actual_close_prices, ultracemco_forecast_close)
ultracemco_close_rmse = np.sqrt(ultracemco_close_mse)

ultracemco_open_mae = mean_absolute_error(ultracemco_actual_open_prices, ultracemco_forecast_open)
ultracemco_open_mse = mean_squared_error(ultracemco_actual_open_prices, ultracemco_forecast_open)
ultracemco_open_rmse = np.sqrt(ultracemco_open_mse)

ultracemco_high_mae = mean_absolute_error(ultracemco_actual_high_prices, ultracemco_forecast_high)
ultracemco_high_mse = mean_squared_error(ultracemco_actual_high_prices, ultracemco_forecast_high)
ultracemco_high_rmse = np.sqrt(ultracemco_high_mse)

ultracemco_low_mae = mean_absolute_error(ultracemco_actual_low_prices, ultracemco_forecast_low)
ultracemco_low_mse = mean_squared_error(ultracemco_actual_low_prices, ultracemco_forecast_low)
ultracemco_low_rmse = np.sqrt(ultracemco_low_mse)

ultracemco_close_mape = mean_absolute_percentage_error(ultracemco_actual_close_prices, ultracemco_forecast_close)
ultracemco_open_mape = mean_absolute_percentage_error(ultracemco_actual_open_prices, ultracemco_forecast_open)
ultracemco_high_mape = mean_absolute_percentage_error(ultracemco_actual_high_prices, ultracemco_forecast_high)
ultracemco_low_mape = mean_absolute_percentage_error(ultracemco_actual_low_prices, ultracemco_forecast_low)

print("Close Forecasts:", ultracemco_forecast_close)
print(f"Close Mean Absolute Error (MAE): {ultracemco_close_mae}")
print(f"Close Mean Squared Error (MSE): {ultracemco_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {ultracemco_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {ultracemco_close_mape}%")

print("Open Forecasts:", ultracemco_forecast_open)
print(f"Open Mean Absolute Error (MAE): {ultracemco_open_mae}")
print(f"Open Mean Squared Error (MSE): {ultracemco_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {ultracemco_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {ultracemco_open_mape}%")

print("High Forecasts:", ultracemco_forecast_high)
print(f"High Mean Absolute Error (MAE): {ultracemco_high_mae}")
print(f"High Mean Squared Error (MSE): {ultracemco_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {ultracemco_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {ultracemco_high_mape}%")

print("Low Forecasts:", ultracemco_forecast_low)
print(f"Low Mean Absolute Error (MAE): {ultracemco_low_mae}")
print(f"Low Mean Squared Error (MSE): {ultracemco_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {ultracemco_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {ultracemco_low_mape}")


# In[251]:


uniphos_y_close = uniphos['Close'].values
uniphos_y_open = uniphos['Open'].values
uniphos_y_high = uniphos['High'].values
uniphos_y_low = uniphos['Low'].values

uniphos_y_close_scaled = scaler.fit_transform(uniphos_y_close.reshape(-1, 1))
uniphos_y_open_scaled = scaler.fit_transform(uniphos_y_open.reshape(-1, 1))
uniphos_y_high_scaled = scaler.fit_transform(uniphos_y_high.reshape(-1, 1))
uniphos_y_low_scaled = scaler.fit_transform(uniphos_y_low.reshape(-1, 1))

uniphos_close_model = auto_arima(
    uniphos_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

uniphos_open_model = auto_arima(
    uniphos_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

uniphos_high_model = auto_arima(
    uniphos_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

uniphos_low_model = auto_arima(
    uniphos_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

uniphos_close_best_order = uniphos_close_model.get_params()['order']
uniphos_open_best_order = uniphos_open_model.get_params()['order']
uniphos_high_best_order = uniphos_high_model.get_params()['order']
uniphos_low_best_order = uniphos_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {uniphos_close_best_order}")
print(f"Best ARIMA Order for Open: {uniphos_open_best_order}")
print(f"Best ARIMA Order for High: {uniphos_high_best_order}")
print(f"Best ARIMA Order for Low: {uniphos_low_best_order}")

uniphos_close_final_model = sm.tsa.ARIMA(
    uniphos_y_close_scaled,
    order=uniphos_close_best_order
)
uniphos_close_final_model = uniphos_close_final_model.fit()
uniphos_close_forecast = uniphos_close_final_model.forecast(steps=forecast_period)
uniphos_close_forecast = uniphos_close_forecast.reshape(-1, 1)
uniphos_close_forecast = scaler.inverse_transform(uniphos_close_forecast)

uniphos_open_final_model = sm.tsa.ARIMA(
    uniphos_y_open_scaled,
    order=uniphos_open_best_order
)
uniphos_open_final_model = uniphos_open_final_model.fit()
uniphos_open_forecast = uniphos_open_final_model.forecast(steps=forecast_period)
uniphos_open_forecast = uniphos_open_forecast.reshape(-1, 1)
uniphos_open_forecast = scaler.inverse_transform(uniphos_open_forecast)

uniphos_high_final_model = sm.tsa.ARIMA(
    uniphos_y_high_scaled,
    order=uniphos_high_best_order
)
uniphos_high_final_model = uniphos_high_final_model.fit()
uniphos_high_forecast = uniphos_high_final_model.forecast(steps=forecast_period)
uniphos_high_forecast = uniphos_high_forecast.reshape(-1, 1)
uniphos_high_forecast = scaler.inverse_transform(uniphos_high_forecast)

uniphos_low_final_model = sm.tsa.ARIMA(
    uniphos_y_low_scaled,
    order=uniphos_low_best_order
)
uniphos_low_final_model = uniphos_low_final_model.fit()
uniphos_low_forecast = uniphos_low_final_model.forecast(steps=forecast_period)
uniphos_low_forecast = uniphos_low_forecast.reshape(-1, 1)
uniphos_low_forecast = scaler.inverse_transform(uniphos_low_forecast)

print("Close Forecasts:", uniphos_close_forecast)
print("Open Forecasts:", uniphos_open_forecast)
print("High Forecasts:", uniphos_high_forecast)
print("Low Forecasts:", uniphos_low_forecast)


# In[252]:


uniphos_tail_50_data = uniphos.tail(forecast_periods)

uniphos_actual_close_prices = uniphos_tail_50_data['Close'].values
uniphos_actual_open_prices = uniphos_tail_50_data['Open'].values
uniphos_actual_high_prices = uniphos_tail_50_data['High'].values
uniphos_actual_low_prices = uniphos_tail_50_data['Low'].values

uniphos_forecast_close = uniphos_close_final_model.forecast(steps=forecast_periods)
uniphos_forecast_close = uniphos_forecast_close.reshape(-1, 1)
uniphos_forecast_close = scaler.inverse_transform(uniphos_forecast_close)

uniphos_forecast_open = uniphos_open_final_model.forecast(steps=forecast_periods)
uniphos_forecast_open = uniphos_forecast_open.reshape(-1, 1)
uniphos_forecast_open = scaler.inverse_transform(uniphos_forecast_open)

uniphos_forecast_high = uniphos_high_final_model.forecast(steps=forecast_periods)
uniphos_forecast_high = uniphos_forecast_high.reshape(-1, 1)
uniphos_forecast_high = scaler.inverse_transform(uniphos_forecast_high)

uniphos_forecast_low = uniphos_low_final_model.forecast(steps=forecast_periods)
uniphos_forecast_low = uniphos_forecast_low.reshape(-1, 1)
uniphos_forecast_low = scaler.inverse_transform(uniphos_forecast_low)

uniphos_close_mae = mean_absolute_error(uniphos_actual_close_prices, uniphos_forecast_close)
uniphos_close_mse = mean_squared_error(uniphos_actual_close_prices, uniphos_forecast_close)
uniphos_close_rmse = np.sqrt(uniphos_close_mse)

uniphos_open_mae = mean_absolute_error(uniphos_actual_open_prices, uniphos_forecast_open)
uniphos_open_mse = mean_squared_error(uniphos_actual_open_prices, uniphos_forecast_open)
uniphos_open_rmse = np.sqrt(uniphos_open_mse)

uniphos_high_mae = mean_absolute_error(uniphos_actual_high_prices, uniphos_forecast_high)
uniphos_high_mse = mean_squared_error(uniphos_actual_high_prices, uniphos_forecast_high)
uniphos_high_rmse = np.sqrt(uniphos_high_mse)

uniphos_low_mae = mean_absolute_error(uniphos_actual_low_prices, uniphos_forecast_low)
uniphos_low_mse = mean_squared_error(uniphos_actual_low_prices, uniphos_forecast_low)
uniphos_low_rmse = np.sqrt(uniphos_low_mse)

uniphos_close_mape = mean_absolute_percentage_error(uniphos_actual_close_prices, uniphos_forecast_close)
uniphos_open_mape = mean_absolute_percentage_error(uniphos_actual_open_prices, uniphos_forecast_open)
uniphos_high_mape = mean_absolute_percentage_error(uniphos_actual_high_prices, uniphos_forecast_high)
uniphos_low_mape = mean_absolute_percentage_error(uniphos_actual_low_prices, uniphos_forecast_low)

print("Close Forecasts:", uniphos_forecast_close)
print(f"Close Mean Absolute Error (MAE): {uniphos_close_mae}")
print(f"Close Mean Squared Error (MSE): {uniphos_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {uniphos_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {uniphos_close_mape}%")

print("Open Forecasts:", uniphos_forecast_open)
print(f"Open Mean Absolute Error (MAE): {uniphos_open_mae}")
print(f"Open Mean Squared Error (MSE): {uniphos_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {uniphos_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {uniphos_open_mape}%")

print("High Forecasts:", uniphos_forecast_high)
print(f"High Mean Absolute Error (MAE): {uniphos_high_mae}")
print(f"High Mean Squared Error (MSE): {uniphos_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {uniphos_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {uniphos_high_mape}%")

print("Low Forecasts:", uniphos_forecast_low)
print(f"Low Mean Absolute Error (MAE): {uniphos_low_mae}")
print(f"Low Mean Squared Error (MSE): {uniphos_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {uniphos_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {uniphos_low_mape}")


# In[253]:


upl_y_close = upl['Close'].values
upl_y_open = upl['Open'].values
upl_y_high = upl['High'].values
upl_y_low = upl['Low'].values

upl_y_close_scaled = scaler.fit_transform(upl_y_close.reshape(-1, 1))
upl_y_open_scaled = scaler.fit_transform(upl_y_open.reshape(-1, 1))
upl_y_high_scaled = scaler.fit_transform(upl_y_high.reshape(-1, 1))
upl_y_low_scaled = scaler.fit_transform(upl_y_low.reshape(-1, 1))

upl_close_model = auto_arima(
    upl_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

upl_open_model = auto_arima(
    upl_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

upl_high_model = auto_arima(
    upl_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

upl_low_model = auto_arima(
    upl_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

upl_close_best_order = upl_close_model.get_params()['order']
upl_open_best_order = upl_open_model.get_params()['order']
upl_high_best_order = upl_high_model.get_params()['order']
upl_low_best_order = upl_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {upl_close_best_order}")
print(f"Best ARIMA Order for Open: {upl_open_best_order}")
print(f"Best ARIMA Order for High: {upl_high_best_order}")
print(f"Best ARIMA Order for Low: {upl_low_best_order}")

upl_close_final_model = sm.tsa.ARIMA(
    upl_y_close_scaled,
    order=upl_close_best_order
)
upl_close_final_model = upl_close_final_model.fit()
upl_close_forecast = upl_close_final_model.forecast(steps=forecast_period)
upl_close_forecast = upl_close_forecast.reshape(-1, 1)
upl_close_forecast = scaler.inverse_transform(upl_close_forecast)

upl_open_final_model = sm.tsa.ARIMA(
    upl_y_open_scaled,
    order=upl_open_best_order
)
upl_open_final_model = upl_open_final_model.fit()
upl_open_forecast = upl_open_final_model.forecast(steps=forecast_period)
upl_open_forecast = upl_open_forecast.reshape(-1, 1)
upl_open_forecast = scaler.inverse_transform(upl_open_forecast)

upl_high_final_model = sm.tsa.ARIMA(
    upl_y_high_scaled,
    order=upl_high_best_order
)
upl_high_final_model = upl_high_final_model.fit()
upl_high_forecast = upl_high_final_model.forecast(steps=forecast_period)
upl_high_forecast = upl_high_forecast.reshape(-1, 1)
upl_high_forecast = scaler.inverse_transform(upl_high_forecast)

upl_low_final_model = sm.tsa.ARIMA(
    upl_y_low_scaled,
    order=upl_low_best_order
)
upl_low_final_model = upl_low_final_model.fit()
upl_low_forecast = upl_low_final_model.forecast(steps=forecast_period)
upl_low_forecast = upl_low_forecast.reshape(-1, 1)
upl_low_forecast = scaler.inverse_transform(upl_low_forecast)

print("Close Forecasts:", upl_close_forecast)
print("Open Forecasts:", upl_open_forecast)
print("High Forecasts:", upl_high_forecast)
print("Low Forecasts:", upl_low_forecast)


# In[254]:


upl_tail_50_data = upl.tail(forecast_periods)

upl_actual_close_prices = upl_tail_50_data['Close'].values
upl_actual_open_prices = upl_tail_50_data['Open'].values
upl_actual_high_prices = upl_tail_50_data['High'].values
upl_actual_low_prices = upl_tail_50_data['Low'].values

upl_forecast_close = upl_close_final_model.forecast(steps=forecast_periods)
upl_forecast_close = upl_forecast_close.reshape(-1, 1)
upl_forecast_close = scaler.inverse_transform(upl_forecast_close)

upl_forecast_open = upl_open_final_model.forecast(steps=forecast_periods)
upl_forecast_open = upl_forecast_open.reshape(-1, 1)
upl_forecast_open = scaler.inverse_transform(upl_forecast_open)

upl_forecast_high = upl_high_final_model.forecast(steps=forecast_periods)
upl_forecast_high = upl_forecast_high.reshape(-1, 1)
upl_forecast_high = scaler.inverse_transform(upl_forecast_high)

upl_forecast_low = upl_low_final_model.forecast(steps=forecast_periods)
upl_forecast_low = upl_forecast_low.reshape(-1, 1)
upl_forecast_low = scaler.inverse_transform(upl_forecast_low)

upl_close_mae = mean_absolute_error(upl_actual_close_prices, upl_forecast_close)
upl_close_mse = mean_squared_error(upl_actual_close_prices, upl_forecast_close)
upl_close_rmse = np.sqrt(upl_close_mse)

upl_open_mae = mean_absolute_error(upl_actual_open_prices, upl_forecast_open)
upl_open_mse = mean_squared_error(upl_actual_open_prices, upl_forecast_open)
upl_open_rmse = np.sqrt(upl_open_mse)

upl_high_mae = mean_absolute_error(upl_actual_high_prices, upl_forecast_high)
upl_high_mse = mean_squared_error(upl_actual_high_prices, upl_forecast_high)
upl_high_rmse = np.sqrt(upl_high_mse)

upl_low_mae = mean_absolute_error(upl_actual_low_prices, upl_forecast_low)
upl_low_mse = mean_squared_error(upl_actual_low_prices, upl_forecast_low)
upl_low_rmse = np.sqrt(upl_low_mse)

upl_close_mape = mean_absolute_percentage_error(upl_actual_close_prices, upl_forecast_close)
upl_open_mape = mean_absolute_percentage_error(upl_actual_open_prices, upl_forecast_open)
upl_high_mape = mean_absolute_percentage_error(upl_actual_high_prices, upl_forecast_high)
upl_low_mape = mean_absolute_percentage_error(upl_actual_low_prices, upl_forecast_low)

print("Close Forecasts:", upl_forecast_close)
print(f"Close Mean Absolute Error (MAE): {upl_close_mae}")
print(f"Close Mean Squared Error (MSE): {upl_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {upl_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {upl_close_mape}%")

print("Open Forecasts:", upl_forecast_open)
print(f"Open Mean Absolute Error (MAE): {upl_open_mae}")
print(f"Open Mean Squared Error (MSE): {upl_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {upl_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {upl_open_mape}%")

print("High Forecasts:", upl_forecast_high)
print(f"High Mean Absolute Error (MAE): {upl_high_mae}")
print(f"High Mean Squared Error (MSE): {upl_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {upl_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {upl_high_mape}%")

print("Low Forecasts:", upl_forecast_low)
print(f"Low Mean Absolute Error (MAE): {upl_low_mae}")
print(f"Low Mean Squared Error (MSE): {upl_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {upl_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {upl_low_mape}")


# In[255]:


sslt_y_close = sslt['Close'].values
sslt_y_open = sslt['Open'].values
sslt_y_high = sslt['High'].values
sslt_y_low = sslt['Low'].values

sslt_y_close_scaled = scaler.fit_transform(sslt_y_close.reshape(-1, 1))
sslt_y_open_scaled = scaler.fit_transform(sslt_y_open.reshape(-1, 1))
sslt_y_high_scaled = scaler.fit_transform(sslt_y_high.reshape(-1, 1))
sslt_y_low_scaled = scaler.fit_transform(sslt_y_low.reshape(-1, 1))

sslt_close_model = auto_arima(
    sslt_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sslt_open_model = auto_arima(
    sslt_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sslt_high_model = auto_arima(
    sslt_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sslt_low_model = auto_arima(
    sslt_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sslt_close_best_order = sslt_close_model.get_params()['order']
sslt_open_best_order = sslt_open_model.get_params()['order']
sslt_high_best_order = sslt_high_model.get_params()['order']
sslt_low_best_order = sslt_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {sslt_close_best_order}")
print(f"Best ARIMA Order for Open: {sslt_open_best_order}")
print(f"Best ARIMA Order for High: {sslt_high_best_order}")
print(f"Best ARIMA Order for Low: {sslt_low_best_order}")

sslt_close_final_model = sm.tsa.ARIMA(
    sslt_y_close_scaled,
    order=sslt_close_best_order
)
sslt_close_final_model = sslt_close_final_model.fit()
sslt_close_forecast = sslt_close_final_model.forecast(steps=forecast_period)
sslt_close_forecast = sslt_close_forecast.reshape(-1, 1)
sslt_close_forecast = scaler.inverse_transform(sslt_close_forecast)

sslt_open_final_model = sm.tsa.ARIMA(
    sslt_y_open_scaled,
    order=sslt_open_best_order
)
sslt_open_final_model = sslt_open_final_model.fit()
sslt_open_forecast = sslt_open_final_model.forecast(steps=forecast_period)
sslt_open_forecast = sslt_open_forecast.reshape(-1, 1)
sslt_open_forecast = scaler.inverse_transform(sslt_open_forecast)

sslt_high_final_model = sm.tsa.ARIMA(
    sslt_y_high_scaled,
    order=sslt_high_best_order
)
sslt_high_final_model = sslt_high_final_model.fit()
sslt_high_forecast = sslt_high_final_model.forecast(steps=forecast_period)
sslt_high_forecast = sslt_high_forecast.reshape(-1, 1)
sslt_high_forecast = scaler.inverse_transform(sslt_high_forecast)

sslt_low_final_model = sm.tsa.ARIMA(
    sslt_y_low_scaled,
    order=sslt_low_best_order
)
sslt_low_final_model = sslt_low_final_model.fit()
sslt_low_forecast = sslt_low_final_model.forecast(steps=forecast_period)
sslt_low_forecast = sslt_low_forecast.reshape(-1, 1)
sslt_low_forecast = scaler.inverse_transform(sslt_low_forecast)

print("Close Forecasts:", sslt_close_forecast)
print("Open Forecasts:", sslt_open_forecast)
print("High Forecasts:", sslt_high_forecast)
print("Low Forecasts:", sslt_low_forecast)


# In[256]:


sslt_tail_50_data = sslt.tail(forecast_periods)

sslt_actual_close_prices = sslt_tail_50_data['Close'].values
sslt_actual_open_prices = sslt_tail_50_data['Open'].values
sslt_actual_high_prices = sslt_tail_50_data['High'].values
sslt_actual_low_prices = sslt_tail_50_data['Low'].values

sslt_forecast_close = sslt_close_final_model.forecast(steps=forecast_periods)
sslt_forecast_close = sslt_forecast_close.reshape(-1, 1)
sslt_forecast_close = scaler.inverse_transform(sslt_forecast_close)

sslt_forecast_open = sslt_open_final_model.forecast(steps=forecast_periods)
sslt_forecast_open = sslt_forecast_open.reshape(-1, 1)
sslt_forecast_open = scaler.inverse_transform(sslt_forecast_open)

sslt_forecast_high = sslt_high_final_model.forecast(steps=forecast_periods)
sslt_forecast_high = sslt_forecast_high.reshape(-1, 1)
sslt_forecast_high = scaler.inverse_transform(sslt_forecast_high)

sslt_forecast_low = sslt_low_final_model.forecast(steps=forecast_periods)
sslt_forecast_low = sslt_forecast_low.reshape(-1, 1)
sslt_forecast_low = scaler.inverse_transform(sslt_forecast_low)

sslt_close_mae = mean_absolute_error(sslt_actual_close_prices, sslt_forecast_close)
sslt_close_mse = mean_squared_error(sslt_actual_close_prices, sslt_forecast_close)
sslt_close_rmse = np.sqrt(sslt_close_mse)

sslt_open_mae = mean_absolute_error(sslt_actual_open_prices, sslt_forecast_open)
sslt_open_mse = mean_squared_error(sslt_actual_open_prices, sslt_forecast_open)
sslt_open_rmse = np.sqrt(sslt_open_mse)

sslt_high_mae = mean_absolute_error(sslt_actual_high_prices, sslt_forecast_high)
sslt_high_mse = mean_squared_error(sslt_actual_high_prices, sslt_forecast_high)
sslt_high_rmse = np.sqrt(sslt_high_mse)

sslt_low_mae = mean_absolute_error(sslt_actual_low_prices, sslt_forecast_low)
sslt_low_mse = mean_squared_error(sslt_actual_low_prices, sslt_forecast_low)
sslt_low_rmse = np.sqrt(sslt_low_mse)

sslt_close_mape = mean_absolute_percentage_error(sslt_actual_close_prices, sslt_forecast_close)
sslt_open_mape = mean_absolute_percentage_error(sslt_actual_open_prices, sslt_forecast_open)
sslt_high_mape = mean_absolute_percentage_error(sslt_actual_high_prices, sslt_forecast_high)
sslt_low_mape = mean_absolute_percentage_error(sslt_actual_low_prices, sslt_forecast_low)

print("Close Forecasts:", sslt_forecast_close)
print(f"Close Mean Absolute Error (MAE): {sslt_close_mae}")
print(f"Close Mean Squared Error (MSE): {sslt_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {sslt_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {sslt_close_mape}%")

print("Open Forecasts:", sslt_forecast_open)
print(f"Open Mean Absolute Error (MAE): {sslt_open_mae}")
print(f"Open Mean Squared Error (MSE): {sslt_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {sslt_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {sslt_open_mape}%")

print("High Forecasts:", sslt_forecast_high)
print(f"High Mean Absolute Error (MAE): {sslt_high_mae}")
print(f"High Mean Squared Error (MSE): {sslt_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {sslt_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {sslt_high_mape}%")

print("Low Forecasts:", sslt_forecast_low)
print(f"Low Mean Absolute Error (MAE): {sslt_low_mae}")
print(f"Low Mean Squared Error (MSE): {sslt_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {sslt_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {sslt_low_mape}")


# In[257]:


vedl_y_close = vedl['Close'].values
vedl_y_open = vedl['Open'].values
vedl_y_high = vedl['High'].values
vedl_y_low = vedl['Low'].values

vedl_y_close_scaled = scaler.fit_transform(vedl_y_close.reshape(-1, 1))
vedl_y_open_scaled = scaler.fit_transform(vedl_y_open.reshape(-1, 1))
vedl_y_high_scaled = scaler.fit_transform(vedl_y_high.reshape(-1, 1))
vedl_y_low_scaled = scaler.fit_transform(vedl_y_low.reshape(-1, 1))

vedl_close_model = auto_arima(
    vedl_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

vedl_open_model = auto_arima(
    vedl_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

vedl_high_model = auto_arima(
    vedl_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

vedl_low_model = auto_arima(
    vedl_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

vedl_close_best_order = vedl_close_model.get_params()['order']
vedl_open_best_order = vedl_open_model.get_params()['order']
vedl_high_best_order = vedl_high_model.get_params()['order']
vedl_low_best_order = vedl_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {vedl_close_best_order}")
print(f"Best ARIMA Order for Open: {vedl_open_best_order}")
print(f"Best ARIMA Order for High: {vedl_high_best_order}")
print(f"Best ARIMA Order for Low: {vedl_low_best_order}")

vedl_close_final_model = sm.tsa.ARIMA(
    vedl_y_close_scaled,
    order=vedl_close_best_order
)
vedl_close_final_model = vedl_close_final_model.fit()
vedl_close_forecast = vedl_close_final_model.forecast(steps=forecast_period)
vedl_close_forecast = vedl_close_forecast.reshape(-1, 1)
vedl_close_forecast = scaler.inverse_transform(vedl_close_forecast)

vedl_open_final_model = sm.tsa.ARIMA(
    vedl_y_open_scaled,
    order=vedl_open_best_order
)
vedl_open_final_model = vedl_open_final_model.fit()
vedl_open_forecast = vedl_open_final_model.forecast(steps=forecast_period)
vedl_open_forecast = vedl_open_forecast.reshape(-1, 1)
vedl_open_forecast = scaler.inverse_transform(vedl_open_forecast)

vedl_high_final_model = sm.tsa.ARIMA(
    vedl_y_high_scaled,
    order=vedl_high_best_order
)
vedl_high_final_model = vedl_high_final_model.fit()
vedl_high_forecast = vedl_high_final_model.forecast(steps=forecast_period)
vedl_high_forecast = vedl_high_forecast.reshape(-1, 1)
vedl_high_forecast = scaler.inverse_transform(vedl_high_forecast)

vedl_low_final_model = sm.tsa.ARIMA(
    vedl_y_low_scaled,
    order=vedl_low_best_order
)
vedl_low_final_model = vedl_low_final_model.fit()
vedl_low_forecast = vedl_low_final_model.forecast(steps=forecast_period)
vedl_low_forecast = vedl_low_forecast.reshape(-1, 1)
vedl_low_forecast = scaler.inverse_transform(vedl_low_forecast)

print("Close Forecasts:", vedl_close_forecast)
print("Open Forecasts:", vedl_open_forecast)
print("High Forecasts:", vedl_high_forecast)
print("Low Forecasts:", vedl_low_forecast)


# In[258]:


vedl_tail_50_data = vedl.tail(forecast_periods)

vedl_actual_close_prices = vedl_tail_50_data['Close'].values
vedl_actual_open_prices = vedl_tail_50_data['Open'].values
vedl_actual_high_prices = vedl_tail_50_data['High'].values
vedl_actual_low_prices = vedl_tail_50_data['Low'].values

vedl_forecast_close = vedl_close_final_model.forecast(steps=forecast_periods)
vedl_forecast_close = vedl_forecast_close.reshape(-1, 1)
vedl_forecast_close = scaler.inverse_transform(vedl_forecast_close)

vedl_forecast_open = vedl_open_final_model.forecast(steps=forecast_periods)
vedl_forecast_open = vedl_forecast_open.reshape(-1, 1)
vedl_forecast_open = scaler.inverse_transform(vedl_forecast_open)

vedl_forecast_high = vedl_high_final_model.forecast(steps=forecast_periods)
vedl_forecast_high = vedl_forecast_high.reshape(-1, 1)
vedl_forecast_high = scaler.inverse_transform(vedl_forecast_high)

vedl_forecast_low = vedl_low_final_model.forecast(steps=forecast_periods)
vedl_forecast_low = vedl_forecast_low.reshape(-1, 1)
vedl_forecast_low = scaler.inverse_transform(vedl_forecast_low)

vedl_close_mae = mean_absolute_error(vedl_actual_close_prices, vedl_forecast_close)
vedl_close_mse = mean_squared_error(vedl_actual_close_prices, vedl_forecast_close)
vedl_close_rmse = np.sqrt(vedl_close_mse)

vedl_open_mae = mean_absolute_error(vedl_actual_open_prices, vedl_forecast_open)
vedl_open_mse = mean_squared_error(vedl_actual_open_prices, vedl_forecast_open)
vedl_open_rmse = np.sqrt(vedl_open_mse)

vedl_high_mae = mean_absolute_error(vedl_actual_high_prices, vedl_forecast_high)
vedl_high_mse = mean_squared_error(vedl_actual_high_prices, vedl_forecast_high)
vedl_high_rmse = np.sqrt(vedl_high_mse)

vedl_low_mae = mean_absolute_error(vedl_actual_low_prices, vedl_forecast_low)
vedl_low_mse = mean_squared_error(vedl_actual_low_prices, vedl_forecast_low)
vedl_low_rmse = np.sqrt(vedl_low_mse)

vedl_close_mape = mean_absolute_percentage_error(vedl_actual_close_prices, vedl_forecast_close)
vedl_open_mape = mean_absolute_percentage_error(vedl_actual_open_prices, vedl_forecast_open)
vedl_high_mape = mean_absolute_percentage_error(vedl_actual_high_prices, vedl_forecast_high)
vedl_low_mape = mean_absolute_percentage_error(vedl_actual_low_prices, vedl_forecast_low)

print("Close Forecasts:", vedl_forecast_close)
print(f"Close Mean Absolute Error (MAE): {vedl_close_mae}")
print(f"Close Mean Squared Error (MSE): {vedl_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {vedl_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {vedl_close_mape}%")

print("Open Forecasts:", vedl_forecast_open)
print(f"Open Mean Absolute Error (MAE): {vedl_open_mae}")
print(f"Open Mean Squared Error (MSE): {vedl_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {vedl_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {vedl_open_mape}%")

print("High Forecasts:", vedl_forecast_high)
print(f"High Mean Absolute Error (MAE): {vedl_high_mae}")
print(f"High Mean Squared Error (MSE): {vedl_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {vedl_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {vedl_high_mape}%")

print("Low Forecasts:", vedl_forecast_low)
print(f"Low Mean Absolute Error (MAE): {vedl_low_mae}")
print(f"Low Mean Squared Error (MSE): {vedl_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {vedl_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {vedl_low_mape}")


# In[259]:


sesagoa_y_close = sesagoa['Close'].values
sesagoa_y_open = sesagoa['Open'].values
sesagoa_y_high = sesagoa['High'].values
sesagoa_y_low = sesagoa['Low'].values

sesagoa_y_close_scaled = scaler.fit_transform(sesagoa_y_close.reshape(-1, 1))
sesagoa_y_open_scaled = scaler.fit_transform(sesagoa_y_open.reshape(-1, 1))
sesagoa_y_high_scaled = scaler.fit_transform(sesagoa_y_high.reshape(-1, 1))
sesagoa_y_low_scaled = scaler.fit_transform(sesagoa_y_low.reshape(-1, 1))

sesagoa_close_model = auto_arima(
    sesagoa_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sesagoa_open_model = auto_arima(
    sesagoa_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sesagoa_high_model = auto_arima(
    sesagoa_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sesagoa_low_model = auto_arima(
    sesagoa_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

sesagoa_close_best_order = sesagoa_close_model.get_params()['order']
sesagoa_open_best_order = sesagoa_open_model.get_params()['order']
sesagoa_high_best_order = sesagoa_high_model.get_params()['order']
sesagoa_low_best_order = sesagoa_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {sesagoa_close_best_order}")
print(f"Best ARIMA Order for Open: {sesagoa_open_best_order}")
print(f"Best ARIMA Order for High: {sesagoa_high_best_order}")
print(f"Best ARIMA Order for Low: {sesagoa_low_best_order}")

sesagoa_close_final_model = sm.tsa.ARIMA(
    sesagoa_y_close_scaled,
    order=sesagoa_close_best_order
)
sesagoa_close_final_model = sesagoa_close_final_model.fit()
sesagoa_close_forecast = sesagoa_close_final_model.forecast(steps=forecast_period)
sesagoa_close_forecast = sesagoa_close_forecast.reshape(-1, 1)
sesagoa_close_forecast = scaler.inverse_transform(sesagoa_close_forecast)

sesagoa_open_final_model = sm.tsa.ARIMA(
    sesagoa_y_open_scaled,
    order=sesagoa_open_best_order
)
sesagoa_open_final_model = sesagoa_open_final_model.fit()
sesagoa_open_forecast = sesagoa_open_final_model.forecast(steps=forecast_period)
sesagoa_open_forecast = sesagoa_open_forecast.reshape(-1, 1)
sesagoa_open_forecast = scaler.inverse_transform(sesagoa_open_forecast)

sesagoa_high_final_model = sm.tsa.ARIMA(
    sesagoa_y_high_scaled,
    order=sesagoa_high_best_order
)
sesagoa_high_final_model = sesagoa_high_final_model.fit()
sesagoa_high_forecast = sesagoa_high_final_model.forecast(steps=forecast_period)
sesagoa_high_forecast = sesagoa_high_forecast.reshape(-1, 1)
sesagoa_high_forecast = scaler.inverse_transform(sesagoa_high_forecast)

sesagoa_low_final_model = sm.tsa.ARIMA(
    sesagoa_y_low_scaled,
    order=sesagoa_low_best_order
)
sesagoa_low_final_model = sesagoa_low_final_model.fit()
sesagoa_low_forecast = sesagoa_low_final_model.forecast(steps=forecast_period)
sesagoa_low_forecast = sesagoa_low_forecast.reshape(-1, 1)
sesagoa_low_forecast = scaler.inverse_transform(sesagoa_low_forecast)

print("Close Forecasts:", sesagoa_close_forecast)
print("Open Forecasts:", sesagoa_open_forecast)
print("High Forecasts:", sesagoa_high_forecast)
print("Low Forecasts:", sesagoa_low_forecast)


# In[260]:


sesagoa_tail_50_data = sesagoa.tail(forecast_periods)

sesagoa_actual_close_prices = sesagoa_tail_50_data['Close'].values
sesagoa_actual_open_prices = sesagoa_tail_50_data['Open'].values
sesagoa_actual_high_prices = sesagoa_tail_50_data['High'].values
sesagoa_actual_low_prices = sesagoa_tail_50_data['Low'].values

sesagoa_forecast_close = sesagoa_close_final_model.forecast(steps=forecast_periods)
sesagoa_forecast_close = sesagoa_forecast_close.reshape(-1, 1)
sesagoa_forecast_close = scaler.inverse_transform(sesagoa_forecast_close)

sesagoa_forecast_open = sesagoa_open_final_model.forecast(steps=forecast_periods)
sesagoa_forecast_open = sesagoa_forecast_open.reshape(-1, 1)
sesagoa_forecast_open = scaler.inverse_transform(sesagoa_forecast_open)

sesagoa_forecast_high = sesagoa_high_final_model.forecast(steps=forecast_periods)
sesagoa_forecast_high = sesagoa_forecast_high.reshape(-1, 1)
sesagoa_forecast_high = scaler.inverse_transform(sesagoa_forecast_high)

sesagoa_forecast_low = sesagoa_low_final_model.forecast(steps=forecast_periods)
sesagoa_forecast_low = sesagoa_forecast_low.reshape(-1, 1)
sesagoa_forecast_low = scaler.inverse_transform(sesagoa_forecast_low)

sesagoa_close_mae = mean_absolute_error(sesagoa_actual_close_prices, sesagoa_forecast_close)
sesagoa_close_mse = mean_squared_error(sesagoa_actual_close_prices, sesagoa_forecast_close)
sesagoa_close_rmse = np.sqrt(sesagoa_close_mse)

sesagoa_open_mae = mean_absolute_error(sesagoa_actual_open_prices, sesagoa_forecast_open)
sesagoa_open_mse = mean_squared_error(sesagoa_actual_open_prices, sesagoa_forecast_open)
sesagoa_open_rmse = np.sqrt(sesagoa_open_mse)

sesagoa_high_mae = mean_absolute_error(sesagoa_actual_high_prices, sesagoa_forecast_high)
sesagoa_high_mse = mean_squared_error(sesagoa_actual_high_prices, sesagoa_forecast_high)
sesagoa_high_rmse = np.sqrt(sesagoa_high_mse)

sesagoa_low_mae = mean_absolute_error(sesagoa_actual_low_prices, sesagoa_forecast_low)
sesagoa_low_mse = mean_squared_error(sesagoa_actual_low_prices, sesagoa_forecast_low)
sesagoa_low_rmse = np.sqrt(sesagoa_low_mse)

sesagoa_close_mape = mean_absolute_percentage_error(sesagoa_actual_close_prices, sesagoa_forecast_close)
sesagoa_open_mape = mean_absolute_percentage_error(sesagoa_actual_open_prices, sesagoa_forecast_open)
sesagoa_high_mape = mean_absolute_percentage_error(sesagoa_actual_high_prices, sesagoa_forecast_high)
sesagoa_low_mape = mean_absolute_percentage_error(sesagoa_actual_low_prices, sesagoa_forecast_low)

print("Close Forecasts:", sesagoa_forecast_close)
print(f"Close Mean Absolute Error (MAE): {sesagoa_close_mae}")
print(f"Close Mean Squared Error (MSE): {sesagoa_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {sesagoa_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {sesagoa_close_mape}%")

print("Open Forecasts:", sesagoa_forecast_open)
print(f"Open Mean Absolute Error (MAE): {sesagoa_open_mae}")
print(f"Open Mean Squared Error (MSE): {sesagoa_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {sesagoa_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {sesagoa_open_mape}%")

print("High Forecasts:", sesagoa_forecast_high)
print(f"High Mean Absolute Error (MAE): {sesagoa_high_mae}")
print(f"High Mean Squared Error (MSE): {sesagoa_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {sesagoa_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {sesagoa_high_mape}%")

print("Low Forecasts:", sesagoa_forecast_low)
print(f"Low Mean Absolute Error (MAE): {sesagoa_low_mae}")
print(f"Low Mean Squared Error (MSE): {sesagoa_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {sesagoa_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {sesagoa_low_mape}")


# In[261]:


wipro_y_close = wipro['Close'].values
wipro_y_open = wipro['Open'].values
wipro_y_high = wipro['High'].values
wipro_y_low = wipro['Low'].values

wipro_y_close_scaled = scaler.fit_transform(wipro_y_close.reshape(-1, 1))
wipro_y_open_scaled = scaler.fit_transform(wipro_y_open.reshape(-1, 1))
wipro_y_high_scaled = scaler.fit_transform(wipro_y_high.reshape(-1, 1))
wipro_y_low_scaled = scaler.fit_transform(wipro_y_low.reshape(-1, 1))

wipro_close_model = auto_arima(
    wipro_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

wipro_open_model = auto_arima(
    wipro_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

wipro_high_model = auto_arima(
    wipro_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

wipro_low_model = auto_arima(
    wipro_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

wipro_close_best_order = wipro_close_model.get_params()['order']
wipro_open_best_order = wipro_open_model.get_params()['order']
wipro_high_best_order = wipro_high_model.get_params()['order']
wipro_low_best_order = wipro_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {wipro_close_best_order}")
print(f"Best ARIMA Order for Open: {wipro_open_best_order}")
print(f"Best ARIMA Order for High: {wipro_high_best_order}")
print(f"Best ARIMA Order for Low: {wipro_low_best_order}")

wipro_close_final_model = sm.tsa.ARIMA(
    wipro_y_close_scaled,
    order=wipro_close_best_order
)
wipro_close_final_model = wipro_close_final_model.fit()
wipro_close_forecast = wipro_close_final_model.forecast(steps=forecast_period)
wipro_close_forecast = wipro_close_forecast.reshape(-1, 1)
wipro_close_forecast = scaler.inverse_transform(wipro_close_forecast)

wipro_open_final_model = sm.tsa.ARIMA(
    wipro_y_open_scaled,
    order=wipro_open_best_order
)
wipro_open_final_model = wipro_open_final_model.fit()
wipro_open_forecast = wipro_open_final_model.forecast(steps=forecast_period)
wipro_open_forecast = wipro_open_forecast.reshape(-1, 1)
wipro_open_forecast = scaler.inverse_transform(wipro_open_forecast)

wipro_high_final_model = sm.tsa.ARIMA(
    wipro_y_high_scaled,
    order=wipro_high_best_order
)
wipro_high_final_model = wipro_high_final_model.fit()
wipro_high_forecast = wipro_high_final_model.forecast(steps=forecast_period)
wipro_high_forecast = wipro_high_forecast.reshape(-1, 1)
wipro_high_forecast = scaler.inverse_transform(wipro_high_forecast)

wipro_low_final_model = sm.tsa.ARIMA(
    wipro_y_low_scaled,
    order=wipro_low_best_order
)
wipro_low_final_model = wipro_low_final_model.fit()
wipro_low_forecast = wipro_low_final_model.forecast(steps=forecast_period)
wipro_low_forecast = wipro_low_forecast.reshape(-1, 1)
wipro_low_forecast = scaler.inverse_transform(wipro_low_forecast)

print("Close Forecasts:", wipro_close_forecast)
print("Open Forecasts:", wipro_open_forecast)
print("High Forecasts:", wipro_high_forecast)
print("Low Forecasts:", wipro_low_forecast)


# In[262]:


wipro_tail_50_data = wipro.tail(forecast_periods)

wipro_actual_close_prices = wipro_tail_50_data['Close'].values
wipro_actual_open_prices = wipro_tail_50_data['Open'].values
wipro_actual_high_prices = wipro_tail_50_data['High'].values
wipro_actual_low_prices = wipro_tail_50_data['Low'].values

wipro_forecast_close = wipro_close_final_model.forecast(steps=forecast_periods)
wipro_forecast_close = wipro_forecast_close.reshape(-1, 1)
wipro_forecast_close = scaler.inverse_transform(wipro_forecast_close)

wipro_forecast_open = wipro_open_final_model.forecast(steps=forecast_periods)
wipro_forecast_open = wipro_forecast_open.reshape(-1, 1)
wipro_forecast_open = scaler.inverse_transform(wipro_forecast_open)

wipro_forecast_high = wipro_high_final_model.forecast(steps=forecast_periods)
wipro_forecast_high = wipro_forecast_high.reshape(-1, 1)
wipro_forecast_high = scaler.inverse_transform(wipro_forecast_high)

wipro_forecast_low = wipro_low_final_model.forecast(steps=forecast_periods)
wipro_forecast_low = wipro_forecast_low.reshape(-1, 1)
wipro_forecast_low = scaler.inverse_transform(wipro_forecast_low)

wipro_close_mae = mean_absolute_error(wipro_actual_close_prices, wipro_forecast_close)
wipro_close_mse = mean_squared_error(wipro_actual_close_prices, wipro_forecast_close)
wipro_close_rmse = np.sqrt(wipro_close_mse)

wipro_open_mae = mean_absolute_error(wipro_actual_open_prices, wipro_forecast_open)
wipro_open_mse = mean_squared_error(wipro_actual_open_prices, wipro_forecast_open)
wipro_open_rmse = np.sqrt(wipro_open_mse)

wipro_high_mae = mean_absolute_error(wipro_actual_high_prices, wipro_forecast_high)
wipro_high_mse = mean_squared_error(wipro_actual_high_prices, wipro_forecast_high)
wipro_high_rmse = np.sqrt(wipro_high_mse)

wipro_low_mae = mean_absolute_error(wipro_actual_low_prices, wipro_forecast_low)
wipro_low_mse = mean_squared_error(wipro_actual_low_prices, wipro_forecast_low)
wipro_low_rmse = np.sqrt(wipro_low_mse)

wipro_close_mape = mean_absolute_percentage_error(wipro_actual_close_prices, wipro_forecast_close)
wipro_open_mape = mean_absolute_percentage_error(wipro_actual_open_prices, wipro_forecast_open)
wipro_high_mape = mean_absolute_percentage_error(wipro_actual_high_prices, wipro_forecast_high)
wipro_low_mape = mean_absolute_percentage_error(wipro_actual_low_prices, wipro_forecast_low)

print("Close Forecasts:", wipro_forecast_close)
print(f"Close Mean Absolute Error (MAE): {wipro_close_mae}")
print(f"Close Mean Squared Error (MSE): {wipro_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {wipro_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {wipro_close_mape}%")

print("Open Forecasts:", wipro_forecast_open)
print(f"Open Mean Absolute Error (MAE): {wipro_open_mae}")
print(f"Open Mean Squared Error (MSE): {wipro_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {wipro_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {wipro_open_mape}%")

print("High Forecasts:", wipro_forecast_high)
print(f"High Mean Absolute Error (MAE): {wipro_high_mae}")
print(f"High Mean Squared Error (MSE): {wipro_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {wipro_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {wipro_high_mape}%")

print("Low Forecasts:", wipro_forecast_low)
print(f"Low Mean Absolute Error (MAE): {wipro_low_mae}")
print(f"Low Mean Squared Error (MSE): {wipro_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {wipro_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {wipro_low_mape}")


# In[263]:


zeetele_y_close = zeetele['Close'].values
zeetele_y_open = zeetele['Open'].values
zeetele_y_high = zeetele['High'].values
zeetele_y_low = zeetele['Low'].values

zeetele_y_close_scaled = scaler.fit_transform(zeetele_y_close.reshape(-1, 1))
zeetele_y_open_scaled = scaler.fit_transform(zeetele_y_open.reshape(-1, 1))
zeetele_y_high_scaled = scaler.fit_transform(zeetele_y_high.reshape(-1, 1))
zeetele_y_low_scaled = scaler.fit_transform(zeetele_y_low.reshape(-1, 1))

zeetele_close_model = auto_arima(
    zeetele_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

zeetele_open_model = auto_arima(
    zeetele_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

zeetele_high_model = auto_arima(
    zeetele_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

zeetele_low_model = auto_arima(
    zeetele_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

zeetele_close_best_order = zeetele_close_model.get_params()['order']
zeetele_open_best_order = zeetele_open_model.get_params()['order']
zeetele_high_best_order = zeetele_high_model.get_params()['order']
zeetele_low_best_order = zeetele_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {zeetele_close_best_order}")
print(f"Best ARIMA Order for Open: {zeetele_open_best_order}")
print(f"Best ARIMA Order for High: {zeetele_high_best_order}")
print(f"Best ARIMA Order for Low: {zeetele_low_best_order}")

zeetele_close_final_model = sm.tsa.ARIMA(
    zeetele_y_close_scaled,
    order=zeetele_close_best_order
)
zeetele_close_final_model = zeetele_close_final_model.fit()
zeetele_close_forecast = zeetele_close_final_model.forecast(steps=forecast_period)
zeetele_close_forecast = zeetele_close_forecast.reshape(-1, 1)
zeetele_close_forecast = scaler.inverse_transform(zeetele_close_forecast)

zeetele_open_final_model = sm.tsa.ARIMA(
    zeetele_y_open_scaled,
    order=zeetele_open_best_order
)
zeetele_open_final_model = zeetele_open_final_model.fit()
zeetele_open_forecast = zeetele_open_final_model.forecast(steps=forecast_period)
zeetele_open_forecast = zeetele_open_forecast.reshape(-1, 1)
zeetele_open_forecast = scaler.inverse_transform(zeetele_open_forecast)

zeetele_high_final_model = sm.tsa.ARIMA(
    zeetele_y_high_scaled,
    order=zeetele_high_best_order
)
zeetele_high_final_model = zeetele_high_final_model.fit()
zeetele_high_forecast = zeetele_high_final_model.forecast(steps=forecast_period)
zeetele_high_forecast = zeetele_high_forecast.reshape(-1, 1)
zeetele_high_forecast = scaler.inverse_transform(zeetele_high_forecast)

zeetele_low_final_model = sm.tsa.ARIMA(
    zeetele_y_low_scaled,
    order=zeetele_low_best_order
)
zeetele_low_final_model = zeetele_low_final_model.fit()
zeetele_low_forecast = zeetele_low_final_model.forecast(steps=forecast_period)
zeetele_low_forecast = zeetele_low_forecast.reshape(-1, 1)
zeetele_low_forecast = scaler.inverse_transform(zeetele_low_forecast)

print("Close Forecasts:", zeetele_close_forecast)
print("Open Forecasts:", zeetele_open_forecast)
print("High Forecasts:", zeetele_high_forecast)
print("Low Forecasts:", zeetele_low_forecast)


# In[264]:


zeetele_tail_50_data = zeetele.tail(forecast_periods)

zeetele_actual_close_prices = zeetele_tail_50_data['Close'].values
zeetele_actual_open_prices = zeetele_tail_50_data['Open'].values
zeetele_actual_high_prices = zeetele_tail_50_data['High'].values
zeetele_actual_low_prices = zeetele_tail_50_data['Low'].values

zeetele_forecast_close = zeetele_close_final_model.forecast(steps=forecast_periods)
zeetele_forecast_close = zeetele_forecast_close.reshape(-1, 1)
zeetele_forecast_close = scaler.inverse_transform(zeetele_forecast_close)

zeetele_forecast_open = zeetele_open_final_model.forecast(steps=forecast_periods)
zeetele_forecast_open = zeetele_forecast_open.reshape(-1, 1)
zeetele_forecast_open = scaler.inverse_transform(zeetele_forecast_open)

zeetele_forecast_high = zeetele_high_final_model.forecast(steps=forecast_periods)
zeetele_forecast_high = zeetele_forecast_high.reshape(-1, 1)
zeetele_forecast_high = scaler.inverse_transform(zeetele_forecast_high)

zeetele_forecast_low = zeetele_low_final_model.forecast(steps=forecast_periods)
zeetele_forecast_low = zeetele_forecast_low.reshape(-1, 1)
zeetele_forecast_low = scaler.inverse_transform(zeetele_forecast_low)

zeetele_close_mae = mean_absolute_error(zeetele_actual_close_prices, zeetele_forecast_close)
zeetele_close_mse = mean_squared_error(zeetele_actual_close_prices, zeetele_forecast_close)
zeetele_close_rmse = np.sqrt(zeetele_close_mse)

zeetele_open_mae = mean_absolute_error(zeetele_actual_open_prices, zeetele_forecast_open)
zeetele_open_mse = mean_squared_error(zeetele_actual_open_prices, zeetele_forecast_open)
zeetele_open_rmse = np.sqrt(zeetele_open_mse)

zeetele_high_mae = mean_absolute_error(zeetele_actual_high_prices, zeetele_forecast_high)
zeetele_high_mse = mean_squared_error(zeetele_actual_high_prices, zeetele_forecast_high)
zeetele_high_rmse = np.sqrt(zeetele_high_mse)

zeetele_low_mae = mean_absolute_error(zeetele_actual_low_prices, zeetele_forecast_low)
zeetele_low_mse = mean_squared_error(zeetele_actual_low_prices, zeetele_forecast_low)
zeetele_low_rmse = np.sqrt(zeetele_low_mse)

zeetele_close_mape = mean_absolute_percentage_error(zeetele_actual_close_prices, zeetele_forecast_close)
zeetele_open_mape = mean_absolute_percentage_error(zeetele_actual_open_prices, zeetele_forecast_open)
zeetele_high_mape = mean_absolute_percentage_error(zeetele_actual_high_prices, zeetele_forecast_high)
zeetele_low_mape = mean_absolute_percentage_error(zeetele_actual_low_prices, zeetele_forecast_low)

print("Close Forecasts:", zeetele_forecast_close)
print(f"Close Mean Absolute Error (MAE): {zeetele_close_mae}")
print(f"Close Mean Squared Error (MSE): {zeetele_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {zeetele_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {zeetele_close_mape}%")

print("Open Forecasts:", zeetele_forecast_open)
print(f"Open Mean Absolute Error (MAE): {zeetele_open_mae}")
print(f"Open Mean Squared Error (MSE): {zeetele_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {zeetele_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {zeetele_open_mape}%")

print("High Forecasts:", zeetele_forecast_high)
print(f"High Mean Absolute Error (MAE): {zeetele_high_mae}")
print(f"High Mean Squared Error (MSE): {zeetele_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {zeetele_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {zeetele_high_mape}%")

print("Low Forecasts:", zeetele_forecast_low)
print(f"Low Mean Absolute Error (MAE): {zeetele_low_mae}")
print(f"Low Mean Squared Error (MSE): {zeetele_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {zeetele_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {zeetele_low_mape}")


# In[265]:


zeel_y_close = zeel['Close'].values
zeel_y_open = zeel['Open'].values
zeel_y_high = zeel['High'].values
zeel_y_low = zeel['Low'].values

zeel_y_close_scaled = scaler.fit_transform(zeel_y_close.reshape(-1, 1))
zeel_y_open_scaled = scaler.fit_transform(zeel_y_open.reshape(-1, 1))
zeel_y_high_scaled = scaler.fit_transform(zeel_y_high.reshape(-1, 1))
zeel_y_low_scaled = scaler.fit_transform(zeel_y_low.reshape(-1, 1))

zeel_close_model = auto_arima(
    zeel_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

zeel_open_model = auto_arima(
    zeel_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

zeel_high_model = auto_arima(
    zeel_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

zeel_low_model = auto_arima(
    zeel_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

zeel_close_best_order = zeel_close_model.get_params()['order']
zeel_open_best_order = zeel_open_model.get_params()['order']
zeel_high_best_order = zeel_high_model.get_params()['order']
zeel_low_best_order = zeel_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {zeel_close_best_order}")
print(f"Best ARIMA Order for Open: {zeel_open_best_order}")
print(f"Best ARIMA Order for High: {zeel_high_best_order}")
print(f"Best ARIMA Order for Low: {zeel_low_best_order}")

zeel_close_final_model = sm.tsa.ARIMA(
    zeel_y_close_scaled,
    order=zeel_close_best_order
)
zeel_close_final_model = zeel_close_final_model.fit()
zeel_close_forecast = zeel_close_final_model.forecast(steps=forecast_period)
zeel_close_forecast = zeel_close_forecast.reshape(-1, 1)
zeel_close_forecast = scaler.inverse_transform(zeel_close_forecast)

zeel_open_final_model = sm.tsa.ARIMA(
    zeel_y_open_scaled,
    order=zeel_open_best_order
)
zeel_open_final_model = zeel_open_final_model.fit()
zeel_open_forecast = zeel_open_final_model.forecast(steps=forecast_period)
zeel_open_forecast = zeel_open_forecast.reshape(-1, 1)
zeel_open_forecast = scaler.inverse_transform(zeel_open_forecast)

zeel_high_final_model = sm.tsa.ARIMA(
    zeel_y_high_scaled,
    order=zeel_high_best_order
)
zeel_high_final_model = zeel_high_final_model.fit()
zeel_high_forecast = zeel_high_final_model.forecast(steps=forecast_period)
zeel_high_forecast = zeel_high_forecast.reshape(-1, 1)
zeel_high_forecast = scaler.inverse_transform(zeel_high_forecast)

zeel_low_final_model = sm.tsa.ARIMA(
    zeel_y_low_scaled,
    order=zeel_low_best_order
)
zeel_low_final_model = zeel_low_final_model.fit()
zeel_low_forecast = zeel_low_final_model.forecast(steps=forecast_period)
zeel_low_forecast = zeel_low_forecast.reshape(-1, 1)
zeel_low_forecast = scaler.inverse_transform(zeel_low_forecast)

print("Close Forecasts:", zeel_close_forecast)
print("Open Forecasts:", zeel_open_forecast)
print("High Forecasts:", zeel_high_forecast)
print("Low Forecasts:", zeel_low_forecast)


# In[266]:


zeel_tail_50_data = zeel.tail(forecast_periods)

zeel_actual_close_prices = zeel_tail_50_data['Close'].values
zeel_actual_open_prices = zeel_tail_50_data['Open'].values
zeel_actual_high_prices = zeel_tail_50_data['High'].values
zeel_actual_low_prices = zeel_tail_50_data['Low'].values

zeel_forecast_close = zeel_close_final_model.forecast(steps=forecast_periods)
zeel_forecast_close = zeel_forecast_close.reshape(-1, 1)
zeel_forecast_close = scaler.inverse_transform(zeel_forecast_close)

zeel_forecast_open = zeel_open_final_model.forecast(steps=forecast_periods)
zeel_forecast_open = zeel_forecast_open.reshape(-1, 1)
zeel_forecast_open = scaler.inverse_transform(zeel_forecast_open)

zeel_forecast_high = zeel_high_final_model.forecast(steps=forecast_periods)
zeel_forecast_high = zeel_forecast_high.reshape(-1, 1)
zeel_forecast_high = scaler.inverse_transform(zeel_forecast_high)

zeel_forecast_low = zeel_low_final_model.forecast(steps=forecast_periods)
zeel_forecast_low = zeel_forecast_low.reshape(-1, 1)
zeel_forecast_low = scaler.inverse_transform(zeel_forecast_low)

zeel_close_mae = mean_absolute_error(zeel_actual_close_prices, zeel_forecast_close)
zeel_close_mse = mean_squared_error(zeel_actual_close_prices, zeel_forecast_close)
zeel_close_rmse = np.sqrt(zeel_close_mse)

zeel_open_mae = mean_absolute_error(zeel_actual_open_prices, zeel_forecast_open)
zeel_open_mse = mean_squared_error(zeel_actual_open_prices, zeel_forecast_open)
zeel_open_rmse = np.sqrt(zeel_open_mse)

zeel_high_mae = mean_absolute_error(zeel_actual_high_prices, zeel_forecast_high)
zeel_high_mse = mean_squared_error(zeel_actual_high_prices, zeel_forecast_high)
zeel_high_rmse = np.sqrt(zeel_high_mse)

zeel_low_mae = mean_absolute_error(zeel_actual_low_prices, zeel_forecast_low)
zeel_low_mse = mean_squared_error(zeel_actual_low_prices, zeel_forecast_low)
zeel_low_rmse = np.sqrt(zeel_low_mse)

zeel_close_mape = mean_absolute_percentage_error(zeel_actual_close_prices, zeel_forecast_close)
zeel_open_mape = mean_absolute_percentage_error(zeel_actual_open_prices, zeel_forecast_open)
zeel_high_mape = mean_absolute_percentage_error(zeel_actual_high_prices, zeel_forecast_high)
zeel_low_mape = mean_absolute_percentage_error(zeel_actual_low_prices, zeel_forecast_low)

print("Close Forecasts:", zeel_forecast_close)
print(f"Close Mean Absolute Error (MAE): {zeel_close_mae}")
print(f"Close Mean Squared Error (MSE): {zeel_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {zeel_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {zeel_close_mape}%")

print("Open Forecasts:", zeel_forecast_open)
print(f"Open Mean Absolute Error (MAE): {zeel_open_mae}")
print(f"Open Mean Squared Error (MSE): {zeel_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {zeel_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {zeel_open_mape}%")

print("High Forecasts:", zeel_forecast_high)
print(f"High Mean Absolute Error (MAE): {zeel_high_mae}")
print(f"High Mean Squared Error (MSE): {zeel_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {zeel_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {zeel_high_mape}%")

print("Low Forecasts:", zeel_forecast_low)
print(f"Low Mean Absolute Error (MAE): {zeel_low_mae}")
print(f"Low Mean Squared Error (MSE): {zeel_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {zeel_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {zeel_low_mape}")


# In[267]:


mundraport_y_close = mundraport['Close'].values
mundraport_y_open = mundraport['Open'].values
mundraport_y_high = mundraport['High'].values
mundraport_y_low = mundraport['Low'].values

mundraport_y_close_scaled = scaler.fit_transform(mundraport_y_close.reshape(-1, 1))
mundraport_y_open_scaled = scaler.fit_transform(mundraport_y_open.reshape(-1, 1))
mundraport_y_high_scaled = scaler.fit_transform(mundraport_y_high.reshape(-1, 1))
mundraport_y_low_scaled = scaler.fit_transform(mundraport_y_low.reshape(-1, 1))

mundraport_close_model = auto_arima(
    mundraport_y_close_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

mundraport_open_model = auto_arima(
    mundraport_y_open_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

mundraport_high_model = auto_arima(
    mundraport_y_high_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

mundraport_low_model = auto_arima(
    mundraport_y_low_scaled,
    seasonal=True,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_order=None,
    trace=True
)

mundraport_close_best_order = mundraport_close_model.get_params()['order']
mundraport_open_best_order = mundraport_open_model.get_params()['order']
mundraport_high_best_order = mundraport_high_model.get_params()['order']
mundraport_low_best_order = mundraport_low_model.get_params()['order']

print(f"Best ARIMA Order for Close: {mundraport_close_best_order}")
print(f"Best ARIMA Order for Open: {mundraport_open_best_order}")
print(f"Best ARIMA Order for High: {mundraport_high_best_order}")
print(f"Best ARIMA Order for Low: {mundraport_low_best_order}")

mundraport_close_final_model = sm.tsa.ARIMA(
    mundraport_y_close_scaled,
    order=mundraport_close_best_order
)
mundraport_close_final_model = mundraport_close_final_model.fit()
mundraport_close_forecast = mundraport_close_final_model.forecast(steps=forecast_period)
mundraport_close_forecast = mundraport_close_forecast.reshape(-1, 1)
mundraport_close_forecast = scaler.inverse_transform(mundraport_close_forecast)

mundraport_open_final_model = sm.tsa.ARIMA(
    mundraport_y_open_scaled,
    order=mundraport_open_best_order
)
mundraport_open_final_model = mundraport_open_final_model.fit()
mundraport_open_forecast = mundraport_open_final_model.forecast(steps=forecast_period)
mundraport_open_forecast = mundraport_open_forecast.reshape(-1, 1)
mundraport_open_forecast = scaler.inverse_transform(mundraport_open_forecast)

mundraport_high_final_model = sm.tsa.ARIMA(
    mundraport_y_high_scaled,
    order=mundraport_high_best_order
)
mundraport_high_final_model = mundraport_high_final_model.fit()
mundraport_high_forecast = mundraport_high_final_model.forecast(steps=forecast_period)
mundraport_high_forecast = mundraport_high_forecast.reshape(-1, 1)
mundraport_high_forecast = scaler.inverse_transform(mundraport_high_forecast)

mundraport_low_final_model = sm.tsa.ARIMA(
    mundraport_y_low_scaled,
    order=mundraport_low_best_order
)
mundraport_low_final_model = mundraport_low_final_model.fit()
mundraport_low_forecast = mundraport_low_final_model.forecast(steps=forecast_period)
mundraport_low_forecast = mundraport_low_forecast.reshape(-1, 1)
mundraport_low_forecast = scaler.inverse_transform(mundraport_low_forecast)

print("Close Forecasts:", mundraport_close_forecast)
print("Open Forecasts:", mundraport_open_forecast)
print("High Forecasts:", mundraport_high_forecast)
print("Low Forecasts:", mundraport_low_forecast)


# In[268]:


mundraport_tail_50_data = mundraport.tail(forecast_periods)

mundraport_actual_close_prices = mundraport_tail_50_data['Close'].values
mundraport_actual_open_prices = mundraport_tail_50_data['Open'].values
mundraport_actual_high_prices = mundraport_tail_50_data['High'].values
mundraport_actual_low_prices = mundraport_tail_50_data['Low'].values

mundraport_forecast_close = mundraport_close_final_model.forecast(steps=forecast_periods)
mundraport_forecast_close = mundraport_forecast_close.reshape(-1, 1)
mundraport_forecast_close = scaler.inverse_transform(mundraport_forecast_close)

mundraport_forecast_open = mundraport_open_final_model.forecast(steps=forecast_periods)
mundraport_forecast_open = mundraport_forecast_open.reshape(-1, 1)
mundraport_forecast_open = scaler.inverse_transform(mundraport_forecast_open)

mundraport_forecast_high = mundraport_high_final_model.forecast(steps=forecast_periods)
mundraport_forecast_high = mundraport_forecast_high.reshape(-1, 1)
mundraport_forecast_high = scaler.inverse_transform(mundraport_forecast_high)

mundraport_forecast_low = mundraport_low_final_model.forecast(steps=forecast_periods)
mundraport_forecast_low = mundraport_forecast_low.reshape(-1, 1)
mundraport_forecast_low = scaler.inverse_transform(mundraport_forecast_low)

mundraport_close_mae = mean_absolute_error(mundraport_actual_close_prices, mundraport_forecast_close)
mundraport_close_mse = mean_squared_error(mundraport_actual_close_prices, mundraport_forecast_close)
mundraport_close_rmse = np.sqrt(mundraport_close_mse)

mundraport_open_mae = mean_absolute_error(mundraport_actual_open_prices, mundraport_forecast_open)
mundraport_open_mse = mean_squared_error(mundraport_actual_open_prices, mundraport_forecast_open)
mundraport_open_rmse = np.sqrt(mundraport_open_mse)

mundraport_high_mae = mean_absolute_error(mundraport_actual_high_prices, mundraport_forecast_high)
mundraport_high_mse = mean_squared_error(mundraport_actual_high_prices, mundraport_forecast_high)
mundraport_high_rmse = np.sqrt(mundraport_high_mse)

mundraport_low_mae = mean_absolute_error(mundraport_actual_low_prices, mundraport_forecast_low)
mundraport_low_mse = mean_squared_error(mundraport_actual_low_prices, mundraport_forecast_low)
mundraport_low_rmse = np.sqrt(mundraport_low_mse)

mundraport_close_mape = mean_absolute_percentage_error(mundraport_actual_close_prices, mundraport_forecast_close)
mundraport_open_mape = mean_absolute_percentage_error(mundraport_actual_open_prices, mundraport_forecast_open)
mundraport_high_mape = mean_absolute_percentage_error(mundraport_actual_high_prices, mundraport_forecast_high)
mundraport_low_mape = mean_absolute_percentage_error(mundraport_actual_low_prices, mundraport_forecast_low)

print("Close Forecasts:", mundraport_forecast_close)
print(f"Close Mean Absolute Error (MAE): {mundraport_close_mae}")
print(f"Close Mean Squared Error (MSE): {mundraport_close_mse}")
print(f"Close Root Mean Squared Error (RMSE): {mundraport_close_rmse}")
print(f"Close Mean Absolute Percentage Error (MAPE): {mundraport_close_mape}%")

print("Open Forecasts:", mundraport_forecast_open)
print(f"Open Mean Absolute Error (MAE): {mundraport_open_mae}")
print(f"Open Mean Squared Error (MSE): {mundraport_open_mse}")
print(f"Open Root Mean Squared Error (RMSE): {mundraport_open_rmse}")
print(f"Open Mean Absolute Percentage Error (MAPE): {mundraport_open_mape}%")

print("High Forecasts:", mundraport_forecast_high)
print(f"High Mean Absolute Error (MAE): {mundraport_high_mae}")
print(f"High Mean Squared Error (MSE): {mundraport_high_mse}")
print(f"High Root Mean Squared Error (RMSE): {mundraport_high_rmse}")
print(f"High Mean Absolute Percentage Error (MAPE): {mundraport_high_mape}%")

print("Low Forecasts:", mundraport_forecast_low)
print(f"Low Mean Absolute Error (MAE): {mundraport_low_mae}")
print(f"Low Mean Squared Error (MSE): {mundraport_low_mse}")
print(f"Low Root Mean Squared Error (RMSE): {mundraport_low_rmse}")
print(f"Low Mean Absolute Percentage Error (MAPE): {mundraport_low_mape}")


# In[ ]:




