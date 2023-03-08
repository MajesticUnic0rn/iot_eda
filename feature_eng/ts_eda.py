import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import statistics as stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.arima_model import ARIMA #might not be used for this project
import ruptures as rpt

def copy_df(df:pd.DataFrame)-> pd.DataFrame:
   return df.copy(deep=False)

def time_series_ewm(input_df:pd.DataFrame,column:str)-> pd.DataFrame: 
    data_copy = copy_df(input_df)
    data_copy[column+"_ewm"] = data_copy[column].ewm(span=12, adjust=False).mean() # take a review of span and adjust 
    return data_copy

def time_series_rolling(input_df:pd.DataFrame, column:str)-> pd.DataFrame:
    data_copy = copy_df(input_df)
    data_copy[column+"_rolling"] = data_copy[column].rolling(window=12).mean()
    return data_copy

def time_series_lagged(input_df:pd.DataFrame, column:str, window:int)-> pd.DataFrame:
    data_copy = copy_df(input_df)
    data_copy[column+f"_{window}_lagged"] = data_copy[column].shift(window)
    return data_copy

def time_series_log(input_df:pd.DataFrame, column:str)-> pd.DataFrame:
    data_copy = copy_df(input_df)
    data_copy[column+"_log"] = np.log(data_copy[column])
    return data_copy

def time_series_boxcox(input_df:pd.DataFrame, column:str)-> pd.DataFrame:
    data_copy = copy_df(input_df)
    data_copy[column+"_boxcox"] = stats.boxcox(data_copy[column])[0]
    return data_copy

def time_series_seasonal_decompose(input_df:pd.DataFrame, column:str)-> pd.DataFrame:
    data_copy = copy_df(input_df)
    data_copy[column+"_seasonal_decomp_trend"] = seasonal_decompose(data_copy[column], model='additive', period=12).trend
    return data_copy

def time_series_decompose(input_df:pd.DataFrame, column:str)-> pd.DataFrame:
    data_copy = copy_df(input_df)
    data_copy[column+"_seasonal_decomp"] = seasonal_decompose(data_copy[column], model='additive', period=12).resid
    return data_copy

def time_series_difference(input_df:pd.DataFrame, column:str)-> pd.DataFrame:
    data_copy = copy_df(input_df)
    data_copy[column+"_diff"] = data_copy[column].diff()
    return data_copy

def time_series_log_difference(input_df:pd.DataFrame, column:str)-> pd.DataFrame:
    data_copy = copy_df(input_df)
    data_copy[column+"_log_difference"] = np.log(data_copy[column]).diff()
    return data_copy

def time_series_change_point(input_df:pd.DataFrame, column:str)-> pd.DataFrame:
    data_copy = copy_df(input_df)
    data_copy[column+"_pct_change"] = data_copy[column].pct_change(periods=12)
    return 
    
def time_series_change_point_model(input_df:pd.DataFrame, column:str)-> list:
    data_copy = copy_df(input_df)
    input_values=data_copy[column].values
    algo = rpt.Pelt(model="rbf").fit(input_values)
    result = algo.predict(pen=10)
    return result

def time_series_plot(input_df:pd.DataFrame, column:str)-> None:
    data_copy = copy_df(input_df)
    data_copy[column].plot()
    plt.show()
    return None

def time_series_interpolate(input_df:pd.DataFrame, column:str)-> pd.DataFrame:
    data_copy = copy_df(input_df)
    data_copy[column+"_linear_interpolate"] = data_copy[column].interpolate(method='linear')
    return data_copy

def time_series_stationarity_test(input_df:pd.DataFrame, column:str)-> None:
    data_copy = copy_df(input_df)
    result = adfuller(data_copy[column])
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    if result[1] > 0.05:
        result = "nonstationary"
    else:
        result = "stationary"
    return result

def time_series_plot_acf_pacf(input_df:pd.DataFrame, column:str)-> None:
    data_copy = copy_df(input_df)
    fig, ax = plt.subplots(2,1, figsize=(10,6))
    ax[0] = plot_acf(data_copy[column], lags=50, ax=ax[0])
    ax[1] = plot_pacf(data_copy[column], lags=50, ax=ax[1])
    plt.show()
    return None

def time_series_is_weekend(input_df:pd.DataFrame, column:str)-> pd.DataFrame:
    data_copy = copy_df(input_df)
    data_copy[column+"_is_weekend"] = data_copy[column].apply(lambda x: 1 if x.weekday() in [5,6] else 0)
    return data_copy

# ensure that date is index before submitting for this function
def create_time_features(input_df:pd.Dataframe , target=None):
    """
    Creates time series features from datetime index for modeling
    """
    input_df['date'] = input_df.index
    input_df['hour'] = input_df['date'].dt.hour
    input_df['dayofweek'] = input_df['date'].dt.dayofweek
    input_df['quarter'] = input_df['date'].dt.quarter
    input_df['month'] = input_df['date'].dt.month
    input_df['year'] = input_df['date'].dt.year
    input_df['dayofyear'] = input_df['date'].dt.dayofyear
    input_df['sin_day'] = np.sin(input_df['dayofyear'])
    input_df['cos_day'] = np.cos(input_df['dayofyear'])
    input_df['dayofmonth'] = input_df['date'].dt.day
    input_df['weekofyear'] = input_df['date'].dt.isocalendar().week
    X = input_df.drop(['date'], axis=1)
    if target:
        y = input_df[target]
        X = X.drop([target], axis=1)
        return X, y
    return X
