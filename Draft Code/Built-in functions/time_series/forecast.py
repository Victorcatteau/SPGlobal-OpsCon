#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This modules implements two useful classes, :class:`TimeSeries` and
:class:`TimeSeriesDataset`, which come with plenty of useful methods for
handling time series data:

* preprocessing (normalization, differentiating, smoothing);
* correlation analyses;
* useful visualizations (line charts, correlation matrices).

To make sure plotly graphs are displayed correctly in Jupyter notebooks,
don't forget to enable the notebook mode as follows::

    from plotly.offline import init_notebook_mode
    init_notebook_mode(connected=True)

"""



# Usual libraries
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

# Statsmodel
from statsmodels import api as sm

# Dataviz
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import seaborn as sns

# Machine Learning
from scipy.signal import correlate
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

# Custom library
from ekimetrics.time_series.series import *





#=============================================================================================================================
# FORECASTER
#=============================================================================================================================



class Forecaster(object):
    """Wrapper for time series forecasting
    """
    def __init__(self):
        pass


    def predict_prophet(self,ts,
            periods = 30,
            yearly_seasonality = "auto",
            daily_seasonality = "auto",
            apply_log = True,
            show = True,
            changepoint_prior_scale=0.05,
            **kwargs):

        """Time Series forecasting using Facebook Prophet library
        """

        # Imports
        from fbprophet import Prophet

        # Prepare the data
        df = pd.DataFrame(ts).reset_index()
        df.columns = ["ds","y"]
        if apply_log:
            df["y"] = np.log(df["y"])


        # Prepare the model
        self.prophet_model = Prophet(
            yearly_seasonality=yearly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            **kwargs)

        # Add seasonality
        self.prophet_model.add_seasonality(name='monthly', period=30, fourier_order=4)

        # Fit the model
        self.prophet_model.fit(df)



        # Prepare the predictions
        future = self.prophet_model.make_future_dataframe(periods=periods)

        # Prediction
        forecast = self.prophet_model.predict(future)


        if show:
            self.prophet_model.plot(forecast)
            plt.axvline(x = ts.index[-1],color = 'red',linestyle = "-",linewidth=2,label = "now")
            plt.legend()
            plt.title("Prophet prediction")
            plt.show()

            try:
                print(">> Seasonality and trends")
                self.prophet_model.plot_components(forecast)
                plt.show()
            except Exception as e:
                print(e)


        new_ts = forecast[["ds","yhat"]].copy().rename(columns = {"yhat":ts.name,"ds":"dates"}).set_index("dates")[ts.name]
        if apply_log:
            new_ts = np.exp(new_ts)
        new_ts = TimeSeries(new_ts)


        return forecast,new_ts


    def predict_lstm(self):
        pass


    def predict_seq2seq(self):
        pass


    def predict_arima(self):
        pass


    def predict_ml(self):
        pass




