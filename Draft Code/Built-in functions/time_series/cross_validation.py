#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bla bla bla

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
from sklearn.model_selection import TimeSeriesSplit

# Custom library
from ekimetrics.time_series.series import *
from ekimetrics.time_series.dataset import *



def train_test_split(ds, n_splits=3):
    """Just a reimplementation of scikit-learn's TimeSeriesSplit that
    outputs a TimeSeriesDataset instead of numpy arrays.
    """
   
    df = ds.build_dataframe()
    df.sort_index(inplace=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_index, test_index in tscv.split(df.as_matrix()):

        df_train, df_test = df.iloc[train_index, :], df.iloc[test_index]

        ds_train = TimeSeriesDataset(dataframe=df_train, target=ds.target)
        ds_test = TimeSeriesDataset(dataframe=df_test, target=ds.target)

        yield (ds_train, ds_test)


def sliding_split(ds, train_size, test_size, step=1):
    """A cross validation with fixed-size, sliding train & test sets.
    Is meant to test stability over time.
    """
   
    df = ds.build_dataframe()
    df.sort_index(inplace=True)

    i = 0
    while i+train_size+test_size <= len(df):

        df_train = df.iloc[i:i+train_size, :]
        df_test = df.iloc[i+train_size:i+train_size+test_size, :]

        ds_train = TimeSeriesDataset(dataframe=df_train, target=ds.target)
        ds_test = TimeSeriesDataset(dataframe=df_test, target=ds.target)

        yield (ds_train, ds_test)

        i += step


def cv_curve(ds, model, metric, splits, intermediary_plots=True):
    """Evaluates a machine learning model with cross validation iterating over
    a given set of splits, and plots the resulting error/learning curve.
    """

    perf_train = []
    perf_test = []

    # If no intermediary plots are to be drawn, display tqdm progressbar instead
    splits = splits if intermediary_plots else tqdm(splits)

    for ds_train, ds_test in splits:

        train_dates, test_dates = ds_train.index, ds_test.index

        X_train, y_train = ds_train.build_Xy(as_matrix=True)
        X_test, y_test = ds_test.build_Xy(as_matrix=True)

        model.fit(X_train, y_train)
        y_hat_train = model.predict(X_train)
        y_hat_test = model.predict(X_test)

        perf_train.append(metric(y_hat_train, y_train))
        perf_test.append(metric(y_hat_test, y_test))

        if intermediary_plots:
            ds[ds.target].plot(label='Ground trouth')
            plt.plot(train_dates, y_hat_train, label=f'Training set predictions ({perf_train[-1]:.2})')
            plt.plot(test_dates, y_hat_test, label=f'Test set predictions ({perf_test[-1]:.2})')
            plt.legend()
            plt.show()

    plt.plot(perf_train, label='Training set error')
    plt.plot(perf_test, label='Test set error')
    plt.legend()
    plt.show()
