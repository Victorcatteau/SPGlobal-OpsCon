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







#=============================================================================================================================
# TIME SERIES
#=============================================================================================================================





class TimeSeries(pd.Series):
    """Time series class extending the classical :class:`pandas.Series` 
    with specific time-series processing methods.

    Args:
        **kwargs: Keyword arguments from :class:`pandas.Series` such as 
            ``data`` or ``index``. Adding ``normalization=False`` will 
            disable calling :func:`self.normalize()` right after the 
            instantiation.

    """

    def __init__(self,data,name = None,*args, **kwargs):

        if "normalization" in kwargs:
            normalization = kwargs.pop("normalization")
        else:
            normalization = True

        super().__init__(data = data,*args,**kwargs)

        self.index = pd.to_datetime(self.index)

        if name is not None:
            self.name = name

        if normalization:
            self.normalize()



    #-------------------------------------------------------------------------------------
    # ANALYSIS

    def decompose(self, smoothing_factor=6.25, inplace=True):
        """Applies the Hodrick-Prescott filter from statsmodels on the 
        time series and returns its trend and cylce components.

        Args:
            smoothing_factor (float): The Hodrick-Prescott smoothing 
                parameter. A value of 1600 is suggested for quarterly 
                data. Ravn and Uhlig suggest using a value of 6.25 
                (:math:`1600/4^4`) for annual data and 129600 
                (:math:`1600 \\times 3^4`) for monthly data.
            inplace (bool): Store cycle and trend components in 
                ``self.cycle`` and ``self.trends`` instead of returning 
                a time series.

        Returns:
            Tuple of TimeSeries: Cycles and trends components, 
            respectively, if ``inplace=False``.

        """
        c, t = sm.tsa.filters.hpfilter(self, smoothing_factor)
        c, t = TimeSeries(c), TimeSeries(t)

        if inplace:
            self.cycle = c
            self.trends = t
        else:
            return c,t


    def normalize(self,inplace = True):
        """Normalizes time series data by dividing by its maximum value.

        Args: 
            inplace (bool): Store normalized series in ``self.norm`` 
                instead of returning a time series.

        Returns:
            TimeSeries: Normalized time series, if ``inplace=False``.

        """
        s = self/self.max()
        s = TimeSeries(s,normalization = False)
        if inplace:
            self.norm = s
        else:
            return s


    def compute_deltas(self,period = 1,inplace = True):
        """Computes variation ratios using the formula:

        .. math::

            \\frac{x_t - x_{t-k}}{x_{t-k}}

        Args: 
            period (int): Time interval :math:`k`.
            inplace (bool): Store ratios series in ``self.deltas`` 
                instead of returning a time series.

        Returns:
            TimeSeries: Variation ratios, if ``inplace=False``.

        """
        s = (self.diff(period) / self.shift(period)).fillna(0).replace(np.inf, 0)
        s = TimeSeries(s)
        if inplace:
            self.deltas = s
        else:
            return s



    def compute_logreturns(self,period = 1,inplace = True):
        """Computes log returns using the formula:

        .. math::

            \\log \\left( \\frac{x_t}{x_{t-k}} \\right)

        Args: 
            period (int): Time interval :math:`k`.
            inplace (bool): Store log returns series in ``self.deltas`` 
                instead of returning a time series.

        Returns:
            TimeSeries: Log returns, if ``inplace=False``.

        """
        s = self / self.shift(period)
        s = s.replace(np.inf,np.nan).replace(0,np.nan)
        s = np.log(s).fillna(0)
        s = TimeSeries(s)
        if inplace:
            self.logreturns = s
        else:
            return s



    def compute_rolling_mean(self,period = 4,inplace = True):
        """Smoothes the times series by computing a rolling mean using 
        a non-centered window of a given size.

        Args: 
            period (int): Size of the time window.
            inplace (bool): Store results in ``self.rolling_mean`` 
                instead of returning a time series.

        Returns:
            TimeSeries: Rolling means, if ``inplace=False``.

        """
        s = self.rolling(window = period).mean()
        s = TimeSeries(s)
        if inplace:
            self.rolling_mean = s
        else:
            return s


    def compute_differences(self, period=1, inplace=True):
        """Computes a differentiated series using the formula:

        .. math::

            x_t - x_{t-k}

        Args: 
            period (int): Time interval :math:`k`.
            inplace (bool): Store results in ``self.differences`` 
                instead of returning a time series.

        Returns:
            TimeSeries: Differenciated series, if ``inplace=False``.

        """
        s = self.diff(period).fillna(0)
        s = TimeSeries(s[period:])
        if inplace:
            self.differences = s
        else:
            return s


    def preprocessing(self):
        """Calls the different preprocessing methods to fill the 
        ``cycle``, ``trends``, ``deltas``, ``rolling_mean`` and 
        ``logreturns`` attributes.

        """
        self.decompose()
        self.compute_deltas()
        self.compute_rolling_mean()
        if all(self.values > 0):
            self.compute_logreturns()


    #-------------------------------------------------------------------------------------
    # PLOTTING


    def plot(self,figsize = (15,4),label = None,*args,**kwargs):
        """Overrides the inherited plotting method, to support plotly
        visualizations.

        Args:
            *args: Arguments for ``pandas.Series.plot()``.
            **kwargs: Keyword arguments for ``pandas.Series.plot()``.

        """
        if label is None: label = self.name
        return super().plot(label=label,figsize = figsize,*args,**kwargs)




    def iplot(self,dash = False,return_trace = False,layout = None,*args,**kwargs):
        """Plotly visualization

        Args:
            dash (bool): Return figure for dash applications
            *args: Arguments for ``pandas.Series.plot()``.
            **kwargs: Keyword arguments for ``pandas.Series.plot()``.
        """
        trace = go.Scatter(x = list(self.index),y = list(self.values),name = self.name)
        fig = {"data":[trace]}
        if layout is not None: fig["layout"] = layout

        if return_trace:
            return trace
        elif dash:
            return fig
        else:
            iplot(fig)




    def year_on_year(self):
        """Uses plotly to display a year-on-year comparison of the data 
        (the dates used for the x-axis being those of the latest year).

        """
        # List all different years in the index and sort them starting with the latest
        years = np.sort(np.unique(self.index.year)).tolist()
        years.reverse()

        # Default reference year is latest year
        ref_year = years[0]

        # Isolate data corresponding to reference year and generate plotly trace
        ref_data = self[self.index.year == ref_year]
        traces = []
        traces.append(go.Scatter(x=ref_data.index, y=ref_data.values, name=str(ref_year)))

        # Generate traces for all other years iteratively
        for y in years[1:]:

            # Shift data from previous year to match indices of the current year (so that the curves overlap)
            y_data = self[self.index.year == y]
            y_data.index = y_data.index + pd.DateOffset(years=ref_year-y)

            # Generate trace
            traces.append(go.Scatter(x=y_data.index, y=y_data.values, name=str(y)))

        # Plot all traces
        iplot(go.Figure(data=traces, layout=go.Layout(hovermode="x")))






    #-------------------------------------------------------------------------------------
    # CROSS CORRELATION AND LAG


    def get_cross_correlation(self, y, lag=0):
        """Computes the Pearson correlation between the time series 
        :math:`x_t` and the shifted version :math:`y_{t-k}` of a second 
        time series.
        
        Args:
            y (TimeSeries): A second time series to be shifted.
            lag (int): The lag :math:`k` to be applied to ``y``.

        Returns:
            dict: Dictionary containing the following entries:

            * 'correlation': correlation between both series, measured 
              by Pearson's r;
            * 'p-value': the p-values associated to Pearson's r, which 
              indicates the probabily to observe the same correlation 
              with non-correlated data;
            * 'lag': the value of the ``lag`` parameter.

        """
        corr, pvalue = pearsonr(self[lag+1:] , y.shift(lag)[lag+1:])
        return {'correlation': corr, 'p-value': pvalue, 'lag': lag}





    def get_best_lag(self, y, max_lag=None):
        """Computes the lag value yielding the highest cross-correlation
        (in terms of Pearson's r) when applied to a second time series.

        Args:
            y (TimeSeries): A second time series to be shifted and 
            compared to ``self``.
            max_lag (int): An optional upper bound for the output.

        Returns:
            int: The lag value maximizing the correlation.

        """
        corr = correlate(self, y)
        nb_columns = int((len(corr)-1)/2)
        if max_lag is not None:
            lag = np.argmax(corr[nb_columns:nb_columns+max_lag+1])
        else:
            lag = np.argmax(corr[nb_columns:])

        return lag




    def get_best_cross_correlation(self, y, max_lag=None):
        """Performs a search for the optimal lag between ``self`` and 
        ``y`` and returns the corresponding correlation statistics.

        Returns:
            dict: Dictionary containing the following entries:

            * 'correlation': correlation between both series after 
              applying lag, measured by Pearson's r;
            * 'p-value': the p-values associated to Pearson's r;
            * 'lag': the optimal lag.

        """
        lag = self.get_best_lag(y, max_lag=max_lag)
        return self.get_cross_correlation(y, lag)








    #===========================================================================================================
    # EMBEDDING 
    #===========================================================================================================



    def get_mono_image(self,window = 7):
        x = self.norm.as_matrix()
        x = split_array(x,window = window)
        return x


    def show_mono_image(self,window = 7):
        x = self.get_mono_image(window)
        plt.imshow(x)
        plt.title(self.name)
        plt.show()

















#===========================================================================================================
# HELPER FUNCTIONS 
#===========================================================================================================




def split_array(x,window = 7):
    x = [x[i*window:(i+1)*window] for i in range(int(len(x)/window)+1)]
    x = [y for y in x if len(y) == window]
    x = np.vstack(x)
    return x

