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
# TIME SERIES DATASET
#=============================================================================================================================



class TimeSeriesDataset(object):
    """TimeSeriesDataset representation, wrapping multiple time series
    together.

    Args:
        data (list): A list of TimeSeries objects.
        dataframe (pandas.DataFrame): A datetime-indexed data frame.
        target (str): The name of the series/column to be considered as 
            the target variable in computing correlations.

    """

    def __init__(self, data=None, dataframe=None, target=None,file_path = None,raise_index_mismatch = False):
        
        self.target = target

        # Input is a list of time series
        if data is not None:
            if isinstance(data,pd.DataFrame):
                self.parse_data(data)
                self.columns = data.columns
            else:
                self.data = data

        # Input is a file
        elif file_path is not None:
            self.data = self.load_data_from_file(file_path)

        # Input is a dataframe
        else:
            raise Exception("Input is not correct")



        self._assert_correct_entry(raise_index_mismatch)
        self.index = self.data[0].index






    #===========================================================================================================
    # IO 
    #===========================================================================================================


    #-------------------------------------------------------------------------------------
    # LOADING DATA

    def parse_data(self, data):
        """Parses a `pandas.DataFrame` to build a list of `TimeSeries` 
        that is stored into ``self.data``.

        Args:
            data (pandas.DataFrame): The dataframe to parse.

        """
        list_of_ts = [TimeSeries(data.iloc[:, i]) for i in range(len(data.columns))]
        self.data = list_of_ts


    def load_data_from_file(self,file_path):

        file_path = str(file_path)

        if file_path.endswith(".json"):
            data = self.load_data_from_json_file(file_path)
        else:
            pass

        return data


    def load_data_from_json_file(self,file_path):
        """Load json data file
        With a json file storing for each key a list of time series values
        """
        json_data = json.loads(open(file_path,"r").read())
        data = []

        for name,values in json_data.items():
            ts = TimeSeries(values,name)
            data.append(ts)

        return data

        

    #-------------------------------------------------------------------------------------
    # SAVING DATA


    def save_as_excel(self):
        pass


    def save_as_json(self):
        pass


    def save_as_pickle(self):
        pass









    #===========================================================================================================
    # OPERATORS AND ABSTRACT METHODS 
    #===========================================================================================================



    #-------------------------------------------------------------------------------------
    # OPERATORS

    def __iter__(self):
        """
        Iterator
        """
        return iter(self.data)

    def __getitem__(self, key,raise_not_found = True):
        """
        Item getter
        """
        if type(key) == int:
            return self.data[key]
        else:
            if type(key) != list:
                possible_variables = self.get_names()
                if key not in possible_variables:
                    if self.get_variables([key]) is not None:
                        key = self.get_variables([key])[0]

                time_series = [ts for ts in self.data if ts.name == key]
                if len(time_series) > 1:
                    raise ValueError("Multiple time series with the given key")
                elif len(time_series) == 0:
                    if raise_not_found:
                        raise ValueError("No time series with this name")
                    else:
                        return None
                else:
                    return time_series[0]

            else:
                data = []
                for k in key:
                    ts = self.__getitem__(k,raise_not_found = False)
                    if ts is not None:
                        data.append(ts)

                return TimeSeriesDataset(data = data)



    def __add__(self,ts):
        """
        Safely adding new time series
        """
        if type(ts) != TimeSeries:
            raise ValueError("Type of the data must be a Time Series")
        else:
            self.data.append(ts)


    def __repr__(self):
        """
        String representation
        """
        return """{} time series in the dataset""".format(len(self))



    def _repr_html_(self):
        """
        Notebook representation (same as the dataframe made from the dataset)
        """
        return self.build_dataframe()._repr_html_()



    def __len__(self):
        """
        Number of data points in the time series
        """
        return len(self.data)







    #-------------------------------------------------------------------------------------
    # ABSTRACT METHODS




    def _assert_correct_entry(self,raise_index_mismatch = False):
        """
        Check if the input data provided have the same index
        """

        indexes = [ts.index for ts in self]
        common_index = pd.Series([date for index in indexes for date in index]).value_counts().sort_index()
        common_index.index = pd.to_datetime(common_index.index)

        if len(common_index.unique()) > 1:

            # Choosing exception 
            if raise_index_mismatch:
                raise Exception("Index mismatch between the time series")
            else:
                print(">> Index mismatch between the time series")
                print("Finding the best compromise as a common index of dates")

            # Find common set dates shared
            count = common_index.value_counts()
            count /= len(common_index)
            count.plot(kind = "bar",title = "% dates shared on the {} time series".format(len(indexes)))
            plt.xlabel("Number of time series sharing a common set of dates")
            plt.ylabel("% of all dates shared on {} possible dates".format(len(common_index)))
            plt.show()

            # If the number of dates shared is sufficient
            if count.iloc[0] > 0.5:

                # Find the best compromise as a common index of dates
                best_compromise = count.index[0]
                index_compromise = common_index.loc[common_index == best_compromise]
                min_compromise = index_compromise.index[0]
                max_compromise = index_compromise.index[-1]
                print(">> The best common index of dates span from {} to {}".format(min_compromise.date(),max_compromise.date()))

                # Plot the compromise time span 
                common_index.plot(figsize = (15,4),title = "# of time series with values on a given date (on {} total time series)".format(len(indexes)))
                plt.xlabel("Date")
                plt.ylabel("Number of time series with values")
                plt.axvline(x=min_compromise, linewidth=2, color = 'k',linestyle = "--")
                plt.axvline(x=max_compromise, linewidth=2, color = 'k',linestyle = "--")
                plt.show()

                # Set all indexes to the common compromise index
                data = []
                removed = 0
                for ts in self:
                    try:
                        ts = TimeSeries(ts.loc[index_compromise.index])
                        data.append(ts)
                    except Exception as e:
                        removed += 1

                print(">> Removed {} time series with almost no data".format(removed))
                self.data = data

            else:
                raise Exception("The best common index of dates is only on {} of all possible dates, clean your data first".format(count.iloc[0]))





        else:
            pass
        






    def get_variables(self,variables = None,exclude = None):

        all_variables = self.get_names()
        if variables is None:
            variables = all_variables

        def extract_match(x,occurences):
            match = []
            for occurence in occurences :
                if x in occurence:
                    match.append(occurence)
            return match

        match = []
        for x in variables:
            match.extend(extract_match(x,all_variables))
        variables = match

        if exclude is not None:
            exclude = [extract_match(x,all_variables) for x in exclude]
            variables = [x for x in variables if x not in exclude]

        return variables



    def get_names(self):
        return [ts.name for ts in self]
















    #===========================================================================================================
    # DATA MANIPULATION
    #===========================================================================================================


    def head(self, rows=5):
        """ Typical pandas head method, adapted to `TimeSeriesDataset`.

        Args:
            rows (int): Number of head rows to display.

        Returns:
            Pandas DataFrame: Dataset containg only the first few rows.

        """
        return self.build_dataframe().head(rows)


    def tail(self, rows=5):
        """ Typical pandas tail method, adapted to `TimeSeriesDataset`.

        Args:
            rows (int): Number of tail rows to display.

        Returns:
            TimeSeriesDataset: Dataset containg only the last few rows.

        """
        return self.build_dataframe().tail(rows)






    #-------------------------------------------------------------------------------------
    # MANIPULATION

    def build_dataframe(self,variables = None,exclude = None):
        """Create a pandas dataframe from all the time series in the 
        dataset.

        Args:
            variables (list): Names of time series to be kept.
            exclude (list): Names of time series to be excluded.

        Returns:
            pandas.DataFrame: A dataframe containing all the requested 
            times series as its columns.

        """
        df = pd.concat(self.data,axis = 1)
        if variables is not None or exclude is not None:
            df = df[self.get_variables(variables,exclude)]
        return df







    #===========================================================================================================
    # PREPROCESSING 
    #===========================================================================================================




    def drop(self,variables):
        """Removes time series from the dataset.

        Args:
            variables (`str` or `list`): Name (or list of names) of the 
                time series to be removes.

        """
        if type(variables) != list: variables = [variables]
        variables = self.get_variables(variables)

        self.data = [ts for ts in self if ts.name not in variables]




    def preprocessing(self):
        """Calls the ``preprocessing()`` method on each time series.

        """
        for ts in self:
            ts.preprocessing()



    def get_trends(self,exempt = [],**kwargs):
        """Builds a new dataset with only the trends.

        Args:
            exempt (list): Names of time series to be excluded.
            **kwargs: Keyword arguments for the ``decompose()`` method 
                of `TimeSeries`, such as ``smoothing_factor``.

        Returns:
            TimeSeriesDataset: Trends dataset.

        """
        trends = [ts.decompose(inplace = False,**kwargs)[1] if ts.name not in exempt else ts for ts in self]
        return TimeSeriesDataset(trends, target=self.target)


    def get_cycles(self,exempt = [],**kwargs):
        """Builds a new dataset with only the cycles.

        Args:
            exempt (list): Names of time series to be excluded.
            **kwargs: Keyword arguments for the ``decompose()`` method 
                of `TimeSeries`, such as ``smoothing_factor``.

        Returns:
            TimeSeriesDataset: Cycles dataset.

        """
        cycles = [ts.decompose(inplace = False,**kwargs)[0]  if ts.name not in exempt else ts for ts in self]
        return TimeSeriesDataset(cycles, target=self.target)



    def get_deltas(self,exempt = [],**kwargs):
        """Builds a new dataset with only the variation ratios.

        Args:
            exempt (list): Names of time series to be excluded.
            **kwargs: Keyword arguments for the ``compute_deltas()`` 
                method of `TimeSeries`, such as ``period``.

        Returns:
            TimeSeriesDataset: Variation ratios dataset.

        """
        deltas = [ts.compute_deltas(inplace = False,**kwargs) if ts.name not in exempt else ts for ts in self]
        return TimeSeriesDataset(deltas, target=self.target)



    def get_logreturns(self,exempt = [],**kwargs):
        """"Builds a new dataset with only the log returns.

        Args:
            exempt (list): Names of time series to be excluded.
            **kwargs: Keyword arguments for the ``compute_logreturns()`` 
                method of `TimeSeries`, such as ``period``.

        Returns:
            TimeSeriesDataset: Log returns dataset.

        """
        logreturns = [ts.compute_logreturns(inplace = False,**kwargs) if ts.name not in exempt else ts for ts in self]
        return TimeSeriesDataset(logreturns, target=self.target)


    def get_rolling_mean(self,exempt = [],**kwargs):
        """"Builds a new dataset with only the rolling means.

        Args:
            exempt (list): Names of time series to be excluded.
            **kwargs: Keyword arguments for the 
                ``compute_rolling_mean()`` method of `TimeSeries`, such 
                as ``period``.

        Returns:
            TimeSeriesDataset: Rolling means dataset.

        """
        rolling_mean = [ts.compute_rolling_mean(inplace = False,**kwargs) if ts.name not in exempt else ts for ts in self]
        return TimeSeriesDataset(rolling_mean, target=self.target)


    def get_differences(self,exempt = [],**kwargs):
        """"Builds a new dataset with only the differentiated series.

        Args:
            exempt (list): Names of time series to be excluded.
            **kwargs: Keyword arguments for the 
                ``compute_differences()`` method of `TimeSeries`, such 
                as ``period``.

        Returns:
            TimeSeriesDataset: Differentiated dataset.

        """
        differences = [ts.compute_differences(inplace = False,**kwargs) if ts.name not in exempt else ts for ts in self]
        return TimeSeriesDataset(differences, target=self.target)


    def get_normalized(self,exempt = [],**kwargs):
        """"Builds a new dataset with only the normalized time series.

        Args:
            exempt (list): Names of time series to be excluded.
            **kwargs: Keyword arguments for the ``normalize()`` method 
                of `TimeSeries`, such as ``inplace``.

        Returns:
            TimeSeriesDataset: Rolling means dataset.

        """
        norm = [ts.normalize(inplace = False) if ts.name not in exempt else ts for ts in self]
        return TimeSeriesDataset(norm, target=self.target)










    def get_on_pipeline(self,pipeline,exempt = []):
        """"Builds a new dataset using a preprocessing pipeline.

        Args:
            pipeline (list): List of preprocessing steps to be applied 
                in sequence; accepted values include:

                * 'normalized'
                * 'rolling_mean'
                * 'logreturns'
                * 'deltas'
                * 'cycles'
                * 'trends'

            exempt (list): Names of time series to be excluded.

        Returns:
            TimeSeriesDataset: Final dataset after preprocessing.

        """
        assert all([x in ["normalized","rolling_mean","logreturns","deltas","cycles","trends"] for x in pipeline])

        ts = self

        for el in pipeline:
            print(el)
            ts = getattr(ts,"get_{}".format(el))()

        return ts














    #===========================================================================================================
    # VISUALIZATIONS
    #===========================================================================================================



    def plot(self,figsize = (15,4),*args,**kwargs):
        """Plot all the time series in the dataset with matplotlib.
        Args:
            *args: Arguments for ``pandas.Series.plot()``.
            **kwargs: Keyword arguments for ``pandas.Series.plot()``.
        """

        for ts in self:
            ts.plot(figsize = figsize,*args,**kwargs)

        plt.legend()
        plt.show()




    def iplot(self,dash = False,layout = None,*args,**kwargs):
        """Plot all the time series in the dataset with plotly 
        Args:
            layout (dict): Specific plotly layout to be used.
            *args: Arguments for ``pandas.Series.plot()``.
            **kwargs: Keyword arguments for ``pandas.Series.plot()``.
        """
        traces = []
        for ts in self:
            traces.append(ts.iplot(dash = dash,layout = layout,return_trace = True,*args,**kwargs))

        fig = {"data":traces}

        if layout is not None:
            fig["layout"] = layout


        if dash:
            return fig
        else:
            iplot(fig)





    def plot_correlations(self,variables=None, exclude=None, figsize=(15,15), cmap=plt.cm.Blues, annot=True, metric = 'pearson',**kwargs):
        """Plots the correlation matrix between all the time series with
        `seaborn.clustermap()`.

        Args:
            variables (list): List of variables to be considered.
            exclude (list): List of variables to be excluded.
            figsize: Matplotlib figure size.
            cmap: The mapping from data values to color space.
            annot (bool): If True, write the data value in each cell. If
                an array-like with the same shape as data, then use this
                to annotate the heatmap instead of the raw data.
            **kwargs: Keyword arguments for `sns.clustermap()` 

        """
        df = self.build_dataframe(variables, exclude)
        sns.clustermap(df.corr(method = metric).fillna(0.0), figsize=figsize, cmap=cmap, annot=annot,**kwargs)
        plt.show()










    #===========================================================================================================
    # MACHINE LEARNING
    #===========================================================================================================




    def build_Xy(self,target = None,as_matrix = False):
        """Builds a suitable dataset for machine learning.

        Args:
            target (str): Name of a series to be set as target, if not
                already done at the instantiation.
            as_matrix (bool): If ``True``, return numpy arrays instead 
                of pandas dataframes.

        Returns:
            Tuple of `pandas.DataFrame` or `numpy.ndarray`: Two matrices
                ``X`` and ``y`` as required by most ML libraries.

        """

        # Map the target variable
        if target is not None:
            self.target = target
        elif self.target is None:
            raise Exception("You have to provide a target")
        else:
            pass

        # Build the dataframe and select the right columns
        df = self.build_dataframe()
        df[df < 0] = 0.0
        X_cols = [col for col in df.columns if col != self.target]
        X = df[X_cols]
        y = df[self.target]

        # Convert to numpy matrices
        if as_matrix:
            X = X.as_matrix()
            y = y.as_matrix()

        return X,y











    #===========================================================================================================
    # CORRELATIONS ANALYSIS
    #===========================================================================================================



    def get_best_cross_correlations(self, max_lag=None, sort='correlation'):
        """Computes the optimal lags and corresponding correlations.

        Args:
            max_lag (int): Maximum lag allowed.
            sort (str): the key used to sort the results; accepted 
                values: 'alphabetical', 'correlation' (default), 'lag'.

        Returns:
            List of dictionaries containing variable names ('name'), 
            best lags ('lag'), lagged correlations ('correlation') and 
            p-values ('pvalue').

        """
        dict_sort = {
            'alphabetical': 'name',
            'correlation': 'correlation',
            'lag': 'lag'
        }

        key_sort = dict_sort.get(sort, 'correlation')

        target = self[self.target]
        list_columns = self.get_names()

        best_cross_corr = [
            target.get_best_cross_correlation(self[col], max_lag=max_lag) \
            for col in list_columns
        ]

        for name, results in zip(self.get_names(), best_cross_corr):
            results.update({"name": name})

        return sorted(
            best_cross_corr,
            key=lambda x: x[key_sort],
            reverse=True
        )






    def plot_lag(self , width=1200, height=900, margin_left=320):
        """Plot the optimal lag w.r.t. the target series for each series
        in the dataset using plotly.

        Args:
            width (int): Figure width.
            height (int): Figure height.
            margin_left (int): Left margin, can be adjusted to avoid 
                cropped variable names.

        """
        best_corr = self.get_best_cross_correlations(sort='alphabetical')
        x = [var['lag'] for var in best_corr
             if var['name']!=self[self.target].name]
        y = [var['name'] for var in best_corr
             if var['name']!=self[self.target].name]

        layout = go.Layout(
            height=height,
            width=width,
            margin=go.Margin(
                l=margin_left
            ),
        )

        data = [go.Bar(
                x=x,
                y=y,
                orientation = 'h',
                text=x,
                textposition = 'auto',
                textfont=dict(
                    size=12,
                    color='white'
                )
        )]

        fig = go.Figure(data=data, layout=layout)
        iplot(fig)






    def shift(self, lags=None, inplace=False, limit=10, verbose=False):
        """Computes optimal lags w.r.t. the target series and shifts 
        all variables (except target) accordingly. Optionally uses a
        dictionary of manually pre-defined lags as input instead.
        
        Args:
            lags (`int` or `dict`): A single integer to be applied as
                lag to all variables, or a dictionary of lags with 
                variable names as keys and desired lags as values. 
                Not listed variables will be assigned the optimal lags 
                computed via :func:`get_best_cross_correlations()`.
            inplace (bool): Overwrite current dataset or not.
            limit (int): Maximum accepted lag.
            verbose (bool): Print lag values while shifting.

        Returns:
            TimeSeriesDataset: Shifted dataset, either inside of self if
            called with inplace=True or as a new instance otherwise.

        """
        if lags is None:
            lags = {}
        elif type(lags) == int:
            lags = {var_name: lags for var_name in self.get_names()}
        else:
            assert type(lags) == dict

        # Computes difference between x_{t} and x_{t-1}
        diffs = [
            self[col].compute_differences(inplace=False)
            for col in self.get_names()
        ]
        df_diff = TimeSeriesDataset(diffs, target=self.target)

        # Computes best lags
        best_corr = df_diff.get_best_cross_correlations(max_lag=limit)

        # Shifts variables with best lag or pre-defined lag if provided
        shifts = []
        for var in best_corr:
            var_name = var['name']
            var_lag = lags[var_name] if var_name in lags else var['lag']
            if var_name!=self.target:
                shifts += [TimeSeries(self[var_name].shift(var_lag)\
                                                    .fillna(0)\
                                                    .replace(np.inf, 0))]
                if verbose:
                    print('Shifted "{0}" by {1} time steps'\
                        .format(var_name, var_lag))
            else:
                shifts += [TimeSeries(self[var_name])]

        if inplace:
            self = TimeSeriesDataset(shifts, target=self.target)
        else:
            return TimeSeriesDataset(shifts, target=self.target)






    #===========================================================================================================
    # EMBEDDINGS
    #===========================================================================================================




    def build_mono_image_dataset(self,window = 7):
        """Build mono image dataset
        """
        images = [np.expand_dims(ts.get_mono_image(window),axis = 0) for ts in self]
        images = np.vstack(images)
        return images



    def show_mono_image_dataset(self,figsize = (20,6)):
        plt.figure(figsize=figsize)
        image = self.build_mono_image_dataset(window = 1)
        image = image.reshape(*image.shape[:-1])
        plt.imshow(image.T)
        plt.xlabel("time series")
        plt.ylabel("t")
        plt.show()












#===========================================================================================================
# HELPER FUNCTIONS 
#===========================================================================================================




def compute_correlations(data, start, stop, target='sales'):
    series = TimeSeriesDataset(
        data=data[start:stop].diff().dropna(),
        target=target
    )

    result = pd.DataFrame(
        [{'event': e['name'], 'lagged correlation': e['correlation'],
          'lag': e['lag']} for e in series.get_best_cross_correlations()]
    ).set_index('event')

    result['immediate correlation'] = \
        series.build_dataframe()[start:stop].corr()[target]

    return result


def get_correlations(data, start='2017-01-01', stop='2017-10-31', target='sales'):
    if isinstance(data, pd.DataFrame):
        return compute_correlations(data, start, stop, target)
    elif isinstance(data, dict):
        return pd.concat(
            [compute_correlations(data[key], start, stop, target) for key in data],
            axis=1,
            keys=[key for key in data]
        )


def compute_rolling_correlations(data, n_weeks, target='sales'):
    first_week = data.index[0]
    last_week = data.index[-1]

    total_weeks = int((last_week - first_week).days / 7) - n_weeks

    correlations = []
    weeks = []
    for i in tqdm(range(total_weeks)):
        week = first_week + timedelta(days=7*i)
        correlation = compute_correlations(data, start=week, stop=(week + timedelta(days=7*n_weeks)), target=target)

        correlations.append(correlation)
        weeks.append(week)

    return pd.concat(correlations, keys=weeks)


def get_rolling_correlations(data, n_weeks, target='sales'):
    if isinstance(data, pd.DataFrame):
        return compute_rolling_correlations(data, n_weeks, target)
    elif isinstance(data, dict):
        return pd.concat(
            [compute_rolling_correlations(data[key], n_weeks, target) for key in data],
            axis=1,
            keys=[key for key in data]
        )






