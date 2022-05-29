"""
    Linear regression models used to predict the demand
"""

from typing import Dict, Any
from collections import namedtuple

import numpy as np
import pandas as pd
import datetime as dt

from scipy.stats import gamma, binom, poisson

from sklearn import base
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from joblib import Parallel, delayed

TrainResult = namedtuple('result', ['name', 'estimator', 'summary'])

WORLD_SIZE = 16
T_MIN = dt.datetime(2022, 3, 1)


def distance(lat1, lon1, lat2, lon2):
    """ Get distance in km
    Copied from https://stackoverflow.com/questions/4913349
    """

    # approximate radius of earth in km
    R = 6373.0

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = R * c

    return dist


def load_and_clean_data(data_path: str):
    """ Load data and remove outliers """

    df = pd.read_csv(data_path)
    df['start_time'] = pd.to_datetime(df['start_time'], format='%Y-%m-%d %H:%M:%S')
    df['dist'] = distance(df['start_lat'], df['start_lng'], df['end_lat'], df['end_lng'])

    # Remove outliers with bad end coordinates
    cond_lat = df['end_lat'].between(df['start_lat'].min() - .05,
                                     df['start_lat'].max() + .05)
    cond_lon = df['end_lng'].between(df['start_lng'].min() - .05,
                                     df['start_lng'].max() + .05)
    df = df.loc[(cond_lat) & (cond_lon)]

    return df


class TimeExtractor(base.BaseEstimator, base.TransformerMixin):
    """ Get weekday and hour from time series """

    def fit(self, x, y=None):
        return self

    def transform(self, x: pd.DataFrame):
        data = [
            x.iloc[:, 0].dt.weekday.to_numpy(),
            x.iloc[:, 0].dt.hour.to_numpy()
        ]
        return np.stack(data).T


def get_time_cg(
        t: pd.Series,
        t_min: dt.datetime = dt.datetime(2022, 3, 1),
        t_window: int = 600
) -> pd.Series:
    """ Get coarse grained time """

    dt_seconds = (t - t_min).dt.total_seconds()

    # approx `dt` to the highest multiple of `t_window` that is <= `dt`
    dt_seconds_cg = (dt_seconds // t_window) * t_window

    t_cg = t_min + pd.to_timedelta(dt_seconds_cg, unit='seconds')

    return t_cg


def get_cat_pipe() -> Pipeline:
    """ Generator and transformer of all categorical features """

    cat_pipe = Pipeline([
        ('categorical_features', ColumnTransformer(
            transformers=[
                ('weekday_extractor', TimeExtractor(), ['t']),
            ])
         ),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    return cat_pipe


def get_num_pipe() -> Pipeline:
    """ Generator and transformer of all numerical features """

    num_pipe = Pipeline([
        ('numerical_features', ColumnTransformer(
            transformers=[
                ('linear_trend', FunctionTransformer(
                    func=lambda x: (x.iloc[:, 0] - T_MIN).dt.total_seconds().values.reshape(-1, 1),
                    check_inverse=False
                ), ['t']),
            ])
         ),
        ('scaler', StandardScaler())
    ])

    return num_pipe


def get_pipe_ride_demand() -> Pipeline:
    """ Data transformation and model training pipeline to predict ride demand

    Use a Poisson regression
    """

    cat_pipe = get_cat_pipe()
    num_pipe = get_num_pipe()

    pipe = Pipeline([
        ('features_generator', FeatureUnion(transformer_list=[
            ('cat_pipe', cat_pipe),
            ('num_pipe', num_pipe)]
        )),
        ('poisson', PoissonRegressor())
    ])

    return pipe


def train(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        parameter_grid: Dict[str, Any] = None
) -> GridSearchCV:
    """ Hyperparameter tuning of a linear regression model """

    if not parameter_grid:
        parameter_grid = {'poisson__alpha': np.logspace(-3, 1, 5)}

    # split 1: use week 1 for training and week 2 for testing
    # split 2: use week 1,2 for training and week 3 for testing
    tscv = TimeSeriesSplit(n_splits=2)

    pipe = get_pipe_ride_demand()

    gs_el_net = GridSearchCV(
        pipe,
        param_grid=parameter_grid,
        cv=tscv,
        scoring='neg_mean_poisson_deviance',
        n_jobs=-1
    )
    gs_el_net.fit(x_train, y_train)

    return gs_el_net


class RegressorFamily:
    """ Regressors for every area """

    def __init__(self, parameter_grid: Dict[str, Any] = None):
        self.estimators_ = None
        self.summary = None
        self.parameter_grid = parameter_grid

    def _fit_ts(self, x, y) -> TrainResult:
        """ Fit time series model """

        gs_el_net = train(x, y, self.parameter_grid)

        summary = dict(
            name=y.name,
            best_score=gs_el_net.best_score_,
            best_params=gs_el_net.best_params_
        )

        result = TrainResult(
            name=y.name,
            estimator=gs_el_net.best_estimator_,
            summary=summary
        )

        return result

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        """ Fit a model for every area """

        results = Parallel(n_jobs=-1)(
            delayed(self._fit_ts)(x, y.loc[:, c]) for c in y.columns
        )

        self.estimators_ = [r.estimator for r in results]

        self.summary = pd.DataFrame([r.summary for r in results])

    def predict(self, x):
        """ Prediction for every area """

        y_pred = Parallel(n_jobs=-1)(
            delayed(e.predict)(x) for e in self.estimators_
        )

        return np.stack(y_pred, axis=1)


def mle_ride_value_pdf(data: pd.DataFrame):
    """ Maximum likelihood estimation of the ride value pdf """

    mle_params = []

    for i in range(WORLD_SIZE):
        a, loc, scale = gamma.fit(
            data.loc[lambda x: x['a_i'] == i, 'ride_value'].values
        )

        mle_params.append(dict(a=a, loc=loc, scale=scale))

    return mle_params


def get_p_ride_worth(threshold: float, dist_params: dict):
    """ Get the probability that the ride value is above a given threshold """

    gamma_rv = gamma(**dist_params)

    p_ride_worth = 1 - gamma_rv.cdf(threshold)

    return p_ride_worth


def rho(m: int, n: int, threshold: float, dist_ride_value_params: dict):
    """ Get the probability that `m` or more rides out of `n` have a value
    above a certain`threshold` """

    p_worth = get_p_ride_worth(threshold, dist_ride_value_params)

    return 1 - binom(n=n, p=p_worth).cdf(m - 1e-5)


def get_prob_n_worth_rides(
        m: int, threshold: float, dist_ride_value_params: dict, lam: float
):
    """ Get the probability that there will be `n` or more of the rides with a
    value above threshold

    Parameters
    ----------
    m: number of rides; generated from a Poisson distribution with lam param
    threshold: threshold for the ride value
    dist_ride_value_params: params of the distribution generating the ride value
    lam: parameter of the Poisson distribution generating the number of rides
    """

    # distribution that generates the number of rides
    poisson_rv = poisson(mu=lam)

    # The poisson pmf has support over all natural numbers; Cut it!
    m_cut = int(poisson_rv.ppf(0.999) + 1)

    prob = sum(
        poisson_rv.pmf(n) * rho(m, n, threshold, dist_ride_value_params)
        for n in range(m, m_cut)
    )

    return prob
