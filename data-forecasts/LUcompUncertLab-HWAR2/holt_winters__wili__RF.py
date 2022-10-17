#mcandrew

import sys
from collections import namedtuple

import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

import argparse

from submission_times import *

from sklearn.ensemble import RandomForestRegressor as RF 
from statsmodels.tsa.arima.model import ARIMA as arima

def from_forecast_2_dist(forecast,i):
    mu    = forecast.predicted_mean[i]
    sigma = forecast.se_mean[i]

    F = lambda x: forecast.dist.cdf(x,loc=mu,scale=sigma)
    f = lambda x: forecast.dist.pdf(x,loc=mu,scale=sigma)
    P = lambda x: forecast.dist.ppf(x,loc=mu,scale=sigma)
    return F,f,P

def normalize_columns(d,cols):
    remaining_columns = set(d.columns) - set(cols)
    d = d.set_index(list(remaining_columns))

    means = d.apply( lambda x:pd.Series({"mu_hat":np.mean(x)}) ,0)
    sds   = d.apply( lambda x:pd.Series({"mu_hat":np.std(x)}) ,0)

    normed = (d-means.values)/sds.values
    normed = normed.reset_index()
    return normed

def find_next_four_weeks(d):
    from epiweeks import Week

    most_recent_time = location_specific_values.sort_values(["date"]).iloc[-1]
    epiweek = Week.fromstring(str(most_recent_time.EW))

    four_weeks = []
    for _ in range(4):
        epiweek+=1
        four_weeks.append(epiweek.week)
    return four_weeks


if __name__ == "__main__":

    #--accepts one parameter that subsets data to Location
    parser = argparse.ArgumentParser()
    parser.add_argument('--LOCATION',type=str)

    args = parser.parse_args()
    LOCATION     = args.LOCATION

    LOCATION="42"
    
    print("Running {:s}".format(LOCATION))
    
    flu = pd.read_csv("../../data-truth/truth-Incident Hospitalizations.csv")
    flu["date"] = pd.to_datetime(flu.date)

    flu_stacked = pd.read_csv("./stacked_influenza_data.csv")
    
    flu2022 = flu_stacked.loc[flu_stacked.date > "2021-10-01"]

    location_specific_values = flu2022.loc[flu2022.location==LOCATION]
    location_specific_values = location_specific_values.sort_values(["date"])

    #--prepare training
    training = flu2022

    def from_training2lag(training):
        training = training.sort_values(["date"])

        lags=4
        y = training["value"]

        #--lag one ys
        lag_YS = {"lag1":y.shift(1)}
        for _ in np.arange(2,lags+1):
            lag_YS["lag{:d}".format(_)] = y.shift(_)
        lag_YS = pd.DataFrame(lag_YS)

        lag_YS["week"] = training.week
        lag_YS["date"] = training["date"]
        return lag_YS
        
    lags_all_y = training.groupby(["location"]).apply(from_training2lag).reset_index()

    training = training.merge( lags_all_y, on = ["location","week","date"] )

    #--remove beginning NAs from the lag
    orig_training = training
    training = training.dropna()

    #--prepare X
    wili_cols = [x for x in training.columns if 'wili' in x]
    lag_cols  = [x for x in training.columns if 'lag' in x]
    
    X  = training[ wili_cols + lag_cols ]

    #--prepare ys
    ys = training.value.values 
    
    mu_yhat = ys.mean()
    sd_yhat = ys.std()
    c_ys = (ys - mu_yhat)/sd_yhat

    mu_X = X.mean(0)
    sd_X = X.std(0)
    c_X  = (X - mu_X)/sd_X
   
    next_four_weeks = find_next_four_weeks(location_specific_values)
    
    #--detrend by fitting wili to hosps
    model   = RF().fit(c_X,c_ys)

    #--inference for single state
    training = location_specific_values

    lags_all_y = training.groupby(["location"]).apply(from_training2lag).reset_index()
    training = training.merge( lags_all_y, on = ["location","week","date"] )

    #--remove beginning NAs from the lag
    orig_training = training
    training = training.dropna()

    #--prepare X
    wili_cols = [x for x in training.columns if 'wili' in x]
    lag_cols  = [x for x in training.columns if 'lag' in x]
    
    X  = training[ wili_cols + lag_cols ]

    #--prepare ys
    ys = training.value.values
    c_ys = (ys - mu_yhat)/sd_yhat
    c_X  = (X - mu_X)/sd_X
    
    y_infer = model.predict(c_X)*sd_yhat + mu_yhat

    #--prepare prediction X
    Xstar    = orig_training.loc[ orig_training.week == next_four_weeks[0], wili_cols + lag_cols  ]

    prediction_ys = list(ys[-4:])
    for step in [1,2,3,4]:
        for _ in np.arange(1,4+1):
            Xstar["lag{:d}".format(_)] = prediction_ys[-_]

        c_Xstar  = (Xstar-mu_X)/sd_X

        y_forecast = model.predict(c_Xstar)
        prediction_ys.append(y_forecast*sd_yhat + mu_yhat)

    y_forecast = [float(y) for y in prediction_ys[-4:]]
        
    
    xs  = np.arange(0,len(location_specific_values))
    T   = len(ys)

    #--compute residuals from Holt Winters procedure
    resids = ys - y_infer
    
    #--standardize the residuals by subtracting mean and diviiding by std 
    params = namedtuple("params","avg sd")
    stand__params   = params( avg = resids.mean(), sd = resids.std() )
    centered_resids = (resids - stand__params.avg)/stand__params.sd

    #--arima
    model = arima(centered_resids, order = (1,1,1))
    model = model.fit()

    #--build quantile forecasts
    quantiles = {"horizon":[], "quantile":[], "values":[]}
    QUANTILES = np.round([0.01,0.025] + list(np.arange(0.05,0.95+0.05,0.05)) + [0.975,0.99],3)

    for horizon in range(4):
        horizon+=1
        quantiles["horizon"].extend( [horizon]*len(QUANTILES) )
            
        quantiles["quantile"].extend(QUANTILES)

        F,f,P = from_forecast_2_dist( model.get_forecast(horizon), -1 )

        #--need to uncenter these values
        quantiles["values"].extend( P(QUANTILES)*stand__params.sd + stand__params.avg )
    quantiles = pd.DataFrame(quantiles)

    random_forecast_trends = pd.DataFrame({"horizon":[1,2,3,4],"trend":y_forecast } )

    quantiles = quantiles.merge( random_forecast_trends, on = ["horizon"] )
    quantiles["value"] = quantiles["values"] + quantiles["trend"]
    
    forecast = quantiles[["horizon","quantile","value"]]
    forecast.loc[forecast["value"]<0,"value"] = 0

    #--compute all time information for forecast submission
    number_of_days_until_monday = next_monday()
    monday = next_monday(True)
    
    next_sat = next_saturday_after_monday_submission( number_of_days_until_monday )
    target_end_dates = collect_target_end_dates(next_sat)

    TEDS = pd.DataFrame({ "horizon":[1,2,3,4]
                          , "target_end_date":target_end_dates
                          , "target": ["{:d} wk ahead inc flu hosp".format(n+1) for n in range(4) ]  })
    forecast = forecast.merge( TEDS, on = ["horizon"] )
    
    forecast["location"]      = LOCATION
    forecast["type"]          = "quantile"
    forecast["forecast_date"] = monday

    
    #--format quantile to three decimals
    forecast['quantile'] = ["{:0.3f}".format(q) for q in forecast["quantile"]]

    #--format values to three decimals
    forecast['value'] = ["{:0.3f}".format(q) for q in forecast["value"]]
    
    #--output data
    forecast.to_csv("./forecasts/location__{:s}.csv".format(LOCATION),index=False)
