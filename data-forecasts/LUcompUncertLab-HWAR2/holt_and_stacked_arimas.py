#mcandrew

import sys
import numpy as np
import pandas as pd

import argparse

from submission_times import *

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA as arima

from collections import namedtuple

import itertools

def CRPS(model,y):
    mu    = model.predicted_mean
    sigma = model.se_mean

    F = model.dist.cdf
    f = model.dist.pdf

    return y*(2*F(y)-1) + 2*f(y) - np.pi**(-0.5)

def from_forecast_2_dist(forecast,i):
    mu    = forecast.predicted_mean[i]
    sigma = forecast.se_mean[i]

    F = lambda x: forecast.dist.cdf(x,loc=mu,scale=sigma)
    f = lambda x: forecast.dist.pdf(x,loc=mu,scale=sigma)
    P = lambda x: forecast.dist.ppf(x,loc=mu,scale=sigma)
    return F,f,P

def produce_LOO_matrix(ys,CUT,L):
    LOO = {"P":[],"D":[], "Q":[],"model":[],"cut":[],"CRPS":[],"truth":[],"muhat":[]}
    for model_number,(P,D,Q) in enumerate(itertools.product([0,1,2,3],[0,1,2],[0,1])):
        for cut in np.arange(CUT,len(ys)-L):
            #--test/train
            train = ys[:cut]
            test  = ys[cut+L]

            #--fit model
            model = arima(endog = train, order = (P,D,Q))
            model = model.fit()

            #--eval model
            crps = CRPS(model.get_forecast(),test)

            LOO["P"].append( P )
            LOO["D"].append( D )
            LOO["Q"].append( Q )
            LOO["model"].append(model_number)

            LOO["cut"].append(cut)
            LOO["CRPS"].append(crps)

            predictions = [float(x) for x in model.get_forecast(4).predicted_mean]
            LOO["muhat"].append( predictions[L] )

        LOO["truth"].extend( ys[CUT+L:] )
    LOO = pd.DataFrame(LOO)
    return LOO

def L2(w,X,y):
    w = np.array(w).reshape(-1,1)
    return float(sum( (y-X.dot(w))**2 ))


if __name__ == "__main__":

    #--accepts one parameter that subsets data to Location
    parser = argparse.ArgumentParser()
    parser.add_argument('--LOCATION',type=str)

    args = parser.parse_args()
    LOCATION     = args.LOCATION
 
    flu = pd.read_csv("../../data-truth/truth-Incident Hospitalizations.csv")
    flu["date"] = pd.to_datetime(flu.date)
    
    flu2022 = flu.loc[flu.date > "2021-10-01"]

    location_specific_values = flu2022.loc[flu2022.location==LOCATION]

    #--detrend using the Holt Winters procedure
    ys = location_specific_values.value.values
    xs = np.arange(0,len(location_specific_values))
    T = len(ys)

    def fit_HW(ys):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing( ys , trend = 'add')
        model = model.fit()

        y_infer, y_forecast  = model.fittedvalues,model.forecast(4)
        return y_infer, y_forecast
    y_infer, y_forecast = fit_HW(ys)

    #--compute residuals from Holt Winters procedure
    resids = ys - y_infer
    
    #--standardize the residuals by subtracting mean and diviiding by std 
    params = namedtuple("params","avg sd")
    stand__params   = params( avg = resids.mean(), sd = resids.std() )
    centered_resids = (resids - stand__params.avg)/stand__params.sd
    
    #--stacked inverse
    W = {"horizon":[], "model":[], "weights":[]}
    for L in np.arange(0,3+1):
        LOO = produce_LOO_matrix(centered_resids,CUT=10,L=L)
        
        LOO_wide   = pd.pivot_table( index=["cut"], columns = ["P","D","Q"], values = ["muhat"] ,data=LOO )
        truth_wide = pd.pivot_table( index=["cut"], values = ["truth"] ,data=LOO )

        num_models = LOO_wide.shape[-1]
    
        X = LOO_wide.to_numpy()
        y = truth_wide.to_numpy()

        from scipy.optimize import Bounds
        from scipy.optimize import minimize

        bounds = Bounds([0]*num_models, [1]*num_models)
        eq_cons = {'type': 'eq'
                   ,'fun' : lambda x: sum(x)-1
                   }
        res = minimize(lambda w: L2(w,X,y) , x0=np.array([1./num_models]*num_models).reshape(-1,1)
                       ,method='SLSQP'
                       ,constraints=[eq_cons]
                       ,options={'ftol': 1e-9, 'disp': True}
                       ,bounds = bounds
                       )
        weights = res.x
        W["weights"].extend(weights)
        W["horizon"].extend([L+1]*num_models)
        W["model"].extend( np.arange(0,num_models) )
    W = pd.DataFrame(W)

    #--build quantile forecasts
    quantiles = {"horizon":[], "model":[], "quantile":[], "values":[]}
    QUANTILES = np.round([0.01,0.025] + list(np.arange(0.05,0.95+0.05,0.05)) + [0.975,0.99],3)

    for L in np.arange(0,3+1):
        for model_number,(P,D,Q) in enumerate(itertools.product([0,1,2,3],[0,1,2],[0,1])):
            #--fit model
            model = arima(endog = centered_resids, order = (P,D,Q))
            model = model.fit()

            quantiles["horizon"].extend( [L+1]*len(QUANTILES) )
            quantiles["model"].extend( [model_number]*len(QUANTILES) )
            
            quantiles["quantile"].extend(QUANTILES)

            F,f,P = from_forecast_2_dist( model.get_forecast(int(L)+1), -1 )
            quantiles["values"].extend( P(QUANTILES) )
    quantiles = pd.DataFrame(quantiles)

    forecasts = W.merge(quantiles, on = ["horizon","model"])
    ensemble_forecasts = forecasts.groupby(["horizon","quantile"]).apply( lambda x: pd.Series({"value": sum(x.weights*x["values"])}) ).reset_index()

    holt_winters_trend = pd.DataFrame({"horizon":list(np.arange(1,4+1)),"trend": list(y_forecast) })

    ensemble_forecasts = ensemble_forecasts.merge(holt_winters_trend, on = ["horizon"])
    ensemble_forecasts["value"] = ensemble_forecasts["value"] + ensemble_forecasts["trend"]

    #--compute all time information for forecast submission
    number_of_days_until_monday = next_monday()
    monday = next_monday(True)
    
    next_sat = next_saturday_after_monday_submission( number_of_days_until_monday )
    target_end_dates = collect_target_end_dates(next_sat)

    TEDS = pd.DataFrame({ "horizon":[1,2,3,4]
                          , "target_end_date":target_end_dates
                          , "target": ["{:d} wk ahead inc flu hosp".format(n+1) for n in range(4) ]  })
    ensemble_forecasts = ensemble_forecasts.merge( TEDS, on = ["horizon"] )
    
    ensemble_forecasts["location"]      = LOCATION
    ensemble_forecasts["type"]          = "quantile"
    ensemble_forecasts["forecast_date"] = monday

    #--format quantile to three decimals
    ensemble_forecasts['quantile'] = ["{:0.3f}".format(q) for q in ensemble_forecasts["quantile"]]

    #--format values to three decimals
    ensemble_forecasts['value'] = ["{:0.3f}".format(q) for q in ensemble_forecasts["value"]]

    ensemble_forecasts = ensemble_forecasts[ ["target", "target_end_date","quantile","value","location","type","forecast_date"]  ]
    
    #--output data
    forecast.to_csv("./forecasts/location__{:s}.csv".format(LOCATION),index=False)
