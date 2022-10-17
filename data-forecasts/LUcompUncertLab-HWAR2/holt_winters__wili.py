#mcandrew

import sys
from collections import namedtuple

import numpy as np
import pandas as pd

import stan
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

import argparse

from submission_times import *

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

    LOCATION = "01"
    
    flu = pd.read_csv("../../data-truth/truth-Incident Hospitalizations.csv")
    flu["date"] = pd.to_datetime(flu.date)

    flu_stacked = pd.read_csv("./stacked_influenza_data.csv")
    
    flu2022 = flu_stacked.loc[flu_stacked.date > "2021-10-01"]

    location_specific_values = flu2022.loc[flu2022.location==LOCATION]
    location_specific_values = location_specific_values.sort_values(["date"])

    wili_cols = [x for x in location_specific_values.columns if 'wili' in x]
    ys = location_specific_values.value.values
    X  = location_specific_values[ wili_cols ]

    mu_yhat = ys.mean()
    sd_yhat = ys.std()
    c_ys = (ys - mu_yhat)/sd_yhat

    mu_X = X.mean(0)
    sd_X = X.std(0)
    c_X  = (X - mu_X)/sd_X
   
    next_four_weeks = find_next_four_weeks(location_specific_values)
    
    #--detrend by fitting wili to hosps
    model   = OLS(c_ys,c_X).fit()
    y_infer = model.predict()

    Xstar    = location_specific_values.loc[ location_specific_values.week.isin(next_four_weeks), wili_cols  ]
    c_Xstar  = (Xstar-mu_X)/sd_X
    
    y_forecast = model.predict(Xstar)
    
    xs  = np.arange(0,len(location_specific_values))
    T   = len(ys)

    #--compute residuals from Holt Winters procedure
    resids = ys - y_infer
    
    #--standardize the residuals by subtracting mean and diviiding by std 
    params = namedtuple("params","avg sd")
    stand__params   = params( avg = resids.mean(), sd = resids.std() )
    centered_resids = (resids - stand__params.avg)/stand__params.sd
    
    stan_model = '''
        data {
           int T;
           vector [T] y;
        }
        parameters { 
           real beta0; 
           real <lower=-1,upper = 1>beta [2]; 
           real <lower = 0> sigma;
        }
        model {

              beta0~normal(0,10);
              for (l in 1:2){
                 beta[l]~normal(0,10);
              }
              sigma~cauchy(5,5);

             for (t in 3:T){
                 //form predictor
                 real mu = 0;
                 for (l in 1:2){
                    mu+= beta[l]*y[t-l];
                 }
                target+=normal_lpdf(y[t]|beta0+mu,sigma);
             }
        }
        generated quantities {
           vector [T] y_hat; 
           vector [2+4] y_forecast;

           //inference
           for (l in 1:2){
              y_hat[l] = y[l];
           } 
           for (t in 3:T){
             //form predictor
             real mu = 0;
             for (l in 1:2){
                mu+= beta[l]*y[t-l];
             }
             y_hat[t] = normal_rng(beta0+mu,sigma);
           }

          //Forecasting

          for (l in 1:2){
             y_forecast[l] = y[T+1-l];
          }
          for (t in 2+1:2+4){
             //form predictor
             real mu = 0;
             for (l in 1:2){
                mu+= beta[l]*y_forecast[t-l];
             }
             y_forecast[t] = normal_rng(beta0+mu,sigma);
          }
        }  
    '''
    model_data = {"T":len(centered_resids), "y":centered_resids}
    posterior  = stan.build(stan_model, data=model_data)
    fit        = posterior.sample(num_chains=1, num_samples=5000)
    
    uncentered_resids = fit.get("y_hat")*stand__params.sd + stand__params.avg
    
    orig = {"n":[], "pred":[], "t":[]}
    for n,sample in enumerate(uncentered_resids.T):
        T = len(sample)
        times = np.arange(0,T)
        orig["pred"].extend( sample + y_infer )

        T = len(sample)
        orig["t"].extend(times)
        orig["n"].extend( [n]*T )
    inferences = pd.DataFrame(orig)

    reference_time = len(ys)
    uncentered_resid_forecasts = fit.get("y_forecast")*stand__params.sd + stand__params.avg
    orig = {"n":[], "pred":[], "t":[]}
    for n,sample in enumerate(uncentered_resid_forecasts.T):

        T = 4
        times = np.arange(reference_time,reference_time+T)
        orig["pred"].extend( sample[-4:] + y_forecast  )

        orig["t"].extend(times)
        orig["n"].extend( [n]*T )
    predictions = pd.DataFrame(orig)

    predictions.loc[predictions["pred"]<0,"pred"] = 0

    wide = pd.pivot_table(index = "n", columns = "t", values = "pred", data = predictions)

    Q = np.round([0.01,0.025] + list(np.arange(0.05,0.95+0.05,0.05)) + [0.975,0.99],3)
    N = len(Q)
    
    quantiles = np.percentile(wide,100*Q,axis=0)

    #--compute all time information for forecast submission
    number_of_days_until_monday = next_monday()
    monday = next_monday(True)
    
    next_sat = next_saturday_after_monday_submission( number_of_days_until_monday )
    target_end_dates = collect_target_end_dates(next_sat)
    
    #--store all forecasts in a DataFrame
    forecast = {"target":[], "target_end_date":[], "quantile":[], "value":[]}
    for n,week_ahead_prediction in enumerate(quantiles.T):
        forecast["value"].extend(week_ahead_prediction)
        forecast["quantile"].extend(Q)

        #--items that need to be repeated Q times
        forecast["target"].extend( ["{:d} wk ahead inc flu hosp".format(n+1)]*N)
        forecast["target_end_date"].extend( [target_end_dates[n]]*N)
        
    forecast = pd.DataFrame(forecast)
    forecast["location"]      = LOCATION
    forecast["type"]          = "quantile"
    forecast["forecast_date"] = monday

    #--format quantile to three decimals
    forecast['quantile'] = ["{:0.3f}".format(q) for q in forecast["quantile"]]

    #--format values to three decimals
    forecast['value'] = ["{:0.3f}".format(q) for q in forecast["value"]]
    
    #--output data
    forecast.to_csv("./forecasts/location__{:s}.csv".format(LOCATION),index=False)
