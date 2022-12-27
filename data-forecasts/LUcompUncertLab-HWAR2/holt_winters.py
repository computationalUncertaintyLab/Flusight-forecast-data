#mcandrew

import sys
from collections import namedtuple

import numpy as np
import pandas as pd

import stan
import statsmodels.api as sm

from submission_times import *

import argparse
from datetime import datetime

from glob import glob

if __name__ == "__main__":

    #--accepts one parameter that subsets data to Location
    parser = argparse.ArgumentParser()
    parser.add_argument('--LOCATION'     ,type=str) 
    parser.add_argument('--RETROSPECTIVE',type=int, nargs = "?", const=0)
    parser.add_argument('--END_DATE'     ,type=str, nargs = "?", const=0)
    
    args = parser.parse_args()
    LOCATION      = args.LOCATION
    RETROSPECTIVE = args.RETROSPECTIVE

    if RETROSPECTIVE:
        END_DATE      = args.END_DATE

        #--if this file already exists then exit.
        if LOCATION=="US":
            file_to_be_generated = "./retrospective_analysis/location_{:s}_end_{:s}.csv".format("US",END_DATE)
        else:
            file_to_be_generated = "./retrospective_analysis/location_{:02d}_end_{:s}.csv".format(int(LOCATION),END_DATE)
        all_retro_files      = glob("./retrospective_analysis/*.csv")

        if file_to_be_generated in all_retro_files:
            print("This retrospective forecast already exists")
            sys.exit()

    flu = pd.read_csv("../../data-truth/truth-Incident Hospitalizations.csv")
    flu["date"] = pd.to_datetime(flu.date)
    
    flu2022 = flu.loc[flu.date > "2022-03-01"] #start training Oct 2021

    if RETROSPECTIVE:
        flu2022 = flu2022.loc[ flu2022.date <= datetime.strptime(args.END_DATE,"%Y-%m-%d")]

    most_recent_date = sorted(flu2022["date"])[-1]
    print("Training up to {:s}".format(most_recent_date.strftime("%Y-%m-%d")))
        
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
    fit        = posterior.sample(num_chains=4, num_samples=1000)
    
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
    if RETROSPECTIVE:
        number_of_days_until_monday = next_monday(from_date=END_DATE)
        monday = next_monday(True, from_date=END_DATE)

        next_sat = next_saturday_after_monday_submission( number_of_days_until_monday, from_date=END_DATE )    
    else:
        number_of_days_until_monday = next_monday(from_date="2022-12-25")
        monday = next_monday(True, from_date="2022-12-25")

        next_sat = next_saturday_after_monday_submission( number_of_days_until_monday)    
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
    if RETROSPECTIVE:
        forecast.to_csv("./retrospective_analysis/location_{:s}_end_{:s}.csv".format(LOCATION,END_DATE),index=False)
    else:
        forecast.to_csv("./forecasts/location__{:s}.csv".format(LOCATION),index=False)
