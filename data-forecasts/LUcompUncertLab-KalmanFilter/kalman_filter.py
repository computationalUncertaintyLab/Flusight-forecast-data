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
   
    #--standardize the residuals by subtracting mean and diviiding by std 
    params = namedtuple("params","avg sd")
    stand__params   = params( avg = ys.mean(), sd = ys.std() )
    centered_ys     = (ys - stand__params.avg)/stand__params.sd
    
    old_stan_model = '''
        data {
           int T;
           vector [T] y;
        }
        parameters { 
           real beta0; 
           vector [T] latent_state;
 
           real alpha; 
           real beta; 

           real <lower = 0> sigma;
           real <lower = 0> phi;

        }
        model {
             //AR hidden latent
             latent_state[1] ~ normal(0,10);

             sigma~normal(1,10);

             for (t in 2:T){
                latent_state[t] ~  normal(alpha + beta*latent_state[t-1], sigma);
             }
             //observation model
             for (t in 1:T){
                y[t] ~ normal(latent_state[t], phi);   
             }
        }
        generated quantities { 
            vector [T] y_hat;
            vector [4] y_forecast;

            for (t in 1:T){
               real latent   = normal_rng(latent_state[t], sigma);
               y_hat[t]      = normal_rng(latent, phi);
            }

            vector [4] latent;
            latent[1] = normal_rng(alpha+beta*latent_state[T], sigma);
            for (l in 2:4){
               latent[l] = normal_rng(alpha+beta*latent[l-1], sigma);
            }
            for (f in 1:4){
               y_forecast[f] = normal_rng(latent[f], phi);
            }
        }
    '''

    stan_model = '''
        data {
           int T;
           vector [T] y;
        }
        parameters { 
           vector [T] latent_state;
 
           real alpha; 
           vector [10] beta; 

           real <lower = 0> sigma;
           real <lower = 0> phi;

        }
        model {
             //AR hidden latent
             latent_state[1] ~ normal(0,10);

             for(l in 1:10){
                 beta[l]~double_exponential(0,0.01);
             }
             sigma~normal(1,10);

             for (t in 11:T){
                real mu=alpha;
                for(l in 1:10){
                    mu+=beta[l]*latent_state[t-l];
                }
                latent_state[t] ~  normal(mu, sigma);
             }
             //observation model
             for (t in 1:T){
                y[t] ~ normal(latent_state[t], phi);   
             }
        }
        generated quantities { 
            vector [T] y_hat;
            vector [4] y_forecast;

            for (t in 1:T){
               real latent   = normal_rng(latent_state[t], sigma);
               y_hat[t]      = normal_rng(latent, phi);
            }

            vector [14] latent;
            for (t in 1:10){
                 latent[t] = latent_state[T-10+t];
            }

            for (t in 11:14){
               real muhat = alpha;
               for (l in 1:10){
                   muhat+=beta[l]*latent[t-l];
               }
               latent[t] = normal_rng(muhat, sigma);
            }
            for (f in 1:4){
               y_forecast[f] = normal_rng(latent[10+f], phi);
            }
        }
    '''
   
    model_data = {"T":len(centered_ys), "y":centered_ys}
    posterior  = stan.build(old_stan_model, data=model_data)
    fit        = posterior.sample(num_chains=4, num_samples=1000)
    
    uncentered_inferences = fit.get("y_hat")*stand__params.sd + stand__params.avg
    
    orig = {"n":[], "pred":[], "t":[]}
    for n,sample in enumerate(uncentered_inferences.T):
        T = len(sample)
        times = np.arange(0,T)
        orig["pred"].extend(sample)

        T = len(sample)
        orig["t"].extend(times)
        orig["n"].extend( [n]*T )
    inferences = pd.DataFrame(orig)

    reference_time = len(ys)
    uncentered_forecasts = fit.get("y_forecast")*stand__params.sd + stand__params.avg
    orig = {"n":[], "pred":[], "t":[]}
    for n,sample in enumerate(uncentered_forecasts.T):

        T = 4
        times = np.arange(reference_time,reference_time+T)
        orig["pred"].extend(sample)

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
