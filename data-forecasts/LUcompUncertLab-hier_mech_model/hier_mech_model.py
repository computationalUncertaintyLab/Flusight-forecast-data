#mcandrew

import sys
import numpy as np
import pandas as pd

from submission_times import *

import argparse

import stan

def model_single():
    model_code = '''

    data {
       real S0;
       int T;
       int T_forecast;
       int cases [T]; 
    }
    transformed data {
   
        //normalized cases (ie divide by S0)
        vector [T] norm_cases; 
        for (t in 1:T){
           norm_cases[t] = cases[t] / S0;
        }
    }
    parameters {
       vector [T] log_beta;
       real <lower=0> phi;       
       real <lower=0,upper=1>s0;

    }
    transformed parameters {

       // exponentiate log beta
       vector [T] beta;

       for(t in 1:T){
          beta[t] = exp(log_beta[t]);
       }

       //latent process and states
       vector [T] S;
       vector [T] I;
       vector [T] i;
       vector [T] R;

       //set the initial time values for the above states
          i[1] = norm_cases[1];
          I[1] = norm_cases[1];

          S[1] = (1 - i[1])*s0;
          
          R[1] = 0;

          real gamma = 1.;

       //fill in additional values over time.
       for (t in 2:T){
             i[t] = beta[t]*I[t-1]*S[t-1];
             I[t] = I[t-1] + i[t] - gamma*I[t-1];
             
             S[t] = S[t-1] - i[t];
             R[t] = R[t-1] + gamma*I[t-1];
          }
    }
    model {
       
       //setup priors
       log_beta[1]~normal( log(0.5), 10 );
       for (t in 2:T){
          log_beta[t] ~normal( log_beta[t-1], 0.001 );
       }
 
       phi~normal(0,10);
       
       s0~beta(5,5);

       //observation model
          for (t in 2:T){
              cases[t]~neg_binomial_2( S0*(1 - i[1])*s0*i[t], phi );
          }
    }

    generated quantities {
       //latent process and states
       vector [T+28] Shat;
       vector [T+28] Ihat;
       vector [T+28] ihat;
       vector [T+28] Rhat;

       //build beta time series
       vector [T+28] log_beta_hat;
       vector [T+28] beta_hat;

       log_beta_hat[1] = normal_rng( log(0.5), 0.1 );
       for (t in 1:T+28){
           log_beta_hat[t] = normal_rng( log_beta_hat[t-1], 1. );
           beta_hat[t]     = exp(log_beta_hat[t]);
       }
       //set the initial time values for the above states
          ihat[1] = norm_cases[1];
          Ihat[1] = norm_cases[1];

          Shat[1] = (1 - i[1])*s0;
          Rhat[1] = 0;

       //fill in additional values over time.
       for (t in 2:T+28){
             ihat[t] = beta_hat[t]*Ihat[t-1]*Shat[t-1];
             Ihat[t] = Ihat[t-1] + ihat[t] - gamma*Ihat[t-1];
             
             Shat[t] = Shat[t-1] - ihat[t];
             Rhat[t] = Rhat[t-1] + gamma*Ihat[t-1];
          }
       //rescale i to the original population
       vector [T+28] scaled_ihat;
       scaled_ihat = ihat*s0*S0;
    }
'''
    return model_code



def model_seasons():
    model_code = '''
    data {
       real S0;
       int seasons;
       int T [seasons];
       int T_forecast;
       int cases [T[1], seasons ]; 
    }
    transformed data {
        //normalized cases (ie divide by S0)
        real norm_cases [T[1], seasons]; 
        for (s in 1:seasons){
           for (t in 1:T[s]){
              norm_cases[t,s] = cases[t,s] / S0;
           }
        }
    }
    parameters {
       real log_beta [T[1], seasons] ;
       real <lower=0> phi;       
       real <lower=0,upper=1>s0;

    }
    transformed parameters {

       // exponentiate log beta
       real beta [T[1], seasons];

       for (s in 1:seasons){
          for(t in 1:T[s]){
             beta[t,s] = exp(log_beta[t,s]);
          }
       }

       //latent process and states
        real S [T[1], seasons];
        real I [T[1], seasons];
        real i [T[1], seasons];
        real R [T[1], seasons];

       //set the initial time values for the above states
       for (s in 1:seasons){
           i[1,s] = norm_cases[1,1];
           I[1,s] = norm_cases[1,1];

           S[1,s] = (1 - i[1][s])*s0;
          
           R[1,s] = 0;
       }
       real gamma = 1.;

       //fill in additional values over time.
        for (s in 1:seasons){ 
            for (t in 2:T[s]){
                 i[t,s] = beta[t,s]*I[t-1,s]*S[t-1,s];
                 I[t,s] = I[t-1,s] + i[t,s] - gamma*I[t-1,s];

                 S[t,s] = S[t-1,s] - i[t,s];
                 R[t,s] = R[t-1,s] + gamma*I[t-1,s];
            }
        }
    }
    model {
       //setup priors
       for (s in 1:seasons){
           log_beta[1,s]~normal( log(0.5), 10 );
           for (t in 2:T[s]){
              log_beta[t,s] ~normal( log_beta[t-1,s], 0.001 );
           }
       }

       phi~normal(0,10);
       
       s0~beta(5,5);

       //observation model
        for (s in 1:seasons){
           for (t in 2:T[s]){
              cases[t,s]~neg_binomial_2( S0*(1 - i[1,s])*s0*i[t,s], phi );
           }
        }
    }
    generated quantities {

        real Shat [28];
        real Ihat [28];
        real ihat [28];
        real Rhat [28];

        real log_beta_hat = log_beta[T[seasons],seasons];
        real beta_hat     = exp(log_beta_hat);

        //start prediction at the last values in the within sample fit
        ihat[1] = i[T[seasons],seasons];
        Ihat[1] = i[T[seasons],seasons];

        Shat[1] = S[T[seasons],seasons];

        Rhat[1] = R[T[seasons],seasons];

        for (t in 2:28){
          ihat[t] = beta_hat*Ihat[t-1]*Shat[t-1];
          Ihat[t] = Ihat[t-1] + ihat[t] - gamma*Ihat[t-1];

          Shat[t] = Shat[t-1] - ihat[t];
          Rhat[t] = Rhat[t-1] + gamma*Ihat[t-1];
        }   
   }
'''
    return model_code

    
def from_date_to_epiweek(x):
    from datetime import datetime, timedelta
    from epiweeks import Week

    #--remove time past day
    dt = x.date.values[0]
    dt = datetime.strptime(dt,"%Y-%m-%d")

    prev_day = dt - timedelta(days=1)

    wk = Week.fromdate(prev_day)
    
    return pd.Series({"yr":wk.year
                      ,"week":wk.week
                      ,"ew":wk.cdcformat()
                      , "month": prev_day.month
                      , "day":prev_day.day
                      ,"start_date":wk.startdate()
                      ,"end_date":wk.enddate()})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--LOCATION'     ,type=str) 
    parser.add_argument('--RETROSPECTIVE',type=int, nargs = "?", const=0)
    parser.add_argument('--END_DATE'     ,type=str, nargs = "?", const=0)
    
    args = parser.parse_args()
    LOCATION      = args.LOCATION
    RETROSPECTIVE = args.RETROSPECTIVE
   
    hhs_data = pd.read_csv("../../data-truth/truth-Incident Hospitalizations-daily.csv")

    populations = pd.read_csv("../../data-locations/locations.csv")
    S0 = float(populations.loc[populations.location==LOCATION, "population"])
    
    flu = hhs_data.loc[hhs_data.location==LOCATION]
    flu_times = flu.groupby("date").apply(from_date_to_epiweek)

    flu = flu.merge(flu_times, on = ["date"])

    #--subset to october to august
    last_season = flu.loc[ (flu.date >= "2022-01-01") & (flu.date <="2022-08-01")]
    current_season_flu = flu.loc[ (flu.date >= "2022-09-15") ]

    #--prepare training data
    T = last_season.shape[0]
    C = current_season_flu.shape[0]
    
    training_data = np.zeros((2,T))
    training_data[0,:]  = last_season.value.values
    training_data[1,:C] = current_season_flu.value.values + 1 #--adding a one
    training_data[1,C:] = 1. #--adding some small number 

    data = { "S0":S0
             ,"seasons":2
             ,"cases": training_data.astype(int).T
             ,"T": [T,C]
             ,"T_forecast":28
    }
    model_code = model_seasons()
    
    #model = pyjags.Model( model_code, data =data, chains = 4) 
    #samples = model.sample(1000, vars=['i'])

    posterior  = stan.build(model_code, data=data)
    fit        = posterior.sample(num_chains=4, num_samples=1000)
 
    #--current season
    current_season_i = fit.get("ihat")
    s0s = fit.get("s0")[0]

    infections = {"t":[],"sample":[],"value":[]}
    for sample,values in enumerate(current_season_i.T):

        s0 = s0s[sample]
        for time,value in enumerate(values):
            infections['t'].append(time)
            infections['sample'].append(sample)
            infections['value'].append(value*S0*s0)
    infections = pd.DataFrame(infections)
    
    #--forecast next 28 days
    predictions = infections

    weekly_agg = {"t":[],"wk":[]}
    for t in np.arange(0,28):
        weekly_agg['t'].append(t)

        if 0<=t<=6:
            weekly_agg['wk'].append(1)
        elif 7<=t<=13:
            weekly_agg['wk'].append(2)
        elif 14<=t<=20:
            weekly_agg['wk'].append(3)
        elif 21<=t<=28:
            weekly_agg['wk'].append(4)
    weekly_agg = pd.DataFrame(weekly_agg)

    predictions = predictions.merge(weekly_agg, on = ['t'])

    def accumulate_daily_to_weekly(x):
        return pd.Series({"hosps":x.value.sum()})
    predictions_weekly = predictions.groupby(["wk","sample"]).apply( accumulate_daily_to_weekly ).reset_index()

    predictions_weekly__wide = pd.pivot_table(index="sample", columns = ["wk"], values = ["hosps"], data = predictions_weekly) 

    Q = np.round([0.01,0.025] + list(np.arange(0.05,0.95+0.05,0.05)) + [0.975,0.99],3)
    N = len(Q)
    
    quantiles = np.percentile(predictions_weekly__wide,100*Q,axis=0)

    #--compute all time information for forecast submission
    if RETROSPECTIVE:
        number_of_days_until_monday = next_monday(from_date=END_DATE)
        monday = next_monday(True, from_date=END_DATE)

        next_sat = next_saturday_after_monday_submission( number_of_days_until_monday, from_date=END_DATE )    
    else:
        number_of_days_until_monday = next_monday()
        monday = next_monday(True)

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
        print("here")
        forecast.to_csv("./forecasts/location__{:s}.csv".format(LOCATION),index=False)
