#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from submission_times import *

import argparse
import pyjags

def model():
    model_code = '''
    model
    {
       s_total ~ dunif(.1,.99)
      r <- 10000

      for (j in 1:2){
        s[1,j] <- s_total

        i0_[j] ~ dunif(i0[j]/N,(i0[j]+100)/N)
        i[1,j] <- i0_[j]

        I[1,j] <- i0_[j]

        log_beta[1,j] ~ dnorm(log(.5),.1)
        for (t in 2:(T+30)){
          log_beta[t,j] ~ dnorm(log_beta[t-1,j],1000)
        }
        gamma[j] <- .25
        for (t in 2:(T+30)){
            i[t,j] <- exp(log_beta[t,j])*I[t-1,j]*s[t-1,j]
            I[t,j] <- I[t-1,j] - gamma[j]*I[t-1,j]+ i[t,j]
            s[t,j] <- s[t-1,j] - i[t,j]
        }

        for (t in 1:T){
            #log(lambda[t,j]) <- N*max(1,i[t,j])
            p[t,j] <- r/(r+N*i[t,j])

            y[t,j] ~ dnegbin(p[t,j],r)
        }
      } ## Loop over j
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


    data = {"N":S0
            ,"i0": training_data[:,0]
            ,"y":training_data.T
            ,"T":T}

    model_code = model()
    
    model = pyjags.Model( model_code, data =data, chains = 4) 

    samples = model.sample(1000, vars=['i'])

    #--current season
    current_season_i = samples['i'][:,1,:,:]

    infections = {"t":[],"sample":[],"value":[]}
    for time,t in enumerate(current_season_i):
        for sample,value in enumerate(t.flatten()):
            infections['t'].append(time)
            infections['sample'].append(sample)
            infections['value'].append(value*S0)
    infections = pd.DataFrame(infections)
    
    #--forecast next 28 days
    predictions = infections.loc[ (infections.t >=T+1) & (infections.t<=T+1+28)]

    weekly_agg = {"t":[],"wk":[]}
    for n,t in enumerate(np.arange(T+1,T+1+28)):
        weekly_agg['t'].append(t)

        if 0<=n<=6:
            weekly_agg['wk'].append(1)
        elif 7<=n<=13:
            weekly_agg['wk'].append(2)
        elif 14<=n<=20:
            weekly_agg['wk'].append(3)
        elif 21<=n<=28:
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

    print(forecast)
    #--output data
    if RETROSPECTIVE:
        forecast.to_csv("./retrospective_analysis/location_{:s}_end_{:s}.csv".format(LOCATION,END_DATE),index=False)
    else:
        print("here")
        forecast.to_csv("./forecasts/location__{:s}.csv".format(LOCATION),index=False)
