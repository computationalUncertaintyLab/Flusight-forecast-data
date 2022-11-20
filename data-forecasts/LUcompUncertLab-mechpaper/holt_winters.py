#mcandrew

import sys
import pickle

import numpy as np
import pandas as pd

from submission_times import *

import argparse

import numpyro.distributions as dist
import numpyro

from numpyro.infer import MCMC, NUTS, HMC, Predictive
from numpyro.distributions import constraints

import jax
from jax import random
import jax.numpy as jnp

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

    numpyro.enable_x64(True)
    numpyro.validation_enabled(True)
    from jax.config import config
    config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--LOCATION'     ,type=str) 
    parser.add_argument('--END_DATE'     ,type=str, nargs = "?", const=0)
    
    args = parser.parse_args()
    LOCATION      = args.LOCATION
    END_DATE      = args.END_DATE
  
    hhs_data = pd.read_csv("../LUData/hhs_data__daily.csv")
    
    S0 = float(hhs_data.loc[hhs_data.location==LOCATION, "population"].iloc[0])
    
    flu = hhs_data.loc[hhs_data.location==LOCATION]
    flu_times = flu.groupby("date").apply(from_date_to_epiweek)

    flu = flu.merge(flu_times, on = ["date"])

    #--sort by time
    flu = flu.sort_values("date")
    
    #--subset to october to august
    current_season_flu = flu.loc[ (flu.date >= "2022-01-01") & (flu.date <= END_DATE) ]

    #--this will occur when the END_DATE is before 2022-09-15 
    if len(current_season_flu)==0:
        sys.exit()
    
    #--prepare training data
    T = current_season_flu.shape[0]

    #--hosp data
    training_data__hosps  = current_season_flu.hosps.values
    training_data__hosps = training_data__hosps.astype(int).T
    training_data__hosps__norm = training_data__hosps / S0
    
    #--death data
    training_data__deaths  = current_season_flu.deaths.values
    training_data__deaths = training_data__deaths.astype(int).T
    training_data__deaths__norm = training_data__deaths / S0

    def model(y, forecast):
        y = jnp.array(y)
        T = len(y)

        #--priors for Holt Winters
        beta  = numpyro.sample("beta", dist.Beta(2,2))
        alpha = numpyro.sample("alpha", dist.Beta(2,2))

        #--priors over beginning level and trend
        level0 = numpyro.sample("level0", dist.Normal( y[0] , 5 ))
        trend0 = numpyro.sample("trend0", dist.Normal( y[1]-y[0], 5))
        
        def HW_model(carry,array):
            prev_level, prev_trend = carry
            alpha, beta, y = array
            
            level = alpha*y + (1-alpha)*(prev_level + prev_trend)
            trend = beta*( level - prev_level ) + (1 - beta)*prev_trend
            
            next_y = level + trend #--one step ahead

            level_and_trend = jnp.array([level,trend])
            
            return level_and_trend, next_y
        
        final_level_and_trend,yhat = jax.lax.scan( HW_model , jnp.array([level0,trend0]), (alpha*jnp.ones(T,), beta*jnp.ones(T,),y)  )
        numpyro.deterministic("yhat",yhat)

        yhat = jnp.clip( yhat, 1*10**-10,jnp.inf )
        
        #--eval
        ll = numpyro.sample("ll", dist.Poisson( yhat ), obs = y)

        if forecast > 0:
            level, trend = final_level_and_trend
            y_forecast = level + trend*jnp.arange(1,forecast+1)
            numpyro.deterministic("forecast", y_forecast)
            
    nuts_kernel = NUTS(model)
    
    mcmc = MCMC( nuts_kernel , num_warmup=1500, num_samples=2000)
    rng_key = random.PRNGKey(0)

    mcmc.run(rng_key
             , y = training_data__hosps
             , forecast = 28
             , extra_fields=('potential_energy',))
    mcmc.print_summary()
    samples = mcmc.get_samples()

    #--format predictions
    predictions = {"t":[],"sample":[],"value":[]}
    for sample,values in enumerate(samples["forecast"]):
        for time,value in enumerate(values):
            predictions['t'].append(time)
            predictions['sample'].append(sample)
            predictions['value'].append(value*S0)
    predictions = pd.DataFrame(predictions)
 
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
    number_of_days_until_monday = next_monday(from_date=END_DATE)
    monday = next_monday(True, from_date=END_DATE)

    next_sat = next_saturday_after_monday_submission( number_of_days_until_monday, from_date=END_DATE )    
    
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

    model_name = "holt_winters"
    forecast["model_name"] = model_name
    
    #--output data
    forecast.to_csv("./retrospective_analysis/{:s}/model_{:s}_location_{:s}_end_{:s}.csv".format(model_name,model_name,LOCATION,END_DATE),index=False)

    pickle.dump( samples
                 , open("./retrospective_analysis/{:s}/model_{:s}_location_{:s}_end_{:s}.pkl".format(model_name,model_name,LOCATION,END_DATE),'wb' )  )
    
