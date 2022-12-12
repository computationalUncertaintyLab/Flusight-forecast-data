#mcandrew

import sys
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

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--LOCATION'     ,type=str) 
    # parser.add_argument('--RETROSPECTIVE',type=int, nargs = "?", const=0)
    # parser.add_argument('--END_DATE'     ,type=str, nargs = "?", const=0)
    
    # args = parser.parse_args()
    # LOCATION      = args.LOCATION
    # RETROSPECTIVE = args.RETROSPECTIVE
    # END_DATE      = args.END_DATE

    LOCATION = "42"
    RETROSPECTIVE = 0
    END_DATE = 0
    

    hhs_data = pd.read_csv("../LUData/hhs_data__daily.csv")
    
    S0 = float(hhs_data.loc[hhs_data.location==LOCATION, "population"].iloc[0])
    
    flu = hhs_data.loc[hhs_data.location==LOCATION]
    flu_times = flu.groupby("date").apply(from_date_to_epiweek)

    flu = flu.merge(flu_times, on = ["date"])

    #--sort by time
    flu = flu.sort_values("date")
    
    #--subset to october to august
    last_season = flu.loc[ (flu.date >= "2022-01-01") & (flu.date <="2022-08-01")]
    current_season_flu = flu.loc[ (flu.date >= "2022-09-15") ]

    #--prepare training data
    T = last_season.shape[0]
    C = current_season_flu.shape[0]

    #--hosp data
    training_data__hosps = np.zeros((2,T))
    training_data__hosps[0,:]  = last_season.hosps.values
    training_data__hosps[1,:C] = current_season_flu.hosps.values + 1 #--adding a one
    training_data__hosps[1,C:] = 0. #--adding some small number

    training_data__hosps = training_data__hosps.astype(int).T
    training_data__hosps__norm = training_data__hosps / S0
    
    #--death data
    training_data__deaths = np.zeros((2,T))
    training_data__deaths[0,:]  = last_season.deaths.values
    training_data__deaths[1,:C] = current_season_flu.deaths.values + 1 #--adding a one
    training_data__deaths[1,C:] = 0. #--adding some small number

    training_data__deaths = training_data__deaths.astype(int).T
    training_data__deaths__norm = training_data__deaths / S0
    
    times = np.arange(0, C).reshape(-1,1)
    ones  = np.ones( (C,1) ) 
    
    y = training_data__hosps[:C,-1].reshape(-1,1)
    
    def model(X,y,future=0):

        y = y.reshape(-1,)
        T = len(y)

        s = jnp.std(y)
        m = jnp.mean(y)
        y_scaled = (y-m)/s
        x_scaled = jnp.linspace(0,1,T)

        points = np.arange(0,T)
        
        smoothing = numpyro.sample("subset_frac", dist.Normal(10,10))
        def lowess(carry, iterates, alpha,X,y,future=0):
            new_y = carry
            point = iterates

            xval = x_scaled[point] #--xvalue
            xs = x_scaled

            d = jnp.power((xs-xval),2)
            weights = jnp.exp( -d/2*alpha  )
            
            betas = jnp.polyfit(x_scaled, y_scaled, deg=3,w=weights)

            yhat = (xval**3)*betas[0] + (xval**2)*betas[1] + (xval**1)*betas[2] + betas[3] 

            return y,yhat

        specific_lowess = lambda carry,iterates: lowess(carry,iterates, smoothing,x_scaled,y_scaled)
        final,results = jax.lax.scan( specific_lowess
                                      , y_scaled
                                      , points )

        yhat_scaled = numpyro.deterministic("yhat_scaled",results.reshape(-1,))
        yhat        = numpyro.deterministic("yhat",yhat_scaled*s + m)

        #obs_model   = numpyro.sample("obs_model", dist.Poisson(yhat) )
        
        LL = numpyro.sample( "ll", dist.Poisson(yhat), obs = y )

        #--prediction
        if future>0:
            step_size = x_scaled[1] - x_scaled[0]
            new_scaled_x = np.arange(0, 1+(future+1)*step_size, step_size)
            new_points   = np.arange(T,T+future) 
            
            specific_lowess = lambda carry,iterates: lowess(carry,iterates, smoothing,new_scaled_x,y)

            final,results = jax.lax.scan( specific_lowess
                                      , y_scaled
                                      ,  )


            
            pass
           
    nuts_kernel = NUTS(model)
    
    mcmc = MCMC( nuts_kernel , num_warmup=1500, num_samples=2000)
    rng_key = random.PRNGKey(0)

    mcmc.run(rng_key
             , X=X
             , y = y
             , future = 0
             , extra_fields=('potential_energy',))
    mcmc.print_summary()
    samples = mcmc.get_samples()

    sys.exit()

    

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
        forecast.to_csv("./forecasts/location__{:s}.csv".format(LOCATION),index=False)
