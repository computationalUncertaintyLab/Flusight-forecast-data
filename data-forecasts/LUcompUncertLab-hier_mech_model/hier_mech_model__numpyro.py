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

    parser = argparse.ArgumentParser()
    parser.add_argument('--LOCATION'     ,type=str) 
    parser.add_argument('--RETROSPECTIVE',type=int, nargs = "?", const=0)
    parser.add_argument('--END_DATE'     ,type=str, nargs = "?", const=0)
    
    args = parser.parse_args()
    LOCATION      = args.LOCATION
    RETROSPECTIVE = args.RETROSPECTIVE
    END_DATE      = args.END_DATE
   
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
    training_data[1,C:] = 0. #--adding some small number 

    training_data = training_data.astype(int).T

    training_data__normalized = training_data / S0

    def hier_sir_model( T, C, SEASONS, ttl, training_data=None, future=0):
        def one_step(carry, aray_element):
            S,i,I,R = carry
            t,beta  = aray_element
            
            newi = beta*I*S
            r = 0.25*I
            
            newI = I + i - r
            newS = S - i
            newR = R + r

            states = jnp.array([newS,newi,newI,newR] )

            return states, states

        training_data__normalized = training_data / ttl
        
        #--prior for beta and set gamma to be fixed
        log_beta     =  numpyro.sample( "log_beta", dist.Normal( np.log(0.50)*jnp.ones( (T,SEASONS) ) , 0.1 ) )
        
        #cum_log_beta = numpyro.deterministic("c_log_beta", jnp.cumsum(log_beta,0))
        
        beta     = numpyro.deterministic("beta", jnp.exp(log_beta))
        gamma    = 0.25

        #--prior for percent of population that is susceptible
        percent_sus = numpyro.sample("percent_sus", dist.Beta(2,2) )

        #--process model
        S = jnp.zeros( (T,SEASONS) )
        i = jnp.zeros( (T,SEASONS) )
        I = jnp.zeros( (T,SEASONS) )
        R = jnp.zeros( (T,SEASONS) )
        
        #--Run process
        times = np.array([T,C])

        phi = numpyro.sample("phi", dist.Gamma( jnp.array([10,10]),jnp.array([1,1])) )
        
        for s in np.arange(0,SEASONS):
            ts     = np.arange(0,times[s])
            betas  = beta[:times[s],s]

            i0 = training_data__normalized[0,s] 
            I0 = training_data__normalized[0,s]

            S0 = 1.*percent_sus - i0
            R0 = 0.

            final, result = jax.lax.scan( one_step, jnp.array([S0,i0,I0,R0]), (ts,betas) )

            states = numpyro.deterministic("states_{:d}".format(s),result)

            #--observations are assumed to be generated from a negative binomial
            ivals = numpyro.deterministic("ivals_{:d}".format(s), jnp.clip(result[:,1], 1*10**-10, jnp.inf))

            LL  = numpyro.sample("LL_{:d}".format(s), dist.Poisson(ivals*ttl), obs = training_data[:times[s],s] )

        #--prediction
        if future>0:
            forecast_betas = beta[C,SEASONS]*jnp.ones((future,))
            lastS,lasti,lastI,lastR = states[C,:]
            
            final, result = jax.lax.scan( one_step, jnp.array([lastS,lasti,lastI,lastR]), (np.arange(0,future),forecast_betas) )
            
            numpyro.deterministic("forecast", result[:,1] )
            
    nuts_kernel = NUTS(hier_sir_model)
    
    mcmc = MCMC( nuts_kernel , num_warmup=1500, num_samples=2000)
    rng_key = random.PRNGKey(0)

    mcmc.run(rng_key
             , T=T
             , C=C
             , SEASONS = 2
             , ttl = S0
             , training_data = training_data
             , future = 28
             , extra_fields=('potential_energy',))
    mcmc.print_summary()
    samples = mcmc.get_samples()


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
