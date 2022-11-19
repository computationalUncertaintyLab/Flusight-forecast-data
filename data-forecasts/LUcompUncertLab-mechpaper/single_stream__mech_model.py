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

    def hier_sir_model( T, ttl, training_data__hosps=None, training_data__deaths=None, future=0):
        def one_step(carry, aray_element):
            S,I,i2h,H,R,h2d,D = carry
            t,beta,rho,kappa  = aray_element

            rho = 0.25
            
            s2i  = beta*I*S        # S->I
            i2r  = (rho*0.25)*I    # I->R (lambda = 0.25)
            i2h  = (rho*0.75)*I    # I->H 
            h2d  = (kappa*0.01)*H  # H->D
            h2r  = (kappa*0.99)*H  # H->R

            nI = I + s2i - (i2r + i2h)
            nH = H + i2h - (h2d + h2r)
            nD = D + h2d
            nS = S - s2i
            
            nR = R + (i2r + h2r)

            states = jnp.array([nS,nI,i2h,nH,nR,h2d,nD] )

            return states, states

        training_data__hosps__normalized  = training_data__hosps / ttl
        training_data__deaths__normalized = training_data__deaths / ttl
        
        #--prior for beta and set gamma to be fixed
        log_beta =  numpyro.sample( "log_beta", dist.Normal( np.log(0.50)*jnp.ones( (T,) ) , 0.1 ) )
        beta     = numpyro.deterministic("beta", jnp.exp(log_beta))

        log_rho =  numpyro.sample( "log_rho", dist.Normal( np.log(0.25)*jnp.ones( (T,) ) , 0.1 ) )
        rho     =  numpyro.deterministic("rho", jnp.exp(log_rho))

        log_kappa =  numpyro.sample( "log_kappa", dist.Normal( np.log(0.01)*jnp.ones( (T,) ) , 0.1 ) )
        kappa     = numpyro.deterministic("kappa", jnp.exp(log_kappa))
        
        #--prior for percent of population that is susceptible
        percent_sus = numpyro.sample("percent_sus", dist.Beta(2,2) )
       
        #--Run process
        times = np.array([T])

        ts     = np.arange(0,times)
        betas  = beta
        kappas = kappa
        rhos   = rho

        i2h0 = training_data__hosps__normalized[0] 
        H0   = training_data__hosps__normalized[0]

        I0 = training_data__hosps__normalized[0]
        S0 = 1.*percent_sus - I0

        R0 = 0.

        h2d0 = training_data__deaths__normalized[0] 
        D0   = training_data__deaths__normalized[0] 

        final, result = jax.lax.scan( one_step, jnp.array([S0,I0,i2h0,H0,R0,h2d0,D0]), (ts,betas,rhos,kappas) )

        states = numpyro.deterministic("states",result)

        #--observations are assumed to be generated from a negative binomial
        i2h__vals = numpyro.deterministic("i2h__vals", jnp.clip(result[:,2], 1*10**-10, jnp.inf))
        h2d__vals = numpyro.deterministic("h2d__vals", jnp.clip(result[:,5], 1*10**-10, jnp.inf))

        LL1  = numpyro.sample("LL_H", dist.Poisson(i2h__vals*ttl), obs = training_data__hosps )

        #--prediction
        if future>0:
            forecast_betas  = beta[T]*jnp.ones((future,))
            forecast_rhos   = rho[T]*jnp.ones((future,))
            forecast_kappas = kappa[T]*jnp.ones((future,))
            
            final, result = jax.lax.scan( one_step, states[T,:], (np.arange(0,future),forecast_betas, forecast_rhos, forecast_kappas) )
            
            numpyro.deterministic("forecast", result[:,2] )
            
    nuts_kernel = NUTS(hier_sir_model)
    
    mcmc = MCMC( nuts_kernel , num_warmup=1500, num_samples=2000)
    rng_key = random.PRNGKey(0)

    mcmc.run(rng_key
             , T=T
             , ttl = S0
             , training_data__hosps  = training_data__hosps
             , training_data__deaths = training_data__deaths
             , future = 28
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

    model_name = "single_stream"
    forecast["model_name"] = model_name
    
    #--output data
    forecast.to_csv("./retrospective_analysis/{:s}/model_{:s}_location_{:s}_end_{:s}.csv".format(model_name,model_name,LOCATION,END_DATE),index=False)

    pickle.dump( samples
                 , open("./retrospective_analysis/{:s}/model_{:s}_location_{:s}_end_{:s}.pkl".format(model_name,model_name,LOCATION,END_DATE),'wb' )  )
    
