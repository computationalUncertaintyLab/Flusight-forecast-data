#mcandrew

import sys
import numpy as np
import pandas as pd

from submission_times import *

import argparse

import numpyro.distributions as dist
import numpyro

from numpyro.infer import MCMC, NUTS, MixedHMC, HMC
from numpyro.distributions import constraints

from jax import random
import jax.numpy as jnp
import jax

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

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--LOCATION'     ,type=str) 
    # parser.add_argument('--RETROSPECTIVE',type=int, nargs = "?", const=0)
    # parser.add_argument('--END_DATE'     ,type=str, nargs = "?", const=0)
    
    # args = parser.parse_args()
    # LOCATION      = args.LOCATION
    # RETROSPECTIVE = args.RETROSPECTIVE
    # END_DATE      = args.END_DATE

    LOCATION="42"
    RETROSPECTIVE=0
    END_DATE=0
    
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

    from jax.config import config
    config.update("jax_enable_x64", True)
    
    def hier_sir_model( T, C, SEASONS, training_data=None):
        def one_step(carry, aray_element):
            S,i,I,R = carry
            t,beta  = aray_element
            
            i = beta*I*S
            r = 0.25*I
            
            newI = I + i - r
            newS = S - i
            newR = R + r

            states =jnp.array( [newS,i,newI,newR] )

            return states, states

        #--prior for beta and set gamma to be fixed

        log_beta     = numpyro.sample( "log_beta", dist.Normal( np.log(0.30)*np.ones( (T,SEASONS)) , 0.1 ) )
        #cum_log_beta = numpyro.deterministic("c_log_beta", jnp.cumsum(log_beta,0))
        
        beta     = numpyro.deterministic("beta", jnp.exp(log_beta))
        gamma    = 0.25

        #--prior for percent of population that is susceptible
        percent_sus = numpyro.sample("percent_sus", dist.Uniform(0.01 ,0.99) )

        #--process model
        S = jnp.zeros( (T,SEASONS) )
        i = jnp.zeros( (T,SEASONS) )
        I = jnp.zeros( (T,SEASONS) )
        R = jnp.zeros( (T,SEASONS) )
        
        #--Run process
        times = np.array([T,C])

        for s in np.arange(0,SEASONS):
            ts     = np.arange(0,times[s])
            betas  = beta[:times[s],s]

            times_betas = jnp.transpose( jnp.vstack([ts,betas]) )

            i0 = training_data[0,s] 
            I0 = training_data[0,s]
            
            S0 = 1.*percent_sus - i0
            R0 = 0.

            final, result = jax.lax.scan( one_step, jnp.array([S0,i0,I0,R0]), times_betas )

            states = numpyro.deterministic("states_{:d}".format(s),result)
               
            #--observations are assumed to be generated from a negative binomial
            phi = numpyro.sample("phi_{:d}".format(s), dist.Gamma(1,1) )

            ivals = numpyro.deterministic("ivals_{:d}".format(s), jnp.clip(result[:,1], 1*10**-10, jnp.inf))
            
            LL  = numpyro.sample("LL_{:d}".format(s), dist.NegativeBinomial2(ivals, phi), obs = training_data[:times[s],s] )

    nuts_kernel = NUTS(hier_sir_model)
    
    mcmc = MCMC( nuts_kernel , num_warmup=500, num_samples=1000)
    rng_key = random.PRNGKey(0)

    mcmc.run(rng_key
             , T=T
             , C=C
             , SEASONS = 2
             , training_data = training_data__normalized
             , extra_fields=('potential_energy',))
    
