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

#from statsmodels.tsa.holtwinters import ExponentialSmoothing as holt

#from patsy import dmatrix
from joblib import Parallel, delayed
from scoring import *


class comp_model_data(object):

    def __init__(self,LOCATION, HOLDOUTWEEKS):
        self.LOCATION     = LOCATION
        self.HOLDOUTWEEKS = HOLDOUTWEEKS

        self.load_data()
        self.collect_S0()
        self.split_into_seasons()
        self.prepare_training_data()
        

    def load_data(self):
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
                              ,"month": prev_day.month
                              ,"day":prev_day.day
                              ,"start_date":wk.startdate().strftime("%Y-%m-%d")
                              ,"end_date":wk.enddate().strftime("%Y-%m-%d")})

        hhs_data = pd.read_csv("../LUData/hhs_data__daily.csv")
        flu = hhs_data.loc[hhs_data.location==self.LOCATION]
        flu_times = flu.groupby("date").apply(from_date_to_epiweek)

        flu = flu.merge(flu_times, on = ["date"])

        #--sort by time
        flu = flu.sort_values("date")

        #--add in the FLUVIEW data
        fluview = pd.read_csv("../LUData/fluview_ili.csv")
        fluview["location"] = ["{:02d}".format(x) if x != "US" else x for x in fluview.location.values]

        fluview = fluview.loc[fluview.location==self.LOCATION]

        #--drop population from fluview and use the one in HHS
        fluview = fluview.drop(columns = ["population"] )

        flu = flu.merge( fluview, on = ["start_date","end_date","location"] )

        #--return
        self.flu = flu

    def collect_S0(self):
        flu = self.flu
        S0 = float(flu.population.iloc[0])
        self.S0 = S0

    def split_into_seasons(self):
        #HOLDOUTWEEKS = self.HOLDOUTWEEKS
        
        #--subset to october to august
        last_season        = self.flu.loc[ (self.flu.date >= "2022-01-30") & (self.flu.date <="2022-08-01")]
        current_season_flu = self.flu.loc[ (self.flu.date >= "2022-09-15") ]

        #--make sure that these are full weeks
        def week_check(x):
            if len(x)==7:
                return x
        last_season = last_season.groupby(["cdcformat"]).apply(week_check).reset_index(drop=True)
        current_season_flu = current_season_flu.groupby(["cdcformat"]).apply(week_check).reset_index(drop=True)

        #--attach training
        self.last_season        = last_season#.iloc#[:-HOLDOUTWEEKS,:]
        self.current_season_flu = current_season_flu#.iloc#[:-HOLDOUTWEEKS,:]

        #--attach hold out
        #self.last_season__test        = last_season.iloc[HOLDOUTWEEKS:,:]
        #self.current_season_flu__test = current_season_flu.iloc[HOLDOUTWEEKS:,:]
        
    def prepare_training_data(self):
       
        def fill_training_data(lastseason,currentseason,var):
            #--hosp data
            training_data = np.zeros((2,T))
            training_data[0,:]  = lastseason[var].values
            training_data[1,:C] = currentseason[var].values + 1 #--adding a one
            training_data[1,C:] = 0. #--adding zeros

            training_data = training_data.astype(int).T
            training_data__norm = training_data / self.S0

            return training_data, training_data__norm

        HOLDOUTWEEKS = self.HOLDOUTWEEKS

        if HOLDOUTWEEKS==0:
            current_season_flu = self.current_season_flu
        else:
            current_season_flu = self.current_season_flu[:-7*HOLDOUTWEEKS] # remove 7*WEEKS days

        T = self.last_season.shape[0]
        C = current_season_flu.shape[0]
        
        data__hosps , data__hosps__norm   = fill_training_data(self.last_season,current_season_flu, "hosps")
        data__deaths, data__deaths__norm  = fill_training_data(self.last_season,current_season_flu, "deaths")

        #--cases data
        #--last seasons cases
        last_season = self.last_season        
        
        last_season = last_season.reset_index(drop=True)
        last_season_weekly_ili      = last_season.groupby(["cdcformat"]).apply(lambda x: x.iloc[-1]["numili"])
        last_season_week_indicators = last_season.groupby(["cdcformat"]).apply(lambda x: x.index[-1])

        #--current season cases
        current_season_flu = current_season_flu.reset_index(drop=True)
        current_season_flu_weekly_ili      = current_season_flu.groupby(["cdcformat"]).apply(lambda x: x.iloc[-1]["numili"])
        current_season_flu_week_indicators = current_season_flu.groupby(["cdcformat"]).apply(lambda x: x.index[-1])

        weekly_T = len(last_season_week_indicators) 
        weekly_C = len(current_season_flu_week_indicators)

        data__ili = np.zeros((2,weekly_T))
        data__ili[0,:]         = last_season_weekly_ili
        data__ili[1,:weekly_C] = current_season_flu_weekly_ili

        #data__ili = training_data__ili.T

        ili_indicators              = np.zeros((2,weekly_T))
        ili_indicators[0,:]         = last_season_week_indicators
        ili_indicators[1,:weekly_C] = current_season_flu_week_indicators

        ili_indicators = ili_indicators.T.astype(int)

        #--attach
        HOLDOUTWEEKS = self.HOLDOUTWEEKS

        #--TRAINING
        self.training_data__hosps  = data__hosps
        self.training_data__deaths = data__deaths
        self.training_data__ili    = data__ili.T

        self.training_data__hosps__norm = data__hosps__norm
        self.training_data__deaths__norm = data__deaths__norm

        self.ili_indicators = ili_indicators

        self.T = T 
        self.C = C 
        self.weekly_T = weekly_T 
        self.weekly_C = weekly_C

        #--test hosps
        if HOLDOUTWEEKS==0:
            pass
        else:
            current_season_flu = self.current_season_flu[-7*HOLDOUTWEEKS:] # remove 7*WEEKS days
            data__hosps = np.array(current_season_flu.hosps.values).reshape(-1,)
            self.test_data__hosps = data__hosps

def model( T, C, weekly_T,weekly_C, SEASONS, ttl
                , prior_param                     = None
                , training_data__hosps            = None
                , training_data__deaths           = None
                , training_data__cases            = None
                , training_data__cases_indicators = None
                , future = 0):
    
    def one_step(carry, aray_element):
        S,s2i,I,i2h,H,R,h2d,D = carry
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

        states = jnp.array([nS,s2i,nI,i2h,nH,nR,h2d,nD] )

        return states, states

    #--normalize the hosps, deaths, and cases so that they are proportions
    training_data__hosps__normalized  = training_data__hosps  / ttl
    training_data__deaths__normalized = training_data__deaths / ttl
    training_data__cases__normalized  = training_data__cases  / ttl 

    #--prior for beta and set gamma to be fixed
    params   = numpyro.sample( "params", dist.Normal( jnp.zeros((weekly_T,SEASONS)), prior_param*jnp.ones((weekly_T,SEASONS)) ) )
    log_beta = np.repeat(params,7,axis=0) + jnp.log(0.25)
    
    #log_beta = jnp.dot(domain,params) + jnp.log(0.25)
    #log_beta =  numpyro.sample( "log_beta"  , dist.GaussianRandomWalk(prior_for_GRW*jnp.ones(SEASONS,), T) )
    beta     =  numpyro.deterministic("beta", jnp.exp(log_beta))
    #beta     =  beta.T

    #print(beta.shape)
    
    log_rho =  numpyro.sample( "log_rho", dist.Normal( np.log(0.25)*jnp.ones( (T,SEASONS) ) , 0.1 ) )
    rho     =  numpyro.deterministic("rho", jnp.exp(log_rho))

    #print(rho.shape)

    log_kappa =  numpyro.sample( "log_kappa", dist.Normal( np.log(0.01)*jnp.ones( (T,SEASONS) ) , 0.1 ) )
    kappa     = numpyro.deterministic("kappa", jnp.exp(log_kappa))

    #--prior for percent of population that is susceptible
    percent_sus = numpyro.sample("percent_sus", dist.Beta(0.5*20,0.5*20) )

    #--process model
    S   = jnp.zeros( (T,SEASONS) )
    I   = jnp.zeros( (T,SEASONS) )
    i2h = jnp.zeros( (T,SEASONS) ) #--incident hosps
    H   = jnp.zeros( (T,SEASONS) )
    R   = jnp.zeros( (T,SEASONS) )
    h2d = jnp.zeros( (T,SEASONS) ) #--incident deaths
    D   = jnp.zeros( (T,SEASONS) )

    #--fit simpler process to betafit

    #--Run process
    times        = np.array([T,C])
    weekly_times = np.array([weekly_T,weekly_C])

    phi_hosps = numpyro.sample("phi_hosps", dist.TruncatedNormal(low= 0.*jnp.ones(2,) ,loc=0*jnp.ones(2,) ,scale=10*jnp.ones(2,)) )
    phi_cases = numpyro.sample("phi_cases", dist.TruncatedNormal(low= 0.*jnp.ones(2,) ,loc=0*jnp.ones(2,) ,scale=10*jnp.ones(2,)) )

    for s in np.arange(0,SEASONS):
        ts     = np.arange(0,times[s])
        betas  = beta[:times[s],s]
        kappas = kappa[:times[s],s]
        rhos   = rho[:times[s],s]

        i2h0 = training_data__hosps__normalized[0,s] 
        H0   = training_data__hosps__normalized[0,s] 

        I0 = training_data__cases__normalized[0,s]*(1/7)
        S0 = 1.*percent_sus - I0

        R0 = 0.

        h2d0 = training_data__deaths__normalized[0,s] 
        D0   = training_data__deaths__normalized[0,s] 

        case_indicators = training_data__cases_indicators[:weekly_times[s],s]

        final, result = jax.lax.scan( one_step, jnp.array([S0,I0,I0,i2h0,H0,R0,h2d0,D0]), (ts,betas,rhos,kappas) )

        states = numpyro.deterministic("states_{:d}".format(s),result)

        #--observations are assumed to be generated from a negative binomial
        s2i__vals = numpyro.deterministic("s2i__vals_{:d}".format(s), jnp.clip(result[:,1], 1*10**-10, jnp.inf))
        i2h__vals = numpyro.deterministic("i2h__vals_{:d}".format(s), jnp.clip(result[:,3], 1*10**-10, jnp.inf))
        h2d__vals = numpyro.deterministic("h2d__vals_{:d}".format(s), jnp.clip(result[:,6], 1*10**-10, jnp.inf))

        #--LIKELIHOODS

        #--likelihood for hosps
        modeled_hosps = numpyro.deterministic("hosps_at_day_{:d}".format(s), i2h__vals*ttl)

        ll_hosps  = numpyro.sample("LL_H_{:d}".format(s), dist.NegativeBinomial2(i2h__vals*ttl, phi_hosps[s])
                                   , obs = training_data__hosps[:times[s],s] )

        #--likelihood for cases

        #--compute weekly sums
        modeled_weekly_splits     = jnp.split(s2i__vals, case_indicators+1)[:-1]
        modeled_weekly_cases      = jnp.array([sum(x) for x in modeled_weekly_splits])

        modeled_cases_at_week = numpyro.deterministic("cases_at_week_{:d}".format(s), modeled_weekly_cases*ttl)

        ll_cases  = numpyro.sample("LL_C_{:d}".format(s), dist.NegativeBinomial2(modeled_weekly_cases*ttl, phi_cases[s])
                                   ,obs = training_data__cases[:weekly_times[s],s] )

    #--prediction
    if future>0:
        forecast_betas = beta[C,SEASONS]*jnp.ones(future,)  #:C+future,SEASONS]
        forecast_betas = numpyro.deterministic("forecast_betas", forecast_betas)

        forecast_rhos   = rho[C  ,SEASONS]*jnp.ones((future,))
        forecast_kappas = kappa[C,SEASONS]*jnp.ones((future,))

        final, result = jax.lax.scan( one_step, states[C,:], (np.arange(0,future),forecast_betas, forecast_rhos, forecast_kappas) )

        numpyro.deterministic("forecast", result[:,3] )

def from_samples_to_forecast(samples,RETROSPECTIVE=0,S0=0,HOLDOUTWEEKS=0):
    from datetime import datetime, timedelta
    
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
        number_of_days_until_monday = next_monday(from_date="2023-01-01")
        monday = next_monday(True, from_date="2023-01-01")

        next_sat = next_saturday_after_monday_submission( number_of_days_until_monday)    
    target_end_dates = collect_target_end_dates(next_sat)

    print(target_end_dates)
    
    #--HOLDOUT adjustment
    if HOLDOUTWEEKS==0:
        pass
    else:
        target_end_dates = [ (datetime.strptime(x,"%Y-%m-%d") - timedelta(weeks=HOLDOUTWEEKS)).strftime("%Y-%m-%d") for x in target_end_dates]
    
    
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


    return forecast

        

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

    # LOCATION='16'
    # RETROSPECTIVE=0
    # END_DATE=0

    #--MODEL DATA
    model_data = comp_model_data(LOCATION=LOCATION,HOLDOUTWEEKS=2)

    #--RUNNING THE MODEL
    nuts_kernel = NUTS(model)
    mcmc        = MCMC( nuts_kernel , num_warmup=1500, num_samples=2000)
    rng_key     = random.PRNGKey(0)
    
    def model_run(mcmc, prior_param):
        mcmc.run(rng_key
                 , T= model_data.T
                 , C= model_data.C
                 , weekly_T = model_data.weekly_T
                 , weekly_C = model_data.weekly_C
                 , SEASONS = 2
                 , ttl = model_data.S0
                 , prior_param = prior_param
                 , training_data__hosps            = model_data.training_data__hosps
                 , training_data__deaths           = model_data.training_data__deaths
                 , training_data__cases            = model_data.training_data__ili
                 , training_data__cases_indicators = model_data.ili_indicators
                 , future = 28
                 , extra_fields=('potential_energy',))
        mcmc.print_summary()
        samples = mcmc.get_samples()
        return samples


    def score_over_params(P):
        samples = model_run(mcmc, P)
    
        #--BUILDING THE FORECAST DATA FRAME FROM THE MODEL
        forecast = from_samples_to_forecast(samples,RETROSPECTIVE=0,S0=model_data.S0, HOLDOUTWEEKS=2)

        truth = pd.read_csv("../../data-truth/truth-Incident Hospitalizations.csv")
        truth = truth.rename(columns = {"value":"truth"})
    
        forecast = forecast.merge(truth, left_on = ["target_end_date","location"], right_on = ["date","location"])

        forecast["quantile"] = forecast["quantile"].astype(float)
        forecast["value"]    = forecast["value"].astype(float)
    
        scores = forecast.groupby(["target_end_date"]).apply(WIS)

        return (np.mean(scores.values),P)
        
    results = Parallel(n_jobs=10)(delayed(score_over_params)(p) for p in np.linspace(0.01,1.0,10))
    results = sorted(results)

    best_param = results[0][-1]

    #--traiing complete now finish
    samples = model_run(mcmc, best_param)
    forecast = from_samples_to_forecast(samples,RETROSPECTIVE=0,S0=model_data.S0, HOLDOUTWEEKS=0)
    
    #--output data
    if RETROSPECTIVE:
        forecast.to_csv("./retrospective_analysis/location_{:s}_end_{:s}.csv".format(LOCATION,END_DATE),index=False)
    else:
        forecast.to_csv("./forecasts/location__{:s}.csv".format(LOCATION),index=False)

    # #--filter betas for forecasting
    # filtered_betas = []
    # for beta_sample in samples["log_beta"]:
    #     dom = np.arange(0,C)
    #     current_season_beta = beta_sample[-1,:C]
        
    #     hw = holt( np.array(current_season_beta), trend='add' ).fit()

    #     filtered_betas.append( np.exp(hw.forecast(28)) )

    # #--filter rhos
    # filtered_rhos = []
    # for rho_sample in samples["log_rho"]:
    #     dom = np.arange(0,C)
    #     current_season_rho = rho_sample[:C,-1]

    #     hw = holt( np.array(current_season_rho), trend='add' ).fit()
    #     filtered_rhos.append( np.exp(hw.forecast(28)) )

    # #--filter kappa
    # filtered_kappas = []
    # for kappa_sample in samples["log_kappa"]:
    #     dom = np.arange(0,C)
    #     current_season_kappa = kappa_sample[:C,-1]

    #     hw = holt( np.array(current_season_kappa), trend='add' ).fit()
        
    #     filtered_kappas.append( np.exp(hw.forecast(28)) )

    # def one_step(carry, aray_element):
    #     S,s2i,I,i2h,H,R,h2d,D = carry
    #     t,beta,rho,kappa  = aray_element

    #     rho = 0.25
            
    #     s2i  = beta*I*S        # S->I
    #     i2r  = (rho*0.25)*I    # I->R (lambda = 0.25)
    #     i2h  = (rho*0.75)*I    # I->H 
    #     h2d  = (kappa*0.01)*H  # H->D
    #     h2r  = (kappa*0.99)*H  # H->R

    #     nI = I + s2i - (i2r + i2h)
    #     nH = H + i2h - (h2d + h2r)
    #     nD = D + h2d
    #     nS = S - s2i
            
    #     nR = R + (i2r + h2r)

    #     states = jnp.array([nS,s2i,nI,i2h,nH,nR,h2d,nD] )
    #     return states,states
    
    # #--collect last states from fit
    # sum_hosps = np.zeros(28,)

    # hosp_forecast = {"sample":[], 'time':[], "forecast":[]}
    
    # for n,state in enumerate(samples["states_1"]):
    #     last_state = state[-1,:]
    #     nS,s2i,nI,i2h,nH,nR,h2d,nD = last_state

    #     forecast_betas  = filtered_betas[n]
    #     forecast_rhos   = filtered_rhos[n]
    #     forecast_kappas = filtered_kappas[n]

    #     final, result = jax.lax.scan( one_step, last_state, (np.arange(0,28),forecast_betas, forecast_rhos, forecast_kappas) )

    #     hosps = np.array(result[:,3])
    #     hosps[np.isnan(hosps)]    = 0
    #     hosps[abs(hosps)==np.inf] = 0


    #     hosp_forecast['sample'].extend(   [n]*28 )
    #     hosp_forecast['time'].extend( np.arange(0,28) )
    #     hosp_forecast['forecast'].extend( hosps*S0 )
    # hosp_forecast = pd.DataFrame(hosp_forecast)


    
    


    
   
    # predictions = {"t":[],"sample":[],"value":[]}
    # for sample,values in enumerate(samples["forecast"]):
    #     for time,value in enumerate(values):
    #         predictions['t'].append(time)
    #         predictions['sample'].append(sample)
    #         predictions['value'].append(value*S0)
    # predictions = pd.DataFrame(predictions)
 
    # weekly_agg = {"t":[],"wk":[]}
    # for t in np.arange(0,28):
    #     weekly_agg['t'].append(t)

    #     if 0<=t<=6:
    #         weekly_agg['wk'].append(1)
    #     elif 7<=t<=13:
    #         weekly_agg['wk'].append(2)
    #     elif 14<=t<=20:
    #         weekly_agg['wk'].append(3)
    #     elif 21<=t<=28:
    #         weekly_agg['wk'].append(4)
    # weekly_agg = pd.DataFrame(weekly_agg)

    # predictions = predictions.merge(weekly_agg, on = ['t'])

    # def accumulate_daily_to_weekly(x):
    #     return pd.Series({"hosps":x.value.sum()})
    # predictions_weekly = predictions.groupby(["wk","sample"]).apply( accumulate_daily_to_weekly ).reset_index()

    # predictions_weekly__wide = pd.pivot_table(index="sample", columns = ["wk"], values = ["hosps"], data = predictions_weekly) 

    # Q = np.round([0.01,0.025] + list(np.arange(0.05,0.95+0.05,0.05)) + [0.975,0.99],3)
    # N = len(Q)
    
    # quantiles = np.percentile(predictions_weekly__wide,100*Q,axis=0)

    # #--compute all time information for forecast submission
    # if RETROSPECTIVE:
    #     number_of_days_until_monday = next_monday(from_date=END_DATE)
    #     monday = next_monday(True, from_date=END_DATE)

    #     next_sat = next_saturday_after_monday_submission( number_of_days_until_monday, from_date=END_DATE )    
    # else:
    #     number_of_days_until_monday = next_monday(from_date="2023-01-01")
    #     monday = next_monday(True, from_date="2023-01-01")

    #     next_sat = next_saturday_after_monday_submission( number_of_days_until_monday)    
    # target_end_dates = collect_target_end_dates(next_sat)
    
    # #--store all forecasts in a DataFrame
    # forecast = {"target":[], "target_end_date":[], "quantile":[], "value":[]}
    # for n,week_ahead_prediction in enumerate(quantiles.T):
    #     forecast["value"].extend(week_ahead_prediction)
    #     forecast["quantile"].extend(Q)

    #     #--items that need to be repeated Q times
    #     forecast["target"].extend( ["{:d} wk ahead inc flu hosp".format(n+1)]*N)
    #     forecast["target_end_date"].extend( [target_end_dates[n]]*N)
        
    # forecast = pd.DataFrame(forecast)
    # forecast["location"]      = LOCATION
    # forecast["type"]          = "quantile"
    # forecast["forecast_date"] = monday

    # #--format quantile to three decimals
    # forecast['quantile'] = ["{:0.3f}".format(q) for q in forecast["quantile"]]

    # #--format values to three decimals
    # forecast['value'] = ["{:0.3f}".format(q) for q in forecast["value"]]


        #--dep
        
    # last_season_week_indicators = {"cdcformat":[],"ind":[]}
    # last_season_weekly_ili      = {"cdcformat":[],"numili":[]}
    # for idx, x in last_season.groupby(["cdcformat"]):
    #     if len(x)<7:
    #         continue
    #     else:
    #         last_season_week_indicators["cdcformat"].append(idx)
    #         last_season_week_indicators["ind"].append(x.index[-1])

    #         last_season_weekly_ili["cdcformat"].append(idx)
    #         last_season_weekly_ili["numili"].append(x.numili.values[-1])
            
    # last_season_week_indicators = pd.DataFrame(last_season_week_indicators)
    # last_season_weekly_ili = pd.DataFrame(last_season_weekly_ili)

       
       
 
    # flu = hhs_data.loc[hhs_data.location==LOCATION]
    # flu_times = flu.groupby("date").apply(from_date_to_epiweek)

    # flu = flu.merge(flu_times, on = ["date"])

    # #--sort by time
    # flu = flu.sort_values("date")

    # #--add in the FLUVIEW data
    # fluview = pd.read_csv("../LUData/fluview_ili.csv")
    # fluview["location"] = ["{:02d}".format(x) if x != "US" else x for x in fluview.location.values]
    
    # fluview = fluview.loc[fluview.location==LOCATION]

    # #--drop population from fluview and use the one in HHS
    # fluview = fluview.drop(columns = ["population"] )
    
    # flu = flu.merge( fluview, on = ["start_date","end_date","location"] )
    
    #--subset to october to august
    #last_season        = flu.loc[ (flu.date >= "2022-01-30") & (flu.date <="2022-08-01")]
    #current_season_flu = flu.loc[ (flu.date >= "2022-09-15") ]

    #--make sure that these are full weeks
    #def week_check(x):
    #     if len(x)==7:
    #         return x
    # last_season = last_season.groupby(["cdcformat"]).apply(week_check).reset_index(drop=True)
    # current_season_flu = current_season_flu.groupby(["cdcformat"]).apply(week_check).reset_index(drop=True)
    
    #--prepare training data
    # T = last_season.shape[0]
    # C = current_season_flu.shape[0]

    # #--hosp data
    # training_data__hosps = np.zeros((2,T))
    # training_data__hosps[0,:]  = last_season.hosps.values
    # training_data__hosps[1,:C] = current_season_flu.hosps.values + 1 #--adding a one
    # training_data__hosps[1,C:] = 0. #--adding some small number

    # training_data__hosps = training_data__hosps.astype(int).T
    # training_data__hosps__norm = training_data__hosps / S0
    
    # #--death data
    # training_data__deaths = np.zeros((2,T))
    # training_data__deaths[0,:]  = last_season.deaths.values
    # training_data__deaths[1,:C] = current_season_flu.deaths.values + 1 #--adding a one
    # training_data__deaths[1,C:] = 0. #--adding some small number

    # training_data__deaths = training_data__deaths.astype(int).T
    # training_data__deaths__norm = training_data__deaths / S0
   
    # #--cases data
    # #--last seasons cases
    # last_season = last_season.reset_index(drop=True)
    # last_season_weekly_ili      = last_season.groupby(["cdcformat"]).apply(lambda x: x.iloc[-1]["numili"])
    # last_season_week_indicators = last_season.groupby(["cdcformat"]).apply(lambda x: x.index[-1])

    # #--current season cases
    # current_season_flu = current_season_flu.reset_index(drop=True)

    # current_season_flu_weekly_ili      = current_season_flu.groupby(["cdcformat"]).apply(lambda x: x.iloc[-1]["numili"])
    # current_season_flu_week_indicators = current_season_flu.groupby(["cdcformat"]).apply(lambda x: x.index[-1])
   
    # weekly_T = len(last_season_week_indicators) 
    # weekly_C = len(current_season_flu_week_indicators)
    
    # training_data__ili = np.zeros((2,weekly_T))
    # training_data__ili[0,:]         = last_season_weekly_ili
    # training_data__ili[1,:weekly_C] = current_season_flu_weekly_ili

    # training_data__ili = training_data__ili.T
    
    # ili_indicators              = np.zeros((2,weekly_T))
    # ili_indicators[0,:]         = last_season_week_indicators
    # ili_indicators[1,:weekly_C] = current_season_flu_week_indicators

    # ili_indicators = ili_indicators.T.astype(int)
 
