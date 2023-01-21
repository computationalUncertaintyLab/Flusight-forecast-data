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
from jax.experimental.ode import odeint

#from patsy import dmatrix
from joblib import Parallel, delayed
from scoring import *
import itertools

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

        hhs_data  = pd.read_csv("../LUData/hhs_data__daily.csv")
        flu       = hhs_data.loc[hhs_data.location==self.LOCATION]
        flu_times = flu.groupby("date").apply(from_date_to_epiweek)

        flu = flu.merge(flu_times, on = ["date"])

        #--sort by time
        flu = flu.sort_values("date")

        #--add in the FLUVIEW data
        fluview = pd.read_csv("../LUData/fluview_ili.csv")
        #fluview["location"] = ["{:02d}".format(x) if x != "US" and isinstance(x,int) else x for x in fluview.location.values]

        fluview = fluview.loc[fluview.location==self.LOCATION]

        #--drop population from fluview and use the one in HHS
        fluview = fluview.drop(columns = ["population"] )

        flu = flu.merge( fluview, on = ["start_date","end_date","location"], how="left" )

        #--return
        self.flu = flu

    def collect_S0(self):
        flu = self.flu
        S0 = float(flu.population.iloc[0])
        self.S0 = S0

    def split_into_seasons(self):
        #--subset to october to august
        last_season        = self.flu.loc[ (self.flu.date >= "2022-01-30") & (self.flu.date <="2022-08-01")]
        current_season_flu = self.flu.loc[ (self.flu.date >= "2022-09-15") ]
        #current_season_flu = self.flu.loc[ (self.flu.date >= "2022-10-15") ]
        

        #--make sure that these are full weeks
        def week_check(x):
            if len(x)==7 or len(x) == 6:
                return x
        last_season        = last_season.groupby(["ew"]).apply(week_check).reset_index(drop=True)
        current_season_flu = current_season_flu.groupby(["ew"]).apply(week_check).reset_index(drop=True)

        #--attach training
        self.last_season        = last_season#.iloc#[:-HOLDOUTWEEKS,:]
        self.current_season_flu = current_season_flu#.iloc#[:-HOLDOUTWEEKS,:]

    def prepare_training_data(self):
       
        def fill_training_data(lastseason,currentseason,var):
            #--hosp data
            training_data = np.zeros((2,T))
            training_data[0,:]  = lastseason[var].values + 1
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
                , prior_phis                      = None
                , training_data__hosps            = None
                , training_data__deaths           = None
                , training_data__cases            = None
                , training_data__cases_indicators = None
                , future = 0):

    def one_step(carry, aray_element):
            S,s2i,I,i2h,H,R,h2d,D = carry
            t,beta,rho,kappa      = aray_element

            I = I+ 1./ttl #--always infected ppl circulating in the system to allow for an outbreak
            
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

            states = jnp.vstack( (nS,s2i,nI,i2h,nH,nR,h2d,nD) )

            return states, states

    def rkstep(carry,beta_rho_kappa):
        def evolution(states,t,params):
            s,i,h,r,d,_ = states
            beta,rho,kappa = params
            
            ds_dt = -beta*i*s
            di_dt = beta*i*s - i*(rho*0.25 + rho*0.75)
            dh_dt = i*rho*0.75 - h*(kappa*0.01 + kappa*0.99)
            dr_dt = h*kappa*0.99 + i*rho*0.25
            dd_dt = h*kappa*0.01

            dc_dt = i*rho*0.75 #--to keep track
            
            return jnp.stack([ds_dt, di_dt, dh_dt, dr_dt, dd_dt,dc_dt])
        
        last_states = carry

        def grab_params(t, beta_rho_kappa):
            return beta_rho_kappa[t.astype(int),:]

        grab = lambda t: grab_params(t,beta_rho_kappa)
        
        states = odeint(evolution, last_states , jnp.array([0.,1.]), beta_rho_kappa, rtol=1e-6, atol=1e-5, mxstep=1000)
        next_state = states[-1,:,:]

        return next_state, next_state

    
    # def evolution(states,t):
    #     s,i,h,r,d = states
    #     beta,rho,kappa = grab(t)

    #     ds_dt = -beta*i*s
    #     di_dt = beta*i*s - i*(rho*0.25 + rho*0.75)
    #     dh_dt = i*rho*0.75 - h*(kappa*0.01 + kappa*0.99)
    #     dr_dt = h*kappa*0.99 + i*rho*0.25
    #     dd_dt = h*kappa*0.01

    #     return jnp.stack([ds_dt, di_dt, dh_dt, dr_dt, dd_dt])

    # def grab_params(t, beta_rho_kappa):
    #     return beta_rho_kappa[t.astype(int),:]
    
    #--normalize the hosps, deaths, and cases so that they are proportions
    training_data__hosps__normalized  = training_data__hosps  / ttl
    training_data__deaths__normalized = training_data__deaths / ttl
    training_data__cases__normalized  = training_data__cases  / ttl 

    weeks = jnp.arange(0,weekly_T+4).reshape(-1,1)
    ttl_weeks = weekly_T+4
    
    days  = jnp.arange(0.,T+4*7).reshape(-1,1)
    ttl_days = T+4*7
    
    #--prior for beta and set gamma to be fixed
    params   = numpyro.sample( "beta_params", dist.Normal( jnp.zeros((ttl_weeks,SEASONS)), prior_param*jnp.ones((ttl_weeks,SEASONS)) ) )
    log_beta = np.repeat(params,7,axis=0) + jnp.log(0.25)
    beta     = numpyro.deterministic("beta", jnp.exp(log_beta))

    #beta = jnp.vstack([beta[0,:], beta])
    
    training_beta = beta[:-28,:] 
    
    log_rho =  jnp.log(0.25)*jnp.ones((T,SEASONS)) #numpyro.sample( "log_rho", dist.Normal( np.log(0.25)*jnp.ones( (T,SEASONS) ) , 0.1 ) )
    rho     =  numpyro.deterministic("rho", jnp.exp(log_rho))

    #rho = jnp.vstack([rho[0,:], rho])

    log_kappa =  jnp.log(0.05)*jnp.ones((T,SEASONS))
    kappa     =  numpyro.deterministic("kappa", jnp.exp(log_kappa))

    #kappa = jnp.vstack([kappa[0,:], kappa])
    
    #--prior for percent of population that is susceptible
    percent_sus = numpyro.sample("percent_sus", dist.Beta(0.5*20,0.5*20) )

    phi_hosps = numpyro.sample("phi_hosps", dist.TruncatedNormal(low= 0.*jnp.ones(2,) ,loc=0*jnp.ones(2,) ,scale=prior_phis[0]*jnp.ones(2,)) )
    phi_cases = numpyro.sample("phi_cases", dist.TruncatedNormal(low= 0.*jnp.ones(2,) ,loc=0*jnp.ones(2,) ,scale=prior_phis[1]*jnp.ones(2,)) )

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
    ts = jnp.arange(0,T)

    H0   = training_data__hosps__normalized[0,:] 
    I0 = training_data__cases__normalized[0,:]*(1/7)
    S0 = (1.*percent_sus - I0)*jnp.ones(SEASONS,)
    R0 = 0.*jnp.ones(SEASONS,)
    D0   = training_data__deaths__normalized[0,:]

    i2h0 = training_data__hosps__normalized[0,:] 
    h2d0 = training_data__deaths__normalized[0,:]

    #weekly_times = np.array([weekly_T,weekly_C])
    #case_indicators = training_data__cases_indicators[:weekly_times[s],s]
    
    #initial_states = jnp.vstack( (S0,I0,I0,i2h0,H0,R0,h2d0,D0))

    initial_states = jnp.vstack( (S0,I0,H0,R0,D0,  H0) )
    
    #final, result = jax.lax.scan( one_step, initial_states, (ts,training_beta,rho,kappa) ) #--T by STATES by SEASONS
   
    beta_rho_kappa = jnp.stack([training_beta, rho, kappa])
    beta_rho_kappa = np.rollaxis(beta_rho_kappa,1)

    #grab = lambda t: grab_params(t,beta_rho_kappa)
    #result = odeint(evolution, initial_states , jnp.arange(0.,T), rtol=1e-6, atol=1e-5, mxstep=1000)
    
    fnl,result     = jax.lax.scan(rkstep, initial_states, beta_rho_kappa )
    states = numpyro.deterministic("states",result)

    diffs = jnp.diff(states.T,1)

    inc_hosps = jnp.vstack([ H0, diffs[:,-1,:].T])
    

    #--add the initial state
    #inc_hosps = jnp.vstack([H0,inc_hosps])
    
    modeled_hosps = numpyro.deterministic("hosps_at_day", inc_hosps*ttl) #-T by SEASONS
    modeled_hosps = jnp.clip(modeled_hosps,0,np.inf)

    times = jnp.array([T,C])
    ts_vec = np.repeat(ts[:,np.newaxis],2,axis=1)
    
    data_indices  = ts_vec < times
    
    ll_hosps  = numpyro.sample("LL_H", dist.NegativeBinomial2(modeled_hosps*data_indices + 10**-10, phi_hosps), obs = training_data__hosps*data_indices )

        #--likelihood for cases

        #--compute weekly sums
        # modeled_weekly_splits     = jnp.split(s2i__vals, case_indicators+1)[:-1]
        # modeled_weekly_cases      = jnp.array([sum(x) for x in modeled_weekly_splits])

        # modeled_cases_at_week = numpyro.deterministic("cases_at_week_{:d}".format(s), modeled_weekly_cases*ttl)

        # ll_cases  = numpyro.sample("LL_C_{:d}".format(s), dist.NegativeBinomial2(modeled_weekly_cases*ttl, phi_cases[s])
        #                            ,obs = training_data__cases[:weekly_times[s],s] )

    #--prediction
    if future>0:
        forecast_betas  = numpyro.deterministic("forecast_betas", beta[-28 :,-1])
        forecast_rhos   = numpyro.deterministic("forecast_rhos" , rho[-28  :,-1])
        forecast_kappas = numpyro.deterministic("forecast_kappa", kappa[-28:,-1])

        last_state = states[C,:,-1]
        last_state = last_state.reshape(-1,1)
        
        #final, result = jax.lax.scan( one_step, last_state, (np.arange(0,future),forecast_betas, forecast_rhos, forecast_kappas) )

        forecast_beta_rho_kappa = jnp.vstack([forecast_betas, forecast_rhos, forecast_kappas])
        forecast_beta_rho_kappa = np.rollaxis(forecast_beta_rho_kappa,1)
        
        # grab = lambda t: grab_params(t,forecast_beta_rho_kappa)
        # result = odeint(evolution, initial_states , jnp.arange(0.,28), rtol=1e-6, atol=1e-5, mxstep=1000)

        #print(last_state.shape)
        #print(forecast_beta_rho_kappa.shape)
        fnl,result     = jax.lax.scan(rkstep, last_state, forecast_beta_rho_kappa )

        diffs = jnp.diff(result.T,1)
        #print(diffs.shape)
        
        inc_hosps = jnp.vstack([ last_state, diffs[:,-1,:].T])
 
        numpyro.deterministic("forecast", inc_hosps )

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

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--LOCATION'     ,type=str) 
    # parser.add_argument('--RETROSPECTIVE',type=int, nargs = "?", const=0)
    # parser.add_argument('--END_DATE'     ,type=str, nargs = "?", const=0)
    
    # args = parser.parse_args()
    # LOCATION      = args.LOCATION
    # RETROSPECTIVE = args.RETROSPECTIVE
    # END_DATE      = args.END_DATE


    LOCATION = '42'
    RETROSPECTIVE = 0
    END_DATE=0
    
    #--MODEL DATA
    model_data = comp_model_data(LOCATION=LOCATION,HOLDOUTWEEKS=4)

    #--RUNNING THE MODEL
    nuts_kernel = NUTS(model)
    mcmc        = MCMC( nuts_kernel , num_warmup=1500, num_samples=2000,progress_bar=True)
    rng_key     = random.PRNGKey(0)

    def model_run(mcmc, prior_param, prior_phis, model_data):
        mcmc.run(rng_key
                 , T= model_data.T
                 , C= model_data.C
                 , weekly_T = model_data.weekly_T
                 , weekly_C = model_data.weekly_C
                 , SEASONS = 2
                 , ttl = model_data.S0
                 , prior_param = prior_param
                 , prior_phis  = prior_phis
                 , training_data__hosps            = model_data.training_data__hosps
                 , training_data__deaths           = model_data.training_data__deaths
                 , training_data__cases            = model_data.training_data__ili
                 , training_data__cases_indicators = model_data.ili_indicators
                 , future = 28
                 , extra_fields=('potential_energy',))
        mcmc.print_summary()
        samples = mcmc.get_samples()
        return samples

    def score_over_params(P,Q, model_data):
        samples = model_run(mcmc, P, Q, model_data)
    
        #--BUILDING THE FORECAST DATA FRAME FROM THE MODEL
        forecast = from_samples_to_forecast(samples,RETROSPECTIVE=0,S0=model_data.S0, HOLDOUTWEEKS=4)

        truth = pd.read_csv("../../data-truth/truth-Incident Hospitalizations.csv")
        truth = truth.rename(columns = {"value":"truth"})
    
        forecast = forecast.merge(truth, left_on = ["target_end_date","location"], right_on = ["date","location"])

        forecast["quantile"] = forecast["quantile"].astype(float)
        forecast["value"]    = forecast["value"].astype(float)
    
        scores = forecast.groupby(["target_end_date"]).apply(WIS)

        return (np.mean(scores.values),P,Q)

    score_crossval = lambda P,Q: score_over_params(P,Q,model_data) 
    
    # if LOCATION == "US":
    #     combos = [x for x in itertools.product(np.linspace(0.005,0.3,30),[1000], [1000])]
    # else:
    #     combos = [x for x in itertools.product(np.linspace(0.001,0.5,100),[1000], [1000])]
    # results = Parallel(n_jobs=30)(delayed(score_crossval)(p,[q,r]) for (p,q,r) in combos)
    # results = sorted(results)

    #best_beta_param, best_phis = results[0][-2:]

    best_beta_param  = 1.0
    best_phis = [1./10,1./10]

    # print(results[:20])
    # print(best_beta_param)
    # print(best_phis)

    #--traiing complete now finish
    forecast_data = comp_model_data(LOCATION=LOCATION,HOLDOUTWEEKS=0)
    samples = model_run(mcmc,best_beta_param, best_phis, forecast_data)
    forecast = from_samples_to_forecast(samples,RETROSPECTIVE=0,S0=model_data.S0, HOLDOUTWEEKS=0)

    #--output data
    if RETROSPECTIVE:
        forecast.to_csv("./retrospective_analysis/location_{:s}_end_{:s}.csv".format(LOCATION,END_DATE),index=False)
    else:
        forecast.to_csv("./forecasts/location__{:s}.csv".format(LOCATION),index=False)

    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(2,3)

    C = forecast_data.C
    S0 = forecast_data.S0
    
    #--current seasons
    observed_hosps = forecast_data.training_data__hosps[:C,-1]
    infered_hosps  = np.median(samples["hosps_at_day"],0)[:C,-1]
    
    domain = np.arange(0,C)

    ax = axs[0,0]
    
    ax.scatter(domain, observed_hosps,s=5, color = "k")
    ax.plot(   domain, infered_hosps    , color = "r" )

    forecast_horizon = np.arange(C,C+28)
    forecast = samples["forecast"].mean(0)
    
    ax.plot(forecast_horizon, forecast*S0, color = "blue", ls = "--")

    ax = axs[0,1]
    beta_trajectory  = samples["beta"].mean(0)[:,-1]
    ax.plot( beta_trajectory )
    ax.axhline(0.25, color="black")

    ax = axs[0,2]
    for sample in samples["states"]:
        hosps = sample[:,3,-1]
        ax.plot(hosps, alpha=0.1, color="black",lw=1)


    #--last season
    T = forecast_data.T
    observed_hosps = forecast_data.training_data__hosps[:T,0]
    infered_hosps  = samples["hosps_at_day"].mean(0)[:T,0]
    
    domain = np.arange(0,T)

    ax = axs[1,0]
    
    ax.scatter(domain, observed_hosps,s=5, color = "k")
    ax.plot(   domain, infered_hosps    , color = "r" )

    ax = axs[1,1]
    beta_trajectory  = samples["beta"].mean(0)[:,0]
    ax.plot( beta_trajectory )
    ax.axhline(0.25, color="black")

    ax = axs[1,2]
    for sample in samples["states"]:
        hosps = sample[:,3,0]
        ax.plot(hosps, alpha=0.1, color="black",lw=1)

    plt.show()



    # training_data__hosps  = forecast_data.training_data__hosps
    # training_data__cases  = forecast_data.training_data__ili
    # training_data__deaths = forecast_data.training_data__deaths

    # ttl = forecast_data.S0
    







    
