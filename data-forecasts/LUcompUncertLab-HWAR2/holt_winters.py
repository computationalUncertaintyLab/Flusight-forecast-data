#mcandrew

import sys
from collections import namedtuple

import numpy as np
import pandas as pd

import stan
import statsmodels.api as sm

import argparse

#--find next monday
def next_monday(dat=0):
    from datetime import datetime,timedelta
    dt  = datetime.now()
    day = dt.weekday()

    counter = 0
    while day%7 !=0:
        counter+=1
        day+=1

    if dat:
        return (dt+timedelta(days=counter)).strftime("%Y-%m-%d")
    return counter

def next_saturday_after_monday_submission( num_days_until_monday ):
    from datetime import datetime, timedelta
    dt  = datetime.now()+ timedelta(days=num_days_until_monday)

    while dt.weekday() !=5 :
        dt = dt + timedelta(days=1)
    return dt.strftime("%Y-%m-%d")

def collect_target_end_dates(first_saturday):
    from datetime import datetime, timedelta
    sat = datetime.strptime(first_saturday,'%Y-%m-%d')

    target_end_dates = [first_saturday]
    for _ in range(3):
        sat =sat + timedelta(days=7)
        target_end_dates.append( sat.strftime("%Y-%m-%d") )
    return target_end_dates 

def define_submission_date(num_days_until_monday):
    from datetime import datetime, timedelta
    now  = datetime.now()
    monday_submission = now+timedelta(days=num_days_until_monday)

    return monday_submission.strftime("%Y-%m-%d")

if __name__ == "__main__":

    #--accepts one parameter that subsets data to Location
    parser = argparse.ArgumentParser()
    parser.add_argument('--LOCATION',type=str)

    args = parser.parse_args()
    LOCATION     = args.LOCATION
 
    flu = pd.read_csv("../../data-truth/truth-Incident Hospitalizations.csv")
    flu["date"] = pd.to_datetime(flu.date)
    
    flu2022 = flu.loc[flu.date > "2021-10-01"]

    location_specific_values = flu2022.loc[flu2022.location==LOCATION]

    #--detrend using the Holt Winters procedure
    ys = location_specific_values.value.values
    xs = np.arange(0,len(location_specific_values))
    T = len(ys)

    def fit_HW(ys):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing( ys , trend = 'add')
        model = model.fit()

        y_infer, y_forecast  = model.fittedvalues,model.forecast(4)
        return y_infer, y_forecast
    y_infer, y_forecast = fit_HW(ys)

    #--compute residuals from Holt Winters procedure
    resids = ys - y_infer
    
    #--standardize the residuals by subtracting mean and diviiding by std 
    params = namedtuple("params","avg sd")
    stand__params   = params( avg = resids.mean(), sd = resids.std() )
    centered_resids = (resids - stand__params.avg)/stand__params.sd
    
    stan_model = '''
        data {
           int T;
           vector [T] y;
        }
        parameters { 
           real beta0; 
           real <lower=-1,upper = 1>beta [2]; 
           real <lower = 0> sigma;
        }
        model {

              beta0~normal(0,10);
              for (l in 1:2){
                 beta[l]~normal(0,10);
              }
              sigma~cauchy(5,5);

             for (t in 3:T){
                 //form predictor
                 real mu = 0;
                 for (l in 1:2){
                    mu+= beta[l]*y[t-l];
                 }
                target+=normal_lpdf(y[t]|beta0+mu,sigma);
             }
        }
        generated quantities {
           vector [T] y_hat; 
           vector [2+4] y_forecast;

           //inference
           for (l in 1:2){
              y_hat[l] = y[l];
           } 
           for (t in 3:T){
             //form predictor
             real mu = 0;
             for (l in 1:2){
                mu+= beta[l]*y[t-l];
             }
             y_hat[t] = normal_rng(beta0+mu,sigma);
           }

          //Forecasting

          for (l in 1:2){
             y_forecast[l] = y[T+1-l];
          }
          for (t in 2+1:2+4){
             //form predictor
             real mu = 0;
             for (l in 1:2){
                mu+= beta[l]*y_forecast[t-l];
             }
             y_forecast[t] = normal_rng(beta0+mu,sigma);
          }
        }  
    '''
    model_data = {"T":len(centered_resids), "y":centered_resids}
    posterior  = stan.build(stan_model, data=model_data)
    fit        = posterior.sample(num_chains=1, num_samples=5000)
    
    uncentered_resids = fit.get("y_hat")*stand__params.sd + stand__params.avg
    
    orig = {"n":[], "pred":[], "t":[]}
    for n,sample in enumerate(uncentered_resids.T):
        T = len(sample)
        times = np.arange(0,T)
        orig["pred"].extend( sample + y_infer )

        T = len(sample)
        orig["t"].extend(times)
        orig["n"].extend( [n]*T )
    inferences = pd.DataFrame(orig)

    reference_time = len(ys)
    uncentered_resid_forecasts = fit.get("y_forecast")*stand__params.sd + stand__params.avg
    orig = {"n":[], "pred":[], "t":[]}
    for n,sample in enumerate(uncentered_resid_forecasts.T):

        T = 4
        times = np.arange(reference_time,reference_time+T)
        orig["pred"].extend( sample[-4:] + y_forecast  )

        orig["t"].extend(times)
        orig["n"].extend( [n]*T )
    predictions = pd.DataFrame(orig)

    predictions.loc[predictions["pred"]<0,"pred"] = 0

    wide = pd.pivot_table(index = "n", columns = "t", values = "pred", data = predictions)

    Q = np.round([0.01,0.025] + list(np.arange(0.05,0.95+0.05,0.05)) + [0.975,0.99],3)
    N = len(Q)
    
    quantiles = np.percentile(wide,100*Q,axis=0)

    #--compute all time information for forecast submission
    number_of_days_until_monday = next_monday()
    monday = next_monday(True)
    
    next_sat = next_saturday_after_monday_submission( number_of_days_until_monday )
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
    forecast.to_csv("./forecasts/location__{:s}.csv".format(LOCATION),index=False)
