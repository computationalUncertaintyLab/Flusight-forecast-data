#mcandrew

import sys
import numpy as np
import pandas as pd
import pickle

from interface import interface

import numpyro.distributions as dist
import numpyro

from numpyro.infer import MCMC, NUTS, HMC, Predictive
from numpyro.distributions import constraints

import jax
from jax import random
import jax.numpy as jnp



class buildWeightMatrix(object):
    def __init__(self):
        self.locations = self.grabLocations()

    def grabLocations(self):
        import pandas as pd
        return pd.read_csv("communityquantiles__interpolated.csv").location_name.unique()

    def add_data(self,X,Y):
        self.X=X
        self.Y=Y
    
    def fitmodel(self):
        import stan
        modeldesc = '''
        data {
            int locs;
            int sublocs;
            int T;
            matrix [T,sublocs]    X;
            matrix [T,locs] Y;
        }
        parameters {
           vector <lower=0> [locs] sigmas ;
           simplex [sublocs] W [locs];
           vector [locs] c ;  // EXPERIMENTAL
        }
        model {
           for (l in 1:locs){
              sigmas[l] ~ cauchy(1,2.5);
           }
           
          // adding experiental intercept
          for (l in 1:locs){
              c[l] ~ normal(0,500);
           }

           for (l in 1: locs){
              W[l]~dirichlet( rep_vector(1,6) );
              Y[:,l]~normal( c[l]+X*W[l],sigmas[l]);
           }
        }
        '''
        data = {"locs": Y.shape[-1], "T":Y.shape[0], "sublocs":X.shape[-1],"X":X,"Y":Y}
        posterior = stan.build(modeldesc, data=data)
        fit = posterior.sample(num_samples=1000,num_chains=3)

        self.fit = fit
        return fit

    def fitOLSmodel(self):
        X,Y = self.X,self.Y
        import statsmodels.api as sm

        nlocs = Y.shape[-1]
        nsublocs = X.shape[-1]
        
        W = np.zeros((nlocs,nsublocs))

        for n,y in enumerate(Y.T):
            model = sm.OLS(y,X).fit()
            W[n,:] = model.params
        return W


def model(X,Y, quantiles = None):
    T = Y.shape[0]
    L = Y.shape[1]
    
    W      = numpyro.sample( "W", dist.Dirichlet( jnp.ones((L,6)) ) )
    W      = W.T
    
    C      = numpyro.sample("C" , dist.Normal( jnp.zeros((L,)), 10*jnp.ones((L,)) )) 

    sigmas = numpyro.sample("S", dist.HalfCauchy( jnp.ones( (L,)) ) ) 
    
    yhat = numpyro.deterministic("yhat", C + jnp.matmul(X,W))
    
    ll  = numpyro.sample("ll", dist.Normal(yhat, sigmas), obs = Y)

    if quantiles is not None:
        Q = numpyro.deterministic("Q", jnp.matmul(quantiles,W) )
    
    
if __name__ == "__main__":

    inter = interface()
    flu = inter.fluDataBy100

    W = buildWeightMatrix()

    location_names = W.grabLocations()
    
    fluwide = pd.pivot_table(index="date", columns="location_name",values="value",data=flu)
    fluwide = fluwide.dropna(0)

    fluwide = fluwide.loc[fluwide.index>'2021-01-01']
    timepoints = fluwide.shape[0]

    X = fluwide[location_names].values
    Y = fluwide.values

    LAST_T_WEEKS = 5
    
    X = X[-LAST_T_WEEKS:,:]
    Y = Y[-LAST_T_WEEKS:,:]
    
    quantiles = inter.grabInterpolatedQuantiles()

    # scale these to per 100K
    census = pd.read_csv("../../data-locations/locations.csv")

    quantiles = quantiles.merge(census,on=["location_name"])
    quantiles["value"] = quantiles["value"]/ (quantiles.population/1*10**5)
    
    quantilesByValues = pd.pivot_table(index= ["target_end_date","quantile"], columns = ["location_name"], values = ["value"],data = quantiles)
    quantilesByValues.columns = [column  for (_,column) in quantilesByValues.columns]

    targets = inter.getNext4Dates()
    quantilesByValues = quantilesByValues.loc[ [ True if x in targets else False for x,y in quantilesByValues.index] ,:]
    
    nuts_kernel = NUTS(model)
    
    mcmc = MCMC( nuts_kernel , num_warmup=1500, num_samples=2000)
    rng_key = random.PRNGKey(0)

    mcmc.run(rng_key
             ,X = X
             ,Y=Y
             ,quantiles = quantilesByValues.to_numpy()
             )
    mcmc.print_summary()
    samples = mcmc.get_samples()
   
    # quantilesByValues = quantilesByValues.dropna(1)

    Q = samples["Q"].mean(0)

    allQuantiles = pd.DataFrame(Q,columns=fluwide.columns,index=quantilesByValues.index)
    
    #allQuantiles = pd.DataFrame(quantilesByValues.values.dot(W.T),columns=fluwide.columns,index=quantilesByValues.index)
    allQuantiles = allQuantiles.reset_index()
    
    prediction2submit = allQuantiles.melt( id_vars=["target_end_date","quantile"]  )

    # scale back
    prediction2submit = prediction2submit.merge(census,on=["location_name"])
    prediction2submit["value"] = prediction2submit.value*(prediction2submit.population/1*10**5)

    prediction2submit = prediction2submit.drop(columns=["location"])
    
    prediction2submit = inter.addFIPS(prediction2submit)
    prediction2submit = inter.addForecastDate(prediction2submit)

    inter.quantilesOut(prediction2submit)
    
