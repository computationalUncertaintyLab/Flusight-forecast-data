#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import stan

if __name__ == "__main__":

    latent = [np.random.normal(0,1)]

    alpha = -1
    beta = 1

    sigma = 1
    phi = 0.5
    
    for t in np.arange(1,100):
        v = np.random.normal(alpha + beta*latent[-1], sigma)
        latent.append(v )

    obs = []
    for t in range(100):
        obs.append( np.random.normal( latent[t], phi )  )
    
    stan_model = '''
        data {
           int T;
           vector [T] y;
        }
        parameters { 
           real beta0; 
           vector [T] latent_state;
 
           real alpha; 
           real beta; 

           real <lower=0> lambda;    
            
           real <lower = 0> sigma;

           real <lower = 0> phi;
        }
        transformed parameters {
           }
      
        model {
             //AR hidden latent
             latent_state[1] ~ normal(0,10);

             sigma~normal(1,10);
             
             for (t in 2:T){
                latent_state[t] ~  normal(alpha + beta*latent_state[t-1], sigma);
             }

        }
    '''
    
    model_data = {"T":len(obs), "y":obs}
    posterior  = stan.build(stan_model, data=model_data)
    fit        = posterior.sample(num_chains=4, num_samples=1000)
 

        
        
