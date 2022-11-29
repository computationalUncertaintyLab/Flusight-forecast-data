#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.integrate import odeint

from scipy.optimize import root

if __name__ == "__main__":


    def sirhd(y,t, params):
        S,I,H,R,D = y
        beta,rho,kappa = params
        
        
        ds_dt = -beta*I*S
        di_dt = beta*I*S - (rho*I)
        dh_dt = rho*0.75*I - (kappa*H)
        dr_dt = rho*0.25*I + kappa*0.99*H
        dd_dt = kappa*0.01*H 

        return [ds_dt, di_dt, dh_dt, dr_dt, dd_dt]

    beta  = 2./7
    rho   = 1./7 
    kappa = 1./14

    s0,i0,h0,r0,d0 = 1 - 0.01, 0.01, 0., 0., 0.
    states  = odeint( lambda y,t: sirhd(y,t, [beta,rho,kappa])
          ,y0 = [s0,i0,h0,r0,d0]
          ,t = np.linspace(0,100,1000)
                     )
    s,i,h,r,d = states[:,0], states[:,1],states[:,2],states[:,3], states[:,4]
    
    hmax = lambda i: (3/4)*rho/beta*i

    fig,axs = plt.subplots(1,3)

    ax = axs[0]
    ax.plot( i,h )
    
    ax = axs[1]
    ax.plot( s,h )
    ax.plot( s,i )

    ax = axs[2]
    ax.plot( r,h )
    ax.plot( r,i )
    
    plt.show()


    i_s = lambda s:  s0+i0 + (rho/beta)*np.log(s/s0) - s
   
    def hs(h,s,params):
        beta,rho,kappa,s0,i0 = params

        gamma = beta*(s0+i0) - rho*np.log(s0)
        f = kappa/(rho*s*np.log(s) + gamma*s - beta*s**2  )

        c = 0.75*(rho/beta)
        g = 1/s
        
        dh_ds = f*h - c*g
        return dh_ds

    beta  = 2./7
    rho   = 1./7 
    kappa = 1./14

    s0,i0,h0,r0,d0 = 1 - 0.01, 0.01, 0., 0., 0.
    states  = odeint( lambda h,s: hs(h,s, [beta,rho,kappa,s0,i0])
          ,y0 = [0]
          ,t =np.linspace(0.999,0.20,1000) 
                     )
    
    plt.plot( np.linspace(0.999,0.20,1000), states.reshape(-1,) - g(np.linspace(0.999,0.20,1000))  )

    g = lambda s: (0.75*rho/kappa)*(s0+i0+(rho/beta)*np.log(s/s0)-s)

    #max_h = np.max(h)

    #plt.axhline(max_h)
    
    plt.show()
    
    
    
    # def sir(y,t, params):
    #     S,I,R = y
    #     beta,rho = params
        
    #     ds_dt = -beta*I*S
    #     di_dt = beta*I*S - ( rho*I )
    #     dr_dt = rho*I
    #     return [ds_dt, di_dt, dr_dt]

    # beta  = 2./7
    # rho   = 1./7 
    # kappa = 1./14

    # s0,i0,r0 = 1 - 0.01, 0.01,0.
    # states  = odeint( lambda y,t: sir(y,t, [beta,rho])
    #       ,y0 = [s0,i0,r0]
    #       ,t = np.linspace(0,100,1000)
    #                  )

    # s,i,r = states[:,0], states[:,1],states[:,2]
    


    # def sihr(y,t, params):
    #     S,I,H,R = y
    #     beta,rho,kappa = params
        
    #     ds_dt = -beta*I*S
    #     di_dt = beta*I*S - ( rho*I )
    #     dh_dt = 0.75*rho*I - kappa*H
    #     dr_dt = 0.25*rho*I + kappa*H
    #     return [ds_dt, di_dt, dh_dt, dr_dt]

    # beta  = 2./7
    # rho   = 1./7 
    # kappa = 1./14

    # s0,i0,h0,r0 = 1 - 0.01, 0.01, 0., 0.
    # states  = odeint( lambda y,t: sihr(y,t, [beta,rho,kappa])
    #       ,y0 = [s0,i0,h0,r0]
    #       ,t = np.linspace(0,100,1000)
    #                  )

    # s,i,h,r = states[:,0], states[:,1], states[:,2], states[:,3]
 


    
    # def IS(y,s,params):
    #     rho,beta = params
    #     return -1 + (rho/beta)*(1./s)

    # solution = odeint(lambda y,s: IS(y,s,[rho,beta]),i0,np.linspace(1,0.02,100))
    
    
    # def dh_ds(x):
    #     s,i,h = x
    #     return 0.75*(rho/beta)*(1/s) - (kappa/beta)*(h/(s*i)) 
    # sol = root(dh_ds, x0 = [0.33,0.33,0.33])
    
    # plt.show()


    


    
    



    

