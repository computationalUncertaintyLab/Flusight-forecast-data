#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

def lowess(x, y, f=1./3.):
    """
    Basic LOWESS smoother with uncertainty. 
    Note:
        - Not robust (so no iteration) and
             only normally distributed errors. 
        - No higher order polynomials d=1 
            so linear smoother.
    """
    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor
    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest 
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 * 
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
    return y_sm, y_stderr


def mm2inch(x):
    return x/25.4

if __name__ == "__main__":


    d = pd.read_csv("../../data-truth/truth-incident hospitalizations-daily.csv")

    nat = d.loc[d.location=="US"]


    def seasons(x):
        from datetime import datetime

        if  datetime.srtptime("2022-03-01","%Y-%m-%d") <= x <= datetime.srtptime("2022-08-01","%Y-%m-%d"):
            return "2021/2022"
        elif datetime.srtptime("2022-08-01","%Y-%m-%d") <= x <= datetime.srtptime("2023-08-01","%Y-%m-%d"):
            return "2022/2023"
        else:
            return "exclude"
    
    nat = nat.loc[ nat.date>="2022-02-01",:]
    nat = nat.reset_index()

    plt.style.use("fivethirtyeight")
    
    fig,ax = plt.subplots()
    sns.scatterplot( x=nat.index, y = "value", s=10, data = nat, ax=ax, color="black", alpha=0.70 )
    
    y_sm, y_std = lowess(nat.index.values, nat["value"], f=1./7.)

    ax.plot(nat.index.values, y_sm, lw=1.5 )

    plt.fill_between(nat.index.values
                     , y_sm - 1.96*y_std
                     ,y_sm + 1.96*y_std
                     , alpha=0.3)

    ax.set_xlim(0,280)
    ax.set_xticks( np.arange(0,278,50))
    
    ax.set_xticklabels( [ nat.loc[x,"date"] for x in ax.get_xticks()]   )

    ax.set_xlabel("", fontsize=10)
    ax.set_ylabel("Incident confirmed influenza hosps", fontsize=10)

    ax.tick_params(which="both",labelsize=8)


    w = mm2inch(183)
    fig.set_size_inches(w,w/1.5)
    fig.set_tight_layout(True)
    
    plt.savefig("fluhosp_trajectories__nat.pdf")
