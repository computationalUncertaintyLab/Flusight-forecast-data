#mcandrew

import sys
import numpy as np
import pandas as pd

from submission_times import *

from glob import glob

if __name__ == "__main__":

    monday_submission = next_monday(1)
    model_name = collect_model_name()
    for n,fil in enumerate(glob("./forecasts/*.csv")):
        d = pd.read_csv(fil)
        d = d[["target", "target_end_date","quantile","value","location","type","forecast_date"]]

        #--format quantile to three decimals
        d['quantile'] = ["{:0.3f}".format(q) for q in d["quantile"]]

        #--format values to three decimals
        d['value'] = ["{:0.3f}".format(q) for q in d["value"]]

        #--format location
        locations = []
        for loc in d.location:
            if loc=="US":
                pass
            else:
                loc = "{:02d}".format(int(loc))
            locations.append(loc)
        d['location'] = locations
        
        if n==0:
            d.to_csv("./{:s}-{:s}.csv".format(monday_submission,model_name),index=False,header=True,mode="w")
        else: 
            d.to_csv("./{:s}-{:s}.csv".format(monday_submission,model_name),index=False,header=False,mode="a")
