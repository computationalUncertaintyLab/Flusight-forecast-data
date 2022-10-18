#mcandrew

import sys
import numpy as np
import pandas as pd

from glob import glob

def extract_time(fil):
    return fil.split("_")[-1].split(".")[0]

import os

if __name__ == "__main__":

    model = "LUcompUncertLab-KalmanFilter"
    for fil in glob("./retrospective_analysis/*.csv"):
        if "location_" not in fil:
            continue
    
        time = extract_time(fil)

        d = pd.read_csv(fil)
        d = d[ ["target","target_end_date","quantile","value","location","type","forecast_date"]  ]

        forecast_date = d["forecast_date"].iloc[0]
        
        
        file_to_write = "./retrospective_analysis/LUcompUncertLab-KalmanFilter/{:s}-{:s}.csv".format(forecast_date,model)

        if os.path.exists(file_to_write):
            d.to_csv(file_to_write, index=False, header=False, mode = "a")
        else:
            d.to_csv(file_to_write, index=False, header=True, mode = "w")
