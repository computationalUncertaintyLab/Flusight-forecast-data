#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob

if __name__ == "__main__":

    #HWARmodel
    for fil in glob("./component-forecasts/LUcompUncertLab-HWAR2/*.csv"):
        d = pd.read_csv(fil)
        d["location"] = ["{:02d}".format(int(x)) if x!= "US" else x for x in d["location"]]
        d.to_csv(fil,index=False)
    

    #KFmodel
    for fil in glob("./component-forecasts/LUcompUncertLab-KalmanFilter/*.csv"):
        d = pd.read_csv(fil)
        d["location"] = ["{:02d}".format(int(x)) if x!= "US" else x for x in d["location"]]
        d.to_csv(fil,index=False)
    

   

