#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scoring import *

from glob import glob

if __name__ == "__main__":

    truths = pd.read_csv("../../data-truth/truth-Incident Hospitalizations.csv")
    truths = truths.rename(columns = {"value":"truth"})
    
    for n,fil in enumerate(glob("./retrospective_analysis/*.csv")):
        if 'locations' in fil or 'training' in fil or "wis_scores" in fil:
            continue

        forecast = pd.read_csv(fil)
        
        #--format location column
        forecast["location"] = [ "{:02d}".format(int(x)) if x!="US" else x for x in forecast.location.values]
        
        forecast_and_truths = forecast.merge(truths, left_on = ["location","target_end_date"],right_on = ["location","date"])

        #--no truth for this forecast
        if len(forecast_and_truths) == 0:
            continue
        
        WIS_scores = forecast_and_truths.groupby(["target","target_end_date","location","forecast_date"]).apply(lambda x: pd.Series({"WIS":WIS(x)}))
        WIS_scores = WIS_scores.reset_index()

        #--add model name
        WIS_scores["model_name"] = "Holt Winters + AR(2)"
        
        #--specify order of columns
        WIS_scores = WIS_scores[ ["model_name","target","target_end_date","location","forecast_date","WIS"]  ]
        
        #--write
        if n==0:
            WIS_scores.to_csv("./retrospective_analysis/all_retrospective_wis_scores.csv", header=True ,index=False,mode="w") 
        else:
            WIS_scores.to_csv("./retrospective_analysis/all_retrospective_wis_scores.csv", header=False,index=False,mode="a")
