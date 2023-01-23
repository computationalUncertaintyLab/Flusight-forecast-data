#mcandrew

import sys
import os

import numpy as np
import pandas as pd

from scoring import *

from glob import glob

if __name__ == "__main__":

    truths = pd.read_csv("../../data-truth/truth-Incident Hospitalizations.csv")
    truths = truths.rename(columns = {"value":"truth"})

    root = "./retrospective_analysis/"
    n=0
    for folder in os.listdir(root):
        print(folder)

        sub_dir = os.path.join(root,folder)
        if os.path.isdir(sub_dir):
            for fil in glob(sub_dir + "/*.csv"):
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
                model_name = forecast["model_name"].values[0]
                WIS_scores["model_name"] = model_name

                #--specify order of columns
                WIS_scores = WIS_scores[ ["model_name","target","target_end_date","location","forecast_date","WIS"]  ]

                #--write
                if n==0:
                    WIS_scores.to_csv("./retrospective_analysis/all_retrospective_wis_scores.csv", header=True ,index=False,mode="w")
                    n=1
                else:
                    WIS_scores.to_csv("./retrospective_analysis/all_retrospective_wis_scores.csv", header=False,index=False,mode="a")