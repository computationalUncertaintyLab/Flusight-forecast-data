#mcandrew

import sys
import pandas as pd

if __name__ == "__main__":

    flu = pd.read_csv("../../data-truth/truth-incident hospitalizations.csv")
    flu["date"] = pd.to_datetime(flu.date)
    
    flu2022 = flu.loc[flu.date > "2021-10-01"] #start training oct 2021

    locations = flu2022["location"].drop_duplicates()
    locations.to_csv("./retrospective_analysis/locations.csv",index=False,header=False)
