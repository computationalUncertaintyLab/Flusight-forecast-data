#mcandrew

import sys
import pandas as pd

if __name__ == "__main__":

    flu = pd.read_csv("../../data-truth/truth-incident hospitalizations.csv")
    flu["date"] = pd.to_datetime(flu.date)
    
    flu2022 = flu.loc[flu.date > "2021-11-06"] #start training oct 2021

    training_dates = flu2022["date"].drop_duplicates()
    training_dates.to_csv("./retrospective_analysis/training_dates.csv",index=False,header=False)
