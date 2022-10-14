#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from epiweeks import Week

from datetime import datetime,timedelta

def link_HHS_to_state():

    fromHHS2State       = {}
    fromHHS2State[1]    = ['09', '23', '25', '33', '44', '50']
    fromHHS2State[2]    = ['34','36','72','78']
    fromHHS2State[3]    = ['10','11','24','42','51','54']
    fromHHS2State[4]    = ['01','12','13','21','28','37','45','47']
    fromHHS2State[5]    = ['17','18','26','27','39','55']
    fromHHS2State[6]    = ['05','22','35','40','48']
    fromHHS2State[7]    = ['19','20','29','31']
    fromHHS2State[8]    = ['08','30','38','46','49','56']
    fromHHS2State[9]    = ['04','06','15','32','60','69','66']
    fromHHS2State[10]   = ['02','16','41','53']
    fromHHS2State['US'] = ['US']
    
    d = {"HHS":[],"location":[]}
    for HHS,states in fromHHS2State.items():
        for state in states:
            d["HHS"].append(HHS)
            d["location"].append(state)
    d = pd.DataFrame(d)
    return d
    
if __name__ == "__main__":

    HHS_Location = link_HHS_to_state()
    
    flu_hosps = pd.read_csv("../../data-truth/truth-Incident Hospitalizations.csv")

    #--add in epidemic week
    flu_hosps["EW"]   = [ Week.fromdate(datetime.strptime(x,'%Y-%m-%d')).cdcformat() for x in flu_hosps["date"]]
    flu_hosps["week"] = [ Week.fromdate(datetime.strptime(x,'%Y-%m-%d')).week for x in flu_hosps["date"]]

    #--add in HHS
    flu_hosps = flu_hosps.merge(HHS_Location, on = ["location"])
    flu_hosps["HHS"] = flu_hosps.HHS.astype(str)
    
    influenza_like_illness = pd.read_csv("./epidataFormated.csv.gz")

    #--collect most recent ILI
    influenza_like_illness = influenza_like_illness.groupby(["Location","EW"]).apply( lambda x: x.sort_values("releaseEW").iloc[-1] ).reset_index(drop=True)

    #--add in saturdays
    influenza_like_illness["date"] = [ Week.fromstring(str(x)).enddate().strftime("%Y-%m-%d") for x in influenza_like_illness.EW ]
    influenza_like_illness["year"] = [ Week.fromstring(str(x)).year for x in influenza_like_illness.EW ]
    influenza_like_illness["week"] = [ Week.fromstring(str(x)).week for x in influenza_like_illness.EW ]
    
    #--reformat location for ILI dataset
    influenza_like_illness["HHS"] = [ "US" if x==10 else str(int(x)+1) for x in influenza_like_illness.Location ]

    influenza_like_illness = influenza_like_illness[ ["HHS","EW","year","week","date","wili"] ]


    influenza_like_illness__wide = pd.pivot_table(index = ["HHS","week"], columns = "year", values = ["wili"], data= influenza_like_illness)
    influenza_like_illness__wide.columns = [  for (x,y) in influenza_like_illness__wide.columns] 
    
    stacked_flu = pd.DataFrame()
    for year,subset in influenza_like_illness.groupby(["year"]):

        break
    

        subset = subset[["HHS","week","wili"]]
        subset = subset.rename(columns = {"wili":"wili{:d}".format(year)})
       
        combined_data = flu_hosps.merge(subset,on=["HHS","week"], indicator=True,how="left")

        stacked_flu = stacked_flu.append(combined_data)
        
    
    flu_hosps = flu_hosps.merge( influenza_like_illness, on = ["HHS","week"] )


    
