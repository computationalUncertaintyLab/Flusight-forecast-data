#mcandrew

import sys
import numpy as np
import pandas as pd

from sodapy import Socrata

from datetime import datetime, timedelta

def from_date_to_epiweek(x):
    from datetime import datetime, timedelta
    from epiweeks import Week

    wk = Week.fromdate(pd.to_datetime(x.date.values[0]))
    
    return pd.Series({"yr"     :wk.year
                      ,"week"  :wk.week
                      ,"ew"    :wk.cdcformat()
                      ,"start_date":wk.startdate()
                      ,"end_date":wk.enddate()})

if __name__ == "__main__":

    data_url = "healthdata.gov"
    data_set = "g62h-syeh"
    app_token = "R0mx9WRuot5YY31kw7BzVw6n8"
    
    client = Socrata(data_url,None)

    #--page through api and collect all data
    data_row = { "state":[]
                 , "date":[]
                 , "previous_day_admission_influenza_confirmed":[]
                 , "previous_day_deaths_influenza":[]}
    for row in client.get_all(data_set):

        if "previous_day_admission_influenza_confirmed" not in row:
            continue

        if "previous_day_deaths_influenza" not in row:
            continue

        data_row["state"].append(row["state"])

        #--add one day to account for the fact that we collected previous day admissions
        date = datetime.strptime( str(row["date"]).split("T")[0], "%Y-%m-%d")
        data_row["date"].append(date - timedelta(days=1))
        
        data_row["previous_day_deaths_influenza"].append(row["previous_day_deaths_influenza"])
        data_row["previous_day_admission_influenza_confirmed"].append(row["previous_day_admission_influenza_confirmed"])
    hhs_data = pd.DataFrame(data_row)
   
    #--subset to flu hosps
    hhs_data = hhs_data[["state","date","previous_day_admission_influenza_confirmed","previous_day_deaths_influenza"]]

    #--format flu column
    hhs_data["previous_day_deaths_influenza"]                   = hhs_data.previous_day_deaths_influenza.astype(float)
    hhs_data["previous_day_admission_influenza_confirmed"]      = hhs_data.previous_day_admission_influenza_confirmed.astype(float)

    #--rename columns
    hhs_data = hhs_data.rename(columns = { "previous_day_admission_influenza_confirmed":"hosps"
                                          ,"previous_day_deaths_influenza":"deaths"})

    #--add location data
    locations = pd.read_csv("../../data-locations/locations.csv")
    hhs_data = hhs_data.merge(locations, left_on = ["state"], right_on = ["abbreviation"])

    hhs_data = hhs_data[["date","location","population","hosps","deaths"]]
    
    #--add US
    US_level = hhs_data.groupby(["date"]).apply( lambda x: x[ ["hosps","deaths","population"] ].sum() ).reset_index()
    US_level["location"] = "US"

    hhs_data = hhs_data.append(US_level)

    #--cut at 2021
    hhs_data = hhs_data[hhs_data.date>="2021-01-01"]

    #--format location
    hhs_data["location"] = [ "{:02d}".format(int(x)) if x !="US" else x for x in hhs_data.location ]
    
    
    hhs_data.to_csv("hhs_data__daily.csv", index=False)

    #--append epidemic week
    time_info = hhs_data.groupby(["date"]).apply( from_date_to_epiweek).reset_index()
    hhs_data  = hhs_data.merge(time_info, on =["date"])

    #--aggregate by week
    hhs_data_week = hhs_data.groupby(["location","yr","week","start_date","end_date","ew"]).apply( lambda x: pd.Series({"weekly_hosps":x.hosps.sum()}) )
    hhs_data_week = hhs_data_week.reset_index()

    hhs_data_week.to_csv("hhs_data__weekly.csv", index=False)
    

