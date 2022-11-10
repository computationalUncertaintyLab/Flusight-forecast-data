#mcandrew

import sys
import numpy as np
import pandas as pd

from sodapy import Socrata

def from_date_to_epiweek(x):
    from datetime import datetime, timedelta
    from epiweeks import Week

    #--remove time past day
    dt = x.date.values[0].split("T")[0]
    dt = datetime.strptime(dt,"%Y-%m-%d")

    prev_day = dt - timedelta(days=1)

    wk = Week.fromdate(prev_day)
    
    return pd.Series({"yr":wk.year
                      ,"week":wk.week
                      ,"ew":wk.cdcformat()
                      , "month": prev_day.month
                      , "day":prev_day.day
                      ,"start_date":wk.startdate()
                      ,"end_date":wk.enddate()})

if __name__ == "__main__":

    data_url = "healthdata.gov"
    data_set = "g62h-syeh"
    app_token = "R0mx9WRuot5YY31kw7BzVw6n8"
    
    client = Socrata(data_url,None)

    #--page through api and collect all data
    data_row = { "state":[], "date":[], "previous_day_admission_influenza_confirmed":[], "total_patients_hospitalized_confirmed_influenza":[] }
    for row in client.get_all(data_set):

        if "total_patients_hospitalized_confirmed_influenza" not in row:
            continue
        
        data_row["state"].append(row["state"])
        data_row["date"].append(row["date"])
        data_row["previous_day_admission_influenza_confirmed"].append(row["previous_day_admission_influenza_confirmed"])
        data_row["total_patients_hospitalized_confirmed_influenza"].append(row["total_patients_hospitalized_confirmed_influenza"])
    hhs_data = pd.DataFrame(data_row)
   
    #--subset to flu hosps
    hhs_data = hhs_data[["state","date","total_patients_hospitalized_confirmed_influenza"]]

    #--format flu column
    hhs_data["total_patients_hospitalized_confirmed_influenza"] = hhs_data.total_patients_hospitalized_confirmed_influenza.astype(float)
    
    #--append epidemic week
    time_info = hhs_data.groupby(["date"]).apply( from_date_to_epiweek).reset_index()
    hhs_data  = hhs_data.merge(time_info, on =["date"])

    #--aggregate by week
    hhs_data_week = hhs_data.groupby(["state","yr","week","start_date","end_date","ew"]).apply( lambda x: pd.Series({"weekly_hosps":x.total_patients_hospitalized_confirmed_influenza.sum()}) )
    hhs_data_week = hhs_data_week.reset_index()
    
    

