#mcandrew

import sys
import numpy as np
import pandas as pd
from epiweeks import Week

if __name__ == "__main__":

    fluview = pd.read_csv("./ILINet.csv", skiprows=1)

    #--mod columns
    fluview = fluview[["REGION","YEAR","WEEK","ILITOTAL"]]
    fluview = fluview.rename(columns = {"REGION":"location_name", "YEAR":"Y", "WEEK":"W","ILITOTAL":"numili"})
    
    #--add FIPS data
    fips_data = pd.read_csv("../../data-locations/locations.csv")
    fluview = fluview.merge(fips_data, on = ["location_name"])

    #--time data
    def from_YW_to_timedata(x):
        Y,W = str(x.Y), str(x.W)

        week = Week.fromstring(Y+W)
        start_date = week.startdate()
        end_date   = week.enddate()
        cdcformat  = week.cdcformat()

        return pd.Series({"start_date":start_date, "end_date":end_date,"cdcformat":cdcformat})
    time_data = fluview.apply(from_YW_to_timedata,1)
    
    #--add time data to fluview
    fluview = pd.concat([fluview,time_data], 1)
    fluview["numili"] = fluview.numili.astype(float)

    #--add in a US location which is the sum of all states/territories
    def add_up(x):
        y = pd.DataFrame({"population": [x.population.sum()]
                          , "numili":[x.numili.sum()]
                          , "count_rate1per100k":[x.count_rate1per100k.sum()]
                          , "count_rate2per100k":[x.count_rate2per100k.sum()] } )
        return y
    fluview__US = fluview.groupby(["start_date","end_date","cdcformat","Y","W"]).apply(add_up).reset_index()
    fluview__US["location"]      = "US"
    fluview__US["abbreviation"]  = "US"
    fluview__US["location_name"] = "US"
    
    fluview = fluview.append(fluview__US)
    fluview = fluview[ ["location_name","location","abbreviation","population","numili","count_rate1per100k","count_rate2per100k","Y","W","cdcformat","start_date","end_date"] ]
    
    #--print the latest data
    latest = fluview.cdcformat.max()
    latest_end_date = fluview.end_date.max()
    
    print("Latest week {:s}".format(latest))
    print("End date {:s}".format(latest_end_date.strftime("%Y-%m-%d")))
    fluview.to_csv("./fluview_ili.csv",index=False)
