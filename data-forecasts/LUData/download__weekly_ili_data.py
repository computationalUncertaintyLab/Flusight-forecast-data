#mcandrew

import sys
import numpy as np
import pandas as pd

from delphi_epidata import Epidata

from epiweeks import Week

if __name__ == "__main__":


    states = {'alabama': 'al','alaska': 'ak',
              'arizona': 'az','arkansas': 'ar','california': 'ca','colorado': 'co','connecticut': 'ct',
              'delaware': 'de',
              'florida': 'fl',
              'georgia': 'ga','hawaii': 'hi','idaho': 'id',
    'illinois': 'il','indiana': 'in',
    'iowa': 'ia',
    'kansas': 'ks',
    'kentucky': 'ky',
    'louisiana': 'la',
    'maine': 'me',
    'maryland': 'md',
    'massachusetts': 'ma',
    'michigan': 'mi',
    'minnesota': 'mn',
    'mississippi': 'ms',
    'missouri': 'mo',
    'montana': 'mt',
    'nebraska': 'ne',
    'nevada': 'nv',
    'new hampshire': 'nh',
    'new jersey': 'nj',
    'new mexico': 'nm',
    'new york': 'ny',
    'north carolina': 'nc',
    'north dakota': 'nd',
    'ohio': 'oh',
    'oklahoma': 'ok',
    'oregon': 'or',
    'pennsylvania': 'pa',
    'rhode island': 'ri',
    'south carolina': 'sc',
    'south dakota': 'sd',
    'tennessee': 'tn',
    'texas': 'tx',
    'utah': 'ut',
    'vermont': 'vt',
    'virginia': 'va',
    'washington': 'wa',
    'west virginia': 'wv',
    'wisconsin': 'wi',
    'wyoming': 'wy',
    'american samoa': 'as',
    'commonwealth of the northern mariana islands': 'mp',
    'district of columbia': 'dc',
    'guam': 'gu',
    'puerto rico': 'pr',
    'virgin islands': 'vi',
              }
    
    stat_abb = states.values()

    from_abb_2_state = {v:k for k,v in states.items()}
    
    thisweek = int(Week.thisweek().cdcformat())
    init_week = Week.fromstring(str(202101))  
    weeks = [int(init_week.cdcformat())]

    #--process results
    data = {"state":[],"epiweek":[], "cases":[], "start_date":[], "end_date":[]}

    while int(init_week.cdcformat()) != thisweek:
        init_week+=1
        weeks.append(int(init_week.cdcformat()))

    while weeks:
        batch = []
        for n in range(20):
            if len(weeks)>0:
                batch.append( weeks.pop() )
            else:
                break
        
        res = Epidata.fluview( list(stat_abb) , batch)

        for d in res["epidata"]:
            data["state"].append(from_abb_2_state[d["region"]])
            data["cases"].append(d["num_ili"])
            data["epiweek"].append( d["epiweek"] )

            week = Week.fromstring(str(d["epiweek"]))
            start_date = (week.startdate()).strftime("%Y-%m-%d")
            end_date = (week.enddate()).strftime("%Y-%m-%d")

            data["start_date"].append(start_date)
            data["end_date"].append(end_date)

    data = pd.DataFrame(data)
    

    

