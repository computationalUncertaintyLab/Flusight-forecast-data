#mcandrew

import sys
import numpy as np
import pandas as pd

from submission_times.py import *

from glob import glob

if __name__ == "__main__":

    monday_submission = next_monday(1)
    
    for n,fil in enumerate(glob("./forecasts/*.csv")):
        d = pd.read_csv(fil)
        d[""target	target_end_date	quantile	value	location	type	forecast_date
        if n==0:
            
        
    






    

