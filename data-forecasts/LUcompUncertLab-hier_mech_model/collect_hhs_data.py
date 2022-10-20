#mcandrew

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    data = pd.read_csv("https://healthdata.gov/resource/g62h-syeh.csv")

    locations = pd.read_csv("../../data-locations/locations.csv")

    data = data.merge(locations, right_on=["abbreviation"], left_on = ["state"], indicator=True, how="left")

    data = data.rename(columns = {"location":"fips"})
    
    data.to_csv("hhs_data.csv",index=False)
