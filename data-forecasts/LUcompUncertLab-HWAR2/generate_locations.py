#mcandrew

import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":

    locations = pd.read_csv("../../data-locations/locations.csv")
    locations = locations[["location","location_name"]]

    fout = open("iteration_list.csv","w")
    for idx,row in locations.iterrows():
        fout.write("--export=ALL,LOCATIONqq={:s}\n".format(row.location))
    fout.close()
