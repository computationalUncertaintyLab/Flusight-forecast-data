#mcandrew

import sys
import numpy as np
import pandas as pd

from submission_times import *

from glob import glob

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    monday_submission = next_monday(1)
    model_name = collect_model_name()
    
    forecasts = pd.read_csv("{:s}-{:s}.csv".format(monday_submission,model_name))

    truths    = pd.read_csv("../../data-truth/truth-Incident Hospitalizations.csv")
    truths    = truths.loc[ truths["date"] > "2022-06-01"]

    pdf = matplotlib.backends.backend_pdf.PdfPages("./monday_subission_viz/{:s}-{:s}.pdf".format(monday_submission,model_name))

    all_locations = forecasts.location.unique()

    locations_pieces = np.array_split(all_locations,6)

    for locations in locations_pieces:
        subset = forecasts.loc[forecasts.location.isin(locations)]

        plt.style.use("fivethirtyeight")
        fig,axs = plt.subplots(3,3)
        axs = axs.flatten()
        
        for n,(location, forecast) in enumerate(subset.groupby(["location"])):
            ax = axs[n]

            #--plot truth
            if location !="US":
                location = "{:02d}".format(int(location))
            truth = truths.loc[truths.location== location]

            dates = pd.to_datetime(truth["date"])
            ax.plot( dates, truth["value"], color="black", lw=1 )

            #--plot forecast
            forecast__wide = pd.pivot_table(index="target_end_date",columns = ["quantile"], values = "value", data = forecast)

            dates = pd.to_datetime(forecast__wide.index)
            
            ax.plot(dates, forecast__wide[0.500], lw=1)
            
            ax.fill_between(dates, forecast__wide[0.250],forecast__wide[0.750], alpha=0.10, color="blue")
            ax.fill_between(dates, forecast__wide[0.100],forecast__wide[0.900], alpha=0.10, color="blue")
            ax.fill_between(dates, forecast__wide[0.025],forecast__wide[0.975], alpha=0.10, color="blue")

            ax.tick_params(which="both",labelsize=6)
            ax.set_ylabel("Hosps",fontsize=8)
            
            ax.set_xlabel("")
            if n<6:
                ax.set_xticklabels([])
            if n not in {0,3,6}:
                ax.set_ylabel("")

            location_name = truth.location_name.iloc[0]
            ax.text(0.99,0.99,s="Loc = {:s}/{:s}".format(location,location_name)
                    ,fontsize=8,transform=ax.transAxes,ha="right",va="top")
                
        pdf.savefig(fig)
    pdf.close()
