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
    truths    = truths.loc[ truths["date"] > "2022-03-01"]

    pdf = matplotlib.backends.backend_pdf.PdfPages("./monday_submission_viz/{:s}-{:s}--comparison.pdf".format(monday_submission,model_name))

    all_locations = forecasts.location.unique()

    #--collect all locations and location names
    loc_and_location_names = truths[["location","location_name"]]
    loc2locname = { row.location:row.location_name for idx,row in loc_and_location_names.iterrows()}
    locname2loc = { row.location_name:row.location for idx,row in loc_and_location_names.iterrows()}

    locations_pieces = np.array_split(sorted(locname2loc),6)

    time = "2023-01-23"
    models = ["../GT-FluFNP/", "../PSI-DICE/", "../UMass-gbq/","../CMU-TimeSeries/"]
    
    for location_names in locations_pieces:
        locations = [ locname2loc[name] for name in location_names]
        subset = forecasts.loc[forecasts.location.isin(locations)]

        plt.style.use("fivethirtyeight")
        fig,axs = plt.subplots(3,3)
        axs = axs.flatten()
        
        for n,(location, forecast) in enumerate(subset.groupby(["location"])):
            location_name = loc2locname[location]
            
            ax = axs[n]

            #--plot truth
            if location !="US":
                location = "{:02d}".format(int(location))
            truth = truths.loc[truths.location== location]

            dates = pd.to_datetime(truth["date"])
            ax.plot( dates, truth["value"], color="black", lw=1 )

            all_forecasts = [forecast]
            
            #--load other models
            for model in models:
                f = glob(model+"{:s}*.csv".format(time))
        
                if len(f)==0:
                    continue
        
                d = pd.read_csv(f[0])
                state = d.loc[d.location==location]

                all_forecasts.append(state)
                
            colors = ['blue','red','purple',"green","orange"]
            for n,fcast in enumerate(all_forecasts):

                if len(fcast)==0:
                    continue
                
                #--plot forecast
                forecast__wide = pd.pivot_table(index="target_end_date",columns = ["quantile"], values = "value", data = fcast)

                dates = pd.to_datetime(forecast__wide.index)

                ax.plot(dates, forecast__wide[0.500], lw=1 ,color=colors[n], label = "{:s}".format(models[n]))
                ax.scatter(dates, forecast__wide[0.500],s=4,color=colors[n])

                ax.fill_between(dates, forecast__wide[0.250],forecast__wide[0.750], alpha=0.10, color=colors[n])
                ax.fill_between(dates, forecast__wide[0.100],forecast__wide[0.900], alpha=0.10, color=colors[n])
                ax.fill_between(dates, forecast__wide[0.025],forecast__wide[0.975], alpha=0.10, color=colors[n])

            ax.tick_params(which="both",labelsize=6)
            ax.set_ylabel("inc hosps",fontsize=8)
            
            ax.set_xlabel("")
            if n<6:
                ax.set_xticklabels([])
            if n not in {0,3,6}:
                ax.set_ylabel("")
            ax.tick_params(axis='x', labelrotation = 90, labelsize=6)

            ax.text(0.99,0.99,s="Loc = {:s}/{:s}".format(location,location_name)
                    ,fontsize=8,transform=ax.transAxes,ha="right",va="top")

        fig.set_tight_layout(True)
        pdf.savefig(fig)
    pdf.close()
