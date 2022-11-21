#mcandrew

modelscript=hier_mech_model__numpyro.py

while read -r location;
do
    while read -r training_date;
    do
	echo ${training_date}
	echo ${location}
	
	python3 ${modelscript} --LOCATION ${location} --RETROSPECTIVE 1 --END_DATE ${training_date}

    done < ./retrospective_analysis/training_dates.csv
done < ./retrospective_analysis/locations.csv
    
      


