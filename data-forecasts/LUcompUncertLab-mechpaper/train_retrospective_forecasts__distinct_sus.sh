#mcandrew

modelscript=distinct_sus_mech_model.py

while read -r location;
do
    while read -r training_date;
    do
	echo ${training_date}
	echo ${location}
	
	python3 ${modelscript} --LOCATION ${location} --END_DATE ${training_date}

    done < ./retrospective_analysis/training_dates.csv
done < ./retrospective_analysis/locations.csv
