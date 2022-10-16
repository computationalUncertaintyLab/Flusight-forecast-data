#mcandrew

modelscript=holt_winters.py

while read -r training_date;
do
    while read -r location;
    do
	echo ${training_date}
	echo ${location}
	
	python3 ${modelscript} --LOCATION ${location} --RETROSPECTIVE 1 --END_DATE ${training_date}
    done < ./retrospective_analysis/locations.csv
done < ./retrospective_analysis/training_dates.csv
    
      


