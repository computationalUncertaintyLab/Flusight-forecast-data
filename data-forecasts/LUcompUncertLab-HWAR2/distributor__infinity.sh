#!bin/bash

#--arguments
SLURM_SCRIPT=$1
ITERATION_LIST=$2

#--functions
function check_number_of_jobs(){
    local UNAME=$(whoami)
    local NJOBS=$(squeue -u ${UNAME} | wc -l)
    echo $NJOBS
}

#--main
while read line; do
    while [[ $(check_number_of_jobs) -gt 3975 ]]; do
	sleep 10m
    done
    echo Sending $line           # MARK progress
    sbatch $line ${SLURM_SCRIPT} # SUBMIT JOB
done < ${ITERATION_LIST}
