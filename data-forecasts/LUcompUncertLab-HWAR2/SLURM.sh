#!/bin/bash

#SBATCH --partition=health,lts,hawkcpu,infolab,engi,eng
 
#--Request 1 hour of computing time
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=3
 
#--Give a name to your job to aid in monitoring
#SBATCH --job-name flumodel
 
#--Write Standard Output and Error
#SBATCH --output="myjob.%j.%N.out"
 
cd ${SLURM_SUBMIT_DIR} # cd to directory where you submitted the job
 
#--launch job
module load anaconda3
#module load conda/biostats
export PYTHONPATH=$PYTHONPATH:$HOME/pypkgs

#--export environmental variables
export LOCATION=${LOCATION}
python3 holt_winters.py --LOCATION ${LOCATION}
 
exit
