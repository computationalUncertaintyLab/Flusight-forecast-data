#!/bin/bash

#SBATCH --partition=health,lts,hawkcpu,infolab,engi,eng
 
#--Request 1 hour of computing time
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=24
 
#--Give a name to your job to aid in monitoring
#SBATCH --job-name flumodel
 
#--Write Standard Output and Error
#SBATCH --output="flu.%j.%N.out"
 
cd ${SLURM_SUBMIT_DIR} # cd to directory where you submitted the job
 
#--launch job
module load anaconda3
conda activate $HOME/condaenv/flu

#--export environmental variables
export LOCATION=${LOCATION}
python hier_mech_model__2.0.2.py --LOCATION ${LOCATION} --RETROSPECTIVE 0
 
exit
