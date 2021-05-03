#!/bin/bash
#SBATCH -J PSF_Generate.py     # run's name
#SBATCH -N 1                   # nb of nodes 
#SBATCH -c 1                   # nb of cpus per task
#SBATCH --mem=15GB             # RAM
#SBATCH -t 2:00:00             # walltime
#SBATCH -o resJobG.txt         # output file name
#SBATCH -e errJobG.txt         # error file name
#SBATCH --mail-type=BEGIN,END  
#SBATCH --mail-user=olivier.beltramo-martin@lam.fr

python3 ./dataGeneratorBatch.py --mag=0 --addStat=0 --nPSFperFolder=3500 --savePath='/result/omartin/APPLY/' > 'genStatus.txt'
