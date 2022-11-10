#!/bin/bash
#PBS -l mem=30gb

singularity run -B /home/pablot/mAnDE/res:/res -B /home/pablot/mAnDE/results:/results -B /home/pablot/mAnDE/results-cv:/results-cv /home/pablot/mAnDE/mande_latest.sif $PBS_ARRAY_INDEX $PARAMS "/res/params/hyperparams.txt"
