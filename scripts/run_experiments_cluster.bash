#!/bin/bash
#PBS -l mem=15gb

singularity run -B /home/pablot/mAnDE/res:/res -B /home/pablot/mAnDE/results:/results /home/pablot/mAnDE/mande_latest.sif $PBS_ARRAY_INDEX $PARAMS "/res/params/hyperparams.txt"
