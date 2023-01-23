#!/bin/bash

FILE="$HOME/mAnDE2/target/mAnDE-3.0-jar-with-dependencies.jar"
SCRIPT="$HOME/mAnDE2/scripts/galgo.bash"

PARAMS_FOLDER="$HOME/mAnDE2/res/params/";

PARAMS="${PARAMS_FOLDER}hyperparams.txt";
PARAMS2="${PARAMS_FOLDER}hyperparams2.txt";

#qsub -N mAnDE -J 0-9999 -v CWD="$PWD",PARAMS="$PARAMS",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"

qsub -N others -J 0-3347 -v CWD="$PWD",PARAMS="$PARAMS2",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"

#qsub -N mAnDE -J 10000-19999 -v CWD="$PWD",PARAMS="$PARAMS",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"
#qsub -N mAnDE -J 20000-29999 -v CWD="$PWD",PARAMS="$PARAMS",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"
#qsub -N mAnDE -J 30000-32735 -v CWD="$PWD",PARAMS="$PARAMS",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"