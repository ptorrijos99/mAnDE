#!/bin/bash

FILE="$HOME/mAnDE2/target/mAnDE-3.0-jar-with-dependencies.jar"
SCRIPT="$HOME/mAnDE2/scripts/galgo.bash"

PARAMS_FOLDER="$HOME/mAnDE2/res/params/";

PARAMS="${PARAMS_FOLDER}hyperparams1.txt";
PARAMS1="${PARAMS_FOLDER}hyperparamsRF.txt";
PARAMS2="${PARAMS_FOLDER}hyperparamsB1.txt";
PARAMS3="${PARAMS_FOLDER}hyperparamsB2.txt";
PARAMS4="${PARAMS_FOLDER}hyperparamsBo1.txt";
PARAMS5="${PARAMS_FOLDER}hyperparamsBo2.txt";

PARAMS20="${PARAMS_FOLDER}hyperparams_resto.txt";
PARAMS21="${PARAMS_FOLDER}hyperparams_restoRF.txt";
PARAMS22="${PARAMS_FOLDER}hyperparams_restoB1.txt";
PARAMS23="${PARAMS_FOLDER}hyperparams_restoB2.txt";
PARAMS24="${PARAMS_FOLDER}hyperparams_restoBo1.txt";
PARAMS25="${PARAMS_FOLDER}hyperparams_restoBo2.txt";

# Antiguo
#qsub -N mAnDE_RF -J 0-2479 -v CWD="$PWD",PARAMS="$PARAMS1",FILE="$FILE" -l nodes=1:cluster=galgo2,ppn=8,mem=31gb "$SCRIPT"   	# 8 Hilos


# RF
qsub -N mAnDE_RF -J 0-2479 -v CWD="$PWD",PARAMS="$PARAMS1",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos
qsub -N base_RF -J 0-2479 -v CWD="$PWD",PARAMS="$PARAMS21",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos

# Bagging REPTree
qsub -N mAnDE_B_REP -J 0-2479 -v CWD="$PWD",PARAMS="$PARAMS3",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos
qsub -N base_B_REP -J 0-2479 -v CWD="$PWD",PARAMS="$PARAMS23",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos

# Bagging J48
qsub -N mAnDE_B_J48 -J 0-2479 -v CWD="$PWD",PARAMS="$PARAMS2",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos
qsub -N base_B_J48 -J 0-2479 -v CWD="$PWD",PARAMS="$PARAMS22",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos

# 1 Árbol
qsub -N mAnDE_1 -J 0-619 -v CWD="$PWD",PARAMS="$PARAMS",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos
qsub -N base -J 0-1859 -v CWD="$PWD",PARAMS="$PARAMS20",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos

# AdaBoost Stump
qsub -N mAnDE_AB_S -J 0-2479 -v CWD="$PWD",PARAMS="$PARAMS4",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos
qsub -N base_AB_S -J 0-2479 -v CWD="$PWD",PARAMS="$PARAMS24",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos

# AdaBoost REPTree
qsub -N mAnDE_AB_REP -J 0-2479 -v CWD="$PWD",PARAMS="$PARAMS5",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos
qsub -N base_AB_REP -J 0-2479 -v CWD="$PWD",PARAMS="$PARAMS25",FILE="$FILE" -l select=1:ncpus=8:mem=31gb:cluster=galgo2 "$SCRIPT"   	# 8 Hilos
