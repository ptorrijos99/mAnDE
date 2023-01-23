#!/bin/bash
HOME_FOLDER="$HOME/mAnDE2/";
BBDD_FOLDER="res/bbdd/";
PARAMS_FOLDER="res/params/";

BBDD_NAMES="${PARAMS_FOLDER}bbdd_names.txt"
SAVE_FILE="${PARAMS_FOLDER}hyperparams.txt";
SAVE_FILE2="${PARAMS_FOLDER}hyperparams2.txt";

#seeds=(2 3 5 7 11 13 17 19 23 29)

seeds=(2)
declare -a discretized=("true" "false")
nTrees=(50 100 150 200)
declare -a featureSelection=("none" "InfoGain" "ReliefF")
ns=(1 2)
percentajes=(0 0.01 0.02 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5)

# bbdd,algorithm,seed,folds,discretized,nTrees,featureSelection,baseClass,n,ensemble,bagSize,porNB

# FEATURE SELECTION

for feature in ${featureSelection[@]}; do
    # mAnDE
    for per in ${percentajes[@]}; do
        for n in ${ns[@]}; do
            for seed in ${seeds[@]}; do
                while read bbdd; do
                    for trees in ${nTrees[@]}; do
                        # RF
                        echo $bbdd "mAnDE" $seed "3" "false" $trees $feature "RandomTree" $n "RF" "100" $per >> $SAVE_FILE

                        # Bagging
                        echo $bbdd "mAnDE" $seed "3" "false" $trees $feature "J48" $n "Bagging" "100" $per >> $SAVE_FILE
                    done
                done < "$BBDD_NAMES"
            done
        done
    done

    # others
    for seed in ${seeds[@]}; do
        for dis in ${discretized[@]}; do
            while read bbdd; do
                echo $bbdd "NB" $seed "3" $dis "1" $feature "none" "0" "none" "0" "0" >> $SAVE_FILE2

                for trees in ${nTrees[@]}; do
                    # RF
                    echo $bbdd "RandomForest" $seed "3" $dis $trees $feature "RandomTree" "0" "none" "0" "0" >> $SAVE_FILE2

                    # Bagging
                    echo $bbdd "Bagging" $seed "3" $dis $trees $feature "J48" "0" "none" "0" "0" >> $SAVE_FILE2
                done
            done < "$BBDD_NAMES"
        done
    done
done

