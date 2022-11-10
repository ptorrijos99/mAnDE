#!/bin/bash
HOME_FOLDER="$HOME/mAnDE2/";
BBDD_FOLDER="res/bbdd/";
PARAMS_FOLDER="res/params/";

BBDD_NAMES="${PARAMS_FOLDER}bbdd_names.txt"
SAVE_FILE="${PARAMS_FOLDER}hyperparams1.txt";
SAVE_FILE1="${PARAMS_FOLDER}hyperparamsRF.txt";
SAVE_FILE2="${PARAMS_FOLDER}hyperparamsB1.txt";
SAVE_FILE3="${PARAMS_FOLDER}hyperparamsB2.txt";
SAVE_FILE4="${PARAMS_FOLDER}hyperparamsBo1.txt";
SAVE_FILE5="${PARAMS_FOLDER}hyperparamsBo2.txt";

SAVE_FILE20="${PARAMS_FOLDER}hyperparams_resto.txt";
SAVE_FILE21="${PARAMS_FOLDER}hyperparams_restoRF.txt";
SAVE_FILE22="${PARAMS_FOLDER}hyperparams_restoB1.txt";
SAVE_FILE23="${PARAMS_FOLDER}hyperparams_restoB2.txt";
SAVE_FILE24="${PARAMS_FOLDER}hyperparams_restoBo1.txt";
SAVE_FILE25="${PARAMS_FOLDER}hyperparams_restoBo2.txt";

#seeds=(2 3 5 7 11 13 17 19 23 29)


declare -a algorithms=("NB" "J48" "REPTree") 
seeds=(2)
declare -a discretized=("true" "false")
nTrees=(10 20 50 100)
declare -a featureSelection=("none" "FCBF" "IWSS_NB" "InfoGain" "ReliefF")
declare -a baseClas=("REPTree" "J48" "Stump")
ns=(1 2)

# bbdd, algorithm, seed, folds, discretized, nTrees, featureSelection, baseClas, (n, ensemble, boosting, RF, bagSize)

# FEATURE SELECTION
for feature in ${featureSelection[@]}; do
    # mAnDE
    for n in ${ns[@]}; do
        for seed in ${seeds[@]}; do
            while read bbdd; do
                # 1 árbol
                echo $bbdd "mAnDE" $seed "3" "false" "1" $feature "null" $n "false" "false" "false" "100" >> $SAVE_FILE

                for trees in ${nTrees[@]}; do
                    # RF
                    echo $bbdd "mAnDE" $seed "3" "false" $trees $feature "RandomTree" $n "true" "false" "true" "100" >> $SAVE_FILE1

                    # Bagging
                    echo $bbdd "mAnDE" $seed "3" "false" $trees $feature "J48" $n "true" "false" "false" "100" >> $SAVE_FILE2
                    echo $bbdd "mAnDE" $seed "3" "false" $trees $feature "REPTree" $n "true" "false" "false" "100" >> $SAVE_FILE3

                    # Boosting
                    echo $bbdd "mAnDE" $seed "3" "false" $trees $feature "Stump" $n "true" "true" "false" "100" >> $SAVE_FILE4
                    echo $bbdd "mAnDE" $seed "3" "false" $trees $feature "REPTree" $n "true" "true" "false" "100" >> $SAVE_FILE5
                done
            done < "$BBDD_NAMES"
        done
    done

    # Resto de algoritmos
    for discret in ${discretized[@]}; do
        for seed in ${seeds[@]}; do
            while read bbdd; do
                for algorithm in ${algorithms[@]}; do
                    echo $bbdd $algorithm $seed "3" $discret "1" $feature "null" "0" "false" "false" "false" "0" >> $SAVE_FILE20
                done

                for trees in ${nTrees[@]}; do
                    # RF
                    echo $bbdd "RandomForest" $seed "3" $discret $trees $feature "RandomTree" "0" "false" "false" "false" "0" >> $SAVE_FILE21

                    # Bagging
                    echo $bbdd "Bagging" $seed "3" $discret $trees $feature "J48" "0" "false" "false" "false" "0" >> $SAVE_FILE22
                    echo $bbdd "Bagging" $seed "3" $discret $trees $feature "REPTree" "0" "false" "false" "false" "0" >> $SAVE_FILE23

                    # Boosting
                    echo $bbdd "AdaBoost" $seed "3" $discret $trees $feature "Stump" "0" "false" "false" "false" "0" >> $SAVE_FILE24
                    echo $bbdd "AdaBoost" $seed "3" $discret $trees $feature "REPTree" "0" "false" "false" "false" "0" >> $SAVE_FILE25
                done
            done < "$BBDD_NAMES"
        done
    done
done

