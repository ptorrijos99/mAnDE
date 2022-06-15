#!/bin/bash
HOME_FOLDER="/home/pablot/ParallelBNs/" #/Users/jdls/developer/projects/ParallelBNs/"
NETWORKS_FOLDER=${HOME_FOLDER}"res/networks/";
BBDD_FOLDER=${NETWORKS_FOLDER}"BBDD/";
TEST_FOLDER=${NETWORKS_FOLDER}"BBDD/tests/";
ENDING_NETWORKS=".xbif";
#ENDING_BBDD10K="10k.csv";
#ENDING_BBDD50K="50k.csv";
PARAMS_FOLDER=${HOME_FOLDER}"res/params/";
SAVE_FILE=${PARAMS_FOLDER}"hyperparams.txt"
#declare -a algorithms=("ges" "pges" "hc" "phc" "pfhcbes")

#declare -a net_names=("alarm" "andes" "barley" "cancer" "child" "earthquake" "hailfinder" "hepar2" "insurance" "link" "mildew" "munin" "pigs" "water" "win95pts")
#declare -a net_names=("andes")

declare networks=()
for net in ${net_names[@]}; do
    networks+=(${NETWORKS_FOLDER}$net${ENDING_NETWORKS} )
done

#echo "${networks[@]}"

#declare -a databases=(
#    ${BBDD_FOLDER}"alarm"${ENDING_BBDD10K} ${BBDD_FOLDER}"alarm"${ENDING_BBDD50K}
#    ${BBDD_FOLDER}"cancer"${ENDING_BBDD10K} ${BBDD_FOLDER}"cancer"${ENDING_BBDD50K}
#    ${BBDD_FOLDER}"barley"${ENDING_BBDD10K} ${BBDD_FOLDER}"barley"${ENDING_BBDD50K}
#    ${BBDD_FOLDER}"child"${ENDING_BBDD10K} ${BBDD_FOLDER}"child"${ENDING_BBDD50K}
#    ${BBDD_FOLDER}"insurance"${ENDING_BBDD10K} ${BBDD_FOLDER}"insurance"${ENDING_BBDD50K}
#    ${BBDD_FOLDER}"mildew"${ENDING_BBDD10K} ${BBDD_FOLDER}"mildew"${ENDING_BBDD50K}
#    ${BBDD_FOLDER}"water"${ENDING_BBDD10K} ${BBDD_FOLDER}"water"${ENDING_BBDD50K}
#)
endings=(".xbif_.csv" ".xbif50001_.csv" ".xbif50002_.csv" ".xbif50003_.csv"
".xbif50004_.csv" ".xbif50005_.csv" ".xbif50006_.csv" ".xbif50007_.csv" ".xbif50008_.csv"
".xbif50009_.csv" ".xbif50001246_.csv")

#databases=()
#for net in ${net_names[@]}; do
#    for ending in ${endings[@]}; do
#        databases+=(${BBDD_FOLDER}$net$ending)
#    done
#done

#tests=()
#for net in ${net_names[@]}; do
#    tests+=(${TEST_FOLDER}$net"_test.csv")
#done
seeds=(2 3 5 7 11 13 17 19 23 29)


#echo "${tests[@]}"

#declare -a nThreads=(1 2 4 8)
#declare -a nItInterleavings=(2 3 4)
declare -a nItInterleavings=(5 10 15)
#maxIterations=250


# Saving params
for ending in ${endings[@]};
do
    for nItInterleaving in ${nItInterleavings[@]}
    do
        for seed in ${seeds[@]}
        do
            echo $ending $nthread $nItInterleaving $seed >> $SAVE_FILE
        done
    done
done




# len_nets=${#networks[@]};
# for((i=0;i<$len_nets;i++))
# do
#     netPath=${networks[$i]};
#     testPath=${tests[$i]};
#     net_name=${net_names[$i]}

#     # Deleting files of previous params
#     FILE=${PARAMS_FOLDER}experiments_${net_name}.txt
#     #if test -f "$FILE"; then
#     #    rm $FILE
#     #fi

#     # Creating new File
#     >${FILE}
#     echo "Creating experiments for: ${net_name} at: ${FILE}"
#     #if test -f "$FILE"; then
#     #    echo "File Created: $FILE"
#     #fi
#     # Defining Databases for current network
#     databases=()
#     for ending in ${endings[@]}; do
#         databases+=(${BBDD_FOLDER}${net_name}$ending)
#     done

#     for dataPath in ${databases[@]};
#     do
#         for nthread in ${nThreads[@]};
#         do
#             for nItInterleaving in ${nItInterleavings[@]};
#             do
#                 for alg in ${algorithms[@]};
#                 do
#                     if [ $alg == "ges" ]
#                     then
#                         echo ${net_names[$i]} $alg $netPath $dataPath $testPath >> $FILE
#                     elif [ $alg == "hc" ]
#                         then
#                             echo ${net_names[$i]} $alg $netPath $dataPath $testPath $nItInterleaving $maxIterations >> $FILE
#                     else
#                         for seed in ${seeds[@]};
#                         do
#                             echo ${net_names[$i]} $alg $netPath $dataPath $testPath $nItInterleaving $maxIterations $nthread $seed >> $FILE
#                         done
#                     fi
#                 done
#             done
#         done
#     done
# done

# # Iterate the string array using for loop
# # echo "$net_number $bbdd_number $fusion $nThreads $nItInterleaving" #>> experiments.txt
                    