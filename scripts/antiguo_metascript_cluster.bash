#!/bin/bash

FILE="$HOME/mAnDE2/target/mAnDE-3.0-jar-with-dependencies.jar"
PARAMS="$HOME/mAnDE2/res/params/hyperparams.txt"
PARAMS2="$HOME/mAnDE2/res/params/hyperparams_resto.txt"
SCRIPT="$HOME/mAnDE2/scripts/galgo.bash"
#SCRIPT="$HOME/mAnDE/scripts/run_mAnDE_experiments.bash"


declare -a nodes=("cn65" "cn66" "cn67" "cn68" "cn69" "cn70" "cn71" "cn72" "cn73" "cn74" "cn75" "cn78" "cn79" "cn80" "cn81" "cn82" "cn83" "cn84" "cn85" "cn86" "cn87" "cn88" "cn89" "cn90" "cn91" "cn92" "cn93" "cn94" "cn95" "cn96" "cn97" "cn98" "cn99" "cn100" "cn101" "cn102" "cn103" "cn104" "cn105")
#declare -a nodes=("cn65" "cn66" "cn67" "cn68" "cn69" "cn70" "cn71" "cn72" "cn73" "cn74" "cn75" "cn78" "cn79" "cn80" "cn81" "cn82" "cn83" "cn84" "cn85" "cn86" "cn87" "cn88" "cn89" "cn90" "cn91" "cn92" "cn94" "cn95" "cn96" "cn97" "cn98" "cn99" "cn100" "cn101" "cn102" "cn104" "cn105")

declare -i x=0
declare -i a=0

declare -i y=61
declare -i b=123

for node in ${nodes[@]};
do
    echo $node
    qsub -N mAnDE2 -J $x-$y -v CWD="$PWD",PARAMS="$PARAMS",FILE="$FILE" -l nodes=$node:ppn=8,mem=31gb "$SCRIPT"   	# 8 Hilos

    #qsub -N others2 -J $a-$b -v CWD="$PWD",PARAMS="$PARAMS2",FILE="$FILE" -l nodes=$node:ppn=8,mem=31gb "$SCRIPT"   	# 8 Hilos

    # 2480 experimentos mAnDE, 4960 resto, 39* (falta 76) nodos
    # 62 mAnDE cada uno, 124 resto

    x=$((x+62))
    y=$((y+62))
    a=$((a+124))
    b=$((b+124))
done