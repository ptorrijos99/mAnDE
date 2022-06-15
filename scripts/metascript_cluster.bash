#!/bin/bash

FILENAME="/home/pablot/mAnDE/res/params/databases.txt"

SCRIPT="/home/pablot/mAnDE/scripts/run_experiments_cluster.bash"

while read bbdd; do
    qsub -J 0-129 -v PARAMS="$bbdd" -l nodes=node8:ppn=3 "$SCRIPT"   	# 3 Hilos
done < "$FILENAME"

#qsub -J 0-129 -v PARAMS="2018_Financial_Data" -l nodes=node8:ppn=3 "$SCRIPT"   	# 3 Hilos