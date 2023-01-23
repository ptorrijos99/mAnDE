awk 'FNR==1 && NR!=1{next;}{print}'  *.csv > ../results.csv

awk '(NR == 1) || (FNR == 2)'  *.csv > ../results.csv

echo *.csv | xargs awk '(NR == 1) || (FNR == 2)'  > ../results.csv