for file in *
do
    sed 's/\(.*[0-9][a-zA-Z][a-zA-Z][a-zA-Z]\).*$/\1%/' $file | sed 's/\([a-zA-Z][a-zA-Z][a-zA-Z]%\).*$//' > ../results2/$file
done