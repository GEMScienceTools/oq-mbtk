#!/usr/bin/env bash

path='/Users/mpagani/Repos/venv/src/oq-mbtk/openquake/ghm/out/map*.json'
outfile='./xyz/global.xyz'
if [ -f $outfile ]; then
    echo "Removing global.xyz"
    rm $outfile
fi
touch $outfile

for file in $path
do
    echo $file
    ./cat_json.py $file -o ./xyz >> $outfile
done
