#!/usr/bin/env bash

DAT1=/tmp/ghm/global/map.csv 
DAT2=/Users/mpagani/Data/gem/global_hazard/gshpub_g.dat 

#./../gmt/plot_exceedances.sh -fgem $DAT1 -fgshap $DAT2 -t $THR -g false

THR=0.1
./../gmt/plot_exceedances.sh -fgem $DAT1 -fgshap $DAT2 -t $THR 
cp /tmp/gshap/fig/0pt1.png .

THR=0.3
./../gmt/plot_exceedances.sh -fgem $DAT1 -fgshap $DAT2 -t $THR 
cp /tmp/gshap/fig/0pt3.png .

THR=0.5
./../gmt/plot_exceedances.sh -fgem $DAT1 -fgshap $DAT2 -t $THR 
cp /tmp/gshap/fig/0pt5.png .
