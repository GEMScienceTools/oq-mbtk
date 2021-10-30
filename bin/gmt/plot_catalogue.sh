#!/usr/bin/env bash

# $1 - Catalogue filename
# $2 - Extent in the form "105/120/15/30"
# $2 - Output figure filename
# $3 - The minimum magnitude for which we write its value

ARG1=${1}
ARG2=${2:-0/360/-90/90}
ARG3=${3:-/tmp/catalogue_plot} 
ARG4=${4:-1.0c} # Size of the map
ARG5=${5:-'"'$ARG1'"'} # Title
ARG6=${6:-2.0} # Size of earthquakes
ARG7=${7:-0} # Coloring country
LOWMA=${8:-7.0}

EXTENT=-R$ARG2
PRO=-Jm$ARG4
PRO=-JR$ARG4

TOPO=$GEM_DATA/dem/globalGTOPO30.grd

gmt set MAP_FRAME_TYPE = PLAIN
gmt set MAP_GRID_CROSS_SIZE_PRIMARY = 0.2i
gmt set MAP_FRAME_TYPE = PLAIN
gmt set FONT_TITLE = 8p
gmt set FONT_LABEL = 6p
gmt set FONT_ANNOT_PRIMARY = 6p

gmt begin $ARG3

    gmt coast $EXTENT $PRO -Bp5 -N1  
    gmt makecpt -C150 -T-10000,10000 -N
    gmt grdimage $TOPO -I+d
    gmt coast -Ccyan -Scyan -V
    
    gawk -F, -v a="$ARG6" '{print $1, $2, a ** $4/800}' $ARG1 | gmt plot -Sc -W0.5,red -Gpink -t50
    if [ $LOWMA > 0 ]; then
        gawk -F, -v a="$LOWMA" '{if ($4 > a) print $1, $2, $4}' $ARG1 | gmt text -F+f4p,Helvetica -Dx0.1
    fi

    if [ $ARG7 > 0 ]; then
        gmt coast -E$ARG7+ggreen -t85
    fi 
    
gmt end show
 
rm gmt.*
