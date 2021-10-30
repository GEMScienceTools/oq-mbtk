#!/usr/bin/env bash

# $1 - Catalogue filename
# $2 - Extent in the form "105/120/15/30"
# $2 - Output figure filename

ARG1=${1:-}
ARG2=${2:-R-60/360/-90/90}
ARG3=${3:-/tmp/fig}
ARG4=${4:-3c}
TITLE=${5:-"Catalogue"}
FORMAT=${6:-"pdf"}

EXTENT=-R$ARG2
PRO=-Jm$ARG4

if [ $ARG2 == "auto" ]; then
    EXTENT=$(gmt info $ARG1 -I2 -D1)
fi 

gmt set MAP_FRAME_TYPE = PLAIN
gmt set MAP_GRID_CROSS_SIZE_PRIMARY = 0.2i
gmt set MAP_FRAME_TYPE = PLAIN
gmt set FONT_TITLE = 8p
gmt set FONT_LABEL = 6p
gmt set FONT_ANNOT_PRIMARY = 6p
gmt set GMT_GRAPHICS_FORMAT = $FORMAT

gmt begin $ARG3

    gmt coast $EXTENT $PRO -Bp5 -N1  
    gmt coast -Ccyan -Scyan 
    gmt plot $ARG1 -Sc0.1 -W0.5,red -Gpink -t50
    gmt coast -N1 -t50 -B+t$TITLE
    
gmt end
 
rm gmt.*