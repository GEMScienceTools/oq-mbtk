#!/usr/bin/env bash

add_licence()
{
    gmt pstext -F+f18p,Helvetica,-=0.5p,red << EOF
-35.0 -57.0 © GEM Foundation 2023 (map v2023.1) - License CC BY-SA
EOF
}

set_defaults()
{
    gmt set MAP_GRID_CROSS_SIZE_PRIMARY = 0.2i
    gmt set MAP_FRAME_TYPE = PLAIN
    gmt set PS_MEDIA = a1
    gmt set FONT_TITLE = 16p
    gmt set FONT_LABEL = 12p
}


set_map_extent()
{
    MINLO=-180.0; MAXLO=+180.0 
    MINLA=-60.0; MAXLA=+89.99
    EXTE="$MINLO/$MAXLO/$MINLA/$MAXLA"
    echo $EXTE
}


delete_temporary_folders()
{
    ROOT=$1
    for i in "/tmp" "/grd" "/fig" "/cpt";
    do
        DIRECTORY=$ROOT$i
        if [ -d "$DIRECTORY" ]; then
		echo "Not erasing"
           # rm -rf $DIRECTORY
        fi
    done
}


create_temporary_folders()
{
    ROOT=$1
    for i in "/tmp" "/grd" "/fig" "/cpt";
    do
        DIRECTORY=$ROOT$i
        mkdir -p $DIRECTORY
    done
}


#
# Plotting map
plot()
{
    
    INPUT=$1
    DATA=$2
    ROOT=$3
    GRADIENT=$4

    DISTANCE_SCALE=false

    # Script location
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
    echo $SCRIPT_DIR

    set_defaults
    echo "root: "$ROOT
    delete_temporary_folders $ROOT
    create_temporary_folders $ROOT

    PRO="-JN10/75c"
    ORI=""
    EXTE=$(set_map_extent)

    # Input cpt
    CPTT2=$SCRIPT_DIR"/cpt/gm_1pt5.cpt"
    #CPTT2="./cpt/gm_new.cpt"

    # Input topography
    #GTOPO=$DATA"/gem/gmt/globalGTOPO30.grd"
    GTOPO=$DATA"/dem/ETOPO1_Ice_g_gmt4.grd"
    # Input bathymetry
    bat_grd=$DATA"/dem/ETOPO1_Ice_g_gmt4.grd"

    PS=$ROOT"/fig/out.ps"
    PNG=$ROOT"/fig/out.png"

    CPTT3=$ROOT"/cpt/bathy.cpt"

    GRD0=$ROOT"/grd/g0.grd"
    GRD1=$ROOT"/grd/g1.grd"
    GRDHAZ=$ROOT"/grd/hazard.grd"
    GRDHAZRES=$ROOT"/grd/hazard_resampled.grd"

    bat_grd_cut=$ROOT"/grd/bathymetry_cut.grd"
    bat_shadow=$ROOT"/grd/bathymetry_shadow.grd"

    # Shaded relief
    RES="5k"

    if $GRADIENT; then
        rm $ROOT"/grd/*.*"
        rm $ROOT"/grdtopo/*.*"

        gmt nearneighbor $INPUT -R$EXTE -I$RES -G$GRD0 -V -N4/2 -S20k
        gmt grdsample $GRD0 -I$RES -R$ext -G$GRDHAZ -r

        gmt grdcut $bat_grd -G$GRD0 -R$EXTE -V
        gmt grdgradient $GRD0 -G$bat_grd_cut -Ne0.3 -A45
        gmt grdsample $bat_grd_cut -I$RES -R$EXTE -G$bat_shadow -r
    else
        echo "Skipping gradient creation"
    fi

    # Set output filename
    FOU=${FIN##*/}
    FOU=${FOU%.csv}

    # Plotting coasts and preparing .cpt files
    AREA=500
    gmt pscoast -R$EXTE $PRO $ORI -N1 -Bf10a30/f10a30WSEn -Df -A$AREA/0/1 -K -Gwhite -Wthinnest > $PS

    # Plot bathymetry and topography (the latter will be covered by hazard)
    gmt makecpt -Cgray -T-4/0.5 > $CPTT3
    gmt grdimage $bat_shadow -R$EXTE $PRO $ORI -O -K -C$CPTT3 -V -Q >> $PS

    GRDA=$ROOT"/grd/grd_a.grd"
    GRDB=$ROOT"/grd/grd_b.grd"
    GRDC=$ROOT"/grd/grd_c.grd"
    GRDD=$ROOT"/grd/grd_d.grd"

    # Selecting hazard data
    if true; then
        gmt grdlandmask -NNaN/1 -G$GRDA -R$EXTE -I$RES -A$AREA+as+l -Df -r
        gmt grdlandmask -NNaN/0 -G$GRDC -R$EXTE -I$RES -A$AREA+as+l -Df -r
        gmt grdmath $GRDHAZ $GRDC AND = $GRDD
        gmt grdmath $GRDD $GRDA OR = $GRDB
    else
        echo "Skipping grdmath"
    fi

    # Plot hazard map
    gmt grdimage $GRDB -R$EXTE $PRO -I$bat_shadow -C$CPTT2 -O -K -nb -V -Q >> $PS

    # Finish the map with coasts
    gmt pscoast -R$EXTE $PRO $ORI -Df -ESJ+gwhite -O -K  >> $PS
    gmt psxy $SCRIPT_DIR"/../data/gis/islands_gld.gmt" -R$EXTE $PRO -Gp500/9:BlightgreyFwhite -O -K -V >> $PS

    # Here we have two options 
    if [$DISTANCE_SCALE = true] ; then
        gmt pscoast -R$EXTE $PRO $ORI -Df -A$AREA+as+l -O -K -N1/thinnest,darkgray -Lg-125/-52.7+c0+w5000+f -Wthinnest,black -V >> $PS
    else
        gmt pscoast -R$EXTE $PRO $ORI -Df -A$AREA+as+l -O -K -N1/thinnest,darkgray -Wthinnest,black -V >> $PS
    fi

    # Add the licence text
    #-35.0 -58© GEM Foundation 2023 (map v2023.1) - License CC BY-SA
    # echo "-35.0 -58.0 LM License CC BY-SA" | gmt pstext -R$EXTE $PRO $ORI -F+f14p,Helvetica-Bold+jTL -O -V >> $PS

    # Plotting the colorscale
    gmt psscale -Dg95/-52.7+w13c/0.3c+e+h -O -C$CPTT2 -L -S -R$EXTE $PRO >> $PS

    # Convert .ps to .png
    convert -density 400 -trim $PS -rotate 90 $PNG
    date

    # Clean
    #rm gmt.*
}


DATA=$HOME"/Data"
ROOT='/tmp'
GRADIENT=true

while [ "$1" != "" ]; do
    case $1 in
        -f | --file )           shift
                                FILENAME=$1
                                ;;
        -d | --data )           shift
                                DATA=$1
                                ;;
        -r | --root )           shift
                                ROOT=$1
                                ;;
        -g | --gradient )       shift
                                GRADIENT=$1
                                ;;
    esac
    shift
done

plot $FILENAME $DATA $ROOT $GRADIENT
