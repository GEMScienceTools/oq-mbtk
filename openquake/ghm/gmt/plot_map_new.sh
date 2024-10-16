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
    DTEXT=$5

    echo "File name       : "$INPUT
    echo "Data folder     : "$DATA
    echo "Root folder     : "$ROOT
    echo "Compute gradient: "$GRADIENT
    echo "Adding text     : "$DTEXT

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

    PS=$ROOT"/fig/out"
    # PNG=$ROOT"/fig/out.png"

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

    # Threshold area 
    AREA=500

    # Temporary grids
    GRDA=$ROOT"/grd/grd_a.grd"
    GRDB=$ROOT"/grd/grd_b.grd"
    GRDC=$ROOT"/grd/grd_c.grd"
    GRDD=$ROOT"/grd/grd_d.grd"

    # Create the map
    gmt begin $PS pdf,png

        # Plotting coasts and preparing .cpt files
        gmt coast -R$EXTE $PRO $ORI -N1 -Bf10a30 -BWSEn -Df -A$AREA/0/1 -Gwhite -Wthinnest 

        # Plot bathymetry and topography (the latter will be covered by hazard)
        gmt makecpt -Cgray -T-4/0.5/0.1 -V -H > $CPTT3
        gmt grdimage $bat_shadow -C$CPTT3 -V -Q 

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
        gmt grdimage $GRDB -I$bat_shadow -C$CPTT2 -nb -V -Q >> $PS

        # Finish the map with coasts
        gmt coast -Df -ESJ+gwhite 
        gmt plot $SCRIPT_DIR"/../data/gis/islands_gld.gmt" -Gp500/9:BlightgreyFwhite -V

        # Add coasts.Here we have two options 
        if [ $DISTANCE_SCALE] ; then
            gmt coast -Df -A$AREA+as+l -N1/thinnest,darkgray -Lg-125/-52.7+c0+w5000+f -Wthinnest,darkgray -V
        else
            gmt coast -Df -A$AREA+as+l -N1/thinnest,darkgray -Wthinnest,darkgray -V
        fi

        # Add the licence text and description
        if [ $DTEXT ] ; then
            echo "0.0 -57.0 \251 GEM Foundation 2023 (map v2023.1) - License CC BY-SA" | gmt text -F+f12p,Helvetica+jCM -V 
            text="Peak Ground Acceleration (PGA) in units of g on reference rock conditions with a 10%"
            echo "92.0 -48.5 $text" | gmt text -F+f9p,Helvetica+jCL -V 
            text="probability of being exceeded in 50 years, equivalent to a return period of 475 years."
            echo "93.25 -50.25 $text" | gmt text -F+f9p,Helvetica+jCL -V 
        fi

        # Plot the colorscale
        gmt colorbar -Dg95/-52.7+w13c/0.3c+e+h -C$CPTT2 -L -S

    gmt end show

    # Clean
    #rm gmt.*
}


DATA=$HOME"/Data"
ROOT='/tmp'
GRADIENT=true
DTEXT=false

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
        -t | --text )           shift
                                DTEXT=$1
                                ;;
    esac
    shift
done

plot $FILENAME $DATA $ROOT $GRADIENT $DTEXT
