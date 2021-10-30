#!/usr/bin/env bash

set_defaults()
{
    gmt set MAP_GRID_CROSS_SIZE_PRIMARY = 0.2i
    gmt set MAP_FRAME_TYPE = PLAIN
    gmt set PS_MEDIA = a4
    gmt set FONT_TITLE = 12p
    gmt set FONT_LABEL = 10p
    gmt set FONT_ANNOT_PRIMARY = 8p
}


set_map_extent()
{
    MINLO=-170.0; MAXLO=+190.0 
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
            rm -rf $DIRECTORY
        fi
    done
}


create_temporary_folders()
{
    ROOT=$1
    mkdir $ROOT
    for i in "/tmp" "/grd" "/fig" "/cpt";
    do
        DIRECTORY=$ROOT$i
        mkdir $DIRECTORY
    done
}

#
# Plotting map
plot()
{
    FPATH="$(dirname $0)"
    INPUT1=$1
    INPUT2=$2
    DATA=$3
    ROOT=$4
    GRADIENT=$5

    set_defaults
    ROOT=$ROOT"/global_diff"
    echo "root: "$ROOT

    if $GRADIENT; then
        delete_temporary_folders $ROOT
        create_temporary_folders $ROOT
    fi

    PRO="-JN10/26c"
    ORI=""
    EXTE=$(set_map_extent)

    CPTT3=$FPATH"/cpt/bathy.cpt"

    PS=$ROOT"/fig/global_diff.ps"
    PNG="${PS/\.ps/.png}"
    
    GRD0=$ROOT"/grd/g0.grd"
    GRD1=$ROOT"/grd/g1.grd"
    GRDGSHAP=$ROOT"/grd/gshap.grd"
    GRDGEM=$ROOT"/grd/gem.grd"

    # Input topography
    GTOPO=$DATA"/gem/gmt/globalGTOPO30.grd"
    # Input bathymetry
    bat_grd=$DATA"/gem/gmt/ETOPO1_Ice_g_gmt4.grd"
    bat_grd_cut=$ROOT"/grd/bathymetry_cut.grd"
    bat_shadow=$ROOT"/grd/bathymetry_shadow.grd"

    # Building the colorscale
    CPTTX=$FPATH"/cpt/vals.cpt"
    CPTDFF=$ROOT"/cpt/dff.cpt"

    # Set grids
     GRD0=$ROOT"/tmp/cg0.grd"
     GRD1=$ROOT"/tmp/cg1.grd"
     GRD2=$ROOT"/tmp/cg2.grd"
     GRD3=$ROOT"/tmp/cg3.grd"
     GRD4=$ROOT"/tmp/cg4.grd"
     GRDHAZ1=$ROOT"/tmp/cg5.grd"
     GRDHAZ2=$ROOT"/tmp/cg6.grd"

    # shaded relief
    RES="5k"
    if $GRADIENT; then
        
        gmt nearneighbor $INPUT1 -R$EXTE -I$RES -G$GRD0 -V -N4/2 -S30k
        gmt grdsample $GRD0 -I$RES -R$ext -G$GRDHAZ1 -r

        gmt nearneighbor $INPUT2 -R$EXTE -I$RES -G$GRD0 -V -N4/2 -S30k
        gmt grdsample $GRD0 -I$RES -R$ext -G$GRDHAZ2 -r

        gmt grdlandmask -NNaN/1 -G$GRD3 -R$EXTE -I$RES -A1000+as+l -Df -r
        gmt grdlandmask -NNaN/0 -G$GRD4 -R$EXTE -I$RES -A1000+as+l -Df -r

        gmt grdmath $GRDHAZ1 $GRD4 AND = $ROOT"/grd/ee.grd"
        gmt grdmath $ROOT"/grd/ee.grd" $GRD3 OR = $ROOT"/grd/f1.grd"

        gmt grdmath $GRDHAZ2 $GRD4 AND = $ROOT"/grd/ee.grd"
        gmt grdmath $ROOT"/grd/ee.grd" $GRD3 OR = $ROOT"/grd/f2.grd"

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
    gmt pscoast -R$EXTE $PRO $ORI -N1 -Bf10a30/f10a30WSEn -Df -A1000/0/1 -K -Gwhite -Wthinnest > $PS

    # Plotting bathymetry and topography (the latter will be covered by hazard)
    gmt makecpt -Cgray -T-4/0.5 > $CPTT3
    gmt grdimage $bat_shadow -R$EXTE $PRO $ORI -O -K -C$CPTT3 -V -Q >> $PS

    # Selecting hazard data
    if true; then
        gmt grdmath $ROOT"/grd/f1.grd" $ROOT"/grd/f2.grd" SUB = $ROOT"/grd/ff.grd"
        SPACING=0.1
    else
        echo "Skipping grdmath"
    fi

    gmt grdinfo $ROOT"/grd/ff.grd"

    # Plots hazard map
    gmt makecpt -Cpolar -T-0.5/0.3 -D > $CPTDFF
    gmt grdimage $ROOT"/grd/ff.grd" -R$EXTE $PRO -C$CPTDFF -O -K -nb -V -Q >> $PS

    # Plotting the colorscale
    gmt psscale -Dx13/1+w6c/0.1c+e+h -R$EXTE $PRO -O -K -C$CPTDFF -S -B$SPACING --FONT_ANNOT_PRIMARY=5p,Helvetica >> $PS

    # Finishing
    gmt pscoast -R$EXTE $PRO $ORI -Df -EGL,SJ+gwhite -O -K  >> $PS
    gmt psxy $FPATH"/../data/gis/islands.gmt" -R$EXTE $PRO -Gp500/9:BlightgreyFwhite -O -K -V >> $PS
    gmt pscoast -R$EXTE $PRO $ORI -Df -A1000+as+l -O -Wthinnest,black -V >> $PS

    # Converting .ps to .png
    convert -density 400 -trim $PS -rotate 90 $PNG
    # gmt psconvert $PS -Tj -V
    date
    echo $INPUT
    rm gmt.conf

    echo "Created: "$PNG
}

DATA=$HOME"/Data"
ROOT='/tmp'
GRADIENT=true

while [ "$1" != "" ]; do
    case $1 in
        -f1 | --filegem )       shift
                                FILENAME1=$1
                                ;;
        -f2 | --filegshap )     shift
                                FILENAME2=$1
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

plot $FILENAME1 $FILENAME2 $THRESHOLD $DATA $ROOT $GRADIENT
