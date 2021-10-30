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

write_legend_file()
{
    FILE=$1
    THRES=$2
    echo "H 10 1 GSHAP and GEM models: areas exceeding $THRES [g]" > $FILE
    echo "G 1.2l" >> $FILE
    echo "P" >> $FILE
    echo "S 0.01 c 0.3 red 0.5 .5  Only GEM model exceeds the ground motion threshold" >> $FILE
    echo "G 0.85l" >> $FILE
    echo "S 0.01 c 0.3 blue 0.5 .5 Only GSHAP model exceeds the ground motion threshold" >> $FILE
    echo "G 0.85l" >> $FILE
    echo "S 0.01 c 0.3 green 0.5 .5 Both GEM and GSHAP models exceed the ground motion threshold" >> $FILE
}


#
# Plotting map
plot()
{
    FPATH="$(dirname $0)"
    INPUT_GEM=$1
    INPUT_GSHAP=$2
    THRES=$3
    DATA=$4
    ROOT=$5
    GRADIENT=$6

    set_defaults
    ROOT=$ROOT"/gshap"
    echo "root: "$ROOT
    echo "threshold: "$THRES

    if $GRADIENT; then
        delete_temporary_folders $ROOT
        create_temporary_folders $ROOT
    fi

    PRO="-JN10/26c"
    ORI=""
    EXTE=$(set_map_extent)

    CPTT3=$FPATH"/cpt/bathy.cpt"

    PS=$ROOT"/fig/""${THRES/\./pt}"".ps"
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

    # shaded relief
    RES="5k"
    if $GRADIENT; then
        gmt nearneighbor $INPUT_GSHAP -R$EXTE -I$RES -G$GRD0 -V -N4/2 -S30k
        gmt grdsample $GRD0 -I$RES -R$ext -G$GRDGSHAP -r

        gmt nearneighbor $INPUT_GEM -R$EXTE -I$RES -G$GRD0 -V -N4/2 -S30k
        gmt grdsample $GRD0 -I$RES -R$ext -G$GRDGEM -r

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

    GRDA=$ROOT"/grd/grd_a.grd"
    GRDB=$ROOT"/grd/grd_b.grd"
    GRDC=$ROOT"/grd/grd_c.grd"
    GRDD=$ROOT"/grd/grd_d.grd"
    GRDE1=$ROOT"/grd/grd_e1.grd"
    GRDE2=$ROOT"/grd/grd_e2.grd"
    GRDF=$ROOT"/grd/grd_f.grd"
    GRDG=$ROOT"/grd/grd_g.grd"
    GRDX=$ROOT"/grd/grd_x.grd"
    GRDFGEM=$ROOT"/grd/grd_gem.grd"
    GRDFGSH=$ROOT"/grd/grd_gsh.grd"

    # Selecting hazard data
    if true; then
        gmt grdlandmask -NNaN/1 -G$GRDA -R$EXTE -I$RES -A1000+as+l -Df -r
        gmt grdlandmask -NNaN/0 -G$GRDC -R$EXTE -I$RES -A1000+as+l -Df -r

        gmt grdmath $GRDGEM $GRDC AND = $GRDX
        gmt grdmath $GRDX $GRDA OR = $GRDD
        gmt grdmath $GRDD $THRES GT = $GRDE1
        gmt grdmath $GRDE1 0.0 NAN = $GRDFGEM

        gmt grdmath $GRDGSHAP $GRDC AND = $GRDX
        gmt grdmath $GRDX $GRDA OR = $GRDD
        gmt grdmath $GRDD $THRES GT = $GRDE2
        gmt grdmath $GRDE2 0.0 NAN = $GRDFGSH

        gmt grdmath $GRDE1 2 MUL = $GRDF
        gmt grdmath $GRDF $GRDE2 ADD = $GRDG
    else
        echo "Skipping grdmath"
    fi

    # Plots hazard map
    gmt grdimage $GRDG -R$EXTE $PRO -I$bat_shadow -C$CPTTX -O -K -V -Q -nb+c >> $PS

    TMPF=$ROOT"/tmp/leg.txt"
    write_legend_file $TMPF $THRES
    gmt pslegend $TMPF -Dg0/-40+w8c/0.3c -O -K -R$EXTE $PRO >> $PS

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
        -fgem | --filegem )     shift
                                FILENAME1=$1
                                ;;
        -fgshap | --filegshap ) shift
                                FILENAME2=$1
                                ;;
        -t | --threshold )      shift
                                THRESHOLD=$1
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
