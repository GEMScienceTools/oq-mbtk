
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
            rm -rf $DIRECTORY
        fi
    done
}


create_temporary_folders()
{
    ROOT=$1
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
    
    INPUT=$1
    DATA=$2
    ROOT=$3
    GRADIENT=$4

    set_defaults
    echo "root: "$ROOT
    delete_temporary_folders $ROOT
    create_temporary_folders $ROOT

    PRO="-JN10/75c"
    ORI=""
    EXTE=$(set_map_extent)

    # Input cpt
    CPTT2="./cpt/gm.cpt"
    # Input topography
    GTOPO=$DATA"/gem/gmt/globalGTOPO30.grd"
    # Input bathymetry
    bat_grd=$DATA"/gem/gmt/ETOPO1_Ice_g_gmt4.grd"

    PS=$ROOT"/fig/out.ps"
    PNG=$ROOT"/fig/out.png"

    CPTT3=$ROOT"/cpt/bathy.cpt"

    GRD0=$ROOT"/grd/g0.grd"
    GRD1=$ROOT"/grd/g1.grd"
    GRDHAZ=$ROOT"/grd/hazard.grd"
    GRDHAZRES=$ROOT"/grd/hazard_resampled.grd"

    bat_grd_cut=$ROOT"/grd/bathymetry_cut.grd"
    bat_shadow=$ROOT"/grd/bathymetry_shadow.grd"

    # shaded relief
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

    # Plotting bathymetry and topography (the latter will be covered by hazard)
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

    # Plots hazard map
    gmt grdimage $GRDB -R$EXTE $PRO -I$bat_shadow -C$CPTT2 -O -K -nb -V -Q >> $PS

    # Finishing
    gmt pscoast -R$EXTE $PRO $ORI -Df -EGL,SJ+gwhite -O -K  >> $PS
    gmt psxy ./../data/gis/islands.gmt -R$EXTE $PRO -Gp500/9:BlightgreyFwhite -O -K -V >> $PS
    gmt pscoast -R$EXTE $PRO $ORI -Df -A$AREA+as+l -O -K -N1,thinnest,darkgray -Lg-125/-52.7+c0+w5000+f -Wthinnest,black -V >> $PS

    # Plotting the colorscale
    gmt psscale -Dg95/-52.7+w13c/0.3c+e+h -O -C$CPTT2 -L -S -R$EXTE $PRO >> $PS

    # Converting .ps to .png
    convert -density 400 -trim $PS -rotate 90 $PNG
    date

    # Cleaning
    rm gmt.*
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
