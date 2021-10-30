#!/usr/bin/env bash


###
### Sets the default GMT plotting parameters

set_defaults()
{
    gmt set MAP_GRID_CROSS_SIZE_PRIMARY = 0.2i
    gmt set MAP_FRAME_TYPE = PLAIN
    gmt set PS_MEDIA = a4
    gmt set FONT_TITLE = 12p
    gmt set FONT_LABEL = 10p
    gmt set FONT_ANNOT_PRIMARY = 8p
}

###
### Finds the header in a comma separated file and gets the index of the 
### column matching a given pattern
### usage: get_column_number <pattern> <path_to_filename>

get_column_number()
{
    # Variables
    PATTERN=$1
    FILE=$2
    # Processing
    LINE=$(grep $PATTERN $FILE)
    LINE_NUMBER=$(sed 's/\,/\'$'\n''/g' <<< $LINE | grep -n $PATTERN | cut -f 1 -d :) 
    echo $LINE_NUMBER 
}

###
### Find longitude and latitude extremes
### usage: get_lon_lat_limits_from_file <path_to_filename> <column_number> <skip_rows>

get_lon_lat_limits_from_file()
{
    # Variables
    FILE=$1
    COLUMN=$2
    SKIP_ROWS=$3
    # Processing
    FILEXY='/tmp/xy.txt'
    #gawk -F, 'FNR>$"'{SKIP_ROWS}'"{print $1, $2, $"'${COLUMN}'"}' $FILE > $FILEXYZ
    gawk -F, 'FNR>$"'{SKIP_ROWS}'"{print $1, $2}' $FILE > $FILEXY
    EXT=`gmt info -I-. $FILEXY`
    MINLO="$(echo $EXT | sed -e 's/^-R//' | gawk -F\/ '{print $1}')"
    MAXLO="$(echo $EXT | sed -e 's/^-R//' | gawk -F\/ '{print $2}')"
    MINLA="$(echo $EXT | sed -e 's/^-R//' | gawk -F\/ '{print $3}')"
    MAXLA="$(echo $EXT | sed -e 's/^-R//' | gawk -F\/ '{print $4}')"
    echo "$MINLO/$MAXLO/$MINLA/$MAXLA"
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

###
### Find mid longitude 
### usage: get_mean_longitude <list>

get_mean_longitude()
{
    # Variables
    LIST=$1
    # Local variables
    SMM=0
    # Processing
    NUM=${#LIST[@]}
    for i in "${LIST[@]}"
    do
        VAL1=$(echo $i | sed -e 's/-/_/g') 
        VAL2=$(echo $SMM | sed -e 's/-/_/g')
        SMM=$(dc <<< "$VAL2 $VAL1 + p")
    done
    MEAN=$(dc <<< "$SMM $NUM / p")
    echo $MEAN
}


###
### Find longitude and latitude limits from a list of strings with format 
### <number>/<number>/<number>/<number>
### usage: get_lon_lat_limits <list>

get_lon_lat_limits()
{
    # Variables
    LIST=$1
    # Initialise variables
    LOMIN=10000
    LAMIN=10000
    LOMAX=-10000 
    LAMAX=-10000 
    # Processing
    for i in "${LIST[@]}"
    do
        VALUES=(${i//\// }) 
        # Minimum longitude
        tmp=${VALUES[0]}
        LOMIN=$(bc <<< "if ($tmp < $LOMIN) $tmp else $LOMIN") 
        # Minimum latitude
        tmp=${VALUES[1]}
        LOMAX=$(bc <<< "if ($tmp > $LOMAX) $tmp else $LOMAX") 
        # Maximum longitude
        tmp=${VALUES[2]}
        LAMIN=$(bc <<< "if ($tmp < $LAMIN) $tmp else $LAMIN")   
        # Maximum latitude
        tmp=${VALUES[3]}
        LAMAX=$(bc <<< "if ($tmp > $LAMAX) $tmp else $LAMAX") 
    done
    echo "$LOMIN/$LOMAX/$LAMIN/$LAMAX"
}

###
### This plots the hazard map
### usage: plot <map_file_1> <map_file_2> 

plot()
{
     
    # Input parameters
    INPUT1=$1
    INPUT2=$2
    PATTERN=$3
    ROOT=$4
    DATA=$5
    GRADIENT=$6

    # Set default parameters
    SKIP_ROWS=2
    set_defaults
    
    # Find column number     
    COLUMN_NUMBER1=$(get_column_number $PATTERN $INPUT1)
    COLUMN_NUMBER2=$(get_column_number $PATTERN $INPUT2)

    # Find map limits 
    EXTE1=$(get_lon_lat_limits_from_file $INPUT1 $COLUMN_NUMBER1 $SKIP_ROWS)
    EXTE2=$(get_lon_lat_limits_from_file $INPUT2 $COLUMN_NUMBER2 $SKIP_ROWS)
    LIST=($EXTE1 $EXTE2)
    EXTE=$(get_lon_lat_limits $EXTE1)

    # Find mean longitude
    VALUES=(${EXTE//\// })
    LIST=(${VALUES[0]} ${VALUES[1]})
    MEAN_LON=$(get_mean_longitude $LIST)

    # Info
    echo 'Map limits      : '$EXTE
    echo 'Mean longitude  : '$MEAN_LON

    # Set GMT parameters  
    PRO="-JN$MEAN_LON/20c"
    ORI=""

    # Create temporary folders
    delete_temporary_folders $ROOT
    create_temporary_folders $ROOT

    PS=$ROOT"/fig/compare.ps"
    PNG=$ROOT"/fig/compare.png"

    GRD0=$ROOT"/tmp/cg0.grd"
    GRD1=$ROOT"/tmp/cg1.grd"
    GRD2=$ROOT"/tmp/cg2.grd"
    GRD3=$ROOT"/tmp/cg3.grd"
    GRD4=$ROOT"/tmp/cg4.grd"
    GRDHAZ1=$ROOT"/tmp/cg5.grd"
    GRDHAZ2=$ROOT"/tmp/cg6.grd"

    CPT=$ROOT"/cpt/tmp.cpt"

    bat_grd="/Users/mpagani/Documents/2018/diary/10/19_global_map/dat/ETOPO1_Ice_g_gmt4.grd"
    bat_grd_cut=$ROOT"/tmp/bathymetry_cut.grd"
    bat_shadow=$ROOT"/tmp/bathymetry_shadow.grd"

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

    fi

    # DENAN - Replace NaNs in A with values from B
    #gmt grdmath ./grd/f1.grd ./grd/f2.grd DENAN = ./grd/f1a.grd
    #gmt grdmath ./grd/f1.grd ./grd/f2.grd DENAN = ./grd/f1a.grd
    #gmt grdmath -V ./grd/f1a.grd ./grd/f2.grd SUB = ./grd/ff.grd

    if true; then
        CPTINPUT=$DATA"/gem/cpt/spectral.cpt"
        # gmt makecpt -C$CPTINPUT -T-0.005/0.005/0.001 > $CPT
        gmt makecpt -C$CPTINPUT -T-0.001/0.001/0.0002 > $CPT
        gmt grdmath $ROOT"/grd/f1.grd" $ROOT"/grd/f2.grd" SUB = $ROOT"/grd/ff.grd"
        SPACING=0.0005
    else
        gmt makecpt -Cred2green -T-0.25/0.25/0.005 > $CPT
        # (current - previous) / previous * 100
        gmt grdmath $GRD1 $GRD2 SUB $GRD2 DIV 100 MUL = ./grd/ff.grd
        SPACING=0.1
    fi

    gmt grdinfo $ROOT"/grd/ff.grd"

    # Plotting coasts and preparing .cpt files
    gmt pscoast -R$EXTE $PRO $ORI -N1 -Bf10a30/f10a30WSEn -Df -A1000+as+l -K -Gwhite -Wthinnest > $PS

    # Plots hazard map
    #gmt grdimage ./grd/ff.grd -R$EXTE $PRO -I$bat_shadow -C$CPT -O -K -nb -V -Q >> $PS
    gmt grdimage $ROOT"/grd/ff.grd" -R$EXTE $PRO -C$CPT -O -K -nb -V -Q >> $PS

    # Plotting the colorscale
    gmt psscale -Dx21/0+w6c/0.2c+e -R$EXTE $PRO -O -K -C$CPT -S -B$SPACING >> $PS

    # Finishing
    gmt pscoast -R$EXTE $PRO $ORI -Df -A1000+as+l -O -N1 -Wthinnest >> $PS
    #
    # Converting .ps to .png
    convert -density 400 -trim $PS -rotate 90 $PNG
    date
    rm gmt.conf

    echo "created: "$PNG
}

###
### Main 

# Set variables
PATTERN='PGA-0.002105\|PGA-0.002107'
ROOT='/tmp'
DATA=$HOME"/Data"
GRADIENT=true

while [ "$1" != "" ]; do
    case $1 in
        -f1 | --file1 )         shift
                                FILENAME1=$1
                                ;;
        -f2 | --file2 )         shift
                                FILENAME2=$1
                                ;;
        -r | --root )           shift
                                ROOT=$1
                                ;;
        -d | --data )           shift
                                DATA=$1
                                ;;
        -g | --gradient )       shift
                                GRADIENT=$1
                                ;;

    esac
    shift
done

# Plotting map
plot $FILENAME1 $FILENAME2 $PATTERN $ROOT $DATA $GRADIENT
