gmt begin
gmt figure gmt2/sizescaled pdf
gmt basemap -R18.0/33.0/32.0/43.0 -JM15c -BWSne+t"Seismicity: size scaled" -Bx2.0 -By2.0
gmt coast -Di -R18.0/33.0/32.0/43.0 -JM15c -Wthin -Gwheat
gmt plot gmt2/tmp_dat_sizeDepth-km.csv -Ssc -Gyellow -Wblack
gmt legend gmt2/legend_ss.csv -DJMR -C0.3c --FONT_ANNOT_PRIMARY=12p
gmt end