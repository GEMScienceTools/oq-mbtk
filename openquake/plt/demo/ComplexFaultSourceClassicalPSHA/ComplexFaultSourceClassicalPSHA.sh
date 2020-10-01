gmt begin
gmt figure ComplexFaultSourceClassicalPSHA/ComplexFaultSourceClassicalPSHA jpg
gmt basemap -R-2.5/2.5/-2.5/2.5 -JM15 -BWSne+t"ComplexFaultSourceClassicalPSHA" -Bx2.0 -By2.0
gmt coast -Df -R-2.5/2.5/-2.5/2.5 -JM15 -Wthin -Gwheat
gmt makecpt -Cjet -T0/35.000000013527554/2> ComplexFaultSourceClassicalPSHA/cf_tmp.cpt
gmt plot ComplexFaultSourceClassicalPSHA/mtkComplexFaultPoints.csv -CComplexFaultSourceClassicalPSHA/cf_tmp.cpt -Ss0.1 
gmt colorbar -DJBC -Ba10+l"Depth (km)" -CComplexFaultSourceClassicalPSHA/cf_tmp.cpt
gmt plot ComplexFaultSourceClassicalPSHA/mtkComplexFaultOutline.csv -Wthick,black
gmt end