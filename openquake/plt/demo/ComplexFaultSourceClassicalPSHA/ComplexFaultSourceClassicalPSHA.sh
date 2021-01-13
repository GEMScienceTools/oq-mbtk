gmt begin
gmt figure ComplexFaultSourceClassicalPSHA/ComplexFaultSourceClassicalPSHA jpg
gmt basemap -R-2.5/2.5/-2.5/2.5 -JM15c -BWSne+t"ComplexFaultSourceClassicalPSHA" -Bx2.0 -By2.0
gmt coast -Di -R-2.5/2.5/-2.5/2.5 -JM15c -Wthin -Gwheat
gmt makecpt -Cjet -T0/35.00000001352754/2> ComplexFaultSourceClassicalPSHA/cf_tmp.cpt
gmt plot ComplexFaultSourceClassicalPSHA/mtkComplexFaultPoints.csv -CComplexFaultSourceClassicalPSHA/cf_tmp.cpt -Ss0.075 -t90
gmt colorbar -DJBC -Ba10+l"Depth to complex fault surface (km)" -CComplexFaultSourceClassicalPSHA/cf_tmp.cpt
gmt plot ComplexFaultSourceClassicalPSHA/mtkComplexFaultOutline.csv -Wthick,black
gmt end