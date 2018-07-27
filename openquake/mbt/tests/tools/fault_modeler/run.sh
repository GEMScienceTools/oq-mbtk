PATH="~/VirtualEnv/openquake/src/oq-mbtk/openquake/mbt/tools/fault_modeler":$PATH

INPUT="Data/ne_asia_faults_rates_bvalue.geojson"
OUTPUT="test.xml"

#fault_source_modeler.py -geo $INPUT \
#                        -xml $OUTPUT \
#                        --project-name "PIPPO" \
#                        --select-list [1,2]

fault_source_modeler.py -cfg config.ini