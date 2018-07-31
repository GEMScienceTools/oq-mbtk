ROOT="../../../tools/fault_modeler"

INPUT="Data/ne_asia_faults_rates_bvalue.geojson"
OUTPUT="test.xml"

#fault_source_modeler.py -geo $INPUT \
#                        -xml $OUTPUT \
#                        --project-name "PIPPO" \
#                        --select-list [1,2]

$ROOT/fault_source_modeler.py -cfg config.ini