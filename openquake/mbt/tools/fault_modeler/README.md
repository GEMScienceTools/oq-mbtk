# Fault Model Building Tools

This is a set of

## Getting Started

The fault model builder can be run as a Python module or as a bash script.
As a module, the main function *build_fault_model* can be called with:

```python
from openquake.mbt.tools.fault_modeler.fault_source_modeler import build_fault_model

build_fault_model(geojson_file='FaultDatabase.geojson',
                  xml_output='FaultModel.xml')
```

or, alternatively, by passing a configuration file (.ini):

```python
from openquake.mbt.tools.fault_modeler.fault_source_modeler import build_fault_model

build_fault_model(cfg_file='config.ini')
```

## Configuration File Format

The configuration file (.ini) replaces the need of manually passing each argument to the function. It contains a list of processing information with the following format:

```
[config]
geojson_file = FaultDatabase.geojson
xml_output = FaultModel.xml

[param_map]
source_id = ogc_fid
name = ns_name
average_dip = ns_average_dip
average_rake = ns_average_rake
net_slip_rate = ns_net_slip_rate
vert_slip_rate = ns_vert_slip_rate
strike_slip_rate = ns_strike_slip_rate
shortening_rate = ns_shortening_rate
dip_dir = ns_dip_dir
dip_slip_rate = ns_dip_slip_rate

[defaults]
upper_seismogenic_depth = 0
lower_seismogenic_depth = 15
tectonic_region_type = ActiveShallow
rupture_mesh_spacing = 5
```

The use of any of the three blocks is optional. The first block is referred to the input/output. The second block contains the keyword translation map when using non-standard a geojson fault database. The third block includes a list of settings to be used as default parameters.


# -----------------------------------------------------------------------------

from openquake.mbt.tools.fault_modeler.fault_source_modeler import build_fault_model

if 1:
    param_map = {'source_id': 'ogc_fid',
                 'name': 'ns_name',
                 'average_dip': 'ns_average_dip',
                 'average_rake': 'ns_average_rake',
                 'net_slip_rate': 'ns_net_slip_rate',
                 'vert_slip_rate': 'ns_vert_slip_rate',
                 'strike_slip_rate': 'ns_strike_slip_rate',
                 'shortening_rate': 'ns_shortening_rate',
                 'dip_dir': 'ns_dip_dir',
                 'dip_slip_rate': 'ns_dip_slip_rate'}

    build_fault_model(
        geojson_file='Data/ne_asia_faults_rates_bvalue.geojson',
        xml_output='Data/test.xml',
        select_list=[1,2],
        param_map=param_map)

if 0:
    build_fault_model(cfg_file='config.ini')

## Geojson standard format

```json
{
    "type": "FeatureCollection",
    "name": "Example 1",
    "crs": {
        "type": "name",
        "properties": {
            "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
        }
    },
    "features": [
        {
            "type": "Feature",
            "properties": {
                "fid": 1,
                "catalog_id": "test_1",
                "name": "Mons Olympus Fault System",
                "is_active": 1,
                "exposure_quality": 1,
                "epistemic_quality": 1,
                "accuracy": 400000,
                "slip_type": "Dextral-Normal",
                "average_dip": "(75,60,90)",
                "average_rake": "(-10, -40, 0)",
                "dip_dir": "E",
                "downthrown_side_dir": "E",
                "net_slip_rate": "(6, 3, 10)",
                "strike_slip_rate": "(5, 2, 7)",
                "vert_slip_rate": null,
                "shortening_rate": null,
                "upper_seis_depth": 0.0,
                "lower_seis_depth": 12.0,
                "last_movement": null,
                "reference": null,
                "notes": "Some consideration"
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [
                        79.7391028607983, 32.41641275873077
                    ],
                    [
                        80.0536700043854, 32.033767651233035
                    ],
                    [
                        80.30954924804952, 31.65112254373529
                    ]
                ]
            }
        }
    ]
}
```
