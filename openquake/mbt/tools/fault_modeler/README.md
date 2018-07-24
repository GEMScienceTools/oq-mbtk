# Fault Model Building Tools

This module contains a set of functionalities to create a a Fault source model from a active fault database.

## Getting Started

The fault model builder can be run as a Python module or from the command line.

### In Python

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

### Using the Console

The above example can also be run a shell command as:

```console
fault_source_modeler.py -geo FaultDatabase.geojson -xml FaultModel.xml
```
or identically:

```console
fault_source_modeler.py -cfg config.ini
```

## Using non-standard Geojson file format

In case a geojson database with *non-standard* keyword (e.g. an old version),
it is possible to specify the translation map with a simple dictionary:

```
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

build_fault_model(geojson_file='FaultDatabase.geojson',
                  xml_output='FaultModel.xml')
                  param_map=PM)
```

## Configuration File Format

The configuration file (.ini) replaces the need of manually passing the arguments to the main function.
The file has the following standard format:

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

The first block is referred to the input/output. The second block contains the keyword translation map when using non-standard a geojson fault database. The third block includes a list of settings to be used as default parameters.

The use of any of the three blocks is optional, as well as any of the directives. The combined used of the configuration file and the direct passing of arguments is allowed. Note, however, that in case an option is specified twice, the directive in the configuration file will have priority.

## Using non-standard Geojson file format

In case a geojson database with *non-standard* keyword (e.g. an old version),
it is possible to specify the translation map with a simple dictionary:

```
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

build_fault_model(geojson_file='FaultDatabase.geojson',
                  xml_output='FaultModel.xml')
                  param_map=PM)
```

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
