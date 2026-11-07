Global Hazard Map (ghm) module
##############################

The :index:`Global Hazard Map` module contains code used to produce homogenised hazard maps using results obtained using a collection of PSHA input models. For the most part this is internal code used by GEM personnel for building various versions of the global seismic hazard maps.

Creating a grid of sites for one of the models
**********************************************
Given a model, an almost equally spaced grid of points can be created using the `get_sites.py` tool. Note that this is atool added in 2022. The grids used for the maps created before the end of 2022 were obtained with an inhouse code that we abandoned in favour of the H3 library (see https://h3geo.org/).

1. To learn about the information required by `get_sites.py`, you can run the following::

    > python get_sites.py

2. For example, for the construction of grid of points covering Europe you can use::

   > python get_sites.py 'eur' /tmp/ conf.toml

Note that this requires a configuration file in the .toml format (https://toml.io/en/). An example of configuration file is provided here https://github.com/GEMScienceTools/oq-mbtk/blob/master/openquake/ghm/grid/. It requires: the name of a shapefile (or .geojson) file that provides a mapping between each country and a model in the mosaic, a buffer distance used to add sites around a model and the resolution of the grid, specified as an integer (see https://h3geo.org/docs/core-library/restable).
