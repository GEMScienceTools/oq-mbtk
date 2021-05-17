SUBduction (sub) module
#######################

The :index:`Subduction` module contains software for the construction of subduction earthquake 
sources for the *oq-engine*. The components of this model can be used either
independently or within a workflow similarly to what is described for example
in TODO. 

Defining the geometry of the top of the slab
********************************************

The modelling of earthquake subduction sources starts with the definition of 
the geometry of the slab. 

1. The first step entails the definition of a configuration file. An example
is provided herein:

.. code-block:: language ini

    AA

2. Create a pickled version of your hmtk formatted catalogue
..  code-block::
    pickle_catalogue.py ./catalogues/cac.cat`

3. Create a set of cross-sections from the subduction trench axis

::
    create_multiple_cross_sections.py ./ini/central_america.ini

Check the traces of the cross-sections in the
map created. It's possible to edit the traces or add new traces in the file `cs_traces.cs`

4. Check the new set of traces in a map with the command

::
    plot_multiple_cross_sections_map.py ./ini/central_america.ini cs_traces.cs

5. Create .pdf for all the cross-sections with the available information: earthquake hypocenters, focal mechanims, slab 1.0 geometry, CRUST 1.0 moho 

:: language python
    plot_multiple_cross_sections.py cs_traces.cs

This command will create as many .pdf files as the number of cross-sections specified in the `.cs` file

6. Digitize the contact between the overriding plate and the subducted plate on a cross-section
The information included in the command is the same one in one line of the `.cs` file. This information corresponds to the longitude and the latitude of the origin of the cross-section, the length [km], the azimuth [decimal degrees], the cross-section ID and the name of the `.ini` file. For example:

:: language python
    `plot_cross_section.py -106.479700 21.250800 600.000000 89.098531 0 ./ini/central_america.ini`

Once launched, by clicking on the image it is possble to digitize a sequence of points. Once completed the digitization, the points can be saved to a file whose name corresponds to `cs_<section ID>.csv` by pressing the `f` key on the keyboard. The points can be deleted with the key `d`.

7. Build the surface of the subduction interface using
`create_2pt5_model.py`. The input information in this case is:
    - The name of the folder containing the `cs_` files cerated by the `plot_cross_section.py` command.
    - The maximum sampling distance along a trace [km]
    - The output folder name
example: 

..  code-block:: language python

    create_2pt5_model.py sp_central_america 10. sp_central_america_int/

The output is a set of interpolated profiles and edges that can be used to create a complex fault source for the openquake engine.

8. The results of the code `create_2pt5_model.py` can be plotted using `plot_2pt5_model.py`. Example: 

..  code-block:: language python
    plot_2pt5_model.py sp_central_america_int/ ./ini/central_america.ini



    
