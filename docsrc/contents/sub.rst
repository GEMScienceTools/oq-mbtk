SUBduction (sub) module
#######################

The :index:`Subduction` module contains software for the construction of subduction earthquake sources for the *oq-engine*. The components of this model can be used either independently or within a workflow similarly to what is described in this section.

Defining the geometry of the top of the slab
********************************************

The modeling of earthquake subduction sources starts with the definition of the geometry of the slab. The mbtk subduction module contains tools for the definition of the top of the slab. Two are the approaches available. The first one, the most comprehensive, requires a tedious process of digititazion of the profiles describing the position of the top of the slab versus depth along each cross section (see `Pagani et al. (2020) <https://sp.lyellcollection.org/content/early/2020/03/02/SP501-2019-120.abstract>`__ for a description of the methodology). The second one uses the geometries of the slab proposed by `Hayes et al. (2018) <https://science.sciencemag.org/content/362/6410/58>`__ (`dataset <https://www.sciencebase.gov/catalog/item/5aa1b00ee4b0b1c392e86467>`__).

The result of these two procedures is a folder containing a set of .csv files each one describing a profile. In this context a profile is a curve that lays on top of the slab and, generally, has a direction parallel to the dip.

.. _first approach:
   
First approach
==============

Herein we provide a brief description of the various steps. Note that we use the symbol ``>`` as the prompt in a terminal, hence every time you find some code starting with this symbol this indicate a command you must type in your terminal. 

1. The first step entails the definition of a configuration file. An example is provided herein

.. code-block:: ini

    [data]

    # Path to the text file with the coordinates of the trench axis
    trench_axis_filename = /Users/kjohnson/GEM/Regions/paisl18/data/subduction/trenches/kerton_trench.xy

    # Path to the pickled file (an instance of the hazard modeller's toolkit Catalogue) 
    catalogue_pickle_filename = /Users/kjohnson/GEM/Regions/paisl18/data/catalogues/locations/PI_cat_filt.p

    # Path to the Slab 1.0 text file with the coordinates of the top of the slab
    slab1pt0_filename = /Users/kjohnson/GEM/Regions/paisl18/data/subduction/slab1pt0/ker_slab1.0_clip.xyz

    # Path to the Crust 1.0 text file (see)
    crust1pt0_filename = /Users/kjohnson/GEM/Regions/paisl18/data/crustal_models/crust1pt0/crsthk.xyz

    # Path to the Litho 1.0 text file (see)
    litho_filename = /Users/kjohnson/GEM/Regions/paisl18/data/crustal_models/litho1pt0/litho_moho.xyz

    # Path to the file containing the focal mechanisms from the Global Centroid Moment Tensor project
    gcmt_filename = /Users/kjohnson/GEM/Regions/paisl18/data/catalogues/focal_mechanisms/GCMT_20151231.ndk

    # Path to the file with volcanoes
    volc_filename = /Users/kjohnson/GEM/Regions/paisl18/data/volcanoes/volcano_list.xy

    # Path to the text topography file 
    topo_filename = /Users/kjohnson/GEM/Regions/paisl18/data/topography/GEBCO_2014/pacisl_topobath_nf.xyz

    [section]

    # Length of each profile [km]
    lenght = 700

    # Spacing [km] between the profiles along the axis subduction trench 
    # specified in the ariable `trench_axis_filename`
    interdistance = 100 

    # Azimuth parameter. When equal to a real number in the range [0, 360] all 
    # the profiles will follow that direction. Ortherwise, if `None` the 
    # profiles will have a direction perpendicular to the trench axis
    azimuth = None

    # Maximum depth of each profile [km]
    dep_max = 700


2. Create a pickled version of your hmtk formatted catalog (NB: the catalogue must be in a hmtk format)::

    > pickle_catalogue.py ./catalogues/cac.cat`

3. Create a set of cross-sections from the subduction trench axis::

    > create_multiple_cross_sections.py ./ini/central_america.ini

Check the traces of the cross-sections in the map created. It's possible to edit the traces or add new traces in the file ``cs_traces.cs``

4. Check the new set of traces in a map with the command::

    > plot_multiple_cross_sections_map.py ./ini/central_america.ini cs_traces.cs

5. Create one .pdf file for each cross-section with the available information: e.g., earthquake hypocentres, focal mechanism, slab 1.0 geometry, CRUST 1.0 Moho::

    > plot_multiple_cross_sections.py cs_traces.cs

This command will produce as many ``.pdf`` files as the number of cross-sections specified in the ``.cs`` file

6. Digitize the contact between the overriding plate and the subducted plate in each cross-section. The information in the command below corresponds to the longitude and the latitude of the origin of the cross-section, the length [km], the azimuth [decimal degrees], the cross-section ID and the name of the ``.ini`` file. For example::

    plot_cross_section.py -106.479700 21.250800 600.000000 89.098531 0 ./ini/central_america.ini

Once launched, by clicking on the image it is possible to digitize a sequence of points. Once completed the digitization, the points can be saved to a file whose name corresponds to ``cs_<section ID>.csv`` by pressing the ``f`` key on the keyboard. The points can be deleted with the key ``d``.

.. _second approach:

Second approach
===============

The second approach proposed is simpler than the first one, but it uses directly subduction contours from Slab2.0 (Hayes et al). 

1. Set-up configuration files. This approach requires an input toml file describing the locations of the Slab2.0 data files and some other input parameters as below

.. code-block:: toml
	# Locations for strike and depth information from Slab2pt0
	fname_str ='./slab2pt0/Slab2Distribute_Mar2018/str_grd/kur_slab2_str_02.24.18.grd'
	fname_dep ='./slab2pt0/Slab2Distribute_Mar2018/dep_grd/kur_slab2_dep_02.24.18.grd'
	# Spacing to use for the profiles (in km)
	spacing = 100.0
	# Folder to save profiles to
	folder_out = './cs'
	# Optional: filename to save a figure of the profiles
	fname_fig = 'kur.png'

	
2. Create a pickled version of your hmtk formatted catalog and make sure this is located as in your toml file::

    > pickle_catalogue.py ./catalogues/cac.cat`

3. Make subduction profiles. In this approach we have two choices for creating profiles across the subduction zone. The first is an automatic procedure that defines cross-sections perpendicular to the average strike of the subduction zone with the spacing specified in the toml::

    > sub_get_profiles_from_slab2pt0 <toml_fname>

Alternatively, we can manually define a set of profiles from a geojson file. This method is necessary in areas where a subduction zone is curved, because parallel profiles defined by the first approach will cross the slab at strange angles and the resulting 3D geometry will be misshapen. In this case, the geojson file should be specified in the toml with `fname_geojson` and the function to create the profiles is::

	>sub get_profiles_from_slab2pt0 <toml_fname>

As with the first approach, you can plot maps and cross_sections with the same commands.
4. Check the new set of traces in a map with the command::

    > plot_multiple_cross_sections_map.py ./ini/central_america.ini cs_traces.cs

5. Create one .pdf file for each cross-section with the available information: e.g., earthquake hypocentres, focal mechanism, slab 1.0 geometry, CRUST 1.0 Moho::

    > plot_multiple_cross_sections.py cs_traces.cs

This command will produce as many ``.pdf`` files as the number of cross-sections specified in the ``.cs`` file

Building the top of the slab geometry
*************************************

Now that we have a set of profiles available, we will build the surface of subduction . The output of this procedure will be a new set of profiles and edges that can be used to define the surface of a complex fault modelling the subduction interface earthquakes and to create inslab sources.

This part of the procedure can be completed by running the 

1. Build the surface of the subduction interface using ``create_2pt5_model.py``. The input information in this case is:

    - The name of the folder ``<cs_folder>`` containing the ``cs_`` files created using either the procedure described in the `first approach`_ or `first approach`_ section;
    - The output profile folder ``<profile_folder>``;
    - The maximum sampling distance along a trace [km];


Example::

    > create_2pt5_model.py <cs_folder> <profile_folder> <sampl_distance>

The output is a set of interpolated profiles and edges that can be used to create a complex fault source for the OpenQuake engine.  The results of the code ``create_2pt5_model.py`` can be plotted using ``plot_2pt5_model.py``. Example::

    > plot_2pt5_model.py <output_folder> <configuration_file>

where ``<configuration_file>`` is the configuration file used to build the cross-sections.

2. You can construct surfaces for both the interface and slab components using ``build_complex_surface``, which takes the depth limits of each component. The profiles for the interface and slab components are then stored in two seperate files as specified in the function call::

	> sub build_complex_surface <profile_folder> <sampl_distance> <sfc_in> 0 50
	> sub build_complex_surface <profile_folder> <sampl_distance> <sfc_sl> 50 450

3. You can plot the 3D geometry created here with::

	> sub plot_geometries {ini_fname} False False
	
where the first flag controls whether to plot the catalogue (as specified in the ini file below) and the second specifies whether or not to plot the classification (see next step). 
	


Classifying an earthquake catalog using the top of the slab surface
********************************************************************************

The ``create_2pt5_model.py`` code produces a set of profiles and edges (i.e. .csv files with the 3D coordinates) describing the geometry of the top of the slab. With this information we can separate the seismicity in an earthquake catalog into a few subsets, each one representing a specific tectonic environment (e.g. `Abrahamson and Shedlock, 1997 <https://pubs.geoscienceworld.org/ssa/srl/article/68/1/9/142158/overview>`__ or `Chen et al., 2017 <https://academic.oup.com/gji/article/213/2/1263/4794950?login=true>`__ ). The procedure required to complete this task includes the following steps.

1. Create a configuration file that describes the tectonic environments

The configuration file specifies the geometry of surfaces, along with buffer regions, that are used as references for each tectonic environment, and the catalogue to be classified. Additionally, the configuration includes a ``priority list`` that indicates how hypocenters that can occur in overlapping buffer regions should be labeled. An example configuration file is shown below. The format of the configuration is as follows.
  
The ``[general]`` section, which includes:
    - the directory ``distance_folder`` where the Euclidean distance between each hypocenter and surface will be stored (NB: with the first method this folder must be manually created by the user, but using the second approach it will be automatically created when making the profiles)
    - an .hdf5 file ``treg_filename`` that will store the results of the classfication
    - the .pkl file ``catalogue_filename``, which is the pickeled catalogue in HMTK format to be classified. 
    - an array ``priority`` lists the tectonic regions, sorting the labels in the order of increasing priority, and a later label overrides classification of a hypocenter to a previous label. For example, in the configuration file shown below, an earthquake that could be classified as both ``crustal`` and ``int_prt`` will be labeled as ``int_prt``.

A geometry section for each labelled tectonic environment in the ``priority`` list in ``[general]``. The labels should each contain one of the following four strings, which indicate the way that the surface will be used for classification. 


    - ``int`` or ``slab``: These strings indicate a surface related to subduction or similar. They require at least four configurations: (1) ``label``, which will be used by ``treg_filename`` to indicate which earthquakes correspond to the given tectonic environment; (2) ``folder``, which gives the relative path to the directory (see Step 2) with the geometry .csv files created by ``create_2pt5_model`` for the given surface; and (3) ``distance_buffer_above`` and (4) ``distance_buffer_below``, which are the upper limits of Euclidean distances used to classify hypocenters above or below the surface to the respective tectonic environment. A user can additionally specify ``lower depth`` to bound the surface and buffer region, and ``low_year``, ``upp_year``,  ``low_mag``, and ``upp_mag`` to to select only from a given time period or magnitude range. These latter options are useful when hypocenters from a given bracket are known to include major assumptions, such as when historical earthquake are assigned a depth of 0 km. 
    - ``crustal`` or ``volcanic``: These strings indicate a surface against which the classification compares the relative position of a hypocenter laterally and vertically, for example to isolate crustal or volcanic earthquakes. They require two configurations: (1) ``crust_filename``, which is a tab-delimited .xyz file listing longitude, latitude, and depth (as a negative value), which indicates the lateral extent of the tectonic environment and the depths above which all earthquakes should be classified to the respective tectonic environment; and (2) ``distance_delta``, which specifies the vertical depth below a surface to be used as a buffer region.

 
.. code-block:: ini

    [general]
    root_folder = /home/kbayliss/projects/geese/classification/kur
    
    distance_folder = ./model/catalogue/classification/distances/
    treg_filename = ./model/catalogue/classification/classified.hdf5
    catalogue_filename = ./model/catalogue/csv/catalogue.pkl
    
    priority=[slab, crustal, int]
    
    
    [crustal]
    
    label = crustal
    distance_delta = 20.
    crust_filename = ./model/litho1pt0/litho_crust3bottom.xyz
    
    
    [int]
    
    label = int
    folder = ./sfc_in
    lower_depth = 60.
    distance_buffer_above = 10.
    distance_buffer_below = 10.
    
    [slab]
    
    label = slab
    folder = ./sfc_sl
    distance_buffer_above = 30.
    distance_buffer_below = 30.
    


2. Run the classification 

The classification algorithm is run using the following command::

    > cat_classify.py <configuration_file> <distance_flag> 

Where:
    - ``configuration_file`` is the name of the .ini configuration file 
    - ``distance_flag`` is a flag indicating whether or not the distances to surfaces must be computed (i.e. *True* is used the first time a classification is run for a set of surfaces and tectonic environments, but *False* when only the buffer and delta distances are changed)

3. Check event classifications with::

	> sub plot_geometries {ini_fname} False True

Which will plot the 3D geometry with the events coloured by their classification.   
It may be necessary to manually classify some events (e.g. where the literature supports an event being in the slab despite it being very shallow etc.). Events can be manually re-classified using::

	> ccl change_class cat.pkl classified.hdf5 eqlist.csv
	
where eqlist is a csv with eventIDs to be reassigned and the required classifications. 

4. Separate the classified events into subcatalogues

The user must decide the exact way in which they would like to separate the classified events into subcatalogues for each tectonic environment. For example, one may want to decluster the entire catalogue before separating the events, or to decluster each tectonic environment separately. To create subcatalogues based on the classifications, the function `create_sub_catalogues` can be used::
    
    > ccl create_sub_catalogues cat.pkl classified.hdf5 -p subcats_folder
 
 which will create subcatalogues with each label found in `classified.hdf5` which are in turn those supplied in the classification ini. These catalogues will be stored in the specified subcats_folder. 
Further, these catalogues can be declustered using the `decluster_multiple_TR` function. This function takes its own toml file (decluster.toml) that specifies the different subcatalogues that should be declustered together and the declustering algorithms from within the OQ engine to use, along with the necessary parameters. For example:

.. code-block:: toml
	
	[main]

	catalogue = 'cat.pkl'
	tr_file = 'classified_up.hdf5'
	output = 'subcats_folder/'
        
	create_subcatalogues = 'true'
	save_aftershocks = 'true'
	catalogue_add_defaults = 'true'

	[method1]
	name = 'GardnerKnopoffType1'
	params = {'time_distance_window' = 'UhrhammerWindow', 'fs_time_prop' = 0.1}
	label = 'UH'

	[method2]
	name = 'GardnerKnopoffType1'
	params = {'time_distance_window' = 'GardnerKnopoffWindow', 'fs_time_prop' = 0.1}
	label = 'GK'

	[case1]
	regions = ['int', 'crustal']
	label = 'int_cru'

	[case2]
	regions = ['slab']
	label = 'slab'

This toml file specifies that the window declustering with parameters of Gardner and Knopoff and Uhrhammer should be used to decluster the interface and crustal events jointly, and the slab separately. This will result in four output declustered catalogues (one for each case and each method) stored in the subcats_folder. To run this declustering, we can simply use::
 
    > ccl decluster_multiple_TR decluster.toml


Creating inslab sources for the OpenQuake Engine
*************************************************************

The construction of subduction inslab sources involves the creation of `virtual faults` elongated along the stike of the slab surface and constrained within the slab volume. This requires a configuration file defining some parameters for generating the ruptures.
    
1. Create a configuration file

.. code-block:: slab.ini

    [main]

    reference_folder = /Users/kjohnson/GEM/Regions/paisl18u/

    profile_sd_topsl = 40.
    edge_sd_topsl = 40.
    
    # MFD
    agr = 5.945
    bgr = 1.057
    mmin = 6.5
    mmax = 7.80

    sampling = 10.

    float_strike = -0.5
    float_dip = -1.0

    slab_thickness = 70.
    hspa = 20.
    vspa = 20.

    #profile_folder contains: resampled profiles and edges for the slab
    profile_folder = ./sfc_sl

    # the pickled catalogue has the hmtk format
    catalogue_pickle_fname = ./cat.pkl

    # the file with labels identifying earthquakes belonging to a given class
    treg_fname = ./classified.hdf5
    label = slab

    # output folder
    out_hdf5_fname = ./tmp/ruptures/ruptures_inslab_kerton_1.hdf5

    # output smoothing folder
    out_hdf5_smoothing_fname = ./tmp/smoothing/smoothing_kerton_1.hdf5

    # this is a lists
    dips = [45, 135]

    # this is a dictionary
    aspect_ratios = {2.0: 0.4, 3.0: 0.3, 6.0: 0.2, 8.0: 0.1}

    # this is a dictionary
    uniform_fraction = 1.0

    # magnitude scaling relationship 
    mag_scaling_relation = StrasserIntraslab

 The MFD parameters should be set by the modeller and determined from some combination of the seismicity and tectonics. The `mmin` parameter defines the lower magnitude limit at which to generate slab ruptures. A lower `mmin` will result in many more smaller ruptures which will increase the size of the rupture object, so this parameter should be chosen carefully considering the size of ruptures at slab locations that might be relevant for hazard.  
 
The `sampling` parameter determines the spatial sampling to be used when simulating ruptures. The `float_strike` and `float_dip` parameters specify the strike and dip for floating ruptures, while the list of `dips` instead specifies dip angles used when creating virtual faults inside the rupture.

The `uniform_fraction` determines the percentage of the ruptures to be uniformly distributed across the slab. A higher uniform fraction means that the distribution of ruptures will be randomly uniform, and a lower uniform fraction means that a larger percentage of ruptures will instead be distributed according to the smoothed distribution of seismicity in the slab.  

Ruptures can be created using::

	> calculate_ruptures ini_fname out_hdf5_fname out_hdf5_smoothing_fname

which will create two hdf5 files with the different ruptures. We then need to process these into xml files, which we can do using::

	> create_inslab_nrml(model, out_hdf5_fname, out_path, investigation_t = 1)

You can plot the 3D model of the subduction zone with ruptures as below:

.. code-block:plot_ruptures
xml_fname = "/home/kbayliss/projects/force/code/Pacific_Jan24/mariana_April24/xml/mar_newrup_unif/6.55.xml"
ssm = to_python(xml_fname)

hy_lo, hy_la, hy_d = [],[],[]
mags = []
rates = []
for sg in ssm:
    for src in sg:
        for rup in src.data:
            h = rup[0].hypocenter
            hy_lo.append(h.longitude); hy_la.append(h.latitude); hy_d.append(-h.depth)
            mags.append(rup[0].mag)
            prob1 = rup[1].data[1][0]
            rate = -1*np.log(1-prob1)
            rates.append(rate)

df = pd.DataFrame({'lons': hy_lo, 'lats': hy_la, 'depth': hy_d, 'mag': mags, 'rate': rates})

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.lons, df.lats, df.depth, c=df.rate)
plt.show()
	
This process will create xml files for each magnitude bin, which is a little impractical. To combine these, we can do the following::

path1 = glob.glob('./xml/mar_newrup/*.xml')
data_all = []
for p in path1: 
    ssm = to_python(p)
    for sg in ssm:
        for src in sg:
            print(len(src.data))
            data_all.extend(src.data)
src.data = data_all 
src.name = 'ruptures for src_slab'
src.source_id = 'src_slab'

write_source_model('./xml/mar_newrup/slab.xml', [src], investigation_time=1.0) 

Which creates a single `slab.xml` file containing all the ruptures across all magnitude bins. This file can then be used directly in OQ as a non-parametric rupture source. 
See the process in more detail in this notebook

