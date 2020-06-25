# Site characterization for probabilistic liquefaction analysis

There are many methods to calculate the probabilities and displacements that
result from liquefaction.  In OpenQuake, we have implemented two of these, the
methods developed by the US Federal Emergency Management Agency through their
HAZUS project, and a statistical method developed by Zhu et al (2015).

These methods require different input datasets. The HAZUS methods are 
simplified from older, more comprehensive liquefaction evaluations that would be
made at a single site following in-depth geotechnical analysis; the HAZUS
methods retain their reliance upon geotechnical parameters that may be measured
or inferred at the study sites. The methods by Zhu et al (2015) were developed
to only use data that can be derived from a digital elevation model (DEM), but
in practice, the datasets must be chosen carefully for the statistical relations
to hold. Furthermore, Zhu's methods do not predict displacements from
liquefaction, so the HAZUS site characterizations must be used for displacement
calculations regardless of the methods used to calculate the probabilities of
liquefaction.

## General considerations

### Spatial resolution and accuracy of data and site characterization

Much like traditional seismic hazard analysis, liquefaction analysis may range
from low-resolution analysis over broad regions to very high resolution analysis
of smaller areas. With advances in computing power, it is possible to run
calculations for tens or hundreds of thousands of earthquakes at tens or
hundreds of thousands of sites in a short amount of time on a personal computer,
giving us the ability to work at a high resolution over a broad area, and
considering a very comprehensive suite of earthquake sources. In principle,
the methods should be reasonably scale-independent but in practice this isn't
always the case.

Two of the major issues that can arise are the limited spatial resolutions of
key datasets and the spatial misalignments of different datasets.

Some datasets, particularly those derived from digital elevation models, must be
of a specific resolution or source to be used accurately in these calculations.
For example, if Vs30 is calculated from slope following methods developed by
Wald and Allen (2007), the slope should be calculated from a DEM with a
resolution of around 1 km. Higher resolution DEMs tend to have higher slopes at
a given point because the slope is averaged over smaller areas. The
mathematical correspondance between slope and Vs30 was developed for DEMs of
about 1 km resolution, so if modern DEMs with resolutions of 90 m or less are
used, the resulting Vs30 values will be too high.

In and of itself, this is not necessarily a problem.  The issues can arise when
the average spacing of the sites is much lower than the resolution of the data,
or the characteristics of the sites vary over spatial distances much less than
the data, so that important variability between sites is lost.

The misalignment of datasets is another issue. Datasets derived from geologic
mapping or other vector-type geospatial data may be made at spatial resolutions
much higher or lower than those derived from digital elevation data or other
raster geospatial datasets (particularly for 1 km raster data as discussed
above). This can cause a situation where irregular geographic or geologic
features such as rivers may be in different locations in two datasets, which can



## HAZUS

### Liquefaction probabilities

The HAZUS methods require several variables to characterize the ground shaking
and the site response:
- Earthquake magnitude
- Peak Ground Acceleration (PGA)
- Liquefaction susceptibility category
- Groundwater depth

The magnitude of the earthquake and the resulting PGA may be calculated by
OpenQuake during a scenario or event-based PSHA, or alternatively from ShakeMap
data or similar for real earthquakes, or through other methods. The earthquake
magnitude should be given as the moment magnitude or work magnitude (*M* or
*M<sub>W</sub>*). PGA should be provided for each site in units of *g* (i.e.,
9.81 m/s<sup>2</sup>).


#### Liquefaction suscepibility category

The HAZUS methods require that each site be assigned into a liquefaction
susceptibility category. These categories are ordinal variables ranging from 'no
susceptibility' to 'very high susceptibility'. The categorization is based on
geologic and geotechnical characteristics of the site, including the age, grain
size and strength of the deposits or rock units.

For a regional probabilistic liquefaction analysis, the categorization will be
based on a geologic map focusing on Quaternary geologic units. The analyst will
typically associate each geologic unit with a liquefaction susceptibility class,
based on the description or characteristics of the unit. (Please note that there
will typically be far fewer geologic units than individual unit polygons or
contiguous regions on a geologic map; the associations described here should
generally work for each unit rather than each polygon.)

Please see the [HAZUS manual][hzm], Section 4-21, for more information on
associating geologic units with susceptibility classes. The descriptions of the
susceptibility classes may not align perfectly with the descriptions of the
geologic units, and therefore the association may have some uncertainty.
Consulting a local or regional geotechnical engineer or geologist may be
helpful. Furthermore, may be prudent to run analyses multiple times, changing
the associations to quantify the effects on the final results, and perhaps
creating a final weighted average of the results.

Once each geologic map unit has been associated with a liquefaction
susceptibility class, each site must be associated with a geologic unit. This is
most readily done through a spatial join operation in a GIS program.

#### Groundwater depth

The groundwater depth parameter is the mean depth from the surface of the soil
to the water table, in meters. Estimation of this parameter from remote sensing
data is quite challenging. It may range from less than a meter near major water
bodies in humid regions to tens of meters in dry, rugged areas. Furthermore,
this value may fluctuate with recent or seasonal rainfall. Sensitivity testing
of this parameter throughout reasonable ranges of uncertainty for each site is recommended.

### Lateral displacements

## Zhu et al. 2015

### Liquefaction probabilities



hzm: https://www.hsdl.org/?view&did=12760