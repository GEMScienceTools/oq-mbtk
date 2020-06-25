.. mbt documentation master file, created by
   sphinx-quickstart on Thu Jan 24 16:06:36 2019.
      You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the OpenQuake Model Building Toolkit's documentation!
################################################################

The OpenQuake Model Building Toolkit (*oq-mbt*) is a suite of tools for the 
construction of components of a Probabilistic Seismic Hazard (PSH) model. 
The main contributors to this suite of tools are GEM Hazard Team members. 
Contribution from extena users are very welcome!

*oq-mbt* code is hosted on github at the following link 
https://github.com/GEMScienceTools/oq-mbtk. It is developed in close 
connection with the 
`OpenQuake engine <https://github.com/gem/oq-engine>`_, the 
open-source hazard and risk calculation engine developed primarily by the 
GEM Foundation.

The *oq-mbt* relies on several functionalities included in the Hazard Modeller's
Toolkit library (*oq-hmtk*). The oq-hmtk code is accessible on github at the
following link https://github.com/gem/oq-engine/tree/master/openquake/hmtk,
while documentation for the oq-hmtk can be downloaded  
at https://github.com/GEMScienceTools/hmtk_docs/blob/master/hmtk_tutorial.pdf.

Currently the oq-mbt includes five sub-modules:

* *CATalogue Toolkit (cat)* contains code used for creating a homogenised 
  catalogue;
* *Global Hazard Map (ghm)* contains code used to produce homogenised hazard 
  maps using results obtained using a collection of PSHA input models;
* *Model ANalysis (man)* contains code for analysing oq-engine formattted PSHA 
  input models; 
* *Model Building tool (mbt)* contains code for seismic source 
  characterisation;
* *SUBduction modelling (sub)* contains code for building subduction 
  earthquake sources. 
* *SEcondary Perils (sep)* contains code for calculating secondary earthquake
perils such as liquefaction and coseismic landslides.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   contents/installation
   contents/cat
   contents/man
   contents/mbt
   contents/sub
   contents/sep


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
