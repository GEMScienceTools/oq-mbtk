Model ANalysis (man) module
###########################

The :index:`Model Analysis` module contains a number of tools for analyzing various characteristics of hazard input models. Below we provide a description of the main functionalities available. We start with a brief description of the structure of a Probabilistic Seismic Hazard Analysis (PSHA) Input Model for the OpenQuake Engine. 

The structure of a PSHA input model for the OpenQuake engine
************************************************************

A PSHA Input Model contains two main components: The seismic source characterization and the ground-motion characterization.

The Seismic Source Characterization
===================================

The :index:`Seismic Source Characterisation` (SSC)  contains the information necessary to describe the location of the earthquake sources, their geometries, the process with which they generate earthquakes and the associated (epistemic) uncertainties.

In its simplest form, the Seismic Source Characterisation contains a Seismic Source Model (i.e. a list of earthquake sources) and the Seismic Source Logic Tree with one Branch Set containing one Branch.

The Ground-Motion Characterization
==================================

The :index:`Ground-Motion Characterisation` contains the information necessary to describe the models used to compute shaking at the investigated sites for all ruptures admitted by the SSC and the associated epistemic uncertainties.
