Model ANalysis (man) module
###########################

The :index:`Model Analysis` module contains a suite of capabilities for examining an OpenQuake format PSHA input model, and also
for exploring several of the most commonly used outputs of such models, such as disaggregation results, or hazard curves.

The capabilities here can be divided into 3 key groups.

The first of set of capabilitiesis found within `checking_utils`, which provides tools for the examination of specific components of a
PSHA input model, like checking the faults within a model, or examining the MFDs of the  sources within a model.

The second set of capabilities is found within `single_source_utils`, which provides utility functions for specific OpenQuake source typologies.

The third set of capabilities is found within `tools`, which contains general utility functions, such as for the plotting of disaggregation
results, or the extracting of hazard curves from an OQ datastore.
