#!/usr/bin/env python
# coding: utf-8

# # Basic Fault Network Modeling with Fermi
# 
# Fermi is a tool to create fault sources for PSHA that are capable of 
# subfault to multifault ruptures, treating the faults as a network.
# 
# This is an entry-level tutorial. 

# # Overview of the Fault Network Modeling process
# 
# The modeling process looks like this:
# 
# 1. Creation of subfaults
# 
#     In this step, the fault traces are expanded down-dip to 3D surfaces, and then 
# 
# 1. Creation of single-fault ruptures
# 
#     1. Joining subfaults
# 
# 1. Creation of multifault ruptures
# 
#     1. Graph Search
# 
# 1. (Optional) Plausibility filtering of multifault ruptures
# 
# 1. Rupture rate inversion
# 
#    1. Choosing constraints
# 
#    1. making system of equations
# 
#    1. Solving
# 
# 1. Writing fault sources

# In[1]:


# Start the logging with level INFO to provide
# good information about the progress and results

import logging
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%d-%b-%y %H:%M:%S',
    level=logging.INFO,
)


# Fermi is the `openquake.fnm` module. The basic entry point is the `openquake.fnm.all_together_now` submodule,
# which provides the functions `build_fault_network` and `build_system_of_equations`. 
# 
# However, we are also going to import a number of other functions to allow us
# to use more constraints in solving for the rupture rates, such as magnitude-frequency constraints for all of the faults involved as well as for the region.

# In[98]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from openquake.hazardlib.mfd import TruncatedGRMFD

from openquake.fnm.exporter import (
    make_multifault_source,
    write_multifault_source,
)
from openquake.fnm.all_together_now import (
    build_fault_network,
    build_system_of_equations,
)
from openquake.fnm.inversion.utils import (
    rup_df_to_rupture_dicts,
    get_fault_moment_rate,
    make_fault_mfd,
)
from openquake.fnm.inversion.plots import (
    plot_soln_slip_rates,
    plot_soln_mfd,
    plot_mfd,
)
from openquake.fnm.inversion.soe_builder import (
    make_fault_rel_mfd_equation_components,
)

from openquake.fnm.inversion.solver import solve_nnls_pg


# Fault network modeling is complex; there are a number of parameters that control many of the aspects of the process. 

# In[16]:


settings = {
    # fraction of slip rate released in mainshocks
    "seismic_fraction": 1.0, 
    # Size of subfault length/width in km
    "subsection_size": 10.0, 
    "max_jump_distance": 10.0, # maximum 3d distance between faults that can co-rupture
    "parallel_subfault_build": True, #set to False if you run in IPython interactively
    'full_fault_only_mf_ruptures': True, # Set to False for site-specific analysis with just a few faults
    "filter_by_plausibility": True, #set to True if you want to remove some geometrically implausible ruptures
    "fault_mfd_b_value": 1.0, # change as you like
    "export_fault_mfds": True, # create MFD objects for faults during processing
}


# In[44]:


fault_geojson = "../data/atf_qilian.geojson"


# In[45]:


gpd.read_file(fault_geojson).plot()


# In[46]:


fault_network = build_fault_network(
        fault_geojson=fault_geojson,
        settings=settings,
        )


# In[47]:


rup_key = 'rupture_df'
rup_set = fault_network[rup_key]
rups = rup_df_to_rupture_dicts(rup_set)


# In[48]:


# MFD constraints (b-value only, no a-value) for faults
fault_rel_mfds = make_fault_rel_mfd_equation_components(
    rups,
    fault_network,
    b_value=settings['fault_mfd_b_value'],
    fault_key='faults',
    rup_key=rup_key,
    )


# In[49]:


# absolute MFD for whole model
fault_moment = sum(get_fault_moment_rate(fault) 
                   for i, fault in fault_network['subfault_df'].iterrows())
model_abs_mfd = TruncatedGRMFD.from_moment(
    min_mag=rup_set.mag.min(),
    max_mag=rup_set.mag.max(),
    bin_width=0.1,
    b_val=1.0, # as you'd like
    moment_rate=fault_moment,
)



# In[87]:


lhs, rhs, err = build_system_of_equations(
    rup_set,
    fault_network["subfault_df"],
    fault_rel_mfds=fault_rel_mfds,
    mfd_rel_weight=1e2,
    mfd_rel_mode='shape',
    #mfd_rel_eqns=True, # if you want a regional (relative, no a-value) MFD
    mfd_rel_b_val=1.0, # adjust as you'd like,
    mfd=model_abs_mfd, # set to None if you don't want to use
)


# In[93]:


logging.info("Solving")
rup_rates, misfit_history = solve_nnls_pg(
    lhs,
    rhs,
    weights=err,
    max_iters=int(2e5),
    accept_grad=-1, # continue to completion of iterations
    accept_norm=-1, # continue to completion of iterations
    stall_val=1e-6, # continue until solutions is pretty stable
)
rup_rates = pd.Series(rup_rates, index=rup_set.index)
logging.info("Done solving")


# In[94]:


rup_set["occurrence_rate"] = rup_rates


# In[95]:


plt.figure()
plot_soln_slip_rates(
    rup_rates,
    fault_network['subfault_df'].net_slip_rate,
    lhs,
    errs=fault_network["subfault_df"].net_slip_rate_err,
    units="mm/yr",
)
plt.title("Observed and modeled slip rates")


# In[96]:


plt.figure()
plot_mfd(model_abs_mfd, label="Model MFD")
plot_soln_mfd(rup_rates.values, rups, label="Solution MFD")
plt.legend(loc="lower left")
plt.title("Solution MFD")


# In[97]:


plt.figure()
plt.semilogy(misfit_history)
plt.title("Solution misfit history")


# In[ ]:




