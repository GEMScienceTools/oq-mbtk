#!/usr/bin/env python
# coding: utf-8

import toml
import importlib
from openquake.baselib import sap
from openquake.sub.utils import _read_edges
from openquake.hazardlib.geo.surface import ComplexFaultSurface

MODULE = importlib.import_module('openquake.hazardlib.scalerel')


def main(fname_config: str, edges_folder: str, source_id: str,
         epsilon: float = 0.0, selected_msr: str = None):
    """
    Computes the mmax given the path to the folder containing the files with
    edges that describe the subduction interface surface.
    """

    # Load the config file
    config = toml.load(fname_config)

    # Check
    msg = "The config file does not contain information about this source"
    assert source_id in config["sources"], msg

    # Get the TR
    trt = config["sources"][source_id]["tectonic_region_type"]

    msg = "The config file does not contain magnitude-scaling relatioships"
    assert trt in config["msr"], msg

    edges = _read_edges(edges_folder)
    fault_surface = ComplexFaultSurface.from_fault_data(edges, 10.)
    fault_surface_area = fault_surface.get_area()

    fmt = "{:30s}: {:5.2f} {:5.3f} {:5.2f}"
    for i, msr_lab in enumerate(config["msr"][trt]):
        my_class = getattr(MODULE, msr_lab)
        msr = my_class()
        magnitude = msr.get_median_mag(fault_surface_area, 90.0)
        standard_dev = msr.get_std_dev_mag(90.0)
        mag_plus_eps = magnitude + (epsilon * standard_dev)
        print(fmt.format(msr_lab, magnitude, standard_dev, mag_plus_eps))
        tmp = "{:.3f}".format(mag_plus_eps)
        if selected_msr is None and i == 0:
            trt = config["sources"][source_id]["mmax"] = float(tmp)
        elif selected_msr == msr_lab:
            trt = config["sources"][source_id]["mmax"] = float(tmp)

    # Updating the config file
    with open(fname_config, "w") as toml_file:
        toml.dump(config, toml_file)


descr = 'The configuration file'
main.fname_config = descr
descr = 'The path to the folder containing the edges (files start with edge_)'
main.edges_folder = descr
main.source_id = 'The ID of the source'
main.epsilon = 'Number of std used to compute mmax'
descr = 'The label of the selected msr [default is the first one in the list]'
main.selected_msr = descr

if __name__ == '__main__':
    sap.run(main)
