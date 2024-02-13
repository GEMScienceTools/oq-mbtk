#!/usr/bin/env python
# coding: utf-8
# ------------------- The Model Building Toolkit ------------------------------
#
# Copyright (C) 2022 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

import toml
import shutil
import os
from openquake.wkf.utils import get_list


HERE = os.path.dirname(__file__)

def main(fname_config: str, copy_loc: str, use: str = []):
	""" Check config file contains all necessary parameters and make a working copy
	
	:param fname_config:
		Location of config file describing model
	:param copy_loc:
		location to store an editable copy of the config file

	"""
	model = toml.load(fname_config)

	if all(i in model for i in ('name', 'mmin', 'bin_width', 'rupture_mesh_spacing')):
	    print("Checking model ", model.get('name'))
	else: 
		print("missing default data (name, mmin, bin_width or rupture_mesh_spacing not found)")
		
	# Check for declustering parameters
	if 'declustering' in model:
		declust_params = model['declustering']
		if declust_params['method'] == 'Zaliapin':
			if all(i in declust_params for i in ('fractal_dim', 'b_value', 'threshold', 'depth', 'output_nearest_neighbor_distances')):
				print("All parameters specified for Zaliapin declustering")
			else:
				breakpoint()
				print(declust_params)
				print("Check Zaliapin declustering parameters. Should contain: 'fractal_dim', 'b_value', 'threshold', 'depth', 'output_nearest_neighbor_distances'.")

		elif declust_params['method'] == 'Reasenberg':
			if all(i in declust_params for i in ('taumin', 'taumax', 'p', 'xk', 'xmeff', 'rfact', 'horiz_error', 'depth_error', 'interaction_formula', 'max_interaction_dist')):
				print("All parameters specified for Reasenberg declustering")
			else:
				print(declust_params)
				print("Check Reasenberg parameters. Should contain: 'taumin', 'taumax', 'p', 'xk', 'xmeff', 'rfact', 'horiz_error', 'depth_error', 'interaction_formula', 'max_interaction_dist'.")

		elif declust_params['method'] == 'windowing':
			if all(i in declust_params for i in ('time_distance_window', 'fs_time_prop')):
				print("All parameters specified for windowing declustering")
			else:
				print(declust_params)
				print("Check windowing parameters. Should contain: 'time_distance_window', 'fs_time_prop'.")

		else: 
			print("unrecognised declustering algorithm. Please choose from 'Zaliapin', 'Reasenberg' or 'windowing'" )


	else:
		print("No declustering algorithm found")

	# Check smoothing parameters
	if 'smoothing' in model: 
		gauss = False
		adap = False
		smoothing_params = model['smoothing']
		if all(i in smoothing_params for i in ('n_v', 'kernel', 'd_i_min', 'maxdist', 'h3res')):
			print("Found parameters for adaptive smoothing")
			adap = True
		if all(i in smoothing_params for i in ('kernel_maximum_distance', 'kernel_smoothing')):
			print("Found parameters for Gaussian smoothing")
			gauss = True

		if adap == False and gauss == False:
			print("Smoothing paramaters missing. 'kernel_maximum_distance', 'kernel_smoothing' needed for Gaussian smoothing. 'n_v', 'kernel', 'd_i_min', 'maxdist', 'h3res' needed for adaptive smoothing. ")

	else:
		print("No smoothing parameters found")
		
	# check sources
	if len(use) > 0:
		use = get_list(use)
		for src_id in use:
		# Add sources specified by use
			if src_id not in model['sources']:
				model['sources'][src_id] = {}
		# Should remove sources not in use, but for some reason this is stupidly difficult to find out how to do!
		print("Configured ", len(use), "sources as specified by use argument")
	    
	elif 'sources' in model:
		print("Found ", len(model['sources']), " sources in model config")

	else:
		print("No sources found. Please add some!")

	# Check for msr section
	if 'msr' not in model: 
		print("No magnitude scaling relationships defined")

	print("copying toml to ", copy_loc)
	source = fname_config
	destination = copy_loc
	shutil.copy(source, destination)
	
main.fname_config = 'location of toml file for source model'
main.copy_loc = 'location to store an editable copy of the config'

if __name__ == '__main__':
    sap.run(main)
