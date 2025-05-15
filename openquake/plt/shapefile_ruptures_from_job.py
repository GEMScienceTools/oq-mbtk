#!/usr/bin/env python
# coding: utf-8

#
# Copyright (C) 2024-2025, GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

import os
import h5py
import numpy as np
import pandas as pd
import geojson as gj
from openquake.baselib import sap
from openquake.commonlib import datastore
from openquake.hazardlib.source.rupture import to_arrays


def read_rups_job(id_calc: int):
    """
    Reads rupture data from job files using the datastore.

    Args:
        id_calc (int): Calculation ID for the hazard analysis.

    Returns:
        tuple: Combined GMF data and rupture IDs for shapefile creation.
    """
    dstore = datastore.read(f'/Users/manuela/oqdata/calc_{id_calc}.hdf5')
    ruptures = dstore.read_df('ruptures')
    events = dstore.read_df('events')
    gmf_data = dstore.read_df('gmf_data')

    ruptures.rename(columns={"id": "rup_id"}, inplace=True)
    events.rename(columns={"id": "eid"}, inplace=True)
    gmf_data = gmf_data.merge(events, on='eid', how='left')
    gmf_data = gmf_data.merge(ruptures, on='rup_id', how='left')

    all_ids_for_shp = gmf_data["seed"]
    return gmf_data, all_ids_for_shp


def read_rup_dataset(file_path: str, all_ids_for_shp: np.ndarray):
    """
    Reads rupture dataset and filters it based on the given rupture IDs.

    Args:
        file_path (str): Path to the ruptures dataset.
        all_ids_for_shp (np.ndarray): Array of rupture IDs for filtering.

    Returns:
        tuple: Raw ruptures data, rupture geometry, and filtered ruptures.
    """
    with h5py.File(file_path, "r") as hdf:
        #print("Keys in the file:", list(hdf.keys()))
        rups_geom = hdf["rupgeoms"][:]
        rups_data = hdf["ruptures"][:]
        rups_model = hdf["models"][:]

    if rups_data['seed'].dtype.kind in {'S', 'U'}:
        all_ids_for_shp = all_ids_for_shp.astype(str)
        rups_data['seed'] = rups_data['seed'].astype(str)

    rups_data_mod = rups_data[np.isin(rups_data['seed'], all_ids_for_shp)]
    return rups_data, rups_geom, rups_data_mod


def get_perimeter(msh: np.ndarray) -> np.ndarray:
    """
    Gets the perimeter of a single-section rupture.

    Args:
        msh (np.ndarray): Rupture mesh.

    Returns:
        np.ndarray: Perimeter coordinates.
    """
    if msh.shape[1] == 1:
        return np.array([[msh[0][0, i], msh[1][0, i], msh[2][0, i]] for i in range(msh.shape[2])])

    perimeter = []

    # Top
    perimeter.extend(zip(msh[0][0, :], msh[1][0, :], msh[2][0, :]))

    # Right
    perimeter.extend(zip(msh[0][1:, -1], msh[1][1:, -1], msh[2][1:, -1]))

    # Bottom
    perimeter.extend(zip(msh[0][-1, ::-1], msh[1][-1, ::-1], msh[2][-1, ::-1]))

    # Left
    perimeter.extend(zip(msh[0][::-1, 0], msh[1][::-1, 0], msh[2][::-1, 0]))

    return np.array(perimeter)


def create_geojson(rups_geom: np.ndarray, rups_data_mod: np.ndarray, output_file: str):
    """
    Creates a GeoJSON file of selected ruptures.

    Args:
        rups_geom (np.ndarray): Rupture geometry data.
        rups_data_mod (np.ndarray): Filtered ruptures data.
        output_file (str): Output GeoJSON file path.
    """
    features = []

    for i, (rup_id, mag) in enumerate(zip(rups_data_mod['geom_id'], rups_data_mod['mag'])):
        rup = rups_geom[rup_id]
        arrays = to_arrays(rup)

        for arr in arrays:
            if arr.shape[2] == 4:
                per = get_perimeter(arr)
                per[[2, 3], :] = per[[3, 2], :]
            elif arr.shape[2] < 4:
                per = get_perimeter(arr)
                per[[1, 2], :] = per[[2, 1], :]
            else:
                continue

            coords = [(float(p[0]), float(p[1]), float(p[2])) for p in per]
            poly = gj.Polygon([coords])

            attributes = {
                'id': int(rup_id),
                'source_id': int(rups_data_mod['source_id'][i]),
                'n_occ': int(rups_data_mod['n_occ'][i]),
                'rake': float(rups_data_mod['rake'][i]),
                'mag': float(rups_data_mod['mag'][i]),
            }
            features.append(gj.Feature(geometry=poly, properties=attributes))

    with open(output_file, 'w') as f:
        gj.dump(gj.FeatureCollection(features), f)

    print(f"GeoJSON created with {len(features)} ruptures: {output_file}")


def main(id_calc: int, file_path_rups_dataset: str, output_path: str):
    """
    Main function to read data, filter ruptures, and create GeoJSON.

    Args:
        id_calc (int): Calculation ID.
        file_path_rups_dataset (str): Path to the ruptures dataset.
        output_path (str): Directory for the output file.
    """
    os.makedirs(output_path, exist_ok=True)
    gmf_data, all_ids_for_shp = read_rups_job(id_calc)
    rups_data, rups_geom, rups_data_mod = read_rup_dataset(file_path_rups_dataset, all_ids_for_shp)

    output_file = os.path.join(output_path, f'ruptures_{id_calc}.geojson')
    create_geojson(rups_geom, rups_data_mod, output_file)


if __name__ == '__main__':
    sap.run(main)