# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation
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
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

import os
import shutil
import numpy as np
import pandas as pd
import tempfile

from matplotlib import pyplot
from matplotlib import cm
from matplotlib.patches import Patch

from openquake.baselib import sap, hdf5
from openquake.commands.export import main as export

import warnings
warnings.filterwarnings("ignore")


# base path
base = os.path.dirname(__file__)

# Colormap
cmap = cm.get_cmap('jet')


def get_info(dstore_fname, calc_id, disagg_type, site_id):
    """
    Return for given datastore the required disaggregation information
    """
    # Make an output folder
    disagg_out = os.path.join(base, f'disagg_{disagg_type}_calc_{calc_id}')
    if os.path.exists(disagg_out):
        shutil.rmtree(disagg_out)
    if not os.path.exists(disagg_out):
        os.makedirs(disagg_out)

    # Load the hdf5 of the calculation
    ds = hdf5.File(dstore_fname)

    # Get the sites
    sites = ds["sitecol"]
    if site_id is not None:
        # Get only the site of interest if specified.
        assert len(site_id) == 1
        sites = sites[sites.sids==site_id]

    # Get the imts
    ims = pd.Series(ds["oqparam"].imtls).index

    # Get the investigation time
    inv_t = ds["oqparam"].investigation_time
    
    # poes
    poes = ds["oqparam"].poes

    # Export the disagg into a tmp file
    export_info = {'exports': 'csv', 'export_dir': tempfile.mkdtemp()}
    export('disagg-stats', dstore_fname, **export_info)

    return ds, sites, ims, inv_t, poes, export_info, disagg_out


def disagg_MRE(dstore_fname, disagg_type, site_id, azimuth):
    """
    Make 3D mag-dist-epsilon disagg plots for an OQ PSHA calculation's
    mean disaggregation results.
    """
    assert disagg_type == "Mag_Dist_Eps"

    # Get calc ID number
    calc_id = int(dstore_fname.split("calc_")[1].split('.')[0])

    # Get the disagg info
    ds, sites, ims, inv_t, poes, export_info, disagg_out =\
         get_info(dstore_fname, calc_id, disagg_type, site_id)

    for idx_site, site in enumerate(sites):

        # Get a tmp file of the M-R-e disagg results
        disagg_filename = f'Mag_Dist_Eps-mean-{idx_site}_{calc_id}.csv'
        disagg_path = os.path.join(export_info['export_dir'], disagg_filename)

        # Load the tmp file
        df = pd.read_csv(disagg_path, header=1)
        poes = sorted(np.unique(df['poe']), reverse=True)

        # Get binning params + color mapping
        Mbin = float(ds["oqparam"].mag_bin_width)
        Dbin = float(ds["oqparam"].distance_bin_width)
        for imt in ims:
            mode_vals, mean_vals = [], []
            RP, apoe_norm = [], []
            all_M, all_R, all_eps = [], [], []

            for poe in poes:
                RP.append(round(-inv_t / np.log(1 - poe)))
                mask_df = (df['poe'] == poe) & (df['imt'] == imt)

                data = pd.DataFrame({
                    'mag': df.loc[mask_df, 'mag'],
                    'eps': df.loc[mask_df, 'eps'],
                    'dist': df.loc[mask_df, 'dist'],
                    'rate': -np.log(1 - df.loc[mask_df, 'mean']) / inv_t
                })

                data['rate_norm'] = data['rate'] / data['rate'].sum()
                apoe_norm.append(data['rate_norm'].values)

                # Modal (highest contribution)
                mode_row = data.sort_values(by='rate_norm', ascending=False).iloc[0]
                mode_vals.append([mode_row['mag'], mode_row['dist'], mode_row['eps']])

                # Mean values weighted by normalised rate
                mean_vals.append([
                    np.sum(data['mag'] * data['rate_norm']),
                    np.sum(data['dist'] * data['rate_norm']),
                    np.sum(data['eps'] * data['rate_norm'])
                ])

                # Store
                all_M.append(data['mag'].values)
                all_R.append(data['dist'].values)
                all_eps.append(data['eps'].values)

            # Epsilon range for normalization
            eps_all = np.concatenate(all_eps)
            unique_eps = np.unique(eps_all)
            n_RP, n_eps = len(RP), len(unique_eps)
            min_eps, max_eps = unique_eps.min(), unique_eps.max()

            for i in range(n_RP):
                if mean_vals[i][0] == 0.0: # Mag here is idx 0
                    continue  # Skip if mag is zero (no contribution)

                # Make fig
                fig = pyplot.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1, projection='3d')

                # Get colorbar for epsilon
                colors = [cmap((eps - min_eps) / (max_eps * 2)) for eps in unique_eps]

                # Stack contributions over the unique epsilons
                Z = np.zeros(len(all_R[i]) // n_eps)
                for eps_idx, eps_val in enumerate(unique_eps):
                    idx = np.arange(eps_idx, len(all_R[i]), n_eps)
                    X = all_R[i][idx] - Dbin / 4
                    Y = all_M[i][idx] - Mbin / 4
                    dx = np.full_like(X, Dbin / 2)
                    dy = np.full_like(Y, Mbin / 2)
                    dz = apoe_norm[i][idx] * 100
                    mask = dz > 0
                    if np.any(mask):
                        ax.bar3d(
                            X[mask], Y[mask], Z[mask],
                            dx[mask], dy[mask], dz[mask],
                            color=colors[eps_idx],
                            alpha=1.0)
                        
                    Z += dz
                assert np.sum(Z) == 100.0

                ax.view_init(elev=23, azim=float(azimuth))
                ax.set_xlabel('R (km)', fontsize=12)
                ax.set_ylabel('$M_{w}$', fontsize=12)
                ax.set_zlabel('Hazard Contribution (%)', fontsize=12, rotation=90)

                # Axis params
                ax.set_xlim(np.min(all_R) - Dbin / 2, np.max(all_R) + Dbin / 2)
                ax.set_ylim(np.min(all_M) - Mbin / 2, np.max(all_M) + Mbin / 2)
                ax.set_xticks(np.round(np.arange(np.min(all_R), np.max(all_R) + Dbin, Dbin), 0))
                ax.set_yticks(np.arange(np.min(all_M), np.max(all_M) + Mbin, Mbin))

                # Get a legend for the epsilon
                lg_elm = [
                    Patch(facecolor=colors[n_eps - j - 1],
                          label=f"\u03B5 = {unique_eps[n_eps - j - 1]:.2f}") for j in range(n_eps)]
                fig.legend(handles=lg_elm, loc="lower center", borderaxespad=0.20, ncol=n_eps, fontsize=12)

                # Export
                rp_str = int(RP[i] + 1)
                filename = f'{disagg_type}_mean_site_{site.id}_{imt}_PSHA_{rp_str}_year_RP.png'
                output_path = os.path.join(disagg_out, filename)
                pyplot.savefig(output_path, format='png')
                pyplot.close(fig)


def disagg_MLL(dstore_fname, disagg_type, site_id, azimuth):
    """
    Make 3D mag-lon-lat disagg plots for an OQ PSHA calculation's
    mean disaggregation results.
    """
    assert disagg_type == "Mag_Lon_Lat"

    # Get calc ID number
    calc_id = int(dstore_fname.split("calc_")[1].split('.')[0])

    # Get the disagg info
    ds, sites, ims, inv_t, poes, export_info, disagg_out =\
         get_info(dstore_fname, calc_id, disagg_type, site_id)

    for idx_site, site in enumerate(sites):

        # Get a tmp file of the M-R-e disagg results
        disagg_filename = f'Mag_Lon_Lat-mean-{idx_site}_{calc_id}.csv'
        disagg_path = os.path.join(export_info['export_dir'], disagg_filename)

        # Load the tmp file
        df = pd.read_csv(disagg_path, header=1)
        poes = sorted(np.unique(df['poe']), reverse=True)

        # Get binning params + color mapping
        Cbin = float(ds["oqparam"].coordinate_bin_width)
        for imt in ims:
            mode_vals, mean_vals = [], []
            RP, apoe_norm = [], []
            all_M, all_lon, all_lat = [], [], []

            for poe in poes:
                RP.append(round(-inv_t / np.log(1 - poe)))
                mask_df = (df['poe'] == poe) & (df['imt'] == imt)

                data = pd.DataFrame({
                    'lon': df.loc[mask_df, 'lon'],
                    'lat': df.loc[mask_df, 'lat'],
                    'mag': df.loc[mask_df, 'mag'],
                    'rate': -np.log(1 - df.loc[mask_df, 'mean']) / inv_t
                })

                data['rate_norm'] = data['rate'] / data['rate'].sum()
                apoe_norm.append(data['rate_norm'].values)

                # Modal (highest contribution)
                mode_row = data.sort_values(by='rate_norm', ascending=False).iloc[0]
                mode_vals.append([mode_row['lon'], mode_row['lat'], mode_row['mag']])

                # Mean values weighted by normalised rate
                mean_vals.append([
                    np.sum(data['lon'] * data['rate_norm']),
                    np.sum(data['lat'] * data['rate_norm']),
                    np.sum(data['mag'] * data['rate_norm'])
                ])

                # Store
                all_lon.append(data['lon'].values)
                all_lat.append(data['lat'].values)
                all_M.append(data['mag'].values)

            # Magnitude range for normalization
            mag_all = np.concatenate(all_M)
            unique_mag = np.unique(mag_all)
            n_RP, n_mag = len(RP), len(unique_mag)
            min_mag, max_mag = unique_mag.min(), unique_mag.max()

            for i in range(n_RP):
                if mean_vals[i][2] == 0.0: # Mag here is idx 2
                    continue  # Skip if mag is zero (no contribution)

                # Make figure
                fig = pyplot.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1, projection='3d')

                # Get colorbar for magnitudes
                colors = [cmap((eq_mag - min_mag) / (max_mag * 2)) for eq_mag in unique_mag]

                # Stack contributions over the unique mags
                Z = np.zeros(len(all_lon[i]) // n_mag)
                for mag_idx, mag_val in enumerate(unique_mag):
                    idx = np.arange(mag_idx, len(all_lon[i]), n_mag)
                    X = all_lon[i][idx] - Cbin / 4
                    Y = all_lat[i][idx] - Cbin / 4
                    dx = np.full_like(X, Cbin / 2)
                    dy = np.full_like(Y, Cbin / 2)
                    dz = apoe_norm[i][idx] * 100
                    mask = dz > 0
                    if np.any(mask):
                        ax.bar3d(
                            X[mask], Y[mask], Z[mask],
                            dx[mask], dy[mask], dz[mask],
                            color=colors[mag_idx],
                            alpha=1.0)
                    Z += dz
                assert np.sum(Z) == 100.0

                ax.view_init(elev=23, azim=float(azimuth))
                ax.set_xlabel('Longitude', fontsize=12)
                ax.set_ylabel('Latitude', fontsize=12)
                ax.set_zlabel('Hazard Contribution (%)', fontsize=12, rotation=90)

                # Axis params
                ax.set_xlim(np.min(all_lon) - Cbin / 2, np.max(all_lon) + Cbin / 2)
                ax.set_ylim(np.min(all_lat) - Cbin / 2, np.max(all_lat) + Cbin / 2)
                ax.set_xticks(np.round(np.arange(np.min(all_lon), np.max(all_lon) + Cbin, Cbin), 1))
                ax.set_yticks(np.round(np.arange(np.min(all_lat), np.max(all_lat) + Cbin, Cbin), 1))

                # Get a legend for the epsilon
                lg_elm = [
                    Patch(facecolor=colors[n_mag - j - 1],
                          label='$M_{w}$' + f" = {unique_mag[n_mag - j - 1]:.2f}") for j in range(n_mag)]
                fig.legend(handles=lg_elm,loc="lower center", borderaxespad=0.20, ncol=n_mag, fontsize=12)

                # Export
                rp_str = int(RP[i] + 1)
                filename = f'{disagg_type}_mean_site_{site.id}_{imt}_PSHA_{rp_str}_year_RP.png'
                output_path = os.path.join(disagg_out, filename)
                pyplot.savefig(output_path, format='png')
                pyplot.close(fig)


def main(dstore_fname, disagg_type="Mag_Dist_Eps", site_id=None, azimuth=-30):
    """
    Generate 3D plots for given disaggregation type for all sites,
    all intensity measures and all return periods (from poes in given
    investigation time) in datastore's OQparam (i.e. job file inputs).
    By default plotting is done for magnitude-distance-epsilon.

    The plots can be generated for a single site by specifying the
    site_id (each site in the SiteCollection object has a site_id).

    :param calc_id: Name of the datastore containing the calculation results.

    :param disagg_type: Can be Mag_Dist_Eps, Mag_Lon_Lat or TRT_Lon_Lat.

    :param site_id: ID of the site of interest. If None it generate the
                    plots for every site in SiteCollection of the calc.

    :param azimuth: Azimuth angle for the 3D plot (sometimes the default
                    value can cause visual issues in the bar alignment).
    """
    assert disagg_type in ["Mag_Dist_Eps", "Mag_Lon_Lat", "TRT_Lon_Lat"]

    if str(site_id).lower() == "none":
        site_id = None
    
    if str(azimuth).lower() == 'none':
        azimuth = 45

    if disagg_type == "Mag_Dist_Eps":
        disagg_MRE(dstore_fname, disagg_type, site_id, azimuth)

    elif disagg_type == "Mag_Lon_Lat":
        disagg_MLL(dstore_fname, disagg_type, site_id, azimuth)

    else:
        raise NotImplementedError

    print(f"Finished plotting {disagg_type} disagg. results for {dstore_fname}")

if __name__ == '__main__':
    sap.run(main)