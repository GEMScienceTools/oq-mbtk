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
import copy

from matplotlib import pyplot
from matplotlib import cm
from matplotlib.patches import Patch

from openquake.baselib import sap, hdf5
from openquake.commands.export import main as export
from openquake.hazardlib.site import SiteCollection

import warnings
warnings.filterwarnings("ignore")


# base path
base = os.path.dirname(__file__)

# Colormap
cmap = cm.get_cmap('jet')

# Mw string
mw_str = "$M_{w}$"


def export_plot(RP, disagg_type, site_id, imt, disagg_out, fig):
    """
    Export the given disagg plot
    """
    rp_str = int(RP + 1)
    filename = f'{disagg_type}_mean_site_{site_id}_{imt}_PSHA_{rp_str}_year_RP.png'
    output_path = os.path.join(disagg_out, filename)
    pyplot.savefig(output_path, format='png')
    pyplot.close(fig)


def get_disagg(disagg_type, calc_id, idx_site, export_info):
    """
    Return dataframe of disaggregation results for given disagg
    type and given site
    """
    # Get a tmp file of the disagg results for given disagg type
    disagg_filename = f'{disagg_type}-mean-{idx_site}_{calc_id}.csv'
    disagg_path = os.path.join(export_info['export_dir'], disagg_filename)

    # Load the tmp
    df = pd.read_csv(disagg_path, header=1)

    # Get sorted POEs
    poes = sorted(np.unique(df['poe']), reverse=True)

    return df, poes


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
    sites = copy.deepcopy(ds["sitecol"])
    if site_id is not None:
        # Get only the site of interest if specified.
        assert len([site_id]) == 1
        sites = sites.filtered([site_id])

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
    
    # Per site in the datastore
    for idx_site, site in enumerate(sites):

        # Get disagg results
        df, poes = get_disagg(disagg_type, calc_id, idx_site, export_info)

        # Get binning params
        Mbin = float(ds["oqparam"].mag_bin_width)
        Dbin = float(ds["oqparam"].distance_bin_width)

        # Per imt
        for imt in ims:
            mode_vals, mean_vals = [], []
            RP, apoe_norm = [], []
            all_mag, all_R, all_eps = [], [], []

            # Per poe
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

                # Modal (highest contribution NOTE: not used here but useful)
                mode_row = data.sort_values(by='rate_norm', ascending=False).iloc[0]
                mode_vals.append([mode_row['mag'], mode_row['dist'], mode_row['eps']])

                # Mean values weighted by normalised rate
                mean_vals.append([
                    np.sum(data['mag'] * data['rate_norm']),
                    np.sum(data['dist'] * data['rate_norm']),
                    np.sum(data['eps'] * data['rate_norm'])
                ])

                all_mag.append(data['mag'].values)
                all_R.append(data['dist'].values)
                all_eps.append(data['eps'].values)

            # Epsilon range for normalization
            eps_all = np.concatenate(all_eps)
            unique_eps = np.unique(eps_all)
            min_eps, max_eps = unique_eps.min(), unique_eps.max()
            n_RP, n_eps = len(RP), len(unique_eps)

            # Get colorbar for unique epsilons
            colors = [cmap((eps - min_eps) / (max_eps - min_eps)) for eps in unique_eps]

            for i in range(n_RP):
                if mean_vals[i][0] == 0.0:
                    continue  # Skip if mag is zero

                fig = pyplot.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1, projection='3d')

                # Loop over epsilons
                stack_base = {}
                for eps_idx, eps_val in enumerate(unique_eps):

                    # Filter by epsilon
                    eps_mask = all_eps[i] == eps_val
                    X = all_R[i][eps_mask] - Dbin / 4
                    Y = all_mag[i][eps_mask] - Mbin / 4
                    dz = apoe_norm[i][eps_mask] * 100

                    if len(X) == 0:
                        continue

                    dx = np.full_like(X, Dbin / 2)
                    dy = np.full_like(Y, Mbin / 2)

                    Z = np.zeros_like(dz)
                    for j in range(len(X)):
                        key = (X[j], Y[j])
                        Z[j] = stack_base.get(key, 0.0)
                        stack_base[key] = Z[j] + dz[j]

                    mask = dz > 0
                    if np.any(mask):
                        ax.bar3d(X[mask], Y[mask], Z[mask], dx[mask], dy[mask], dz[mask],
                                color=colors[eps_idx], alpha=1.0)

                assert abs(sum(stack_base.values()) - 100.0) < 1e-6

                # Labels and azimuth
                ax.view_init(elev=23, azim=azimuth)
                ax.set_xlabel('R (km)', fontsize=14)
                ax.set_ylabel(mw_str, fontsize=14)
                ax.set_zlabel('Hazard Contribution (%)', fontsize=14, rotation=90)

                # Axis params
                ax.set_xlim(np.min(all_R) - Dbin / 2, np.max(all_R) + Dbin / 2)
                ax.set_ylim(np.min(all_mag) - Mbin / 2, np.max(all_mag) + Mbin / 2)
                ax.set_xticks(np.round(np.arange(np.min(all_R), np.max(all_R) + Dbin, Dbin), 0))
                ax.set_yticks(np.arange(np.min(all_mag), np.max(all_mag) + Mbin, Mbin))

                # Legend
                lg_elm = [
                    Patch(facecolor=colors[n_eps - j - 1],
                        label=f"\u03B5 = {unique_eps[n_eps - j - 1]:.2f}") for j in range(n_eps)]

                fig.legend(handles=lg_elm,
                           loc="upper left",
                           borderaxespad=0.40,
                           ncol=1,
                           fontsize=14)

                # Also provide info on modal and mean mag-dist-eps
                modal_mw = np.round(mode_vals[i][0], 2)
                modal_r = int(mode_vals[i][1])
                modal_eps = np.round(mode_vals[i][2], 2)
                mean_mw = np.round(mean_vals[i][0], 2)
                mean_r = int(mean_vals[i][1])
                mean_eps = np.round(mean_vals[i][2], 2)
                pyplot.title((f"MODAL: {mw_str} = {modal_mw}, R = {modal_r} km, \u03B5 = {modal_eps}"
                              f"\nMEAN: {mw_str} = {mean_mw}, R = {mean_r} km, \u03B5 = {mean_eps}"),
                              fontsize=18, loc='center', va='top', x=0.65, y=1.2)

                # Export
                export_plot(RP[i], disagg_type, site.id, imt, disagg_out, fig)


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

    # Per site in the datastore
    for idx_site, site in enumerate(sites):

        # Get disagg results
        df, poes = get_disagg(disagg_type, calc_id, idx_site, export_info)

        # Get binning params
        Cbin = float(ds["oqparam"].coordinate_bin_width)

        # Per imt
        for imt in ims:
            mode_vals, mean_vals = [], []
            RP, apoe_norm = [], []
            all_mag, all_lon, all_lat = [], [], []

            # Per poe
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

                all_lon.append(data['lon'].values)
                all_lat.append(data['lat'].values)
                all_mag.append(data['mag'].values)

            # Magnitude range for normalization
            mag_all = np.concatenate(all_mag)
            unique_mag = np.unique(mag_all)
            min_mag, max_mag = unique_mag.min(), unique_mag.max()
            n_RP, n_mag = len(RP), len(unique_mag)

            # Get colorbar for unique magnitudes
            colors = [cmap((eq_mag - min_mag) / (max_mag - min_mag)) for eq_mag in unique_mag]

            for i in range(n_RP):
                if mean_vals[i][2] == 0.0:
                    continue  # Skip if mag is zero (no contribution)

                # Make figure
                fig = pyplot.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1, projection='3d')

                # Loop over magnitudes
                stack_base = {}
                for mag_idx, mag_val in enumerate(unique_mag):

                    # Filter by magnitude
                    mag_mask = all_mag[i] == mag_val
                    X = all_lon[i][mag_mask]
                    Y = all_lat[i][mag_mask]
                    dz = apoe_norm[i][mag_mask] * 100

                    if len(X) == 0:
                        continue

                    dx = np.full_like(X, Cbin / 2)
                    dy = np.full_like(Y, Cbin / 2)

                    Z = np.zeros_like(dz)
                    for j in range(len(X)):
                        key = (X[j], Y[j])
                        Z[j] = stack_base.get(key, 0.0)
                        stack_base[key] = Z[j] + dz[j]

                    mask = dz > 0
                    if np.any(mask):
                        ax.bar3d(X[mask], Y[mask], Z[mask], dx[mask], dy[mask], dz[mask],
                                color=colors[mag_idx], alpha=1.0)

                assert abs(sum(stack_base.values()) - 100.0) < 1e-6

                # Labels and azimuth
                ax.view_init(elev=23, azim=azimuth)
                ax.set_xlabel('Longitude', fontsize=14)
                ax.set_ylabel('Latitude', fontsize=14)
                ax.set_zlabel('Hazard Contribution (%)', fontsize=14, rotation=90)

                # Axis params
                ax.set_xlim(np.min(all_lon) - Cbin / 2, np.max(all_lon) + Cbin / 2)
                ax.set_ylim(np.min(all_lat) - Cbin / 2, np.max(all_lat) + Cbin / 2)
                ax.set_xticks(np.round(np.arange(np.min(all_lon), np.max(all_lon) + Cbin, Cbin), 1))
                ax.set_yticks(np.round(np.arange(np.min(all_lat), np.max(all_lat) + Cbin, Cbin), 1))

                # Legend
                lg_elm = [
                    Patch(facecolor=colors[n_mag - j - 1],
                        label='$M_{w}$' + f" = {unique_mag[n_mag - j - 1]:.2f}") for j in range(n_mag)]

                fig.legend(handles=lg_elm,
                           loc="upper left",
                           borderaxespad=0.40,
                           ncol=1,
                           fontsize=14)

                # Also provide info on modal and mean mag-lon-lat
                modal_lon = np.round(mode_vals[i][0], 3)
                modal_lat = np.round(mode_vals[i][1], 3)
                modal_mw = np.round(mode_vals[i][2], 2)
                mean_lon = np.round(mean_vals[i][0], 3)
                mean_lat = np.round(mean_vals[i][1], 3)
                mean_mw = np.round(mean_vals[i][2], 2)
                pyplot.title((f"MODAL: {mw_str} = {modal_mw}, lon = {modal_lon}, lat = {modal_lat}"
                              f"\nMEAN: {mw_str} = {mean_mw}, lon = {mean_lon}, lat = {mean_lat}"),
                              fontsize=18, loc='center', va='top', x=0.65, y=1.2)

                # Export
                export_plot(RP[i], disagg_type, site.id, imt, disagg_out, fig)


def disagg_TLL(dstore_fname, disagg_type, site_id, azimuth):
    """
    Make 3D trt-lon-lat disagg plots for an OQ PSHA calculation's
    mean disaggregation results.
    """
    assert disagg_type == "TRT_Lon_Lat"

    # Get calc ID number
    calc_id = int(dstore_fname.split("calc_")[1].split('.')[0])

    # Get the disagg info
    ds, sites, ims, inv_t, poes, export_info, disagg_out =\
         get_info(dstore_fname, calc_id, disagg_type, site_id)

    # Per site in the datastore
    for idx_site, site in enumerate(sites):

        # Get disagg results
        df, poes = get_disagg(disagg_type, calc_id, idx_site, export_info)

        # Map each TRT in the df to an identifying integer
        trt_maps = {trt: idx for idx, trt in enumerate(df['trt'].unique())}
        df['trt_conv'] = [trt_maps[trt] for trt in df['trt']]

        # Get binning params
        Cbin = float(ds["oqparam"].coordinate_bin_width)

        # Per imt
        for imt in ims:
            mode_vals, mean_vals = [], []
            RP, apoe_norm = [], []
            all_trt, all_lon, all_lat = [], [], []

            # Per poe
            for poe in poes:
                RP.append(round(-inv_t / np.log(1 - poe)))
                mask_df = (df['poe'] == poe) & (df['imt'] == imt)

                data = pd.DataFrame({
                    'lon': df.loc[mask_df, 'lon'],
                    'lat': df.loc[mask_df, 'lat'],
                    'trt': df.loc[mask_df, 'trt_conv'],
                    'rate': -np.log(1 - df.loc[mask_df, 'mean']) / inv_t
                })

                data['rate_norm'] = data['rate'] / data['rate'].sum()
                apoe_norm.append(data['rate_norm'].values)

                # Modal (highest contribution)
                mode_row = data.sort_values(by='rate_norm', ascending=False).iloc[0]
                mode_vals.append([mode_row['lon'], mode_row['lat'], mode_row['trt']])

                # Mean values weighted by normalised rate
                mean_vals.append([
                    np.sum(data['lon'] * data['rate_norm']),
                    np.sum(data['lat'] * data['rate_norm']),
                    np.sum(data['trt'] * data['rate_norm'])
                ])

                all_lon.append(data['lon'].values)
                all_lat.append(data['lat'].values)
                all_trt.append(data['trt'].values)

            # TRT "range" for normalization
            trt_all = np.concatenate(all_trt)
            unique_trt = np.unique(trt_all)
            min_trt, max_trt = unique_trt.min(), unique_trt.max()
            n_RP, n_trt = len(RP), len(unique_trt)

            # Get colorbar for unique magnitudes
            colors = [cmap((eq_trt - min_trt) / (max_trt - min_trt)) for eq_trt in unique_trt]
            
            for i in range(n_RP):
                if mean_vals[i][2] == 0.0:
                    continue  # Skip if TRT is zero (no contribution)

                # Make figure
                fig = pyplot.figure(figsize=(12, 12))
                ax = fig.add_subplot(1, 1, 1, projection='3d')

                # Loop over TRTs
                stack_base = {}
                for trt_idx, trt_val in enumerate(unique_trt):

                    # Filter by magnitude
                    trt_mask = all_trt[i] == trt_val
                    X = all_lon[i][trt_mask]
                    Y = all_lat[i][trt_mask]
                    dz = apoe_norm[i][trt_mask] * 100

                    if len(X) == 0:
                        continue

                    dx = np.full_like(X, Cbin / 2)
                    dy = np.full_like(Y, Cbin / 2)

                    Z = np.zeros_like(dz)
                    for j in range(len(X)):
                        key = (X[j], Y[j])
                        Z[j] = stack_base.get(key, 0.0)
                        stack_base[key] = Z[j] + dz[j]

                    mask = dz > 0
                    if np.any(mask):
                        ax.bar3d(X[mask], Y[mask], Z[mask], dx[mask], dy[mask], dz[mask],
                                color=colors[trt_idx], alpha=1.0)

                assert abs(sum(stack_base.values()) - 100.0) < 1e-6

                # Labels and azimuth
                ax.view_init(elev=23, azim=azimuth)
                ax.set_xlabel('Longitude', fontsize=14)
                ax.set_ylabel('Latitude', fontsize=14)
                ax.set_zlabel('Hazard Contribution (%)', fontsize=14, rotation=90)

                # Axis params
                ax.set_xlim(np.min(all_lon) - Cbin / 2, np.max(all_lon) + Cbin / 2)
                ax.set_ylim(np.min(all_lat) - Cbin / 2, np.max(all_lat) + Cbin / 2)
                ax.set_xticks(np.round(np.arange(np.min(all_lon), np.max(all_lon) + Cbin, Cbin), 1))
                ax.set_yticks(np.round(np.arange(np.min(all_lat), np.max(all_lat) + Cbin, Cbin), 1))

                # Legend
                trt_map_inv = {trt_maps[val]: val for val in trt_maps}
                lg_elm = [
                    Patch(facecolor=colors[n_trt - j - 1],
                        label=f"{trt_map_inv[unique_trt[n_trt - j - 1]]}") for j in range(n_trt)]

                fig.legend(handles=lg_elm,
                           loc="upper left",
                           borderaxespad=0.40,
                           ncol=1,
                           fontsize=14)

                # Also provide info on modal and mean trt-lon-lat
                modal_lon = np.round(mode_vals[i][0], 3)
                modal_lat = np.round(mode_vals[i][1], 3)
                modal_trt = trt_map_inv[mode_vals[i][2]] # Cannot provide a "mean" TRT!
                pyplot.title(f"MODAL: TRT = {modal_trt}, lon = {modal_lon}, lat = {modal_lat}",
                            fontsize=18, loc='center', va='top', x=0.65, y=1.2)                    

                # Export
                export_plot(RP[i], disagg_type, site.id, imt, disagg_out, fig)


def main(dstore_fname, disagg_type, site_id=None, azimuth=-30):
    """
    Generate 3D plots for given disaggregation type for all sites,
    all intensity measures and all return periods (from poes in given
    investigation time) in datastore's OQparam (i.e. job file inputs).

    The plots can be generated for a single site by specifying the
    site_id (each site in the SiteCollection object has a site_id).

    :param dstore_fname: Name of the datastore containing the calculation results.

    :param disagg_type: Can be Mag_Dist_Eps, Mag_Lon_Lat or TRT_Lon_Lat.

    :param site_id: ID of the site of interest. If None it generate the
                    plots for every site in SiteCollection of the calc.

    :param azimuth: Azimuth angle for the 3D plot (sometimes the default
                    value can cause visual issues in the bar alignment).
    """
    assert disagg_type in ["Mag_Dist_Eps", "Mag_Lon_Lat", "TRT_Lon_Lat"]

    if str(site_id).lower() == "none":
        site_id = None
    else:
        site_id = int(site_id)

    if str(azimuth).lower() == 'none':
        azimuth = -30
    else:
        azimuth = float(azimuth)

    if disagg_type == "Mag_Dist_Eps":
        disagg_MRE(dstore_fname, disagg_type, site_id, azimuth)

    elif disagg_type == "Mag_Lon_Lat":
        disagg_MLL(dstore_fname, disagg_type, site_id, azimuth)

    else:
        disagg_TLL(dstore_fname, disagg_type, site_id, azimuth)

    print(f"Finished plotting {disagg_type} disagg. results for {dstore_fname}")

if __name__ == '__main__':
    sap.run(main)