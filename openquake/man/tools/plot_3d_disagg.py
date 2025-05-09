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
from openquake.commonlib.logs import get_datadir

import warnings
warnings.filterwarnings("ignore")


# base path
base = os.path.dirname(__file__)


def get_info(calc_id, disagg_type, site_id):
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
    oq_pth = get_datadir()
    dstore_name = os.path.join(oq_pth, f'calc_{calc_id}.hdf5')
    ds = hdf5.File(dstore_name)

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
    export('disagg-stats', dstore_name, **export_info)

    return ds, sites, ims, inv_t, poes, export_info, disagg_out


def disagg_MRE(calc_id, disagg_type, site_id, azimuth):
    """
    Make 3D M-R-e disagg plots for each OQ PSHA calculation's
    mean disaggregation results.
    """
    # Get the disagg info
    ds, sites, ims, inv_t, poes, export_info, disagg_out =\
         get_info(calc_id, disagg_type, site_id)
    
    # disagg bins for MRE
    distance_bin_width = ds["oqparam"].distance_bin_width
    mag_bin_width = ds["oqparam"].mag_bin_width

    for idx_site, site in enumerate(sites):

        # Get the disagg csv (input)
        disagg_file = f'Mag_Dist_Eps-mean-{idx_site}_{calc_id}.csv'
        disagg_path = os.path.join(export_info['export_dir'], disagg_file)

        # Set some params
        Mbin = float(mag_bin_width)
        Dbin = float(distance_bin_width)
        cmap = cm.get_cmap('jet')

        # Load the mean disagg results
        df = pd.read_csv(disagg_path, header=1)
        poes = np.unique(df['poe']).tolist()
        poes.sort(reverse=True)

        # Loop through IMTs and POEs
        for imt_v in ims:
            mode_vals, mean_vals = [], []
            RP = []
            apoe_norm = []
            M, R, eps = [], [], []
            for poe in poes:

                # Get corresponding RP of the POE
                RP.append(round(-inv_t / np.log(1 - poe)))

                # Get data for this IMT and POE
                data = {}
                find = (df['poe'] == poe) & (df['imt'] == imt_v)
                data['mag'] = df['mag'][find]
                data['eps'] = df['eps'][find]
                data['dist'] = df['dist'][find]
                data['rate'] = (-np.log(1 - df['mean'][find]) / inv_t)

                # Normalise the rates
                apoe_norm.append(np.array(data['rate'] / data['rate'].sum()))
                data['rate_norm'] = apoe_norm[-1]
                data = pd.DataFrame(data)

                # Compute the modal M-R-epsilon (highest contribution)
                mode = data.sort_values(by='rate_norm', ascending=False)[0:1]
                mode_vals.append([mode['mag'].values[0],
                                mode['dist'].values[0],
                                mode['eps'].values[0]])
                
                # Also compute the mean M-R-epsilon
                mean_vals.append([np.sum(data['mag'] * data['rate_norm']),
                                np.sum(data['dist'] * data['rate_norm']),
                                np.sum(data['eps'] * data['rate_norm'])])

                # Store the individual mags, dists and epsilons
                M.append(np.array(data['mag']))
                R.append(np.array(data['dist']))
                eps.append(np.array(data['eps']))
                
            # Get number RPs, number of epsilons and min and max epsilon
            n_RP = len(RP)
            n_eps = len(np.unique(np.asarray(eps)))
            min_eps = np.min(np.unique(np.asarray(eps)))  # Get range of colorbars 
            max_eps = np.max(np.unique(np.asarray(eps)))  # so we can normalize

            for i in range(n_RP):
                
                if mean_vals[i][0] == 0.0: # Skip if mean mag is 0 (no haz to disagg)
                    continue

                # Make plot
                fig = pyplot.figure(figsize=(12, 12))
                ax1 = fig.add_subplot(1, 1, 1, projection='3d')

                # Scale each epsilon between 0 and 1 in the cmap
                uni_eps = np.unique(np.asarray(eps))
                rgba = [cmap((k - min_eps) / max_eps / 2) for k in (uni_eps)]
                num_triads_M_R_eps = len(R[i])
                Z = np.zeros(int(num_triads_M_R_eps / n_eps))

                # Make the 3D histogram
                for l in range(n_eps):
                    idx = np.arange(l, num_triads_M_R_eps, step=n_eps)
                    X = np.array(R[i][idx])
                    Y = np.array(M[i][idx])
                    X = X - Dbin/4 # Center on bin midpoints (works for Mbin = 0.5)
                    Y = Y - Mbin/4 # Center on bin midpoints (works for Mbin = 0.5)
                    dx = np.ones(int(num_triads_M_R_eps / n_eps)) * Dbin / 2
                    dy = np.ones(int(num_triads_M_R_eps / n_eps)) * Mbin / 2
                    dz = np.array(
                        apoe_norm[i][np.arange(l, num_triads_M_R_eps, n_eps)]) * 100

                    # Plot the bars
                    mask = dz > 0
                    if np.any(mask): # Only plot if M-R-e provides haz contribution
                        ax1.bar3d(X[mask], Y[mask], Z[mask],
                                dx[mask], dy[mask], dz[mask],
                                color=rgba[l], alpha=1.0)
                    Z += dz # Add height of each bar

                # Axis
                ax1.view_init(elev=23, azim=float(azimuth))
                ax1.set_xlabel('R (km)')
                ax1.set_ylabel('$M_{w}$')
                ax1.set_zlabel('Hazard Contribution (%)')
                ax1.zaxis.set_rotate_label(False)
                ax1.set_zlabel('Hazard Contribution (%)', rotation=90)
                ax1.zaxis._axinfo['juggled'] = (1, 2, 0)
                xmin = np.array(R).min()-Dbin/2
                xmax = np.array(R).max()+Dbin/2
                ymin = np.array(M).min()-Mbin/2
                ymax = np.array(M).max()+Mbin/2
                xticks = np.round(np.arange(xmin, xmax, step=Dbin-1e-09),0)
                yticks = np.arange(ymin, ymax, step=Mbin-1e-09)
                ax1.set_xlim(xmin, xmax)
                ax1.set_xticks(xticks)
                ax1.set_ylim(ymin, ymax)
                ax1.set_yticks(yticks)
                pyplot.tight_layout()
                            
                # Legend for epsilon
                leg_elem = []
                for j in range(n_eps):
                    label = f"\u03B5 = {np.unique(np.asarray(eps))[n_eps - j - 1]:.2f}"
                    ptch = Patch(facecolor=rgba[n_eps - j - 1], label=label)      
                    leg_elem.append(ptch)
                fig.legend(handles=leg_elem, loc="lower center", borderaxespad=0.20,
                        ncol=n_eps, fontsize=12)

                # Title with RP + intensity type
                rp_s = int(RP[i]+1)
                fn = f'MRE_mean_site_{site.id}_{imt_v}_PSHA_{rp_s}_year_RP.png'

                # Export
                fname = os.path.join(disagg_out, fn)
                pyplot.savefig(fname, format='png')
                pyplot.close(fig)


def main(calc_id, disagg_type="Mag_Dist_Eps", site_id=None, azimuth=-45):
    """
    Generate 3D plots for given disaggregation type for all sites,
    all intensity measures and all return periods (from poes in
    investigation time) in OQ job file. By default plotting is done
    for magnitude-distance-epsilon.

    The plots can be generated for a single site by specifying the
    site_id (each site in the SiteCollection object has a site_id).

    :param calc_id: Number of the calculation in your oqdata e.g. calc_451.hdf5
                    would be 451, so to run we would enter into the command line
                    "python mag_dist_eps_disagg_3d_plots.py 451".

    :param disagg_type: Can be Mag_Dist_Eps, Mag_Lon_Lat or TRT_Lon_Lat

    :param site_id: ID of the site of interest. If None it generate the
                    plots for every site in SiteCollection.

    :param azimuth: Azimuth angle for the 3D plot
    """
    assert disagg_type in ["Mag_Dist_Eps", "Mag_Lon_Lat", "TRT_Lon_Lat"]

    if str(site_id).lower() == "none":
        site_id = None
    
    if str(azimuth).lower() == 'none':
        azimuth = 45

    if disagg_type == "Mag_Dist_Eps":
        disagg_MRE(calc_id, disagg_type, site_id, azimuth)

    elif disagg_type == "Mag_Lon_Lat":
        raise NotImplementedError

    else:
        assert disagg_type == "TRT_Lon_Lat"
        raise NotImplementedError

    print(f"Finished plotting {disagg_type} disagg. results for calc {calc_id}")

if __name__ == '__main__':
    sap.run(main)