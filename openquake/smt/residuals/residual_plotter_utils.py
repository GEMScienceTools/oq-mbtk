#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2025 GEM Foundation and G. Weatherill
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
"""
Module managing residual plotting data.
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress


def _get_residuals_density_distribution(residuals, gmpe, imt, bin_width=0.5):
    """
    Returns the density distribution of the given gmpe and imt

    :param residuals: instance of :class: openquake.smt.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'mean' (float) and 'Std Dev' (float) representing
    the mean and standard deviation of the data
    """
    statistics = residuals.get_residual_statistics_for(gmpe, imt)
    plot_data = {}
    data = residuals.residuals[gmpe][imt]

    for res_type in data.keys():

        vals, bins = _get_histogram_data(data[res_type], bin_width=bin_width)

        mean = statistics[res_type]["Mean"]
        stddev = statistics[res_type]["Std Dev"]
        x = bins[:-1]
        y = vals

        plot_data[res_type] = \
            {'x': x, 'y': y, 'mean': mean, 'stddev': stddev,
             'xlabel': "Z (%s)" % imt, 'ylabel': "Frequency"}

    return plot_data


def _get_histogram_data(data, bin_width=0.5):
    """
    Retreives the histogram of the residuals
    """
    # Ignore nans otherwise max and min raise
    bins = np.arange(
        np.floor(np.nanmin(data)),
        np.ceil(np.nanmax(data)) + bin_width,
        bin_width
        )
    # Work on finite numbers to prevent np.histogram raising
    vals = np.histogram(data[np.isfinite(data)], bins, density=True)[0]
    return vals.astype(float), bins


def _get_likelihood_data(residuals, gmpe, imt, bin_width=0.1):
    """
    Returns the likelihood of the given gmpe and imt

    :param residuals: instance of :class: openquake.smt.gmpe_residuals.Likelihood
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'median' (float) representing the median of the data
    """
    plot_data = {}
    data = residuals._compute_likelihood_values_for(gmpe, imt)
    for res_type in data.keys():
        lh_vals, median_lh = data[res_type]
        vals, bins = _get_lh_histogram_data(lh_vals, bin_width=bin_width)
        plot_data[res_type] = {
            'x': bins[:-1],
            'y': vals,
            'median': median_lh,
            'xlabel': "LH (%s)" % imt,
            'ylabel': "Frequency"
            }

    return plot_data


def _get_lh_histogram_data(lh_values, bin_width=0.1):
    """
    Retreives the histogram of the likelihoods
    """
    bins = np.arange(0.0, 1.0 + bin_width, bin_width)
    # Work on finite numbers to prevent np.histogram raising:
    vals = np.histogram(
        lh_values[np.isfinite(lh_values)], bins, density=True)[0]
    return vals.astype(float), bins


def _get_magnitudes(residuals, gmpe, imt, res_type):
    """
    Returns an array of magnitudes equal in length to the number of
    residuals
    """
    magnitudes = np.array([])
    for i, ctx in enumerate(residuals.contexts):
        keep = ctx["Retained"][imt]
        if res_type == "Inter event":
            nval = np.ones(len(residuals.unique_indices[gmpe][imt][i]))
        else:
            nval = np.ones(len(ctx["Ctx"].repi))
            nval = nval[keep]
        magnitudes = np.hstack([magnitudes, ctx["Ctx"].mag * nval])
    return magnitudes


def _get_depths(residuals, gmpe, imt, res_type):
    """
    Returns an array of magnitudes equal in length to the number of
    residuals
    """
    depths = np.array([])
    for i, ctx in enumerate(residuals.contexts):
        keep = ctx["Retained"][imt]
        if res_type == "Inter event":
            nvals = np.ones(len(residuals.unique_indices[gmpe][imt][i]))
        else:
            nvals = np.ones(len(ctx["Ctx"].repi))
            nvals = nvals[keep]
        depths = np.hstack([depths, ctx["Ctx"].hypo_depth * nvals])
    return depths


def _get_vs30(residuals, gmpe, imt, res_type):
    """
    Return required vs30 values
    """
    vs30 = np.array([])
    for i, ctx in enumerate(residuals.contexts):
        keep = ctx["Retained"][imt]
        if res_type == "Inter event":
            vs30 = np.hstack([vs30, ctx["Ctx"].vs30[
                residuals.unique_indices[gmpe][imt][i]]])
        else:
            vs30_vals = ctx["Ctx"].vs30[keep]
            vs30 = np.hstack([vs30, vs30_vals])
        
    return vs30


def _get_distances(residuals, gmpe, imt, res_type, distance_type):
    """
    Return required distances
    """
    distances = np.array([])
    for i, ctx in enumerate(residuals.contexts):
        keep = ctx["Retained"][imt]
        # Get the distances
        if res_type == "Inter event":
            dists = getattr(ctx["Ctx"], distance_type)[
                residuals.unique_indices[gmpe][imt][i]]
            distances = np.hstack([distances, dists])
        else:
            dist_vals = getattr(ctx["Ctx"], distance_type)
            dist_vals = dist_vals[keep]
            distances = np.hstack([distances, dist_vals])
            
    return distances


def get_scatter_vals(var, residuals, gmpe, imt, res_type, distance_type):
    """
    Return values for given explanatory variable matching the 
    length of the given residuals
    """
    if var == "magnitude":
        return _get_magnitudes(residuals, gmpe, imt, res_type)
    elif var == "depth":
        return _get_depths(residuals, gmpe, imt, res_type)
    elif var == "vs30":
        return _get_vs30(residuals, gmpe, imt, res_type)
    else:
        assert var == "distance"
        return _get_distances(residuals, gmpe, imt, res_type, distance_type)


def get_scatter_data(residuals, gmpe, imt, var, distance_type=None):
    """
    Get plot data for a scatter plot of residuals (y-axis)
    and given explanatory variable (x-axis)
    """
    plot_data = {}
    
    mean_res_df, sigma_res_df = bin_res_wrt_var(residuals, gmpe, imt, var)
    
    data = residuals.residuals[gmpe][imt]
    for res_type in data.keys():
    
        if res_type in ["vals"]:
            continue

        x = get_scatter_vals(var, residuals, gmpe, imt, res_type, distance_type)
        y = data[res_type]

        slope, intercept, _, pval, _ = _nanlinregress(x, y)

        plot_data[res_type] = {
                               'x': x,
                               'y': y,
                               'slope': slope,
                               'intercept': intercept,
                               'pvalue': pval,
                               'ylabel': "Z (%s)" % imt,
                               'bin_midpoints': mean_res_df.x_data,
                               'mean_res': mean_res_df[res_type],
                               'sigma_res': sigma_res_df[res_type]
                               }
            
        if var == "magnitude":
            plot_data[res_type]['xlabel'] = "Magnitude (Mw)"
        elif var == "depth":
            plot_data[res_type]["xlabel"] = "Hypocentral Depth (km)"
        elif var == "vs30":
            plot_data[res_type]["xlabel"] = "Vs30 (m/s)"
        else:
            assert var == "distance"
            plot_data[res_type]["xlabel"] = f"{distance_type} (km)"

    return plot_data


def residuals_with_magnitude(residuals, gmpe, imt):
    """
    Returns the residuals of the given gmpe and imt vs. magnitude

    :param residuals: instance of openquake.smt.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    """
    return get_scatter_data(residuals, gmpe, imt, "magnitude")


def residuals_with_depth(residuals, gmpe, imt):
    """
    Returns the residuals of the given gmpe and imt vs. depth

    :param residuals: instance of openquake.smt.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    """
    return get_scatter_data(residuals, gmpe, imt, "depth")


def residuals_with_vs30(residuals, gmpe, imt):
    """
    Returns the residuals of the given gmpe and imt vs. vs30

    :param residuals: instance of :class: openquake.smt.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    """
    return get_scatter_data(residuals, gmpe, imt, "vs30")


def residuals_with_distance(residuals, gmpe, imt, distance_type="rjb"):
    """
    Returns the residuals of the given gmpe and imt vs. distance

    :param residuals: instance of :class: openquake.smt.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    """
    return get_scatter_data(residuals, gmpe, imt, "distance", distance_type)


def _nanlinregress(x, y):
    """
    Calls scipy linregress only on finite numbers of x and y
    """
    finite = np.isfinite(x) & np.isfinite(y)
        
    if not finite.any():
        # Empty arrays passed to linreg raise ValueError
        # so force returning an object with nans
        return linregress([np.nan], [np.nan])
    else:
        return linregress(x[finite], y[finite])


### Utils for binning residuals w.r.t. a given GMM input variable
def get_ctx_vals(var_type, ctx, distance_type):
    """
    Get value(s) of the given ctx corresponding to the variable we
    are plotting the residuals against
    """
    if var_type == 'magnitude':
        event_val = ctx.mag
    elif var_type == 'vs30':
        event_val = ctx.vs30
    elif var_type == 'distance':
        event_val = getattr(ctx, distance_type)
    elif var_type == 'depth':
        event_val = ctx.hypo_depth

    return event_val


def _get_residual_means_and_stds(residuals):
    """
    Get the mean and sigma of the distributions of residuals
    for each gmpe and imt
    """
    # Get all residuals for all GMPEs at all IMTs
    res_statistics = {}
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            res_statistics[gmpe, imt] = residuals.get_residual_statistics_for(
                gmpe, imt)
    
    # Now get into dataframes
    mean_sigma_intra, mean_sigma_inter, mean_sigma_total = {}, {}, {}
    dummy_values = {'Mean': np.nan, 'Std Dev': np.nan} # Assign if only total sigma
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            mean_sigma_total[gmpe, imt] = res_statistics[gmpe, imt]['Total']
            if ('Inter event' in residuals.residuals[gmpe][imt]
                and
                'Intra event' in residuals.residuals[gmpe][imt]):
                mean_sigma_inter[
                    gmpe, imt] = res_statistics[gmpe, imt]['Inter event']
                mean_sigma_intra[
                    gmpe, imt] = res_statistics[gmpe, imt]['Intra event']
            else:
                mean_sigma_inter[gmpe, imt] = dummy_values
                mean_sigma_intra[gmpe, imt] = dummy_values

    intra = pd.DataFrame(mean_sigma_intra)
    inter = pd.DataFrame(mean_sigma_inter)
    total = pd.DataFrame(mean_sigma_total)

    return intra, inter, total


def mean_and_sigma_per_bin(df, idx_res_per_val_bin):
    """
    Computes the mean and standard deviation for residuals per value bin.
    """
    # Set stores of mean and sigma per bin of the given variable
    total_mean, total_sigma = {}, {}
    intra_mean, intra_sigma = {}, {}
    inter_mean, inter_sigma = {}, {}

    # Get the mean and sigma for each component of the res assoc with each bin
    for val_bin, indices in idx_res_per_val_bin.items():
        idx_vals = pd.Series(indices.keys())
        df_bin = df.iloc[idx_vals]
        total_mean[val_bin] = df_bin["Total"].mean()
        total_sigma[val_bin] = df_bin["Total"].std()

        if 'Inter event' in df_bin.columns:
            intra_mean[val_bin] = df_bin["Intra event"].mean()
            inter_mean[val_bin] = df_bin["Inter event"].mean()
            intra_sigma[val_bin] = df_bin["Intra event"].std()
            inter_sigma[val_bin] = df_bin["Inter event"].std()

    return {"total_mean": total_mean,
            "total_sigma": total_sigma,
            "intra_mean": intra_mean,
            "intra_sigma": intra_sigma,
            "inter_mean": inter_mean,
            "inter_sigma": inter_sigma}


def get_binning_params(var_type, vals):
    """
    Get the params for the binning of the given variable we are plotting
    the residuals with respect to
    """
    # Get values for given variable
    var_bins = {
                'magnitude': 0.25, # Mw
                'depth': 5,        # km
                'distance': 10,    # km
                'vs30': 100        # m/s
                }
    val_bin = var_bins[var_type]

    # Create bins and make last interval fill up to max var value
    val_bins = np.arange(np.min(vals), np.max(vals), val_bin)
    val_bins[len(val_bins) - 1] = np.max(vals)
    bin_bounds = {}
    for idx, val_bin in enumerate(val_bins):
        if idx == len(val_bins) - 1:
            pass
        else:
            bin_bounds[idx] = [val_bins[idx], val_bins[idx+1]]

    # Get midpoint of each val bin for plotting
    bin_mid_points = {val_bin: bounds[0] + 0.5 * (
        bounds[1] - bounds[0]) for val_bin, bounds in bin_bounds.items()}
        
    return bin_bounds, bin_mid_points


def get_res_df(var_type, residuals, gmpe, imt, distance_type):
    """
    Return a dataframe with the total, inter-event and intra event
    residuals w.r.t. the variable of interest for plotting
    """
    store = []
    for ctx in residuals.contexts:

        # Set a dict for this eq
        eq = {}

        # Get idx of recs that are not null for given IMT
        retain = ctx["Retained"][imt]

        # Get values of the explanatory variable for given ctx
        vals = get_ctx_vals(var_type, ctx["Ctx"], distance_type)
        if var_type in ["magnitude", "depth"]:
            eq["vals"] = np.full(len(retain), vals)
        else:
            eq["vals"] = vals[retain]

        if "Inter event" in ctx['Residual'][gmpe][imt].keys():

            # Inter event residual    
            eq["Inter event"] = ctx['Residual'][gmpe][imt]['Inter event']

            # Inter event residual
            eq["Intra event"] = ctx['Residual'][gmpe][imt]['Intra event']

        # Total residual
        eq["Total"] = ctx['Residual'][gmpe][imt]['Total']

        # Into df for given ctx
        eq_df = pd.DataFrame(eq)

        # Store the eq df
        store.append(eq_df)

    return pd.concat(store).sort_values(by="vals")


def bin_res_wrt_var(residuals, gmpe, imt, var_type, distance_type='repi'):
    """
    Compute mean total, inter-event and inter-event residual within bins
    for a given explanatory variable. These binned residuals are plotted
    within the scatter plots of residuals (y-axis) w.r.t. the given
    explanatory var (x-axis).
    :param var_type: Specifies variable which residuals are plotted against
    """
    # Get residuals and the variable (per record) in a dataframe
    df = get_res_df(var_type, residuals, gmpe, imt, distance_type)
    
    # Get bin bounds
    bin_bounds, bin_mid_points, = get_binning_params(var_type, df.vals)

    # Get indices for the residuals in each bin
    idx_res_per_val_bin = {idx: {} for idx in bin_bounds}
    for idx in bin_bounds:
        for idx_dp, data_point in enumerate(df.vals):
            if (data_point >= bin_bounds[idx][0]
                and
                data_point <= bin_bounds[idx][1]): 
                idx_res_per_val_bin[idx][idx_dp] = data_point

    # Get the mean and std per res assoc with each bin of the given var
    means_and_sigmas = mean_and_sigma_per_bin(df, idx_res_per_val_bin)
    
    # Get final data to plot
    if 'Inter event' in df.columns:
        mean_res_wrt_val = pd.DataFrame({
            'x_data': bin_mid_points,
            'Total': means_and_sigmas['total_mean'],
            'Inter event': means_and_sigmas['inter_mean'], 
            'Intra event': means_and_sigmas['intra_mean']})
        
        sigma_res_wrt_val = pd.DataFrame({
            'x_data': bin_mid_points,
            'Total': means_and_sigmas['total_sigma'],
            'Inter event': means_and_sigmas['inter_sigma'], 
            'Intra event': means_and_sigmas['intra_sigma']})
    else:
        mean_res_wrt_val = pd.DataFrame(
            {'x_data':bin_mid_points, 'Total': means_and_sigmas['total_mean']})
        
        sigma_res_wrt_val = pd.DataFrame({
            'x_data':bin_mid_points, 'Total': means_and_sigmas['total_sigma']})
        
    return mean_res_wrt_val, sigma_res_wrt_val