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
Module managing GMPE+IMT residual plot data.
This module avoids the use of classes and inhertances as simple functions
accomplish the task without unnecessary overhead.
All non-private functions should return the same dicts (see docstrings
for details)
"""
import numpy as np
import pandas as pd
from scipy.stats import linregress


def residuals_density_distribution(residuals, gmpe, imt, bin_width=0.5):
    """
    Returns the density distribution of the given gmpe and imt

    :param residuals:
            Residuals as instance of :class: openquake.smt.gmpe_residuals.Residuals
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
    # ignore nans otherwise max and min raise:
    bins = np.arange(np.floor(np.nanmin(data)),
                     np.ceil(np.nanmax(data)) + bin_width,
                     bin_width)
    # work on finite numbers to prevent np.histogram raising:
    vals = np.histogram(data[np.isfinite(data)], bins, density=True)[0]
    return vals.astype(float), bins


def likelihood(residuals, gmpe, imt, bin_width=0.1):
    """
    Returns the likelihood of the given gmpe and imt

    :param residuals:
            Residuals as instance of :class: openquake.smt.gmpe_residuals.Likelihood
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'median' (float) representing
    the median of the data
    """
    plot_data = {}
    data = residuals._get_likelihood_values_for(gmpe, imt)

    for res_type in data.keys():
        lh_vals, median_lh = data[res_type]
        vals, bins = _get_lh_histogram_data(lh_vals, bin_width=bin_width)

        x = bins[:-1]
        y = vals

        plot_data[res_type] = \
            {'x': x, 'y': y, 'median': median_lh,
             'xlabel': "LH (%s)" % imt, 'ylabel': "Frequency"}

    return plot_data


def _get_lh_histogram_data(lh_values, bin_width=0.1):
    """
    Retreives the histogram of the likelihoods
    """
    bins = np.arange(0.0, 1.0 + bin_width, bin_width)
    # work on finite numbers to prevent np.histogram raising:
    vals = np.histogram(lh_values[np.isfinite(lh_values)],
                        bins, density=True)[0]
    return vals.astype(float), bins


def residuals_with_magnitude(residuals, gmpe, imt):
    """
    Returns the residuals of the given gmpe and imt vs. magnitude

    :param residuals:
            Residuals as instance of :class: openquake.smt.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    """
    plot_data = {}
    data = residuals.residuals[gmpe][imt]
    
    var_type = 'magnitude'
    
    mean_res_df, sigma_res_df = _get_mean_res_wrt_var(residuals, gmpe,
                                                      imt, var_type)
    
    for res_type in data.keys():

        x = _get_magnitudes(residuals, gmpe, imt, res_type)
        slope, intercept, _, pval, _ = _nanlinregress(x, data[res_type])
        y = data[res_type]

        plot_data[res_type] = \
            {'x': x, 'y': y,
             'slope': slope, 'intercept': intercept, 'pvalue': pval,
             'xlabel': "Magnitude", 'ylabel': "Z (%s)" % imt,
             'bin_midpoints': mean_res_df.x_data,
             'mean_res': mean_res_df[res_type],
             'sigma_res': sigma_res_df[res_type] }
            
    return plot_data


def _get_magnitudes(residuals, gmpe, imt, res_type):
    """
    Returns an array of magnitudes equal in length to the number of
    residuals
    """
    magnitudes = np.array([])
    for i, ctxt in enumerate(residuals.contexts):
        if res_type == "Inter event":

            nval = np.ones(
                len(residuals.unique_indices[gmpe][imt][i])
                )
        else:
            nval = np.ones(len(ctxt["Ctx"].repi))

        magnitudes = np.hstack([magnitudes, ctxt["Ctx"].mag * nval])

    return magnitudes


def residuals_with_vs30(residuals, gmpe, imt):
    """
    Returns the residuals of the given gmpe and imt vs. vs30

    :param residuals:
            Residuals as instance of :class: openquake.smt.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    """
    plot_data = {}
    data = residuals.residuals[gmpe][imt]

    var_type = 'vs30'    
    
    mean_res_df, sigma_res_df = _get_mean_res_wrt_var(residuals, gmpe, imt, var_type)
    
    for res_type in data.keys():

        x = _get_vs30(residuals, gmpe, imt, res_type)
        slope, intercept, _, pval, _ = _nanlinregress(x, data[res_type])
        y = data[res_type]

        plot_data[res_type] = \
            {'x': x, 'y': y,
             'slope': slope, 'intercept': intercept, 'pvalue': pval,
             'xlabel': "Vs30 (m/s)", 'ylabel': "Z (%s)" % imt,
             'bin_midpoints': mean_res_df.x_data,
             'mean_res': mean_res_df[res_type],
             'sigma_res': sigma_res_df[res_type]  }

    return plot_data


def _get_vs30(residuals, gmpe, imt, res_type):
    """
    Return required vs30 values
    """
    vs30 = np.array([])
    for i, ctxt in enumerate(residuals.contexts):
        if res_type == "Inter event":
            vs30 = np.hstack([vs30, ctxt["Ctx"].vs30[
                residuals.unique_indices[gmpe][imt][i]]])
        else:
            vs30 = np.hstack([vs30, ctxt["Ctx"].vs30])
    return vs30


def residuals_with_distance(residuals, gmpe, imt, distance_type="rjb"):
    """
    Returns the residuals of the given gmpe and imt vs. distance

    :param residuals:
            Residuals as instance of :class: openquake.smt.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    """
    plot_data = {}
    data = residuals.residuals[gmpe][imt]

    var_type = 'distance'    
    
    mean_res_df, sigma_res_df = _get_mean_res_wrt_var(residuals, gmpe, imt, var_type)

    for res_type in data.keys():

        x = _get_distances(residuals, gmpe, imt, res_type, distance_type)
        slope, intercept, _, pval, _ = _nanlinregress(x, data[res_type])
        y = data[res_type]

        plot_data[res_type] = \
            {'x': x, 'y': y,
             'slope': slope, 'intercept': intercept, 'pvalue': pval,
             'xlabel': "%s Distance (km)" % distance_type,
             'ylabel': "Z (%s)" % imt, 'bin_midpoints': mean_res_df.x_data,
             'mean_res': mean_res_df[res_type],
             'sigma_res': sigma_res_df[res_type] }

    return plot_data


def _get_distances(residuals, gmpe, imt, res_type, distance_type):
    """
    Return required distances
    """
    distances = np.array([])
    for i, ctxt in enumerate(residuals.contexts):
        # Get the distances
        if res_type == "Inter event":
            ctxt_dist = getattr(ctxt["Ctx"], distance_type)[
                residuals.unique_indices[gmpe][imt][i]]
            distances = np.hstack([distances, ctxt_dist])
        else:
            distances = np.hstack([
                distances, getattr(ctxt["Ctx"], distance_type)])
    return distances


def residuals_with_depth(residuals, gmpe, imt):
    """
    Returns the residuals of the given gmpe and imt vs. depth

    :param residuals:
            Residuals as instance of :class: openquake.smt.gmpe_residuals.Residuals
    :param gmpe: (string) the gmpe/gsim
    :param imt: (string) the intensity measure type

    :return: a dict mapping each residual type (string, e.g. 'Intra event') to
    a dict with (at least) the mandatory keys 'x', 'y', 'xlabel', 'ylabel'
    representing the plot data.
    Additional keys: 'slope' (float), 'intercept' (float) and 'pvalue' (float)
    representing the linear regression of the data
    """
    plot_data = {}
    data = residuals.residuals[gmpe][imt]

    var_type = 'depth'    
    mean_res_df, sigma_res_df = _get_mean_res_wrt_var(residuals, gmpe, imt, var_type)

    for res_type in data.keys():

        x = _get_depths(residuals, gmpe, imt, res_type)
        slope, intercept, _, pval, _ = _nanlinregress(x, data[res_type])
        y = data[res_type]
            
        plot_data[res_type] = \
            {'x': x, 'y': y,
             'slope': slope, 'intercept': intercept, 'pvalue': pval,
             'xlabel': "Hypocentral Depth (km)", 'ylabel': "Z (%s)" % imt,
             'bin_midpoints': mean_res_df.x_data,
             'mean_res': mean_res_df[res_type],
             'sigma_res': sigma_res_df[res_type] }

    return plot_data


def _get_depths(residuals, gmpe, imt, res_type):
    """
    Returns an array of magnitudes equal in length to the number of
    residuals
    """
    depths = np.array([])
    for i, ctxt in enumerate(residuals.contexts):
        if res_type == "Inter event":
            nvals = np.ones(len(residuals.unique_indices[gmpe][imt][i]))
        else:
            nvals = np.ones(len(ctxt["Ctx"].repi))
        depths = np.hstack([depths, ctxt["Ctx"].hypo_depth * nvals])
    return depths


def _nanlinregress(x, y):
    """
    Calls scipy linregress only on finite numbers of x and y
    """
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        # empty arrays passed to linreg raise ValueError:
        # force returning an object with nans:
        return linregress([np.nan], [np.nan])
    return linregress(x[finite], y[finite])


### Utils for binning residuals w.r.t. a given GMM input variable ###
def get_binning_params(var_type, residuals, res_per_gmc, imts, distance_type):
    """
    Get the params for the binning of the given variable we are plotting
    the residuals with respect to
    """
    # Get values for given variable
    if var_type == 'magnitude':
        vals = _get_magnitudes(
            residuals, res_per_gmc, imts,'Total')
        val_bin = 0.25
    elif var_type == 'vs30':
        vals = _get_vs30(
            residuals, res_per_gmc, imts, 'Total')
        val_bin = 100
    elif var_type == 'distance':
        vals = _get_distances(
            residuals, res_per_gmc, imts, 'Total', distance_type)
        val_bin = 10     
    elif var_type == 'depth':
        vals = _get_depths(
            residuals, res_per_gmc, imts, 'Total')        
        val_bin = 5

    # Create bins and make last interval fill up to max var value
    val_bins = np.arange(np.min(vals), np.max(vals), val_bin)
    val_bins[len(val_bins) - 1] = np.max(vals)
    bins_bounds = {}
    for idx, val_bin in enumerate(val_bins):
        if idx == len(val_bins) - 1:
            pass
        else:
            bins_bounds[idx] = [val_bins[idx], val_bins[idx+1]]
        
    return bins_bounds, vals


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


def get_res_df(var_type, vals, res_per_imt, residuals, gmpe, imt, distance_type):
    """
    Return a dataframe with the total, inter-event and intra event
    residuals w.r.t. the variable of interest for plotting
    """
    total_res = res_per_imt['Total'].tolist()
    if 'Intra event' in res_per_imt:
        intra_res = res_per_imt['Intra event'].tolist()
        df = pd.DataFrame({
            'vals': vals, 'total_res': total_res, 'intra_res': intra_res})
    else:
        df = pd.DataFrame({
            'vals': vals, 'total_res': total_res}).sort_values(['vals'])
    df = df.sort_values(by="vals")

    if 'Inter event' in res_per_imt:
        # Inter-event is mean residual per eq but need to allocate to each rec
        store = []
        for idx_eq, ctx in enumerate(residuals.contexts):
        
            # Get values of the var for given the given ctx
            ctx_vals = get_ctx_vals(var_type, ctx["Ctx"], distance_type)

            # Get the Inter event residual (mean res per event)    
            inter_res = ctx['Residual'][gmpe][imt]['Inter event']

            # Get value per rec for given var_type into an array
            inter_res_val = np.full(len(inter_res), ctx_vals) 

            # Make a df for the given event
            eq_df = pd.DataFrame({"inter": inter_res, "vals": inter_res_val})

            store.append(eq_df)

        # Into a df for all eqs with inter res and required vals
        df_inter = pd.concat(store).sort_values(by="vals")

        # Into single df with total and intra res
        assert all(df_inter.vals.values == df.vals.values)
        df["inter_res"] = df_inter["inter"].values

    return df


def mean_sigma_per_bin(df, idx_res_per_val_bin, res_per_imt):
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

        total_mean[val_bin] = df_bin.total_res.mean()
        total_sigma[val_bin] = df_bin.total_res.std()

        if 'Intra event' in res_per_imt and 'Inter event' in res_per_imt:
            intra_mean[val_bin] = df_bin.intra_res.mean()
            inter_mean[val_bin] = df_bin.inter_res.mean()
            intra_sigma[val_bin] = df_bin.intra_res.std()
            inter_sigma[val_bin] = df_bin.inter_res.std()

    return {"total_mean": total_mean,
            "total_sigma": total_sigma,
            "intra_mean": intra_mean,
            "intra_sigma": intra_sigma,
            "inter_mean": inter_mean,
            "inter_sigma": inter_sigma}


def _get_mean_res_wrt_var(residuals, gmpe, imt, var_type, distance_type=None):
    """
    Compute mean total, inter-event and inter-event residual within bin for
    given variable. This is plotted within the scatter plots of residuals
    w.r.t. the given variable
    :param var_type: Specifies variable which residuals are plotted against
    """
    # If no distance type use repi
    if distance_type is None: distance_type = 'repi'
    res_per_gmc = residuals.residuals[gmpe]
    res_per_imt = residuals.residuals[gmpe][imt]
    imts = residuals.imts

    # Get bin bounds and values to be binned
    bins_bounds, vals = get_binning_params(
        var_type, residuals, res_per_gmc, imts, distance_type)

    # Get residuals and the variable (per record) in a dataframe
    df = get_res_df(
        var_type, vals, res_per_imt, residuals, gmpe, imt, distance_type)

    # Get indices for the residuals in each bin
    idx_res_per_val_bin = {idx: {} for idx in bins_bounds}
    for idx in bins_bounds:
        for data_point in df.index:
            if (df.vals.iloc[data_point] >= bins_bounds[idx][0]
                and
                df.vals.iloc[data_point] <= bins_bounds[idx][1]): 
                idx_res_per_val_bin[idx][data_point] = data_point

    # Get the mean and std per res assoc with each bin of the given var
    means_and_sigmas = mean_sigma_per_bin(df, idx_res_per_val_bin, res_per_imt)

    # Get midpoint of each val bin for plotting
    bin_mid_points = {val_bin: bounds[0] + 0.5 * (
        bounds[1] - bounds[0]) for val_bin, bounds in bins_bounds.items()}
    
    # Get final data to plot
    if 'Intra event' in res_per_imt and 'Inter event' in res_per_imt:
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