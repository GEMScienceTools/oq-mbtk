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
Module to manage GMPE residual plotting functions
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy
from math import floor, ceil
from scipy.stats import norm
from cycler import cycler

from openquake.hazardlib.imt import from_string
from openquake.hazardlib import valid
from openquake.smt.residuals.gmpe_residuals import Residuals, SingleStationAnalysis
from openquake.smt.residuals.residual_plotter_utils import (
                                                    _get_residuals_density_distribution,
                                                    residuals_with_magnitude,
                                                    residuals_with_vs30,
                                                    residuals_with_distance,
                                                    residuals_with_depth,
                                                    _get_residual_means_and_stds)


COLORS = ['r', 'g', 'b', 'y', 'lime', 'dodgerblue', 'gold', '0.8', 'm', 'k',
          'mediumseagreen', 'tab:orange', 'tab:purple', 'tab:brown', '0.5']


### General Utils
def manage_imts(residuals):
    """
    Removes the non-acceleration IMTs from the imts attribute of the residuals
    object and create an array of the remaining IMTs. This is a utility function
    used for plotting of GMM ranking metrics vs period.
    """
    # Preserve original residuals.imts
    preserve_imts = deepcopy(residuals.imts)

    # Remove IMTs if they are not PGA or (non-average) SA
    idx_to_drop = []
    for imt_idx, imt in enumerate(preserve_imts):
        if imt != 'PGA' and 'SA' not in imt or imt == "AvgSA":
            idx_to_drop.append(imt_idx)
    residuals.imts = pd.Series(preserve_imts).drop(idx_to_drop).values

    # Get ordinals for original IMTs
    x_with_imt = pd.DataFrame(
    [(from_string(imt).period, imt) for imt in preserve_imts], columns=['imt_float', 'imt_str']
    )
    return residuals, x_with_imt


class BaseResidualPlot(object):
    """
    Abstract-like class to create a Residual plot of strong ground motion
    residuals
    """
    # Class attributes passed to matplotlib xlabel, ylabel and title methods
    xlabel_styling_kwargs = dict(fontsize=12)
    ylabel_styling_kwargs = dict(fontsize=12)
    title_styling_kwargs = dict(fontsize=12)

    def __init__(self,
                 residuals,
                 gmpe,
                 imt,
                 filename,
                 **kwargs):
        """
        Initializes a BaseResidualPlot

        :param residuals:
            Residuals as instance of :class: openquake.smt.gmpe_residuals.Residuals
        :param str gmpe: Choice of GMPE
        :param str imt: Choice of IMT
        :param kwargs: optional keyword arguments.
        """
        self._assertion_check(residuals)
        self.residuals = residuals
        if gmpe not in residuals.gmpe_list:
            raise ValueError("No residual data found for GMPE %s" % gmpe)
        if imt not in residuals.imts:
            raise ValueError("No residual data found for IMT %s" % imt)
        if hasattr(residuals,'residuals') == True:
             if not residuals.residuals[gmpe][imt]:
                raise ValueError("No residuals found for %s (%s)" % (gmpe, imt))
        self.gmpe = gmpe
        self.imt = imt
        self.filename = filename
        self.num_plots = len(residuals.types[gmpe][imt])
        
        # Adjust plot aspect ratio if only total residual for GMPE
        if hasattr(residuals,'residuals') == True:
            if 'Inter event' and 'Intra event' not in residuals.residuals[
                    gmpe][imt] == True:
                self.figure_size = kwargs.get("figure_size",(8,6))
            else:
                self.figure_size = kwargs.get("figure_size",(8,8))
        elif hasattr(residuals,'site_residuals') == True:
            if 'Inter event' and 'Intra event' not in residuals.site_residuals[
                    0].residuals[gmpe][imt]:
                self.figure_size = kwargs.get("figure_size",(8,6))
            else:
                self.figure_size = kwargs.get("figure_size",(9,9))
            
        self.create_plot()

    def _assertion_check(self, residuals):
        """
        Checks that residuals is an instance of the residuals class
        """
        assert isinstance(residuals, Residuals)

    def create_plot(self):
        """
        Creates a residual plot
        """
        data = self.get_plot_data()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_layout_engine('tight')
        nrow, ncol = self.get_subplots_rowcols()
        for tloc, res_type in enumerate(data.keys(), 1):
            self._residual_plot(plt.subplot(nrow, ncol, tloc), data[res_type], res_type)
        plt.savefig(self.filename)
        plt.close()

    def _residual_plot(self, ax, res_data, res_type):
        """
        Plots the residual data on the given axis. This method should in
        principle not be overridden by sub-classes
        """
        self.draw(ax, res_data, res_type)
        ax.set_xlim(*self.get_axis_xlim(res_data, res_type))
        ax.set_ylim(*self.get_axis_ylim(res_data, res_type))
        ax.set_xlabel(res_data['xlabel'], **self.xlabel_styling_kwargs)
        ax.set_ylabel(res_data['ylabel'], **self.ylabel_styling_kwargs)
        title_string = self.get_axis_title(res_data, res_type)
        if title_string:
            ax.set_title(title_string, **self.title_styling_kwargs)

    def get_axis_xlim(self, res_data, res_type):
        """
        Sets the x-axis limit for each `Axes` object (sub-plot).
        This method can be overridden by subclasses, by default it returns
        None, None (i.e., automatic axis limit).

        :param res_data: the residual data to be plotted. It's one of
            the values of the dict returned by `self.get_plot_data`
            (`res_type` is the corresponding key): it is a dict with
            at least the mandatory keys 'x', 'y' (both numeric arrays),
            'xlabel' and 'ylabel' (both strings). Other keys, if present,
            should be handled by sub-classes implementation, if needed
        :param res_type: string denoting the residual type such as, e.g.
            "Inter event". It's one of the keys of the dict returned by
            `self.get_plot_data` (`res_data` is the corresponding value)

        :return: a numeric tuple denoting the axis minimum and maximum.
            None's are allowed and delegate matplotlib for calculating the
            limits
        """
        return None, None

    def get_axis_ylim(self, res_data, res_type):
        """
        Sets the y-axis limit for each plot.
        This method can be overridden by subclasses, by default it returns
        None, None (i.e., automatic axis limit).

        :param res_data: the residual data to be plotted. It's one of
            the values of the dict returned by `self.get_plot_data`
            (`res_type` is the corresponding key): it is a dict with
            at least the mandatory keys 'x', 'y' (both numeric arrays),
            'xlabel' and 'ylabel' (both strings). Other keys, if present,
            should be handled by sub-classes implementation, if needed
        :param res_type: string denoting the residual type such as, e.g.
            "Inter event". It's one of the keys of the dict returned by
            `self.get_plot_data` (`res_data` is the corresponding value)

        :return: a numeric tuple denoting the axis minimum and maximum.
            None's are allowed and delegate matplotlib for calculating the
            limits
        """
        return None, None

    def get_axis_title(self, res_data, res_type):
        """
        Sets the title for each plot.
        This method can be overridden by subclasses, by default it returns
        "" (i.e., no title).

        :param res_data: the residual data to be plotted. It's one of
            the values of the dict returned by `self.get_plot_data`
            (`res_type` is the corresponding key): it is a dict with
            at least the mandatory keys 'x', 'y' (both numeric arrays),
            'xlabel' and 'ylabel' (both strings). Other keys, if present,
            should be handled by sub-classes implementation, if needed
        :param res_type: string denoting the residual type such as, e.g.
            "Inter event". It's one of the keys of the dict returned by
            `self.get_plot_data` (`res_data` is the corresponding value)

        :return: a string denoting the axis title
        """
        return ""


class ResidualHistogramPlot(BaseResidualPlot):
    """
    Abstract-like class to create histograms of strong ground motion residuals
    """
    def __init__(self,
                 residuals,
                 gmpe,
                 imt,
                 filename,
                 bin_width=0.5,
                 **kwargs):
        """
        Initializes a ResidualHistogramPlot object. Sub-classes need to
        implement (at least) the method `get_plot_data`.

        All arguments not listed below are described in
        `BaseResidualPlot.__init__`.

        :param bin_width: float denoting the bin width of the histogram.
            defaults to 0.5
        """
        self.bin_width = bin_width
        super(ResidualHistogramPlot, self).__init__(residuals,
                                                    gmpe,
                                                    imt,
                                                    filename=filename,
                                                    **kwargs)

    def get_subplots_rowcols(self):
        if self.num_plots > 1:
            nrow = 3
            ncol = 1
        else:
            nrow = 1
            ncol = 1
        return nrow, ncol

    def draw(self, ax, res_data, res_type):
        bin_width = self.bin_width
        x, y = res_data['x'], res_data['y']
        ax.bar(x, y, width=0.95 * bin_width,
               color="LightSteelBlue", edgecolor="k")


class ResidualPlot(ResidualHistogramPlot):
    """
    Class to create a simple histrogram of strong ground motion residuals
    """
    def get_plot_data(self):
        return _get_residuals_density_distribution(
            self.residuals, self.gmpe, self.imt, self.bin_width)

    def draw(self, ax, res_data, res_type):
        # Draw histogram
        super(ResidualPlot, self).draw(ax, res_data, res_type)
        # Draw normal distributions
        mean = res_data["mean"]
        stddev = res_data["stddev"]
        x = res_data['x']
        xdata = np.arange(x[0], x[-1] + self.bin_width + 0.01, 0.01)
        xdata_norm_pdf = np.arange(-3,3,0.01)
        ax.plot(xdata,
                norm.pdf(xdata, mean, stddev),
                linestyle='-',
                color="LightSlateGrey",
                linewidth=2.0,
                label='Empirical')
        ax.plot(xdata_norm_pdf,
                norm.pdf(xdata_norm_pdf, 0.0, 1.0),
                linestyle='-',
                color='k',
                linewidth=2.0, 
                label='Standard. Norm. Dist.')
        ax.legend(loc='best', fontsize='xx-small')
        x_limit = max(abs(x))
        ax.set_xlim(x_limit*-1,x_limit)

    def get_axis_title(self, res_data, res_type):
        sigma_type = res_type
        if res_type == 'Total':
            sigma_type = 'Total Res.'
        elif res_type == 'Inter event':
            sigma_type = 'Between-Event Res.'
        elif res_type == 'Intra event':
            sigma_type = 'Within-Event Res.'
        
        mean, stddev = res_data["mean"], res_data["stddev"]
        return "%s - %s\n Mean = %7.3f, Std Dev = %7.3f" % (str(
            self.residuals.gmpe_list[self.gmpe]).split('(')[0].replace(
                ']\n', '] - ').replace('sigma_model','Sigma').replace(
                    'sigma_model','Sigma'),sigma_type,mean, stddev)


class ResidualScatterPlot(BaseResidualPlot):
    """
    Abstract-like class to create scatter plots of strong ground motion
    residuals
    """
    def __init__(self,
                 residuals,
                 gmpe,
                 imt,
                 filename,
                 plot_type='',
                 **kwargs):
        """
        Initializes a ResidualScatterPlot object. Sub-classes need to
        implement (at least) the method `get_plot_data`.

        All arguments not listed below are described in
        `BaseResidualPlot.__init__`.

        :param plot_type: string denoting if the plot x axis should be
            logarithmic (provide 'log' in case). Default: '' (no log x axis)
        """
        self.plot_type = plot_type
        super(ResidualScatterPlot, self).__init__(
            residuals, gmpe, imt, filename=filename, **kwargs)
        
    def get_subplots_rowcols(self):
        if self.num_plots > 1:
            nrow = 3
            ncol = 1
        else:
            nrow = 1
            ncol = 1
        return nrow, ncol

    def get_axis_xlim(self, res_data, res_type):
        x = res_data['x']
        return floor(np.min(x)), ceil(np.max(x))

    def get_axis_ylim(self, res_data, res_type):
        y = res_data['y']
        max_lim = ceil(np.nanmax(np.fabs(y)))
        return -max_lim, max_lim
    
    def get_axis_title(self, res_data, res_type):
        sigma_type = res_type
        if res_type == 'Total':
            sigma_type = 'Total Res.'
        elif res_type == 'Inter event':
            sigma_type = 'Between-Event Res.'
        elif res_type == 'Intra event':
            sigma_type = 'Within-Event Res.'
        return "%s - %s" %(str(self.residuals.gmpe_list[self.gmpe]).split('(')[
            0].replace(']\n', '] - ').replace('sigma_model','Sigma'),sigma_type)

    def draw(self, ax, res_data, res_type):
        x, y = res_data['x'], res_data['y']
        x_zero = np.arange(np.floor(np.nanmin(x))-20, np.ceil(np.nanmax(x))+20, 0.001)
        zero_line = np.zeros(len(x_zero))
        pts_styling_kwargs = dict(
            markeredgecolor='Gray', markerfacecolor='LightSteelBlue', label='residual')
        
        if self.plot_type == "log":
            ax.semilogx(x, y, 'o', **pts_styling_kwargs)
            
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'],
                       marker='s', color='b', label='mean', zorder=4)
            
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'] + (
                -1*res_data['sigma_res']), marker='x', color='b', zorder=4)
            
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'] + (
                res_data['sigma_res']), marker='x', color='b',
                label='+/- 1 Std.', zorder=4)
            
            ax.plot(x_zero, zero_line, color='k', linestyle='--',
                    linewidth=1.25)
        else:
            ax.plot(x, y, 'o', **pts_styling_kwargs)
            
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'],
                       marker='s', color='b', label='mean', zorder=4)
        
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'] + (
                -1*res_data['sigma_res']), marker='x', color='b', zorder=4)
        
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'] + (
                res_data['sigma_res']), marker='x', color='b',
                label='+/- 1 Std.', zorder=4)
            
            ax.plot(x_zero, zero_line, color='k', linestyle='--',
                    linewidth=1.25)
            
        ax.legend(loc='upper right', fontsize='xx-small')


class ResidualWithDistance(ResidualScatterPlot):
    """
    Class to create a simple scatter plot of strong ground motion
    residuals (y-axis) versus distance (x-axis)
    """
    def __init__(self,
                 residuals,
                 gmpe, 
                 imt,
                 filename,
                 plot_type='linear',
                 distance_type="rjb",
                 **kwargs):
        """
        Initializes a ResidualWithDistance object

        All arguments not listed below are described in
        `ResidualScatterPlot.__init__`. Note that `plot_type` is 'log' by
        default.

        :param distance_type: string denoting the distance type to be
            used. Defaults to 'rjb'
        """
        self.distance_type = distance_type
        super(ResidualWithDistance, self).__init__(residuals, gmpe, imt,
                                                   filename=filename,
                                                   plot_type=plot_type,
                                                   **kwargs)

    def get_plot_data(self):
        return residuals_with_distance(self.residuals,
                                       self.gmpe,
                                       self.imt,
                                       self.distance_type)

    def get_axis_xlim(self, res_data, res_type):
        x = res_data['x']
        if self.plot_type == "log":
            return 0.1, 10.0 ** (ceil(np.log10(np.nanmax(x))))
        else:
            if self.distance_type == "rcdpp":
                return np.nanmin(x), np.nanmax(x)
            else:
                return 0, np.nanmax(x)


class ResidualWithMagnitude(ResidualScatterPlot):
    """
    Class to create a simple scatter plot of strong ground motion
    residuals (y-axis) versus magnitude (x-axis)
    """
    def get_plot_data(self):
        return residuals_with_magnitude(self.residuals, self.gmpe, self.imt)


class ResidualWithDepth(ResidualScatterPlot):
    """
    Class to create a simple scatter plot of strong ground motion
    residuals (y-axis) versus depth (x-axis)
    """
    def get_plot_data(self):
        return residuals_with_depth(self.residuals, self.gmpe, self.imt)


class ResidualWithVs30(ResidualScatterPlot):
    """
        Class to create a simple scatter plot of strong ground motion
        residuals (y-axis) versus Vs30 (x-axis)
    """
    def get_plot_data(self):
        return residuals_with_vs30(self.residuals, self.gmpe, self.imt)

    def get_axis_xlim(self, res_data, res_type):
        x = res_data['x']
        return np.min(x)-20, np.max(x)+20


### Plotting of ranking metrics vs period
def plot_llh_with_period(residuals, filename):
    """
    Create a simple plot of loglikelihood values of Scherbaum et al. 2009
    (y-axis) versus period (x-axis)
    """
    # Check have computed LLH
    if not hasattr(residuals, "llh"):
        raise ValueError("The user must first compute LLH.")

    # Check enough IMTs to plot w.r.t. period
    if len(residuals.imts) == 1:
        raise ValueError('Cannot plot w.r.t. period (only 1 IMT).')
                
    # Manage IMTs
    residuals, x_llh = manage_imts(residuals)

    # Define colours for GMMs
    colour_cycler = (cycler(color=COLORS)*cycler(linestyle=['-']))
        
    # Plot LLH values w.r.t. period
    llh_with_imt = pd.DataFrame(residuals.llh).drop('all')
    fig_llh, ax_llh = plt.subplots(figsize=(10, 8))
    ax_llh.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        y_llh = np.array(llh_with_imt[gmpe])
        ax_llh.scatter(x_llh.imt_float, y_llh)
        tmp = str(residuals.gmpe_list[gmpe]).split('(')[0]
        ax_llh.plot(x_llh.imt_float, y_llh, label=tmp)
    ax_llh.set_xlabel('Period (s)', fontsize='12')
    ax_llh.set_ylabel('LLH', fontsize='12')
    ax_llh.legend(loc='upper right', ncol=2, fontsize='12')
    plt.savefig(filename)
    plt.close()
    

def plot_edr_with_period(residuals, filename):
    """
    Create plots of EDR, the median pred. correction factor and normalised MDE
    computed using Kale and Akkar (2013) (y-axis) versus period (x-axis)
    """
    # Check have computed EDR
    if not hasattr(residuals, "edr_values_wrt_imt"):
        raise ValueError("The user must first compute EDR.")

    # Check enough IMTs to plot w.r.t. period
    if len(residuals.imts) == 1:
        raise ValueError('Cannot plot w.r.t. period (only 1 IMT).')
    
    # Manage IMTs
    residuals, x_with_imt = manage_imts(residuals)

    # Define colours for GMMs
    colour_cycler = (cycler(color=COLORS)*cycler(linestyle=['-']))
    
    # Plot EDR w.r.t. period
    EDR_with_imt = {}
    fig_EDR, ax_EDR = plt.subplots(figsize=(10, 8))
    ax_EDR.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        EDR_with_imt = pd.DataFrame(residuals.edr_values_wrt_imt[gmpe])
        y_EDR = EDR_with_imt.EDR
        tmp = str(residuals.gmpe_list[gmpe]).split('(')[0]
        ax_EDR.scatter(x_with_imt.imt_float, y_EDR)
        ax_EDR.plot(x_with_imt.imt_float, y_EDR, label=tmp)
    ax_EDR.set_xlabel('Period (s)', fontsize='12')
    ax_EDR.set_ylabel('EDR', fontsize='12')
    ax_EDR.legend(loc = 'upper right', ncol=2, fontsize=12)
    parts = filename.split(".")
    plt.savefig(parts[0] + "_value." + parts[1])
    plt.close()

    # Plot median pred. correction factor w.r.t. period
    kappa_with_imt = {}
    fig_kappa, ax_kappa = plt.subplots(figsize=(10, 8))
    ax_kappa.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        kappa_with_imt = pd.DataFrame(residuals.edr_values_wrt_imt[gmpe])
        y_kappa = kappa_with_imt["sqrt Kappa"]
        tmp = str(residuals.gmpe_list[gmpe]).split('(')[0]
        ax_kappa.scatter(x_with_imt.imt_float, y_kappa)
        ax_kappa.plot(x_with_imt.imt_float, y_kappa, label=tmp)
    ax_kappa.set_xlabel('Period (s)', fontsize='12')
    ax_kappa.set_ylabel('sqrt(k)', fontsize='12')
    ax_kappa.legend(loc = 'upper right', ncol=2, fontsize=12)
    plt.savefig(parts[0] + "_kappa." + parts[1])
    plt.close()

    # Plot MDE w.r.t. period
    MDE_with_imt = {}
    fig_MDE, ax_MDE = plt.subplots(figsize=(10, 8))
    ax_MDE.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        MDE_with_imt = pd.DataFrame(residuals.edr_values_wrt_imt[gmpe])
        y_MDE = MDE_with_imt["MDE Norm"]
        tmp = str(residuals.gmpe_list[gmpe]).split('(')[0]
        ax_MDE.scatter(x_with_imt.imt_float, y_MDE)
        ax_MDE.plot(x_with_imt.imt_float, y_MDE, label=tmp)
    ax_MDE.set_xlabel('Period (s)', fontsize='12')
    ax_MDE.set_ylabel('MDE Norm', fontsize='12')
    ax_MDE.legend(loc = 'upper right', ncol=2, fontsize=12)
    plt.savefig(parts[0] + "_MDE." + parts[1])
    plt.close()


def plot_sto_with_period(residuals, filename):
    """
    Definition to create plot of the stochastic area metric
    computed using Sunny et al. (2021) versus period (x-axis)
    """
    # Check have computed Stochastic Area
    if not hasattr(residuals, "stoch_areas_wrt_imt"):
        raise ValueError("The user must first compute Stochastic Area.")

    # Check enough IMTs to plot w.r.t. period
    if len(residuals.imts) == 1:
        raise ValueError('Cannot plot w.r.t. period (only 1 IMT).')
    
    # Manage IMTs
    residuals, x_with_imt = manage_imts(residuals)
    
    # Define colours for plots
    colour_cycler = (cycler(color=COLORS)*cycler(linestyle=['-']))
    
    # Plot stochastic area w.r.t. period
    sto_with_imt = {}
    fig_sto, ax_sto = plt.subplots(figsize=(10, 8))
    ax_sto.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        sto_with_imt = pd.Series(residuals.stoch_areas_wrt_imt[gmpe])
        y_sto = sto_with_imt.values
        tmp = str(residuals.gmpe_list[gmpe]).split('(')[0]
        ax_sto.scatter(x_with_imt.imt_float, y_sto)
        ax_sto.plot(x_with_imt.imt_float, y_sto, label=tmp)
    ax_sto.set_xlabel('Period (s)', fontsize='12')
    ax_sto.set_ylabel('Stochastic Area', fontsize='12')
    ax_sto.legend(loc='upper right', ncol=2, fontsize=12)
    plt.savefig(os.path.join(filename))
    plt.close()


### Functions for exporting tables of ranking metrics
def llh_table(residuals, filename):
    """
    Create a table of loglikelihood values per gmpe per
    imt (Scherbaum et al. 2009)
    """
    # Check have computed LLH
    if not hasattr(residuals, "llh"):
        raise ValueError("The user must first compute LLH.")
    
    # Get LLH per GMM per IMT
    llh_metrics = pd.DataFrame()
    for gmpe in residuals.gmpe_list:
        llh_metrics["LLH " + gmpe] = residuals.llh[gmpe]

    # Export table
    llh_metrics.to_csv(filename, sep=',')


def llh_weights(residuals, filename=None):
    """
    Create a table of model weights per gmpe per imt based on sample
    loglikelihood (Scherbaum et al. 2009).
    """       
    # Check have computed LLH
    if not hasattr(residuals, "llh"):
        raise ValueError("The user must first compute LLH.")

    # Get LLH weight per GMM per IMT
    llh_df = pd.DataFrame(residuals.llh)[list(residuals.gmpe_list)]
    weights = 2.0 ** -llh_df
    weights = weights.div(weights.sum(axis=1), axis=0)
    residuals.llh_weights = weights.to_dict(orient='index')
    llh_weights = pd.DataFrame(residuals.llh_weights)
    llh_weights = llh_weights.T  # GMMs as cols, IMTs as index

    # Get weight per GMM averaged over the IMTs
    llh_weights.loc['Avg over imts'] = llh_weights.mean(axis=0)
    llh_weights.columns = llh_weights.columns + ' LLH-based weight'
    assert np.abs(llh_weights.loc['Avg over imts'].sum() - 1.0) < 1E-09

    # Export table if required (might just want the weights)
    if filename is not None:
        llh_weights.to_csv(filename, sep=',')

    # Add llh weights to residuals obj
    setattr(residuals, "llh_weights", llh_weights)
    

def edr_table(residuals, filename):
    """
    Create a table of MDE Norm, sqrt(kappa) and EDR gmpe per imt (Kale and Akkar,
    2013)
    """
    # Check have computed EDR
    if not hasattr(residuals, "edr_values_wrt_imt"):
        raise ValueError("The user must first compute EDR.")

    # Get Kale and Akkar (2013) ranking metrics
    edr_dfs = []
    for gmpe in residuals.gmpe_list:
        col = {'MDE Norm':str(gmpe) + ' MDE Norm',
               'sqrt Kappa':str(gmpe) + ' sqrt Kappa',
               'EDR': str(gmpe) + ' EDR'}
        edr = pd.DataFrame(residuals.edr_values_wrt_imt[gmpe]).rename(col)
        means = []
        for metric in edr.columns: # Get average values over IMTs
            mean = edr[metric].mean()
            means.append(mean)
        edr.loc['Avg over imts'] = means
        edr.columns = edr.columns + ' ' + gmpe
        edr_dfs.append(edr)

    # Into final df
    edr_df = pd.concat(edr_dfs, axis=1)
    edr_df.to_csv(filename, sep=',')


def edr_weights(residuals, filename=None):
    """
    Create a table of model weights per imt based on Euclidean
    distance based ranking (Kale and Akkar, 2013)
    """ 
    # Check have computed EDR
    if not hasattr(residuals, "edr_values_wrt_imt"):
        raise ValueError("The user must first compute EDR.")

    # Get the EDR values from the residuals object
    edr_for_weights = residuals.edr_values_wrt_imt

    # Compute EDR based model weights
    edr_per_gmpe = pd.DataFrame({
        gmpe: edr_for_weights[gmpe]['EDR'] for gmpe in edr_for_weights})

    # Get weight per GMM per IMT
    edr_inv = edr_per_gmpe ** -1
    edr_weight = edr_inv.div(edr_inv.sum(axis=1), axis=0)
    
    # Get weight per GMM averaged over the IMTs
    avg_edr_weight = edr_weight.mean().to_frame().T
    avg_edr_weight.index = ['Avg over imts']
    
    # Into final df    
    edr_weights = pd.concat([edr_weight, avg_edr_weight])
    edr_weights.columns = edr_weights.columns + ' EDR-based weight'
    assert np.abs(edr_weights.loc['Avg over imts'].sum() - 1.0) < 1E-09

    # Export table if required (might just want the weights)
    if filename is not None:
        edr_weights.to_csv(filename, sep=',')

    # Add edr weights to residuals obj
    setattr(residuals, "edr_weights", edr_weights)


def sto_table(residuals, filename):
    """
    Create a table of stochastic area ranking metric per GMPE per imt (Sunny et
    al. 2021)
    """
    # Check have computed Stochastic Area
    if not hasattr(residuals, "stoch_areas_wrt_imt"):
        raise ValueError("The user must first compute Stochastic Area.")

    # Get stochastic area value per GMM per IMT
    sto_metrics = pd.DataFrame(residuals.stoch_areas_wrt_imt)
    sto_metrics.loc['Avg over imts'] = sto_metrics.mean()
    sto_metrics.columns = "STO " + sto_metrics.columns

    # Export table
    sto_metrics.to_csv(filename, sep=',')


def sto_weights(residuals, filename=None):
    """
    Create a table of model weights per imt based on sample stochastic area
    (Sunny et al. 2021))
    """       
    # Check have computed Stochastic Area
    if not hasattr(residuals, "stoch_areas_wrt_imt"):
        raise ValueError("The user must first compute Stochastic Area.")

    # Get required values
    sto_per_gmpe = pd.DataFrame(residuals.stoch_areas_wrt_imt)
    
    # Get weight per GMM per IMT
    sto_inv = sto_per_gmpe ** -1
    sto_weight = sto_inv.div(sto_inv.sum(axis=1), axis=0)

    # Get weight per GMM averaged over the IMTs
    avg_sto_weight_per_gmpe = {gmpe: np.mean(
        sto_weight[gmpe]) for gmpe in residuals.gmpe_list}

    # Into final df
    avg_sto_weights = pd.DataFrame(
        avg_sto_weight_per_gmpe, index=['Avg over imts']
        )
    sto_weights = pd.concat([sto_weight, avg_sto_weights])
    sto_weights.columns = sto_weights.columns + " STO-based weight"
    assert np.abs(sto_weights.loc['Avg over imts'].sum() - 1.0) < 1E-09

    # Export if required (might just want the weights)
    if filename is not None:
        sto_weights.to_csv(filename, sep=',')

    # Add stochastic area weights to residuals obj
    setattr(residuals, "sto_weights", sto_weights)


### Functions for plotting mean and sigma of residual dists vs period
def _set_residuals_means_and_stds_plots(residuals, res_dists, imts_to_plot):
    """
    Set the plots for the means and std devs of each residual component
    per gmpe vs period
    """
    # Create figure
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(14, 14)) 

    # Plot mean of zero and sigma of 1 for standard normal dist
    for ax_idx in range(0, 3):
        ax[ax_idx, 0].plot(imts_to_plot.imt_float, np.zeros(len(imts_to_plot)),
                          color = 'k', linestyle = '--')
        ax[ax_idx, 1].plot(imts_to_plot.imt_float, np.ones(len(imts_to_plot)),
                          color = 'k', linestyle = '--')
    
    # Define colour per GMM
    colour_cycler = (cycler(color=COLORS)*cycler(marker=['x']))
    colour_cycler_df = pd.DataFrame(colour_cycler)[:len(residuals.gmpe_list)]
    colour_cycler_df['gmpe'] = residuals.gmpe_list.keys()

    # Set axes limits and axes labels
    means = np.concatenate([res_dists[0].loc['Mean'],
                            res_dists[1].loc['Mean'],
                            res_dists[2].loc['Mean']])
    sigmas = np.concatenate([res_dists[0].loc['Std Dev'],
                             res_dists[1].loc['Std Dev'],
                             res_dists[2].loc['Std Dev']])
    mean_y_bound = np.max([np.abs(np.min(means)), np.abs(np.max(means))])
    sigma_y_bound_non_centered = np.max(
        [np.abs(np.max(sigmas)), np.abs(np.max(sigmas))])
    sigma_y_bound = min(np.abs(1-sigma_y_bound_non_centered),
                        np.abs(1+sigma_y_bound_non_centered))
    for ax_index in range(0, 3):
        ax[ax_index, 0].set_ylim(-mean_y_bound-0.5, mean_y_bound+0.5)
        ax[ax_index, 1].set_ylim(0.9-sigma_y_bound, 1.1+sigma_y_bound)
        ax[ax_index, 0].set_xlabel('Period (s)', fontsize=12)
        ax[ax_index, 1].set_xlabel('Period (s)', fontsize=12)
        ax[ax_index, 0].set_prop_cycle(colour_cycler)
        ax[ax_index, 1].set_prop_cycle(colour_cycler)
    for ax_index in range(0, 2):
        ax[2, ax_index].set_ylabel('Within-Event', fontsize=12)
        ax[1, ax_index].set_ylabel('Between-Event', fontsize=12)
        ax[0, ax_index].set_ylabel('Total', fontsize=12)
    ax[0, 0].set_title('Mean of GMPE Residuals', fontsize=12)
    ax[0, 1].set_title('Std Dev of GMPE Residuals', fontsize=12)

    return fig, ax


def plot_residual_means_and_stds(
        ax,
        res_dists,
        mean_or_std,
        gmpe,
        imts_to_plot,
        marker_inp,
        color_inp):
    """
    Plot means or sigmas for given GMPE.
    """
    # Get axes index
    if mean_or_std == 'Mean':
        i = 0
    elif mean_or_std == 'Std Dev':
        i = 1

    # Get gmpe label
    if '_toml=' in gmpe:
        sqs  = re.findall(r'\[[^\]]+\]', gmpe)
        for sq in sqs:
            try:
                valid.gsim(sq) # Must be the gmm 
                gmpe_label = sq
                break
            except Exception:
                continue
    else:
        gmpe_label = gmpe # If not from toml file

    # Plot mean
    if (res_dists[2][gmpe].loc[mean_or_std].all()==0 and
        res_dists[1][gmpe].loc[mean_or_std].all()==0):
        
        ax[2, i].scatter(imts_to_plot.imt_float,
                         res_dists[0][gmpe].loc[mean_or_std],
                         color='w',
                         marker=marker_inp,
                         zorder=0)
        ax[1, i].scatter(imts_to_plot.imt_float,
                         res_dists[1][gmpe].loc[mean_or_std],
                         color='w',
                         marker=marker_inp,
                         zorder=0)
    else:
        ax[2, i].scatter(imts_to_plot.imt_float,
                         res_dists[0][gmpe].loc[mean_or_std],
                         color=color_inp,
                         marker=marker_inp)
        ax[1, i].scatter(imts_to_plot.imt_float,
                         res_dists[1][gmpe].loc[mean_or_std],                           
                         color=color_inp,
                         marker=marker_inp)
        
    ax[0, i].scatter(imts_to_plot.imt_float,
                     res_dists[2][gmpe].loc[mean_or_std],
                     label=gmpe_label,
                     color=color_inp,
                     marker=marker_inp)
    
    return ax


def plot_residual_means_and_stds_with_period(residuals, filename):
    """
    Create a simple plot of residual mean and residual sigma
    for each GMPE  (y-axis) versus period (x-axis)
    """
    # Check enough IMTs to plot w.r.t. period
    if len(residuals.imts) == 1:
        raise ValueError('Cannot plot w.r.t. period (only 1 IMT).')
        
    # Manage IMTs
    residuals, imts_to_plot = manage_imts(residuals)
            
    # Get distributions of residuals per gmm and imt 
    res_dists = _get_residual_means_and_stds(residuals)

    # Set plots
    fig, ax = _set_residuals_means_and_stds_plots(residuals, res_dists, imts_to_plot)

    # Define colours for GMPEs
    colour_cycler = (cycler(color=COLORS)*cycler(marker=['x']))
    colour_cycler_df = pd.DataFrame(colour_cycler)[:len(residuals.gmpe_list)]
    colour_cycler_df['gmpe'] = residuals.gmpe_list.keys()

    # Plot data
    for gmpe in residuals.gmpe_list.keys():

        # Assign colour and marker to each gmpe
        input_df = pd.DataFrame(
            colour_cycler_df.loc[colour_cycler_df['gmpe']==gmpe]).reset_index()
        color_inp = input_df['color'].iloc[0]
        marker_inp = input_df['marker'].iloc[0]
        
        # Plot means
        ax = plot_residual_means_and_stds(
            ax, res_dists, "Mean", gmpe, imts_to_plot, marker_inp, color_inp)
       
        # Plot sigma
        ax = plot_residual_means_and_stds(
            ax, res_dists, "Std Dev", gmpe, imts_to_plot, marker_inp, color_inp)
        
    # Add grid to each axis
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].grid()

    # Add legend
    ax[0, 0].legend(loc='upper right', ncol=2, fontsize=8)

    plt.savefig(filename)
    plt.close()


def residual_means_and_stds_table(residuals, filename):
    """
    Create a table of mean and standard deviation for total, inter-event and 
    intra-event residual distributions
    """
    # Retrieve mean and stddev for each 
    stats = {}
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            stats[gmpe, imt] = residuals.get_residual_statistics_for(gmpe, imt)
    
    mean_sigma_intra, mean_sigma_inter, mean_sigma_total = {}, {}, {}
    dummy_values = {'Mean': 'Total sigma only', 'Std Dev': 'Total sigma only'}
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            mean_sigma_total[gmpe, imt] = stats[gmpe, imt]['Total']
            if ('Intra event' in residuals.residuals[gmpe][imt] and
                'Inter event' in residuals.residuals[gmpe][imt]):
                mean_sigma_intra[gmpe, imt] = stats[gmpe, imt]['Intra event']
                mean_sigma_inter[gmpe, imt] = stats[gmpe, imt]['Inter event']
            else:
                mean_sigma_intra[gmpe, imt] = dummy_values
                mean_sigma_inter[gmpe, imt] = dummy_values

    mean_sigma_intra_df = pd.DataFrame(mean_sigma_intra)
    mean_sigma_inter_df = pd.DataFrame(mean_sigma_inter)
    mean_sigma_total_df = pd.DataFrame(mean_sigma_total)
    
    combined_df = pd.concat(
        [mean_sigma_total_df, mean_sigma_inter_df, mean_sigma_intra_df])
    combined_df.index = ['Total Mean', 'Total Std Dev',
                         'Inter-Event Mean', 'Inter-Event Std Dev',
                         'Intra-Event Mean', 'Intra-Event Std Dev']
    
    combined_df.to_csv(filename, sep=',')


### Plotting of single station residual analysis results
class ResidualWithSite(ResidualPlot):
    """
    Plot (normalised) total, inter-event and intra-event single-station
    residuals for the site selection, GMPE and intensity measure considered
    """
    def _assertion_check(self, residuals):
        """
        Checks that residuals is an instance of the residuals class
        """
        assert isinstance(residuals, SingleStationAnalysis)
    
    def create_plot(self):
        """
        Create residuals with site plot
        """
        data = self._get_site_data()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_layout_engine('tight')
        if self.num_plots > 1:
            nrow = 3
            ncol = 1
        else:
            nrow = 1
            ncol = 1
        tloc = 1
        for res_type in self.residuals.types[self.gmpe][self.imt]:
            self._residual_plot(
                fig.add_subplot(nrow, ncol, tloc),
                data,
                res_type)
            tloc += 1
        plt.savefig(self.filename)
        plt.close()

    def _residual_plot(self, ax, data, res_type):
        """
        Plot residuals per site
        """
        xmean = np.array([data[site_id]["x-val"][0]
                          for site_id in self.residuals.site_ids])

        yvals = np.array([])
        xvals = np.array([])
        for site_id in self.residuals.site_ids:
            xvals = np.hstack([xvals, data[site_id]["x-val"]])
            yvals = np.hstack([yvals, data[site_id][res_type]])
        ax.scatter(xvals,
                   yvals,
                   marker='o',
                   edgecolor='Gray',
                   facecolor='LightSteelBlue',
                   zorder=-32)
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        xtick_label = self.residuals.site_ids
        ax.set_xticklabels(xtick_label, rotation="vertical")
        
        sigma_type = res_type
        if res_type == 'Total':
            sigma_type = 'Total Res.'
        elif res_type == 'Inter event':
            sigma_type = 'Within-Event Res.'
        elif res_type == 'Intra event':
            sigma_type = 'Between-Event Res.'
        
        max_lim = ceil(np.max(np.fabs(yvals)))
        ax.set_ylim(-max_lim, max_lim)
        ax.set_ylabel("%s" % sigma_type, fontsize=12)
        ax.grid()
        title_string = "%s - %s - %s" % (str(self.residuals.gmpe_list[
            self.gmpe]).split('(')[0].replace(']\n', '] - ').replace(
                'sigma_model','Sigma'),self.imt,sigma_type)
        ax.set_title(title_string, fontsize=11)

    def _get_site_data(self):
        """
        Get single station analysis residual data
        """
        data = {site_id: {} for site_id in self.residuals.site_ids}
        for iloc, site_resid in enumerate(self.residuals.site_residuals):
            resid = deepcopy(site_resid)
            site_id = list(self.residuals.site_ids)[iloc]
            n_events = resid.site_analysis[self.gmpe][self.imt]["events"]
            total_res = resid.site_analysis[self.gmpe][self.imt]["Total"]
            total_exp = resid.site_analysis[self.gmpe][self.imt]["Expected total"]
            data[site_id]["Total"] = np.array(total_res) / np.array(total_exp)
            if "Intra event" in resid.site_analysis[self.gmpe][self.imt].keys():
                inter_res = resid.site_analysis[self.gmpe][self.imt]["Inter event"] 
                intra_res = resid.site_analysis[self.gmpe][self.imt]["Intra event"] 
                inter_exp = resid.site_analysis[self.gmpe][self.imt]["Expected inter"]
                intra_exp = resid.site_analysis[self.gmpe][self.imt]["Expected intra"]
                keep = pd.notnull(inter_res) # Dropping NaN idxs will realign with exp
                data[site_id]["Inter event"] = np.array(inter_res)[keep] / np.array(inter_exp)
                data[site_id]["Intra event"] = np.array(intra_res) / np.array(intra_exp)
            data[site_id]["ID"] = list(self.residuals.site_ids)[iloc]
            data[site_id]["N"] = n_events
            data[site_id]["x-val"] = (float(iloc) + 0.5) * np.ones_like(data[site_id]["Total"])

        return data


class IntraEventResidualWithSite(ResidualPlot):
    """
    Create plots of intra-event residual components for the site selection,
    GMPEs and intensity measures considered
    """     
    def _assertion_check(self, residuals):
        """
        Checks that residuals is an instance of the residuals class
        """
        assert isinstance(residuals, SingleStationAnalysis)
    
    def create_plot(self):
        """
        Creates the plot
        """
        if 'Intra event' in self.residuals.site_residuals[0].residuals[self.gmpe][
                self.imt]:
            # Get data
            self.residuals.station_residual_statistics()
            mean_deltaS2S = self.residuals.mean_deltaS2S
            phi_ss = self.residuals.phi_ss
            phi_S2S = self.residuals.phi_S2S
            data = self._get_site_data()

            # Make plot
            fig = plt.figure(figsize=self.figure_size)
            fig.set_layout_engine('tight')
            self._residual_plot(fig,
                                data,
                                mean_deltaS2S[self.gmpe][self.imt],
                                phi_ss[self.gmpe][self.imt],
                                phi_S2S[self.gmpe][self.imt]
                                )
            plt.savefig(self.filename)
            plt.close()
        else:
            warnings.warn('This implementation of %s GMPE does not have a mixed'
                          ' effects sigma model - plotting skipped' % self.gmpe,
                          stacklevel=10)

    def _residual_plot(self, fig, data, mean_deltaS2S, phi_ss, phi_S2S):
        """
        Creates three plots:
        1) Plot of the intra-event residual per record at each station
        2) Plot of the site term (average intra-event per site)
        3) Plot of the remainder-residual (intra per rec minus avg intra per site)
        """
        deltaW_es, deltaS2S_s, deltaWS_es = np.array([]), np.array([]), np.array([])
        xvals = np.array([])
        for site_id in self.residuals.site_ids:
            xvals = np.hstack([xvals, data[site_id]["x-val"]])
            deltaW_es = np.hstack([deltaW_es, data[site_id]["Intra event"]])
            deltaS2S_s = np.hstack([deltaS2S_s, data[site_id]["deltaS2S_s"]])
            deltaWS_es = np.hstack([deltaWS_es, data[site_id]["deltaWS_es"]])
        ax = fig.add_subplot(311)
        
        # Plot intra-event residuals for given site
        mean = np.array(
            [np.mean(data[site_id]["Intra event"]) for site_id in self.residuals.site_ids])
        stddevs = np.array( # i.e. phi
            [np.std(data[site_id]["Intra event"]) for site_id in self.residuals.site_ids])
        xmean = np.array(
            [data[site_id]["x-val"][0] for site_id in self.residuals.site_ids])

        ax.plot(xvals, deltaW_es,
                'x', markeredgecolor='k', markerfacecolor='k', markersize=8,
                zorder=-32, label=r'$\delta W_{es}$')
        
        ax.errorbar(xmean,
                    mean,
                    yerr=stddevs,
                    ecolor="r",
                    elinewidth=3.0,
                    barsabove=True,
                    fmt="s",
                    mfc="r",
                    mec="k",
                    ms=6,
                    label='Error bar')
        
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        xtick_label = self.residuals.site_ids
        ax.set_xticklabels(xtick_label, rotation="vertical")
        
        max_lim = ceil(np.max(np.fabs(deltaW_es)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta W_{es}$ (%s)' % self.imt, fontsize=12)

        phi = np.std(deltaW_es)
        nxv = np.ones(len(xvals))
        ax.plot(xvals, phi * nxv, 'k--', linewidth=2.)
        ax.plot(xvals, -phi * nxv, 'k--', linewidth=2, label=r'+/- $\phi$') # Not strictly phi because here
                                                                            # not computing on per-eq basis

        title_string = "%s - %s (Std Dev = %8.5f)" % (str(
            self.residuals.gmpe_list[self.gmpe]).split('(')[0].replace(
                ']\n', '] - ').replace('sigma_model','Sigma'), self.imt, phi)
        ax.set_title(title_string, fontsize=11)
        ax.legend(loc='upper right', fontsize=12)
        
        # Plot delta s2ss (avg intra-event per site)
        ax = fig.add_subplot(312)
        nxm = np.ones(len(xmean))
        
        ax.plot(xmean, deltaS2S_s,
                's', markeredgecolor='k', markerfacecolor='LightSteelBlue', markersize=8,        
                zorder=-32, label=r'$\delta S2S_S$')
        
        ax.plot(
            xmean, (mean_deltaS2S - phi_S2S) * nxm, "k--", linewidth=1.5
            )
        
        ax.plot(
            xmean, (mean_deltaS2S + phi_S2S) * nxm,
            "k--", linewidth=1.5, label=r'+/- $\phi_{S2S}$'
            )
        
        ax.plot(
            xmean, mean_deltaS2S * nxm,
            "k-", linewidth=2, label=r'Mean $\phi_{S2S}$'
            )
        
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(xtick_label, rotation="vertical")
        
        max_lim = ceil(np.max(np.fabs(deltaS2S_s)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta S2S_S$ (%s)' % self.imt, fontsize=12)
        
        title_string = r'%s - %s ($\phi_{S2S}$ = %8.5f)' % (str(
            self.residuals.gmpe_list[self.gmpe]).split('(')[0].replace(
                ']\n', '] - ').replace('sigma_model','Sigma'),
            self.imt, phi_S2S)
        ax.set_title(title_string, fontsize=11)
        ax.legend(loc='upper right', fontsize=12)
        
        # Plot deltaWS_es (remainder residual)
        ax = fig.add_subplot(313)
        
        ax.plot(xvals, deltaWS_es, 'x', markeredgecolor='k', markerfacecolor='k',
                markersize=8, zorder=-32, label=r'$\delta W_{o,es}$')
        
        ax.plot(xmean, -phi_ss * nxm, "k--", linewidth=1.5)
        
        ax.plot(xmean, phi_ss * nxm, "k--", linewidth=1.5, label=r'+/- $\phi_{SS}$')
        
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(xtick_label, rotation="vertical")
        
        max_lim = ceil(np.max(np.fabs(deltaWS_es)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta W_{o,es} = \delta W_{es} - \delta S2S_S$', fontsize=12)
        
        title_string = r'%s - %s ($\phi_{SS}$ = %8.5f)' % (str(
            self.residuals.gmpe_list[self.gmpe]).split('(')[0].replace(
                ']\n', '] - ').replace('sigma_model','Sigma'), self.imt, phi_ss)
        ax.set_title(title_string, fontsize=11)
        ax.legend(loc='upper right', fontsize=12)
        
    def _get_site_data(self):
        """
        Get site-specific intra-event residual components for each site for the
        GMPEs and intensity measures considered
        """
        data = {site_id: {} for site_id in self.residuals.site_ids}
        for iloc, site_resid in enumerate(self.residuals.site_residuals):
            resid = deepcopy(site_resid)
            site_id = list(self.residuals.site_ids)[iloc]
            n_events = resid.site_analysis[self.gmpe][self.imt]["events"]
            data[site_id] = resid.site_analysis[self.gmpe][self.imt]
            data[site_id]["ID"] = list(self.residuals.site_ids)[iloc]
            data[site_id]["N"] = n_events
            data[site_id]["Intra event"] =\
                resid.site_analysis[self.gmpe][self.imt]["Intra event"]
            data[site_id]["deltaS2S_s"] =\
                resid.site_analysis[self.gmpe][self.imt]["deltaS2S_s"]
            data[site_id]["deltaWS_es"] =\
                resid.site_analysis[self.gmpe][self.imt]["deltaWS_es"]
            data[site_id]["x-val"] =(float(iloc) + 0.5) *\
                np.ones_like(data[site_id]["Intra event"])
                
        return data
