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
Class to hold GMPE residual plotting functions
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy
from math import floor, ceil
from scipy.stats import norm
from cycler import cycler

from openquake.hazardlib.imt import imt2tup
from openquake.smt.utils_intensity_measures import _save_image
from openquake.smt.residuals.gmpe_residuals import Residuals, SingleStationAnalysis
from openquake.smt.residuals.residual_plotter_utils import (
                                                    residuals_density_distribution,
                                                    likelihood,
                                                    residuals_with_magnitude,
                                                    residuals_with_vs30,
                                                    residuals_with_distance,
                                                    residuals_with_depth)


colors = ['r', 'g', 'b', 'y', 'lime', 'dodgerblue', 'gold', '0.8', 'm', 'k',
          'mediumseagreen', 'tab:orange', 'tab:purple', 'tab:brown', '0.5']


class BaseResidualPlot(object):
    """
    Abstract-like class to create a Residual plot of strong ground motion
    residuals
    """
    # Class attributes to be passed to matplotlib xlabel, ylabel and title
    # methods. Allows DRY (don't repeat yourself) plots customization:
    xlabel_styling_kwargs = dict(fontsize=12)
    ylabel_styling_kwargs = dict(fontsize=12)
    title_styling_kwargs = dict(fontsize=12)

    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png",
                 dpi=300, **kwargs):
        """
        Initializes a BaseResidualPlot

        :param residuals:
            Residuals as instance of :class: openquake.smt.gmpe_residuals.Residuals
        :param str gmpe: Choice of GMPE
        :param str imt: Choice of IMT
        :param kwargs: optional keyword arguments. Supported are:
            'figure_size' (default: (8,8) or (7,5)), 'show' (default: True)
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
        self.filetype = filetype
        self.dpi = dpi
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
            
        self.show = kwargs.get("show", False)
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
        fig.set_tight_layout(True)
        nrow, ncol = self.get_subplots_rowcols()
        for tloc, res_type in enumerate(data.keys(), 1):
            self._residual_plot(plt.subplot(nrow, ncol, tloc), data[res_type],
                                res_type)
        _save_image(self.filename, plt.gcf(), self.filetype, self.dpi)
        if self.show:
            plt.show()

    def get_plot_data(self):
        """
        Builds the data to be plotted.
        This is an abstract-like method which subclasses need to implement.

        :return: a dictionary with keys denoting the residual types
        of the given GMPE (`self.gmpe`) and IMT (`self.imt`).
        Each key (residual type) needs then to be mapped to a residual data
        dict with at least the mandatory keys 'x', 'y' ,'xlabel' and 'ylabel'
        (See :module:`openquake.smt.residuals.residual_plotter_utils` for a list of available
        functions that return these kind of dict's and should be in principle
        be called here)
        """
        raise NotImplementedError()

    def get_subplots_rowcols(self):
        """
        Configures the plot layout (subplots grid).
        This is an abstract-like method which subclasses need to implement.

        :return: the tuple (row, col) denoting the layout of the
        figure to be displayed. The returned tuple should be consistent with
        the residual types available for the given GMPE (`self.gmpe`) and
        IMT (`self.imt`)
        """
        raise NotImplementedError()

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

    def draw(self, ax, res_data, res_type):
        """
        Draws the given residual data into the matplotlib `Axes` object `ax`.
        This is an abstract-like method which subclasses need to implement.

        :param ax: the matplotlib `Axes` object. this method should call
            the Axes plot method such as, e.g. `ax.plot(...)`,
            `ax.semilogx(...)` and so on
        :param res_data: the residual data to be plotted. It's one of
            the values of the dict returned by `self.get_plot_data`
            (`res_type` is the corresponding key): it is a dict with
            at least the mandatory keys 'x', 'y' (both numeric arrays),
            'xlabel' and 'ylabel' (both strings). Other keys, if present,
            should be handled by sub-classes implementation, if needed
        :param res_type: string denoting the residual type such as, e.g.
            "Inter event". It's one of the keys of the dict returned by
            `self.get_plot_data` (`res_data` is the corresponding value)
        """
        raise NotImplementedError()

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

    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png",
                 dpi=300, bin_width=0.5, **kwargs):
        """
        Initializes a ResidualHistogramPlot object. Sub-classes need to
        implement (at least) the method `get_plot_data`.

        All arguments not listed below are described in
        `BaseResidualPlot.__init__`.

        :param bin_width: float denoting the bin width of the histogram.
            defaults to 0.5
        """
        self.bin_width = bin_width
        super(ResidualHistogramPlot, self).__init__(residuals, gmpe, imt,
                                                    filename=filename,
                                                    filetype=filetype,
                                                    dpi=dpi, **kwargs)

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
        return residuals_density_distribution(self.residuals, self.gmpe,
                                              self.imt, self.bin_width)

    def draw(self, ax, res_data, res_type):
        # draw histogram:
        super(ResidualPlot, self).draw(ax, res_data, res_type)
        # draw normal distributions:
        mean = res_data["mean"]
        stddev = res_data["stddev"]
        x = res_data['x']
        xdata = np.arange(x[0], x[-1] + self.bin_width + 0.01, 0.01)
        xdata_norm_pdf = np.arange(-3,3,0.01)
        ax.plot(xdata, norm.pdf(xdata, mean, stddev), '-',
                color="LightSlateGrey", linewidth=2.0, 
                label = 'Empirical')
        ax.plot(xdata_norm_pdf, norm.pdf(xdata_norm_pdf, 0.0, 1.0), '-',
                color='k', linewidth=2.0, 
                label = 'Standard. Norm. Dist.')
        ax.legend(loc = 'best', fontsize = 'xx-small')
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

                    
class LikelihoodPlot(ResidualHistogramPlot):
    """
    Abstract-like class to create a simple histrogram of strong ground motion
    likelihood
    """

    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png",
                 dpi=300, bin_width=0.1, **kwargs):
        """
        Initializes a LikelihoodPlot. Basically calls the superclass
        `__init__` method with a `bin_width` default value of 0.1 instead of
        0.5
        """
        super(LikelihoodPlot, self).__init__(residuals, gmpe, imt,
                                             filename=filename,
                                             filetype=filetype,
                                             dpi=dpi,
                                             bin_width=bin_width,
                                             **kwargs)

    def _assertion_check(self, residuals):
        """
        Overrides the super-class method by asserting we are dealing
        with a `Likelihood` class
        """
        assert isinstance(residuals, Residuals)

    def get_plot_data(self):
        return likelihood(self.residuals, self.gmpe, self.imt, self.bin_width)

    def get_axis_xlim(self, res_data, res_type):
        return 0., 1.0

    def get_axis_title(self, res_data, res_type):
        median_lh = res_data["median"]
        sigma_type = res_type
        if res_type == 'Total':
            sigma_type = 'Total Res.'
        elif res_type == 'Inter event':
            sigma_type = 'Between-Event Res.'
        elif res_type == 'Intra event':
            sigma_type = 'Within-Event Res.'
        return "%s \n%s - Median LH = %7.3f" % (str(self.residuals.gmpe_list[
            self.gmpe]).split('(')[0].replace('\n',' - ').replace(
                'sigma_model','Sigma'),sigma_type,median_lh)


class ResidualScatterPlot(BaseResidualPlot):
    """
    Abstract-like class to create scatter plots of strong ground motion
    residuals
    """

    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png",
                 dpi=300, plot_type='', **kwargs):
        """
        Initializes a ResidualScatterPlot object. Sub-classes need to
        implement (at least) the method `get_plot_data`.

        All arguments not listed below are described in
        `BaseResidualPlot.__init__`.

        :param plot_type: string denoting if the plot x axis should be
            logarithmic (provide 'log' in case). Default: '' (no log x axis)
        """
        self.plot_type = plot_type
        super(ResidualScatterPlot, self).__init__(residuals, gmpe, imt,
                                                  filename=filename,
                                                  filetype=filetype,
                                                  dpi=dpi, **kwargs)
        
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
        x_zero = np.arange(np.floor(np.min(x))-20, np.ceil(np.max(x))+20, 0.001)
        zero_line = np.zeros(len(x_zero))
        pts_styling_kwargs = dict(markeredgecolor='Gray',
                                  markerfacecolor='LightSteelBlue',
                                  label='residual')
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

    def __init__(self, residuals, gmpe, imt, filename=None, filetype="png",
                 dpi=300, plot_type='linear', distance_type="rjb", **kwargs):
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
                                                   filetype=filetype,
                                                   dpi=dpi,
                                                   plot_type=plot_type,
                                                   **kwargs)

    def get_plot_data(self):
        return residuals_with_distance(self.residuals, self.gmpe, self.imt,
                                       self.distance_type)

    def get_axis_xlim(self, res_data, res_type):
        x = res_data['x']
        if self.plot_type == "log":
            return 0.1, 10.0 ** (ceil(np.log10(np.max(x))))
        else:
            if self.distance_type == "rcdpp":
                return np.min(x), np.max(x)
            else:
                return 0, np.max(x)


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


### Plotting of ranking metrics vs spectral period
def manage_imts(residuals):
    """
    Removes the non-acceleration IMTs from the imts attribute of the residuals
    object and create an array of the remaining IMTs. This is a utility function
    used for plotting of GMM ranking metrics vs period.
    """
    # Preserve original residuals.imts
    preserve_imts = residuals.imts

    # Remove non-acceleration imts from residuals.imts for generation of metrics
    idx_to_drop = []
    for imt_idx, imt in enumerate(residuals.imts):
        if imt != 'PGA' and 'SA' not in imt:
            idx_to_drop.append(imt_idx)
    residuals.imts = pd.Series(residuals.imts).drop(idx_to_drop).values

    # Convert imt_list to array
    x_with_imt = pd.DataFrame(
        [imt2tup(imts) for imts in residuals.imts], columns=['imt_str', 'imt_float']
    )
    for imt_idx in range(len(residuals.imts)):
        if x_with_imt.loc[imt_idx, 'imt_str'] == 'PGA':
            x_with_imt.loc[imt_idx, 'imt_float'] = 0

    x_with_imt = x_with_imt.dropna()

    return residuals, preserve_imts, x_with_imt

def plot_loglikelihood_with_spectral_period(residuals, filename, filetype='jpg', dpi=200):
    """
    Create a simple plot of loglikelihood values of Scherbaum et al. 2009
    (y-axis) versus spectral period (x-axis)
    """
    # Check enough imts to plot w.r.t. spectral period
    if len(residuals.imts) == 1:
        raise ValueError('Cannot plot w.r.t. spectral period (only 1 IMT).')
                
    # Manage imts
    residuals, preserve_imts, x_llh = manage_imts(residuals)

    # Define colours for GMMs
    colour_cycler = (cycler(color=colors)*cycler(linestyle=['-']))
        
    # Plot LLH values w.r.t. spectral period
    llh_with_imt = pd.DataFrame(residuals.llh).drop('All')
    fig_llh, ax_llh = plt.subplots(figsize=(10, 8))
    ax_llh.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        y_llh = np.array(llh_with_imt[gmpe])
        ax_llh.scatter(x_llh.imt_float, y_llh)
        tmp = str(residuals.gmpe_list[gmpe])
        ax_llh.plot(x_llh.imt_float, y_llh,label = tmp.split('(')[0])
    ax_llh.set_xlabel('Spectral Period (s)', fontsize='12')
    ax_llh.set_ylabel('Loglikelihood Value', fontsize='12')
    ax_llh.legend(loc='upper right', ncol=2, fontsize='medium')
    _save_image(filename, plt.gcf(), filetype, dpi)
    
    # Reassign original imts to residuals.imts
    residuals.imts = preserve_imts
    
def plot_edr_metrics_with_spectral_period(residuals, filename, filetype='jpg',
                                          dpi=200):
    """
    Create plots of EDR, the median pred. correction factor and normalised MDE
    computed using Kale and Akkar (2013) (y-axis) versus spectral period (x-axis)
    """
    # Check enough imts to plot w.r.t. spectral period
    if len(residuals.imts) == 1:
        raise ValueError('Cannot plot w.r.t. spectral period (only 1 IMT).')
    
    # Manage imts
    residuals, preserve_imts, x_with_imt = manage_imts(residuals)

    # Define colours for GMMs
    colour_cycler = (cycler(color=colors)*cycler(linestyle=['-']))
    
    # Plot EDR w.r.t. spectral period
    EDR_with_imt = {}
    fig_EDR, ax_EDR = plt.subplots(figsize=(10, 8))
    ax_EDR.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        EDR_with_imt = pd.DataFrame(residuals.edr_values_wrt_imt[gmpe])
        y_EDR = EDR_with_imt.EDR
        tmp = str(residuals.gmpe_list[gmpe])
        ax_EDR.scatter(x_with_imt.imt_float, y_EDR)
        ax_EDR.plot(x_with_imt.imt_float, y_EDR, label=tmp.split('(')[0])
    ax_EDR.set_xlabel('Spectral Period (s)', fontsize='12')
    ax_EDR.set_ylabel('EDR', fontsize='12')
    ax_EDR.legend(loc = 'upper right', ncol=2, fontsize='medium')
    _save_image(os.path.join(filename + '_EDR_value'), plt.gcf(), filetype, dpi)

    # Plot median pred. correction factor w.r.t. spectral period
    kappa_with_imt = {}
    fig_kappa, ax_kappa = plt.subplots(figsize=(10, 8))
    ax_kappa.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        kappa_with_imt = pd.DataFrame(residuals.edr_values_wrt_imt[gmpe])
        y_kappa = kappa_with_imt["sqrt Kappa"]
        tmp = str(residuals.gmpe_list[gmpe])
        ax_kappa.scatter(x_with_imt.imt_float, y_kappa)
        ax_kappa.plot(x_with_imt.imt_float, y_kappa, label=tmp.split('(')[0])
    ax_kappa.set_xlabel('Spectral Period (s)', fontsize='12')
    ax_kappa.set_ylabel('sqrt(k)', fontsize='12')
    ax_kappa.legend(loc = 'upper right', ncol=2, fontsize='medium')
    _save_image(os.path.join(filename + '_EDR_correction_factor'), plt.gcf(),
                filetype, dpi)
    
    # Plot MDE w.r.t. spectral period
    MDE_with_imt = {}
    fig_MDE, ax_MDE = plt.subplots(figsize=(10, 8))
    ax_MDE.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        MDE_with_imt = pd.DataFrame(residuals.edr_values_wrt_imt[gmpe])
        y_MDE = MDE_with_imt["MDE Norm"]
        tmp = str(residuals.gmpe_list[gmpe])
        ax_MDE.scatter(x_with_imt.imt_float, y_MDE)
        ax_MDE.plot(x_with_imt.imt_float, y_MDE, label=tmp.split('(')[0])
    ax_MDE.set_xlabel('Spectral Period (s)', fontsize='12')
    ax_MDE.set_ylabel('MDE Norm', fontsize='12')
    ax_MDE.legend(loc = 'upper right', ncol=2, fontsize='medium')
    _save_image(os.path.join(filename + '_MDE'), plt.gcf(), filetype, dpi)
    
    # Reassign original imts to residuals.imts
    residuals.imts = preserve_imts
    
def plot_stochastic_area_with_spectral_period(residuals, filename,
                                              filetype='jpg', dpi=200):
    """
    Definition to create plot of the stochastic area metric computed using Sunny
    et al. (2021) versus spectral period (x-axis)
    """
    # Check enough imts to plot w.r.t. spectral period
    if len(residuals.imts) == 1:
        raise ValueError('Cannot plot w.r.t. spectral period (only 1 IMT).')
    
    # Manage imts
    residuals, preserve_imts, x_with_imt = manage_imts(residuals)
    
    # Define colours for plots
    colour_cycler = (cycler(color=colors)*cycler(linestyle=['-']))
    
    # Plot stochastic area w.r.t. spectral period
    sto_with_imt = {}
    fig_sto, ax_sto = plt.subplots(figsize=(10, 8))
    ax_sto.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        sto_with_imt = pd.Series(residuals.stoch_areas_wrt_imt[gmpe])
        y_sto = sto_with_imt.values
        tmp = str(residuals.gmpe_list[gmpe])
        ax_sto.scatter(x_with_imt.imt_float, y_sto)
        ax_sto.plot(x_with_imt.imt_float, y_sto, label=tmp.split('(')[0])
    ax_sto.set_xlabel('Spectral Period (s)', fontsize='12')
    ax_sto.set_ylabel('Stochastic Area', fontsize='12')
    ax_sto.legend(loc = 'upper right', ncol=2, fontsize='medium')
    _save_image(os.path.join(filename), plt.gcf(), filetype, dpi)
        
    # Reassign original imts to residuals.imts
    residuals.imts = preserve_imts
    

### Functions for exporting tables of ranking metrics
def llh_table(residuals, filename):
    """
    Create a table of loglikelihood values per gmpe per imt (Scherbaum et al.
    2009)
    """
    # Get loglikelihood values per imt per gmpe
    llh_metrics = pd.DataFrame()
    for gmpe in residuals.gmpe_list:
        llh_metrics[gmpe + ' LLH'] = residuals.llh[gmpe]

    # Export table
    llh_metrics.to_csv(filename, sep=',')

def llh_weights_table(residuals, filename):
    """
    Create a table of model weights per gmpe per imt based on sample
    loglikelihood (Scherbaum et al. 2009)
    """       
    # Get weights based on llh and export table
    imt_idx = []
    for imt in residuals.imts:
        imt_idx.append(imt)
    imt_idx.append('Avg over imts') # Avg over all imts
    llh_weights = pd.DataFrame({}, columns=residuals.gmpe_list, index=imt_idx)
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            llh_weights.loc[imt, gmpe] = \
                residuals.model_weights_with_imt[imt][gmpe]
        llh_weights.loc['Avg over imts', gmpe] = llh_weights[gmpe].mean()
    llh_weights.columns = llh_weights.columns + ' LLH-based weights'

    # Export table
    llh_weights.to_csv(filename, sep=',')

def edr_table(residuals, filename):
    """
    Create a table of MDE Norm, sqrt(kappa) and EDR gmpe per imt (Kale and Akkar,
    2013)
    """
    # Get Kale and Akkar (2013) ranking metrics
    edr_dfs = []
    for gmpe in residuals.gmpe_list:
        col = {'MDE Norm':str(gmpe) + ' MDE Norm',
               'sqrt Kappa':str(gmpe) + ' sqrt Kappa',
               'EDR': str(gmpe) + ' EDR'}
        edr = pd.DataFrame(residuals.edr_values_wrt_imt[gmpe]).rename(col)
        means = []
        for metric in edr.columns: # Get average values over imts
            mean = edr[metric].mean()
            means.append(mean)
        edr.loc['Avg over imts'] = means
        edr.columns = edr.columns + ' ' + gmpe
        edr_dfs.append(edr)

    # Combine and export
    edr_df = pd.concat(edr_dfs, axis=1)
    edr_df.to_csv(filename, sep=',')

def edr_weights_table(residuals, filename):
    """
    Create a table of model weights per imt based on Euclidean distance based
    ranking (Kale and Akkar, 2013)
    """     
    # Get the EDR values from the residuals object
    edr_for_weights = residuals.edr_values_wrt_imt

    # Compute EDR based model weights
    edr_per_gmpe = {}
    for gmpe in edr_for_weights.keys():
        edr_per_gmpe[gmpe] = edr_for_weights[gmpe]['EDR']
    edr_per_gmpe_df = pd.DataFrame(edr_per_gmpe)

    gmpe_edr_weight = {gmpe: {} for gmpe in residuals.gmpe_list}
    for imt in edr_per_gmpe_df.index:
        total_edr_per_imt = np.sum(edr_per_gmpe_df.loc[imt]**-1)
        for gmpe in edr_for_weights.keys():
            gmpe_edr_weight[gmpe][imt] = \
                edr_per_gmpe_df.loc[imt][gmpe]**-1/total_edr_per_imt
    gmpe_edr_weight_df = pd.DataFrame(gmpe_edr_weight)
    
    avg_edr_weight_per_gmpe = {}
    for gmpe in residuals.gmpe_list:
        avg_edr_weight_per_gmpe[gmpe] = np.mean(gmpe_edr_weight_df[gmpe])
    avg_gmpe_edr_weight_df = \
        pd.DataFrame(avg_edr_weight_per_gmpe, index=['Avg over imts'])

    # Export table
    final_edr_weight_df = pd.concat([gmpe_edr_weight_df, avg_gmpe_edr_weight_df])
    final_edr_weight_df.to_csv(filename, sep=',')
        
def stochastic_area_table(residuals, filename):
    """
    Create a table of stochastic area ranking metric per GMPE per imt (Sunny et
    al. 2021)
    """
    # Get stochastic area value per imt
    imt_idx = []
    for imt in residuals.imts:
        imt_idx.append(imt)
    imt_idx.append('Avg over imts')
    sto_metrics = pd.DataFrame({}, columns=residuals.gmpe_list, index=imt_idx)
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            sto_metrics.loc[imt, gmpe] = \
                residuals.stoch_areas_wrt_imt[gmpe][imt]
        sto_metrics.loc['Avg over imts', gmpe] = sto_metrics[gmpe].mean()
    sto_metrics.columns = sto_metrics.columns + ' stochastic area'

    # Export table
    sto_metrics.to_csv(filename, sep=',')

def stochastic_area_weights_table(residuals, filename):
    """
    Create a table of model weights per imt based on sample stochastic area
    (Sunny et al. 2021))
    """       
    # Get required values
    sto_for_weights = residuals.stoch_areas_wrt_imt
    sto_per_gmpe_df = pd.DataFrame(sto_for_weights)

    # Get weights
    gmpe_sto_weight = {gmpe: {} for gmpe in residuals.gmpe_list}
    for imt in sto_per_gmpe_df.index:
        total_sto_per_imt = np.sum(sto_per_gmpe_df.loc[imt]**-1)
        for gmpe in sto_for_weights.keys():
            gmpe_sto_weight[gmpe][imt] = (sto_per_gmpe_df.loc[imt][gmpe]**-1)/total_sto_per_imt
    gmpe_sto_weight_df = pd.DataFrame(gmpe_sto_weight)

    # Get average per gmpe over the imts
    avg_sto_weight_per_gmpe = {}
    for gmpe in residuals.gmpe_list:
        avg_sto_weight_per_gmpe[gmpe] = np.mean(gmpe_sto_weight_df[gmpe])

    # Export table
    avg_gmpe_sto_weights = \
        pd.DataFrame(avg_sto_weight_per_gmpe, index=['Avg over imts'])
    final_sto_weights = pd.concat([gmpe_sto_weight_df, avg_gmpe_sto_weights])
    final_sto_weights.to_csv(filename, sep=',')


### Functions for plotting mean and sigma of residual dists vs spectral period
def get_res_dists(residuals):
    """
    Get the mean and sigma of the distributions of residuals per gmpe and imt
    """
    # Get all residuals for all GMPEs at all imts
    res_statistics = {}
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            res_statistics[gmpe, imt] = residuals.get_residual_statistics_for(
                gmpe, imt)
    
    # Now get into dataframes
    mean_sigma_intra, mean_sigma_inter, mean_sigma_total = {}, {}, {}
    dummy_values = {'Mean': float(0), 'Std Dev': float(0)} # Assign if only total
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

def set_res_pdf_plots(residuals, res_dists, imts_to_plot):
    """
    Set the plots for the means and std devs of each residual component per
    gmpe vs spectral period
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
    colour_cycler = (cycler(color=colors)*cycler(marker=['x']))
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
        ax[ax_index, 0].set_xlabel('Spectral Period (s)')
        ax[ax_index, 1].set_xlabel('Spectral Period (s)')
        ax[ax_index, 0].set_prop_cycle(colour_cycler)
        ax[ax_index, 1].set_prop_cycle(colour_cycler)
    for ax_index in range(0, 2):
        ax[2, ax_index].set_ylabel('Within-Event')
        ax[1, ax_index].set_ylabel('Between-Event')
        ax[0, ax_index].set_ylabel('Total')
    ax[0, 0].set_title('Mean of GMPE Residuals')    
    ax[0, 1].set_title('Sigma of GMPE Residuals')

    return fig, ax

def plot_res_pdf(ax, res_dists, dist_comp, gmpe, imts_to_plot, marker_input,
                 color_input):
    """
    Plot mean for each residual distribution for a given GMPE
    """
    # Get axes index and gmpe label
    if dist_comp == 'Mean':
        i = 0
    elif dist_comp == 'Std Dev':
        i = 1
    try:
        gmpe_label = gmpe.split('_toml=')[1].replace(')','')
    except:
        gmpe_label = gmpe # If not from toml file can't split

    # Plot mean
    if (res_dists[2][gmpe].loc[dist_comp].all()==0 and
        res_dists[1][gmpe].loc[dist_comp].all()==0):
        
        ax[2, i].scatter(imts_to_plot.imt_float,
                         res_dists[0][gmpe].loc[dist_comp],
                         color='w',
                         marker=marker_input,
                         zorder=0)
        ax[1, i].scatter(imts_to_plot.imt_float,
                         res_dists[1][gmpe].loc[dist_comp],
                         color='w',
                         marker=marker_input,
                         zorder=0)
    else:
        ax[2, i].scatter(imts_to_plot.imt_float,
                         res_dists[0][gmpe].loc[dist_comp],
                         color=color_input,
                         marker=marker_input)
        ax[1, i].scatter(imts_to_plot.imt_float,
                         res_dists[1][gmpe].loc[dist_comp],
                         color=color_input,
                         marker=marker_input)
        
    ax[0, i].scatter(imts_to_plot.imt_float,
                     res_dists[2][gmpe].loc[dist_comp],
                     label=gmpe_label,
                     color=color_input,
                     marker=marker_input)
    return ax

def plot_residual_pdf_with_spectral_period(
        residuals, filename, filetype='jpg', dpi=200):
    """
    Create a simple plot of residual mean and residual sigma for each GMPE 
    (y-axis) versus spectral period (x-axis)
    """
    # Check enough imts to plot w.r.t. spectral period
    if len(residuals.imts) == 1:
        raise ValueError('Cannot plot w.r.t. spectral period (only 1 IMT).')
        
    # Manage imts
    residuals, preserve_imts, imts_to_plot = manage_imts(residuals)
            
    # Get distributions of residuals per gmm and imt 
    res_dists = get_res_dists(residuals)

    # Set plots
    fig, ax = set_res_pdf_plots(residuals, res_dists, imts_to_plot)

    # Define colours for GMPEs
    colour_cycler = (cycler(color=colors)*cycler(marker=['x']))
    colour_cycler_df = pd.DataFrame(colour_cycler)[:len(residuals.gmpe_list)]
    colour_cycler_df['gmpe'] = residuals.gmpe_list.keys()

    # Plot data
    for gmpe in residuals.gmpe_list.keys():

        # Assign colour and marker to each gmpe
        input_df = pd.DataFrame(
            colour_cycler_df.loc[colour_cycler_df['gmpe']==gmpe]).reset_index()
        color_input = input_df['color'].iloc[0]
        marker_input = input_df['marker'].iloc[0]
        
        # Plot means
        dist_comp = 'Mean'
        ax = plot_res_pdf(ax, res_dists, dist_comp, gmpe, imts_to_plot,
                          marker_input, color_input)
       
        # Plot sigma
        dist_comp = 'Std Dev'
        ax = plot_res_pdf(ax, res_dists, dist_comp, gmpe, imts_to_plot,
                          marker_input, color_input)
        
    ax[0, 0].legend(loc='upper right', ncol=2, fontsize=6)
    _save_image(filename, plt.gcf(), filetype, dpi)
    
    # Reassign original imts to residuals.imts
    residuals.imts = preserve_imts

def pdf_table(residuals, filename):
    """
    Create a table of mean and standard deviation for total, inter-event and 
    intra-event residual distributions
    """
    # Get all residuals for all GMPEs at all imts
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
    
    # Export table
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
        fig.set_tight_layout(True)
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
        _save_image(self.filename, plt.gcf(), self.filetype, self.dpi)
        if self.show:
            plt.show()
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
            data[site_id]["Total"] = total_res/total_exp
            if "Intra event" in resid.site_analysis[self.gmpe][self.imt].keys():
                inter_res = resid.site_analysis[self.gmpe][self.imt]["Inter event"] 
                intra_res = resid.site_analysis[self.gmpe][self.imt]["Intra event"] 
                inter_exp = resid.site_analysis[self.gmpe][self.imt]["Expected inter"]
                intra_exp = resid.site_analysis[self.gmpe][self.imt]["Expected intra"]
                keep = pd.notnull(inter_res) # Dropping NaN idxs will realign with exp
                data[site_id]["Inter event"] = inter_res[keep]/inter_exp
                data[site_id]["Intra event"] = intra_res/intra_exp
            data[site_id]["ID"] = list(self.residuals.site_ids)[iloc]
            data[site_id]["N"] = n_events
            data[site_id]["x-val"] = (
                float(iloc) + 0.5) * np.ones_like(data[site_id]["Total"])
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
            phi_ss, phi_s2ss = self.residuals.station_residual_statistics()
            data = self._get_site_data()
            fig = plt.figure(figsize=self.figure_size)
            fig.set_tight_layout(True)
            self._residual_plot(fig, data,phi_ss[self.gmpe][self.imt],
                                phi_s2ss[self.gmpe][self.imt])
            _save_image(self.filename, plt.gcf(), self.filetype, self.dpi)
            if self.show:
                plt.show()
        else:
            warnings.warn('This implementation of %s GMPE does not have a mixed'
                         ' effects sigma model - plotting skipped' %self.gmpe,
                         stacklevel=10)

    def _residual_plot(self, fig, data, phi_ss, phi_s2ss):
        """
        Creates three plots:
        1) Plot of the intra-event residual (not normalised by GMPE intra-event)
           for each station (i.e. per EQ and site combination = 1 per record)
        2) Plot of the site term (average non-normalised intra-event per site)
        3) Plot of the remainder-residual ( = intra per rec - avg intra per site)
        """
        dwess = np.array([])
        dwoess = np.array([])
        ds2ss = []
        xvals = np.array([])
        for site_id in self.residuals.site_ids:
            xvals = np.hstack([xvals, data[site_id]["x-val"]])
            dwess = np.hstack([dwess, data[site_id]["Intra event"]])
            dwoess = np.hstack([dwoess, data[site_id]["dWo,es"]])
            ds2ss.append(data[site_id]["dS2ss"])
        ds2ss = np.array(ds2ss)
        ax = fig.add_subplot(311)
        
        # Show non-normalised intra-event residuals
        mean = np.array([np.mean(data[site_id]["Intra event"])
                         for site_id in self.residuals.site_ids])
        stddevs = np.array([np.std(data[site_id]["Intra event"])
                            for site_id in self.residuals.site_ids])
        xmean = np.array([data[site_id]["x-val"][0]
                          for site_id in self.residuals.site_ids])

        ax.plot(xvals, dwess, 'x', markeredgecolor='k', markerfacecolor='k',
                markersize=8, zorder=-32, label = '$\delta W_{es}$')
        ax.errorbar(xmean, mean, yerr=stddevs, ecolor="r", elinewidth=3.0,
                    barsabove=True, fmt="s", mfc="r", mec="k", ms=6,
                    label='Error bar')
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        xtick_label = self.residuals.site_ids
        ax.set_xticklabels(xtick_label, rotation="vertical")
        
        max_lim = ceil(np.max(np.fabs(dwess)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta W_{es}$ (%s)' % self.imt, fontsize=12)
        phi = np.std(dwess)
        ax.plot(xvals, phi * np.ones(len(xvals)), 'k--', linewidth=2.)
        ax.plot(xvals, -phi * np.ones(len(xvals)), 'k--', linewidth=2,
                label = 'Std dev')
        title_string = "%s - %s (Std Dev = %8.5f)" % (str(
            self.residuals.gmpe_list[self.gmpe]).split('(')[0].replace(
                ']\n', '] - ').replace('sigma_model','Sigma'), self.imt, phi)
        ax.set_title(title_string, fontsize=11)
        ax.legend(loc = 'upper right', fontsize = 'medium')
        
        # Show delta s2ss (avg non-normalised intra-event per site)
        ax = fig.add_subplot(312)
        ax.plot(xmean, ds2ss, 's', markeredgecolor='k',
                markerfacecolor='LightSteelBlue', markersize=8, zorder=-32,
                label='$\delta S2S_S$')
        ax.plot(xmean,(phi_s2ss["Mean"]-phi_s2ss["StdDev"])*np.ones(len(xmean)),
                "k--", linewidth=1.5)
        ax.plot(xmean,(phi_s2ss["Mean"]+phi_s2ss["StdDev"])*np.ones(len(xmean)),
                "k--", linewidth=1.5, label='+/- 1 $\phi_{S2S}$')
        ax.plot(xmean, (phi_s2ss["Mean"])*np.ones(len(xmean)), "k-",
                linewidth=2, label='Mean $\phi_{S2S}$')
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(xtick_label, rotation="vertical")
        max_lim = ceil(np.max(np.fabs(ds2ss)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta S2S_S$ (%s)' % self.imt, fontsize=12)
        title_string = r'%s - %s ($\phi_{S2S}$ = %8.5f)' % (str(
            self.residuals.gmpe_list[self.gmpe]).split('(')[0].replace(
                ']\n', '] - ').replace('sigma_model','Sigma'),
            self.imt, phi_s2ss["StdDev"])
        ax.set_title(title_string, fontsize=11)
        ax.legend(loc = 'upper right', fontsize = 'medium')
        
        # Show dwoes (remainder residual)
        ax = fig.add_subplot(313)
        ax.plot(xvals, dwoess, 'x', markeredgecolor='k', markerfacecolor='k',
                markersize=8, zorder=-32, label = '$\delta W_{o,es}$')
        ax.plot(xmean, -phi_ss * np.ones(len(xmean)), "k--", linewidth=1.5)
        ax.plot(xmean, phi_ss * np.ones(len(xmean)), "k--", linewidth=1.5,
                label = '$\phi_{SS}$')
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(xtick_label, rotation="vertical")
        max_lim = ceil(np.max(np.fabs(dwoess)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta W_{o,es} = \delta W_{es} - \delta S2S_S$',
                      fontsize=12)
        title_string = r'%s - %s ($\phi_{SS}$ = %8.5f)' % (str(
            self.residuals.gmpe_list[self.gmpe]).split('(')[0].replace(
                ']\n', '] - ').replace('sigma_model','Sigma'),self.imt,phi_ss)
        ax.set_title(title_string, fontsize=11)
        ax.legend(loc = 'upper right', fontsize = 'medium')
        
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
            data[site_id]["dS2ss"] =\
                resid.site_analysis[self.gmpe][self.imt]["dS2ss"]
            data[site_id]["dWo,es"] =\
                resid.site_analysis[self.gmpe][self.imt]["dWo,es"]
            data[site_id]["x-val"] =(float(iloc) + 0.5) *\
                np.ones_like(data[site_id]["Intra event"])
                
        return data