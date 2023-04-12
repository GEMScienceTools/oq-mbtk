# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2014-2017 GEM Foundation and G. Weatherill
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
from copy import deepcopy
from collections import OrderedDict
from math import floor, ceil
from scipy.stats import norm
from openquake.smt.sm_utils import _save_image
from openquake.smt.residuals.gmpe_residuals import Residuals, SingleStationAnalysis
from IPython.display import display
from cycler import cycler

from openquake.smt.residuals.residual_plots import (residuals_density_distribution,
                                           likelihood, residuals_with_magnitude,
                                           residuals_with_vs30,
                                           residuals_with_distance,
                                           residuals_with_depth)

from openquake.hazardlib.imt import imt2tup

class BaseResidualPlot(object):
    """
    Abstract-like class to create a Residual plot of strong ground motion
    residuals
    """

    # class attributes to be passed to matplotlib xlabel, ylabel and title
    # methods. Allows DRY (don't repet yourself) plots customization:
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
            'figure_size' (default: (7,5)), 'show' (default: True)
        """
        self._assertion_check(residuals)
        self.residuals = residuals
        if gmpe not in residuals.gmpe_list:
            raise ValueError("No residual data found for GMPE %s" % gmpe)
        if imt not in residuals.imts:
            raise ValueError("No residual data found for IMT %s" % imt)
        if not residuals.residuals[gmpe][imt]:
            raise ValueError("No residuals found for %s (%s)" % (gmpe, imt))
        self.gmpe = gmpe
        self.imt = imt
        self.filename = filename
        self.filetype = filetype
        self.dpi = dpi
        self.num_plots = len(residuals.types[gmpe][imt])
        self.figure_size = kwargs.get("figure_size",  (8, 8))
        self.show = kwargs.get("show", True)
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
        # statistics = self.residuals.get_residual_statistics()
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
        (See :module:`openquake.smt.residuals.residual_plots` for a list of available
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
        Plots the reisudal data on the given axis. This method should in
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
        elif res_type == 'Inter-Event Res.':
            sigma_type = 'Within-Event Res.'
        elif res_type == 'Intra-Event Res.':
            sigma_type = 'Between-Event Res.'
        
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
            overrides the super-class method by asserting we are dealing
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
        max_lim = ceil(np.max(np.fabs(y)))
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
                                  label = 'residual')
        if self.plot_type == "log":
            ax.semilogx(x, y, 'o', **pts_styling_kwargs)
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'],
                       marker = 's', color = 'b', label = 'mean', zorder = 4)
            
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'] + (
                -1*res_data['sigma_res']), marker = 'x', color = 'b', zorder = 4)
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'] + (
                res_data['sigma_res']), marker = 'x', color = 'b',
                label = '+/- 1 Std.', zorder = 4)
            
            ax.plot(x_zero, zero_line, color = 'k', linestyle = '--',
                    linewidth = 1.25)
        else:
            ax.plot(x, y, 'o', **pts_styling_kwargs)
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'],
                       marker = 's', color = 'b', label = 'mean', zorder = 4)
            
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'] + (
                -1*res_data['sigma_res']), marker = 'x', color = 'b', zorder = 4)
            ax.scatter(res_data['bin_midpoints'],res_data['mean_res'] + (
                res_data['sigma_res']), marker = 'x', color = 'b',
                label = '+/- 1 Std.', zorder = 4)
            
            ax.plot(x_zero, zero_line, color = 'k', linestyle = '--',
                    linewidth = 1.25)
        ax.legend(loc = 'upper right', fontsize = 'xx-small')
    

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
    

# FIXME: code below not tested and buggy (at least ResidualWithSite)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class ResidualWithSite(ResidualPlot):
    """
    Class uses Single-Station residuals to plot residual for specific sites
    """
    def _assertion_check(self, residuals):
        """
        Checks that residuals is en instance of the residuals class
        """
        assert isinstance(residuals, SingleStationAnalysis)
    
    def create_plot(self):
        """

        """
        phi_ss, phi_s2ss = self.residuals.residual_statistics()
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


    def _residual_plot(self, ax, data, res_type):
        """

        """
        xmean = np.array([data[site_id]["x-val"][0]
                          for site_id in self.residuals.site_ids])

        yvals = np.array([])
        xvals = np.array([])
        for site_id in self.residuals.site_ids:
            xvals = np.hstack([xvals, data[site_id]["x-val"]])
            yvals = np.hstack([yvals, data[site_id][res_type]])
        ax.plot(xvals,
                yvals,
                'o',
                markeredgecolor='Gray',
                markerfacecolor='LightSteelBlue',
                zorder=-32)
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(self.residuals.site_ids, rotation="vertical")

        max_lim = ceil(np.max(np.fabs(yvals)))
        ax.set_ylim(-max_lim, max_lim)
        ax.set_ylabel("%s" % res_type, fontsize=12)
        ax.grid()
        title_string = "%s - %s - %s Residual" % (str(self.residuals.gmpe_list[
            self.gmpe]).split('(')[0].replace(']\n', '] - ').replace(
                'sigma_model','Sigma'),self.imt,res_type)
        ax.set_title(title_string, fontsize=12)

    def _get_site_data(self):
        """

        """
        data = OrderedDict([(site_id, {}) 
                            for site_id in self.residuals.site_ids])
        for iloc, site_resid in enumerate(self.residuals.site_residuals):
            resid = deepcopy(site_resid)

            site_id = self.residuals.site_ids[iloc]
            n_events = resid.site_analysis[self.gmpe][self.imt]["events"]
            data[site_id]["Total"] = (
                resid.site_analysis[self.gmpe][self.imt]["Total"] /
                resid.site_analysis[self.gmpe][self.imt]["Expected Total"])
            if "Intra event" in\
                resid.site_analysis[self.gmpe][self.imt].keys():
                data[site_id]["Inter event"] = (
                    resid.site_analysis[self.gmpe][self.imt]["Inter event"] /
                    resid.site_analysis[self.gmpe][self.imt]["Expected Inter"])
                data[site_id]["Intra event"] = (
                    resid.site_analysis[self.gmpe][self.imt]["Intra event"] /
                    resid.site_analysis[self.gmpe][self.imt]["Expected Intra"])

            data[site_id]["ID"] = self.residuals.site_ids[iloc]
            data[site_id]["N"] = n_events
            data[site_id]["x-val"] =(float(iloc) + 0.5) *\
                np.ones_like(data[site_id]["Intra event"])
        return data


class IntraEventResidualWithSite(ResidualPlot):
    """

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
        phi_ss, phi_s2ss = self.residuals.residual_statistics()
        data = self._get_site_data()
        fig = plt.figure(figsize=self.figure_size)
        fig.set_tight_layout(True)
        
        self._residual_plot(fig, data,
                            phi_ss[self.gmpe][self.imt],
                            phi_s2ss[self.gmpe][self.imt])
        _save_image(self.filename, plt.gcf(), self.filetype, self.dpi)
        if self.show:
            plt.show()


    def _residual_plot(self, fig, data, phi_ss, phi_s2ss):
        """
        Creates three plots:
        1) Plot of the intra-event residual for each station
        2) Plot of the site term
        3) Plot of the remainder-residual
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
        # Show intra-event residuals
        mean = np.array([np.mean(data[site_id]["Intra event"])
                         for site_id in self.residuals.site_ids])
        stddevs = np.array([np.std(data[site_id]["Intra event"])
                            for site_id in self.residuals.site_ids])
        xmean = np.array([data[site_id]["x-val"][0]
                          for site_id in self.residuals.site_ids])

        ax.plot(xvals,
                dwess,
                'x',
                markeredgecolor='k',
                markerfacecolor='k',
                markersize=8,
                zorder=-32)
        ax.errorbar(xmean, mean, yerr=stddevs,
                    ecolor="r", elinewidth=3.0, barsabove=True,
                    fmt="s", mfc="r", mec="k", ms=6)
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(self.residuals.site_ids, rotation="vertical")
        max_lim = ceil(np.max(np.fabs(dwess)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta W_{es}$ (%s)' % self.imt, fontsize=12)
        phi = np.std(dwess)
        ax.plot(xvals, phi * np.ones(len(xvals)), 'k--', linewidth=2.)
        ax.plot(xvals, -phi * np.ones(len(xvals)), 'k--', linewidth=2.)
        title_string = "%s - %s (Std Dev = %8.5f)" % (str(
            self.residuals.gmpe_list[self.gmpe]).split('(')[0].replace(
                ']\n', '] - ').replace('sigma_model','Sigma'), self.imt, phi)
        ax.set_title(title_string, fontsize=16)
        # Show delta s2ss
        ax = fig.add_subplot(312)
        ax.plot(xmean,
                ds2ss,
                's',
                markeredgecolor='k',
                markerfacecolor='LightSteelBlue',
                markersize=8,
                zorder=-32)
        ax.plot(xmean,
                (phi_s2ss["Mean"] - phi_s2ss["StdDev"]) * np.ones(len(xmean)),
                "k--", linewidth=1.5)
        ax.plot(xmean,
                (phi_s2ss["Mean"] + phi_s2ss["StdDev"]) * np.ones(len(xmean)),
                "k--", linewidth=1.5)
        ax.plot(xmean,
                (phi_s2ss["Mean"]) * np.ones(len(xmean)),
                "k-", linewidth=2)
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(self.residuals.site_ids, rotation="vertical")
        max_lim = ceil(np.max(np.fabs(ds2ss)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta S2S_S$ (%s)' % self.imt, fontsize=12)
        title_string = r'%s - %s ($\phi_{S2S}$ = %8.5f)' % (str(
            self.residuals.gmpe_list[self.gmpe]).split('(')[0].replace(
                ']\n', '] - ').replace('sigma_model','Sigma'),
            self.imt, phi_s2ss["StdDev"])
        ax.set_title(title_string, fontsize=16)
        # Show dwoes
        ax = fig.add_subplot(313)
        ax.plot(xvals,
                dwoess,
                'x',
                markeredgecolor='k',
                markerfacecolor='k',
                markersize=8,
                zorder=-32)
        ax.plot(xmean, -phi_ss * np.ones(len(xmean)), "k--", linewidth=1.5)
        ax.plot(xmean, phi_ss * np.ones(len(xmean)), "k--", linewidth=1.5)
        ax.set_xlim(0, len(self.residuals.site_ids))
        ax.set_xticks(xmean)
        ax.set_xticklabels(self.residuals.site_ids, rotation="vertical")
        max_lim = ceil(np.max(np.fabs(dwoess)))
        ax.set_ylim(-max_lim, max_lim)
        ax.grid()
        ax.set_ylabel(r'$\delta W_{o,es} = \delta W_{es} - \delta S2S_S$ (%s)'
                      % self.imt, 
                      fontsize=12)
        title_string = r'%s - %s ($\phi_{SS}$ = %8.5f)' % (str(
            self.residuals.gmpe_list[self.gmpe]).split('(')[0].replace(
                ']\n', '] - ').replace('sigma_model','Sigma'),self.imt,phi_ss)
        ax.set_title(title_string, fontsize=16)

    def _get_site_data(self):
        """

        """
        data = OrderedDict([(site_id, {}) 
                            for site_id in self.residuals.site_ids])
        for iloc, site_resid in enumerate(self.residuals.site_residuals):
            resid = deepcopy(site_resid)

            site_id = self.residuals.site_ids[iloc]
            n_events = resid.site_analysis[self.gmpe][self.imt]["events"]
            data[site_id] = resid.site_analysis[self.gmpe][self.imt]
            data[site_id]["ID"] = self.residuals.site_ids[iloc]
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
        
def plot_loglikelihood_with_spectral_period(residuals, filename, custom_cycler = 0,
                                        filetype = 'jpg', dpi = 200):
    """
    Definition to create a simple plot of loglikelihood values of Scherbaum
    et al. 2009 (y-axis) versus spectral period (x-axis)
    
    :param custom_cycler: Default set to 0, to assign specific colours
    and linestyle to each GMPE, a cycler can instead be specified manually,
    where index value in the cycler corresponds to gmpe in residuals.gmpe_list
    """
    # Check enough imts to plot w.r.t. spectral period
    if len(residuals.imts) == 1:
        raise ValueError('Cannot plot w.r.t. spectral period (only 1 imt).')
                
    # Preserve original residuals.imts
    preserve_imts = residuals.imts
    
    # Remove non-acceleration imts from residuals.imts for generation of metrics
    imt_append = pd.DataFrame(residuals.imts, index = residuals.imts)
    imt_append.columns = ['imt']
    for imt_idx in range(0, np.size(residuals.imts)):
         if residuals.imts[imt_idx] == 'PGV':
             imt_append = imt_append.drop(imt_append.loc['PGV'])
         if residuals.imts[imt_idx] == 'PGD':
             imt_append = imt_append.drop(imt_append.loc['PGD'])
         if residuals.imts[imt_idx] == 'CAV':
             imt_append = imt_append.drop(imt_append.loc['CAV'])
         if residuals.imts[imt_idx] == 'Ia':
             imt_append = imt_append.drop(imt_append.loc['Ia'])
        
    imt_append_list = pd.DataFrame()
    for idx in range(0,len(imt_append)):
         imt_append_list[idx] = imt_append.iloc[idx]
    imt_append = imt_append.reset_index()
    imt_append_list.columns = imt_append.imt
    residuals.imts = list(imt_append_list)

    # Convert imt_list to array
    x_llh = pd.DataFrame([imt2tup(imts) for imts in residuals.imts], columns =
                       ['imt_str', 'imt_float'])
    for imt_idx in range(0, np.size(residuals.imts)):
         if x_llh.imt_str[imt_idx] == 'PGA':
            x_llh.imt_float.iloc[imt_idx] = 0
    x_llh=x_llh.dropna() # Remove any non-acceleration imt   
    
    # Generate attributes required from residuals for table
    residuals.get_loglikelihood_values(residuals.imts)

    # Produce series of residuals.gmpe_list used to index llh_with_imt
    gmpe_list_series = pd.Series(pd.DataFrame(
        residuals.gmpe_list, index = np.arange(0, len(residuals.gmpe_list), 1)).
        columns)
    
    # Define colours for plots
    colour_cycler = (cycler(color = ['r', 'g', 'b', 'y', 'lime', 'k', 'dodgerblue',
                                   'gold', '0.8', 'mediumseagreen', '0.5',
                                   'tab:orange', 'tab:purple', 'tab:brown',
                                   'tab:pink'])*cycler(linestyle = ['-']))
    
    if type(custom_cycler) == type(cycler(colour = 'b')):
       colour_cycler = custom_cycler
        
    # Plot LLH values w.r.t. spectral period
    llh_with_imt = pd.DataFrame(residuals.llh).drop('All')
    fig_llh, ax_llh = plt.subplots(figsize=(10, 8))
    ax_llh.set_prop_cycle(colour_cycler)
    for gmpe in range(0, len(gmpe_list_series)):
        y_llh = np.array(llh_with_imt[gmpe_list_series[gmpe]])
        ax_llh.scatter(x_llh.imt_float, y_llh)
        tmp = str(residuals.gmpe_list[gmpe_list_series[gmpe]])
        ax_llh.plot(x_llh.imt_float, y_llh,label = tmp.split('(')[0])
    ax_llh.set_xlabel('Spectral Period (s)')
    ax_llh.set_ylabel('Loglikelihood Value')
    ax_llh.set_title('Scherbaum et al. (2009) Loglikelihood Values', fontsize = '16')
    ax_llh.legend(loc = 'upper right', ncol = 2, fontsize = 'xx-small')
    _save_image(filename, plt.gcf(), filetype, dpi)
    
    # Reassign original imts to residuals.imts
    residuals.imts = preserve_imts
    
def plot_edr_metrics_with_spectral_period(residuals, filename, custom_cycler = 0,
                              filetype = 'jpg', dpi = 200):
    """
    Definition to create a simple plots of EDR, the median pred. correction
    factor and normalised MDE computed using Kale and Akkar (2013) (y-axis)
    versus spectral period (x-axis)
    
    :param custom_cycler: Default set to 0, to assign specific colours
    and linestyle to each GMPE, a cycler can instead be specified manually,
    where index value in the cycler corresponds to gmpe in residuals.gmpe_list
    """
    
    # Check enough imts to plot w.r.t. spectral period
    if len(residuals.imts) == 1:
        raise ValueError('Cannot plot w.r.t. spectral period (only 1 imt).')
        
    # Generate attributes required from residuals for table    
    residuals.get_edr_values_wrt_spectral_period()
    
    # Convert imt_list to array
    x_with_imt = pd.DataFrame([imt2tup(imts) for imts in residuals.imts],
                                columns=['imt_str', 'imt_float'])
    for imt_idx in range(0, np.size(residuals.imts)):
         if x_with_imt.imt_str[imt_idx] == 'PGA':
            x_with_imt.imt_float.iloc[imt_idx] = 0
    x_with_imt = x_with_imt.dropna() #Remove any non-acceleration imt

    # Define colours for plots
    colour_cycler = (cycler(color=['r', 'g', 'b', 'y', 'lime', 'k', 'dodgerblue',
                                   'gold', '0.8', 'mediumseagreen','0.5',
                                   'tab:orange', 'tab:purple', 'tab:brown',
                                   'tab:pink'])*cycler(linestyle = ['-']))
    if type(custom_cycler) == type(cycler(colour = 'b')):
       colour_cycler = custom_cycler
    
    # Plot EDR w.r.t. spectral period
    EDR_with_imt = {}
    fig_EDR, ax_EDR = plt.subplots(figsize = (10, 8))
    ax_EDR.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        EDR_with_imt = pd.DataFrame(residuals.edr_values_wrt_imt[gmpe])
        y_EDR = EDR_with_imt.EDR
        tmp = str(residuals.gmpe_list[gmpe])
        ax_EDR.scatter(x_with_imt.imt_float, y_EDR)
        ax_EDR.plot(x_with_imt.imt_float, y_EDR, label = tmp.split('(')[0])
    ax_EDR.set_xlabel('Spectral Period (s)')
    ax_EDR.set_ylabel('EDR')
    ax_EDR.set_title('Euclidean-Based Distance Ranking (Kale and Akkar, 2013)',
                     fontsize = '16')
    ax_EDR.legend(loc = 'upper right', ncol = 2, fontsize = 'xx-small')
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
        ax_kappa.plot(x_with_imt.imt_float, y_kappa, label = tmp.split('(')[0])
    ax_kappa.set_xlabel('Spectral Period (s)')
    ax_kappa.set_ylabel('k^0.5')
    ax_kappa.set_title('Median Pred. Correction Factor (k) (Kale and Akkar, 2013)',
                       fontsize = '16')
    ax_kappa.legend(loc = 'upper right', ncol = 2, fontsize = 'xx-small')
    _save_image(os.path.join(filename + '_EDR_correction_factor'), plt.gcf(),
                filetype, dpi)
    
    # Plot MDE w.r.t. spectral period
    MDE_with_imt = {}
    fig_MDE, ax_MDE = plt.subplots(figsize = (10, 8))
    ax_MDE.set_prop_cycle(colour_cycler)
    for gmpe in residuals.gmpe_list:
        MDE_with_imt = pd.DataFrame(residuals.edr_values_wrt_imt[gmpe])
        y_MDE = MDE_with_imt["MDE Norm"]
        tmp = str(residuals.gmpe_list[gmpe])
        ax_MDE.scatter(x_with_imt.imt_float, y_MDE)
        ax_MDE.plot(x_with_imt.imt_float, y_MDE, label = tmp.split('(')[0])
    ax_MDE.set_xlabel('Spectral Period (s)')
    ax_MDE.set_ylabel('MDE Norm')
    ax_MDE.set_title('Normalised MDE (Kale and Akkar, 2013)',fontsize='16')
    ax_MDE.legend(loc = 'upper right', ncol = 2, fontsize = 'xx-small')
    _save_image(os.path.join(filename + '_MDE'), plt.gcf(), filetype, dpi)
    
def plot_residual_pdf_with_spectral_period(residuals, filename, custom_cycler = 0,
                                      filetype = 'jpg', dpi = 200):
    """
    Definition to create a simple plot of residual mean and residual sigma 
    for each GMPE (y-axis) versus spectral period (x-axis)
    
    :param custom_cycler: Default set to 0, to assign specific colours
    and linestyle to each GMPE, a cycler can instead be specified manually,
    where index value in the cycler corresponds to gmpe in residuals.gmpe_list
    """
    
    # Check enough imts to plot w.r.t. spectral period
    if len(residuals.imts) == 1:
        raise ValueError('Cannot plot w.r.t. spectral period (only 1 imt).')
        
    # Preserve original residuals.imts
    preserve_imts = residuals.imts
    
    # Remove non-acceleration imts from residuals.imts for generation of metrics
    imt_append = pd.DataFrame(residuals.imts, index = residuals.imts)
    imt_append.columns = ['imt']
    for imt_idx in range(0, np.size(residuals.imts)):
        if residuals.imts[imt_idx] == 'PGV':
            imt_append = imt_append.drop(imt_append.loc['PGV'])
        if residuals.imts[imt_idx] == 'PGD':
           imt_append = imt_append.drop(imt_append.loc['PGD'])
        if residuals.imts[imt_idx] == 'CAV':
           imt_append = imt_append.drop(imt_append.loc['CAV'])
        if residuals.imts[imt_idx] == 'Ia':
           imt_append = imt_append.drop(imt_append.loc['Ia'])
        
    imt_append_list = pd.DataFrame()
    for idx in range(0,len(imt_append)):
        imt_append_list[idx] = imt_append.iloc[idx]
    imt_append = imt_append.reset_index()
    imt_append_list.columns = imt_append.imt
    residuals.imts = list(imt_append_list)
    
    # Convert imt_list to array
    imts_to_plot = pd.DataFrame([imt2tup(imts) for imts in residuals.imts],
                              columns=['imt_str', 'imt_float'])
    for imt_idx in range(0, np.size(residuals.imts)):
        if imts_to_plot.imt_str[imt_idx] == 'PGA':
           imts_to_plot.imt_float.iloc[imt_idx] = 0
    imts_to_plot = imts_to_plot.dropna() # Remove any non-acceleration imt

    # Get all residuals for all gmpes at all imts
    res_statistics = {}
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            res_statistics[gmpe, imt] = residuals.get_residual_statistics_for(
                gmpe, imt)
    
    Mean_Sigma_Intra = {}
    Mean_Sigma_Inter = {}
    Mean_Sigma_Total = {}
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            Mean_Sigma_Intra[gmpe, imt] = res_statistics[gmpe, imt]['Intra event']
            Mean_Sigma_Inter[gmpe, imt] = res_statistics[gmpe, imt]['Inter event']
            Mean_Sigma_Total[gmpe, imt] = res_statistics[gmpe, imt]['Total']

    Mean_Sigma_Intra_df = pd.DataFrame(Mean_Sigma_Intra)
    Mean_Sigma_Inter_df = pd.DataFrame(Mean_Sigma_Inter)
    Mean_Sigma_Total_df = pd.DataFrame(Mean_Sigma_Total)

    # Create figure
    fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize = (14, 14)) 

    # Plot mean of zero and sigma of 1 for standard normal dist
    for ax_idx in range(0, 3):
        ax[ax_idx, 0].plot(imts_to_plot.imt_float, np.zeros(len(imts_to_plot)),
                          color = 'k', linestyle = '--')
        ax[ax_idx, 1].plot(imts_to_plot.imt_float, np.ones(len(imts_to_plot)),
                          color = 'k', linestyle = '--')

    # Produce series of residuals.gmpe_list
    gmpe_list_series = pd.Series(pd.DataFrame(
        residuals.gmpe_list, index = np.arange(0, len(residuals.gmpe_list) ,1)).
        columns)
    
    # Define colours for plots
    colour_cycler = (cycler(color=['r', 'g', 'b', 'y','lime', 'dodgerblue', 'k',
                                   'gold', '0.8', 'mediumseagreen', '0.5',
                                   'tab:orange', 'tab:purple','tab:brown',
                                   'tab:pink'])*cycler(marker = ['x']))
    
    if type(custom_cycler) == type(cycler(colour = 'b')):
       colour_cycler = custom_cycler
    colour_cycler_df=pd.DataFrame(colour_cycler)
    
    gmpe_colour = {}
    gmpe_marker = {}
    for gmpe in range(0,np.size(gmpe_list_series)):
        gmpe_colour[gmpe] = colour_cycler_df.color.iloc[gmpe]
        gmpe_marker[gmpe] = colour_cycler_df.marker.iloc[gmpe]
    colour_cycler_df_appended = colour_cycler_df[:len(residuals.gmpe_list)]
    colour_cycler_df_appended['gmpe'] = residuals.gmpe_list

    # Set plots
    Mean_ymax = np.max([Mean_Sigma_Intra_df.loc['Mean'],
                        Mean_Sigma_Inter_df.loc['Mean'],
                        Mean_Sigma_Total_df.loc['Mean']])
    Mean_ymin = np.min([Mean_Sigma_Intra_df.loc['Mean'],
                        Mean_Sigma_Inter_df.loc['Mean'],
                        Mean_Sigma_Total_df.loc['Mean']])
    Mean_y_bound = np.max([np.abs(Mean_ymax), np.abs(Mean_ymin)])
    Sigma_ymax = np.max([Mean_Sigma_Intra_df.loc['Std Dev'],
                         Mean_Sigma_Inter_df.loc['Std Dev'],
                         Mean_Sigma_Total_df.loc['Std Dev']])
    Sigma_ymin = np.min([Mean_Sigma_Intra_df.loc['Std Dev'],
                         Mean_Sigma_Inter_df.loc['Std Dev'],
                         Mean_Sigma_Total_df.loc['Std Dev']])
    Sigma_y_bound_non_centered = np.max([np.abs(Sigma_ymax), np.abs(Sigma_ymin)])
    Sigma_y_bound = min(np.abs(1-Sigma_y_bound_non_centered),
                      np.abs(1+Sigma_y_bound_non_centered))
    for ax_index in range(0, 3):
        ax[ax_index, 0].set_ylim(-Mean_y_bound-0.5, Mean_y_bound+0.5)
        ax[ax_index, 1].set_ylim(0.9-Sigma_y_bound, 1.1+Sigma_y_bound)
        ax[ax_index, 0].set_xlabel('Spectral Period (s)')
        ax[ax_index, 1].set_xlabel('Spectral Period (s)')
        ax[ax_index, 0].set_prop_cycle(colour_cycler)
        ax[ax_index, 1].set_prop_cycle(colour_cycler)
    for ax_index in range(0, 2):
        ax[2, ax_index].set_ylabel('Intra-Event')
        ax[1, ax_index].set_ylabel('Inter-Event')
        ax[0, ax_index].set_ylabel('Total')
    ax[0, 0].set_title('Mean of GMPE Residuals')    
    ax[0, 1].set_title('Sigma of GMPE Residuals')

    # Plot data
    for gmpe in residuals.gmpe_list:
        
        # Assign colour and marker to each gmpe
        input_df = pd.DataFrame(colour_cycler_df_appended.loc[
            colour_cycler_df_appended['gmpe'] == gmpe])
        input_df.reset_index()
        color_input = input_df['color'].iloc[0]
        marker_input = input_df['marker'].iloc[0]
        
        # Plot residual data
        ax[2, 0].scatter(imts_to_plot.imt_float,Mean_Sigma_Intra_df[gmpe].loc[
            'Mean'], color = color_input, marker = marker_input)
        ax[1, 0].scatter(imts_to_plot.imt_float, Mean_Sigma_Inter_df[gmpe].loc[
            'Mean'], color = color_input, marker = marker_input)
        ax[0, 0].scatter(imts_to_plot.imt_float, Mean_Sigma_Total_df[gmpe].loc[
            'Mean'], label = residuals.gmpe_list[gmpe], color = color_input, 
            marker = marker_input)
        ax[2, 1].scatter(imts_to_plot.imt_float, Mean_Sigma_Intra_df[gmpe].loc[
            'Std Dev'], color = color_input, marker = marker_input)
        ax[1, 1].scatter(imts_to_plot.imt_float, Mean_Sigma_Inter_df[gmpe].loc[
            'Std Dev'], color = color_input, marker = marker_input)
        ax[0, 1].scatter(imts_to_plot.imt_float, Mean_Sigma_Total_df[gmpe].loc[
            'Std Dev'], color = color_input, marker = marker_input)
        ax[0, 0].legend(loc = 'upper right', ncol = 2, fontsize = 'xx-small')

        _save_image(filename, plt.gcf(), filetype, dpi)
    
        # Reassign original imts to residuals.imts
        residuals.imts = preserve_imts  
    
def loglikelihood_table(residuals, filename):
    """
    Definition to create a table of loglikelihood values per GMPE per imt
    (Scherbaum et al. 2009)
    """
    
    # Preserve original residuals.imts
    preserve_imts = residuals.imts
    
    # Remove non-acceleration imts from residuals.imts for generation of metrics
    imt_append = pd.DataFrame(residuals.imts, index = residuals.imts)
    imt_append.columns = ['imt']
    for imt_idx in range(0, np.size(residuals.imts)):
         if residuals.imts[imt_idx] == 'PGV':
             imt_append = imt_append.drop(imt_append.loc['PGV'])
         if residuals.imts[imt_idx] == 'PGD':
             imt_append = imt_append.drop(imt_append.loc['PGD'])
         if residuals.imts[imt_idx] == 'CAV':
             imt_append = imt_append.drop(imt_append.loc['CAV'])
         if residuals.imts[imt_idx] == 'Ia':
             imt_append = imt_append.drop(imt_append.loc['Ia'])
        
    imt_append_list = pd.DataFrame()
    for idx in range(0, len(imt_append)):
         imt_append_list[idx] = imt_append.iloc[idx]
    imt_append = imt_append.reset_index()
    imt_append_list.columns = imt_append.imt
    residuals.imts = list(imt_append_list)
    
    # Generate attributes required from residuals for table
    residuals.get_loglikelihood_values(residuals.imts)

    # Produce series of residuals.gmpe_list
    gmpe_list_series = pd.Series(pd.DataFrame(residuals.gmpe_list, index=
                                    np.arange(0, len(residuals.gmpe_list)
                                              , 1)).columns)

    # Get loglikelihood values per spectral period
    llh_metrics = {}
    llh_metrics_df = {} #First as dictionary
    for gmpe in range(0, np.size(gmpe_list_series)):
        llh_columns = pd.Series(gmpe_list_series)[gmpe]+ ' LLH'
        llh_metrics[gmpe] = pd.DataFrame({llh_columns:(pd.Series(residuals.llh))
                                        [gmpe]}, index=residuals.imts)
        llh_metrics_df[gmpe] = pd.DataFrame(llh_metrics[gmpe],
                                          index = residuals.imts)

    llh_metrics_df = pd.DataFrame() # Then to true dataframe once indexing by gmpe
    for gmpe in range(0, np.size(gmpe_list_series)):
        llh_metrics_df[gmpe] = pd.DataFrame(llh_metrics[gmpe], index=
                                          residuals.imts)
    llh_columns_all = pd.Series(gmpe_list_series) +' LLH' 
    llh_metrics_df.columns = llh_columns_all
    
    average_llh_over_imts = np.arange(0, np.size(gmpe_list_series), 1,
                                    dtype = 'float')
    for gmpe in range(0, np.size(gmpe_list_series)):
        average_llh_over_imts[gmpe] = np.array(np.mean(
            llh_metrics_df[llh_columns_all[gmpe]]))
    average_llh_over_imts_df = pd.DataFrame([average_llh_over_imts],
                                          index = ['Avg over all periods'])
    average_llh_over_imts_df.columns = llh_columns_all
    final_llh_df = pd.concat([llh_metrics_df, average_llh_over_imts_df])
    
    #Table and display outputs (need to assign simplified GMPE names)
    final_llh_df_output = final_llh_df
    llh_columns_all_output = {}
    for gmpe in range(0, len(residuals.gmpe_list)):
         tmp = str(residuals.gmpe_list[gmpe_list_series[gmpe]])
         llh_columns_all_output[gmpe] =  tmp.split('(')[0].replace(
             '\n',', ') + ' LLH'
    final_llh_df_output.columns = list(pd.Series(llh_columns_all_output))
    final_llh_df_output.to_csv(filename, sep = ',')
    display(final_llh_df_output)
    
    # Reassign original imts to residuals.imts
    residuals.imts = preserve_imts
    
    # Assign final llh dataframe to residuals object (for test unit case use)
    residuals.final_llh_df = final_llh_df
    return residuals.final_llh_df
    
def llh_weights_table(residuals, filename):
    """
    Definition to create a table of model weights per imt based
    on sample loglikelihood (Scherbaum et al. 2009)
    """     
       
    # Preserve original residuals.imts
    preserve_imts = residuals.imts
    
    # Remove non-acceleration imts from residuals.imts for generation of metrics
    imt_append = pd.DataFrame(residuals.imts, index = residuals.imts)
    imt_append.columns = ['imt']
    for imt_idx in range(0, np.size(residuals.imts)):
         if residuals.imts[imt_idx] == 'PGV':
             imt_append = imt_append.drop(imt_append.loc['PGV'])
         if residuals.imts[imt_idx] == 'PGD':
             imt_append = imt_append.drop(imt_append.loc['PGD'])
         if residuals.imts[imt_idx] == 'CAV':
             imt_append = imt_append.drop(imt_append.loc['CAV'])
         if residuals.imts[imt_idx] == 'Ia':
             imt_append = imt_append.drop(imt_append.loc['Ia'])
        
    imt_append_list = pd.DataFrame()
    for idx in range(0,len(imt_append)):
         imt_append_list[idx] = imt_append.iloc[idx]
    imt_append = imt_append.reset_index()
    imt_append_list.columns = imt_append.imt
    residuals.imts = list(imt_append_list)
    
    # Generate attributes required from residuals for table
    residuals.get_loglikelihood_values(residuals.imts)

    # Produce series of residuals.gmpe_list
    gmpe_list_series = pd.Series(pd.DataFrame(residuals.gmpe_list, index=
                                        np.arange(0, len(residuals.gmpe_list)
                                                  , 1)).columns)
    
    # Get model weights per spectral period
    model_weights_columns = pd.Series(gmpe_list_series) +' weighting'
    model_weights = pd.DataFrame(residuals.model_weights_with_imt)

    model_weights_df = pd.DataFrame()
    for gmpe in residuals.gmpe_list:
        model_weights_df[gmpe] = pd.Series(model_weights.loc[gmpe])
    model_weights_df.columns = model_weights_columns
    
    model_weights_avg = np.array(np.mean(model_weights_df))
    model_weights_avg_df = pd.DataFrame([model_weights_avg],index=
                                      ['Avg over all periods'])
    model_weights_avg_df.columns = model_weights_columns
    final_model_weights_df = pd.concat([model_weights_df, model_weights_avg_df])
    
    # Table and display outputs (need to assign simplified GMPE names)
    final_model_weights_df_output = final_model_weights_df
    model_weights_columns_all_output = {}
    for gmpe in range(0, len(residuals.gmpe_list)):
         tmp = str(residuals.gmpe_list[gmpe_list_series[gmpe]])
         model_weights_columns_all_output[gmpe] = tmp.split('(')[0].replace(
             '\n',', ') + ' LLH Based Model Weight'
    final_model_weights_df_output.columns = list(pd.Series(
        model_weights_columns_all_output))
    final_model_weights_df_output.to_csv(filename, sep = ',')
    display(final_model_weights_df_output)
    
    # Reassign original imts to residuals.imts
    residuals.imts = preserve_imts
    
    # Assign final model weights dataframe to residuals object (for unit test)
    residuals.final_LLH_model_weights_df = final_model_weights_df
    return residuals.final_LLH_model_weights_df

def edr_weights_table(residuals, filename):
    """
    Definition to create a table of model weights per imt based
    on Euclidean distance based ranking (Kale and Akkar, 2013)
    """     
       
    # Preserve original residuals.imts
    preserve_imts = residuals.imts
    
    # Remove non-acceleration imts from residuals.imts for generation of metrics
    imt_append = pd.DataFrame(residuals.imts, index = residuals.imts)
    imt_append.columns = ['imt']
    for imt_idx in range(0, np.size(residuals.imts)):
         if residuals.imts[imt_idx] == 'PGV':
             imt_append = imt_append.drop(imt_append.loc['PGV'])
         if residuals.imts[imt_idx] == 'PGD':
             imt_append = imt_append.drop(imt_append.loc['PGD'])
         if residuals.imts[imt_idx] == 'CAV':
             imt_append = imt_append.drop(imt_append.loc['CAV'])
         if residuals.imts[imt_idx] == 'Ia':
             imt_append = imt_append.drop(imt_append.loc['Ia'])
        
    imt_append_list = pd.DataFrame()
    for idx in range(0, len(imt_append)):
         imt_append_list[idx] = imt_append.iloc[idx]
    imt_append = imt_append.reset_index()
    imt_append_list.columns = imt_append.imt
    residuals.imts = list(imt_append_list)
    
    # Produce series of residuals.gmpe_list
    gmpe_list_series = pd.Series(pd.DataFrame(residuals.gmpe_list, index=
                                        np.arange(0, len(residuals.gmpe_list)
                                                  , 1)).columns)
    
    # Compute EDR based model weights
    EDR_for_weights = residuals.get_edr_values_wrt_spectral_period()
    EDR_per_gmpe = {}
    for gmpe in EDR_for_weights.keys():
        EDR_per_gmpe[gmpe] = EDR_for_weights[gmpe]['EDR']
    EDR_per_gmpe_df = pd.DataFrame(EDR_per_gmpe)
    
    GMPE_EDR_weight = OrderedDict([(gmpe, {}) for gmpe in residuals.gmpe_list])
    for imt in EDR_per_gmpe_df.index:
        total_EDR_per_imt = np.sum(EDR_per_gmpe_df.loc[imt]**-1)
        for gmpe in EDR_for_weights.keys():
            GMPE_EDR_weight[gmpe][imt] = EDR_per_gmpe_df.loc[imt][
                gmpe]**-1/total_EDR_per_imt
    GMPE_EDR_weight_df = pd.DataFrame(GMPE_EDR_weight)
    
    Avg_EDR_weight_per_GMPE = {}
    for gmpe in residuals.gmpe_list:
        Avg_EDR_weight_per_GMPE[gmpe] = np.mean(GMPE_EDR_weight_df[gmpe])
        
    Avg_GMPE_EDR_weight_df = pd.DataFrame(Avg_EDR_weight_per_GMPE, index = [
        'Avg over all periods'])
    Final_EDR_weight_df = pd.concat([GMPE_EDR_weight_df, Avg_GMPE_EDR_weight_df])

    # Table and display outputs (need to assign simplified GMPE names)
    final_model_weights_df_output = Final_EDR_weight_df
    model_weights_columns_all_output={}
    for gmpe in range(0, len(residuals.gmpe_list)):
         tmp = str(residuals.gmpe_list[gmpe_list_series[gmpe]])
         model_weights_columns_all_output[gmpe] = tmp.split('(')[0].replace(
             '\n',', ') + ' EDR Based Model Weight'
    final_model_weights_df_output.columns = list(pd.Series(
        model_weights_columns_all_output))
    final_model_weights_df_output.to_csv(filename, sep = ',')
    display(final_model_weights_df_output)
    
    # Reassign original imts to residuals.imts
    residuals.imts = preserve_imts

def edr_table(residuals, filename):
    """
    Definition to create a table of MDE Norm, sqrt(kappa) and 
    EDR per imt per spectal period (Kale and Akkar, 2013)
    """
    
    # Generate attributes required from residuals for table
    residuals.get_edr_values_wrt_spectral_period()
    
    # Get Kale and Akkar (2013) ranking metrics
    EDR_metrics = {}
    for gmpe in residuals.gmpe_list:
        EDR_metrics[gmpe] = pd.DataFrame(
            residuals.edr_values_wrt_imt[gmpe]).rename(
                columns = {'MDE Norm':str(gmpe) + ' MDE Norm', 'sqrt Kappa': 
                         str(gmpe) + ' sqrt Kappa', 'EDR': str(gmpe) + ' EDR'})
        EDR_metrics_df = EDR_metrics[gmpe]
        average_EDR_metrics_per_gmpe = np.arange(0, len(EDR_metrics_df.columns),
                                               1, dtype = 'float')
        for EDR_metric in range(0, len(EDR_metrics_df.columns)):
            average_EDR_metrics_per_gmpe[EDR_metric] = np.array(np.mean(
                EDR_metrics_df[EDR_metrics_df.columns[EDR_metric]]))      
            average_EDR_metrics_over_imts_df = pd.DataFrame(
                [average_EDR_metrics_per_gmpe], index=['Avg over all periods'])
            average_EDR_metrics_over_imts_df.columns = EDR_metrics_df.columns
            final_EDR_metrics_df = pd.concat([EDR_metrics_df,
                                            average_EDR_metrics_over_imts_df])
            
            # Rename to assign simplified GMPE names for outputted files
            final_EDR_metrics_df_output = final_EDR_metrics_df
            tmp = str(residuals.gmpe_list[gmpe])
            final_EDR_metrics_df_output.columns = list(pd.Series({
                'MDE Norm':tmp.split('(')[0].replace('\n',' ') + ' MDE Norm',
                'sqrt Kappa':tmp.split('(')[0].replace('\n',' ') +' sqrt Kappa',
                'EDR':tmp.split('(')[0].replace('\n',' ') + ' EDR'}))
            final_EDR_metrics_df_output.to_csv(filename.replace(
                '.csv','') + '_%s' %(
                str(residuals.gmpe_list[gmpe]).replace('\n','_').replace(
                    ' ','')).replace('"','') + '.csv', sep = ',')
        display(final_EDR_metrics_df_output)
        
def pdf_table(residuals, filename):
    """
    Definition to create a table of mean and standard deviation for total,
    inter- and intra-event residual distributions (for each GMPE at each
    spectral period
    """  
        
    # Preserve original residuals.imts
    preserve_imts = residuals.imts
    
    # Remove non-acceleration imts from residuals.imts for generation of metrics
    imt_append = pd.DataFrame(residuals.imts, index = residuals.imts)
    imt_append.columns = ['imt']
    for imt_idx in range(0, np.size(residuals.imts)):
        if residuals.imts[imt_idx] == 'PGV':
            imt_append = imt_append.drop(imt_append.loc['PGV'])
        if residuals.imts[imt_idx] == 'PGD':
           imt_append = imt_append.drop(imt_append.loc['PGD'])
        if residuals.imts[imt_idx] == 'CAV':
           imt_append = imt_append.drop(imt_append.loc['CAV'])
        if residuals.imts[imt_idx] == 'Ia':
           imt_append = imt_append.drop(imt_append.loc['Ia'])
        
    imt_append_list = pd.DataFrame()
    for idx in range(0, len(imt_append)):
        imt_append_list[idx] = imt_append.iloc[idx]
    imt_append = imt_append.reset_index()
    imt_append_list.columns = imt_append.imt
    residuals.imts = list(imt_append_list)
    
    # Convert imt_list to array
    if len(residuals.imts) == 1 and residuals.imts[0] == 'PGA':
        input_imt_string = [('PGA', 0)]
        imts_to_plot = pd.DataFrame(input_imt_string, columns=[
            'imt_str','imt_float'])
    else:
        imts_to_plot=pd.DataFrame([imt2tup(imts) for imts in residuals.imts],
                              columns=['imt_str', 'imt_float'])
    for imt_idx in range(0, np.size(residuals.imts)):
        if imts_to_plot.imt_str[imt_idx] == 'PGA':
           imts_to_plot.imt_float.iloc[imt_idx] = 0
    imts_to_plot = imts_to_plot.dropna() # Remove any non-acceleration imt

    # Get all residuals for all gmpes at all imts
    res_statistics = {}
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            res_statistics[gmpe, imt] = residuals.get_residual_statistics_for(
                gmpe, imt)
    
    Mean_Sigma_Intra = {}
    Mean_Sigma_Inter = {}
    Mean_Sigma_Total = {}
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            Mean_Sigma_Intra[gmpe, imt] = res_statistics[gmpe, imt]['Intra event']
            Mean_Sigma_Inter[gmpe, imt] = res_statistics[gmpe, imt]['Inter event']
            Mean_Sigma_Total[gmpe, imt] = res_statistics[gmpe, imt]['Total']

    Mean_Sigma_Intra_df = pd.DataFrame(Mean_Sigma_Intra)
    Mean_Sigma_Inter_df = pd.DataFrame(Mean_Sigma_Inter)
    Mean_Sigma_Total_df = pd.DataFrame(Mean_Sigma_Total)
    
    combined_df = pd.concat([Mean_Sigma_Total_df,
                           Mean_Sigma_Inter_df,
                           Mean_Sigma_Intra_df])
    combined_df.index = ['Total Mean', 'Total Std Dev', 'Inter-Event Mean',
                         'Inter-Event Std Dev','Intra-Event Mean',
                         'Intra-Event Std Dev']
    
    # Create additional dataframe for renaming of headers
    combined_df_output = combined_df
    gmpe_headers = {}
    for gmpe in residuals.gmpe_list:
        for imt in residuals.imts:
            tmp = str(residuals.gmpe_list[gmpe])
            gmpe_headers[gmpe,imt] = str(imt) + '' + tmp.split(
                '(')[0].replace('\n',' ')
            
    combined_df_output.columns = list(pd.Series(gmpe_headers))
    
    combined_df_output.to_csv(filename, sep = ',')
    display(combined_df_output)

    # Reassign original imts to residuals.imts
    residuals.imts = preserve_imts  