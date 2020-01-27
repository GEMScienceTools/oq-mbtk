# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4

#
# LICENSE
#
# Copyright (c) 2017 GEM Foundation
#
# The Catalogue Toolkit is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
#
# You should have received a copy of the GNU Affero General Public License
# with this download. If not, see <http://www.gnu.org/licenses/>

import numpy as np
from copy import deepcopy
from math import fabs
from openquake.cat.isc_homogenisor import MagnitudeConversionRule
from openquake.cat.utils import _to_latex, _set_string


def _piecewise_linear_sigma(sigmas, params, m):
    """
    Simple function to return the sigma for a given magnitude in a
    multi-segment piecewise linear function when sigma changes for each
    segment
    """
    turning_points = params[int(len(params) / 2):-1]
    assert (len(sigmas) - 1) == len(turning_points)
    for sigma, turning_point in zip(sigmas[:-1], turning_points):
        if m < turning_point:
            return sigma
    return sigmas[-1]


def piecewise_linear(params, xval):
    """
    Implements the piecewise linear analysis function as a vector
    """
    n_params = len(params)
    if fabs(float(n_params / 2) - float(n_params) / 2.) > 1E-7:
        raise ValueError(
            'Piecewise Function requires 2 * nsegments parameters')
    n_seg = n_params / 2
    if n_seg == 1:
        return params[1] + params[0] * xval
    gradients = params[0:n_seg]
    turning_points = params[n_seg: -1]
    c_val = params[-1]
    for iloc, slope in enumerate(gradients):
        if iloc == 0:
            yval = (slope * xval) + c_val

        else:
            select = np.where(xval >= turning_points[iloc - 1])[0]
            # Project line back to x = 0
            c_val = c_val - turning_points[iloc - 1] * slope
            yval[select] = (slope * xval[select]) + c_val
        if iloc < (n_seg - 1):
            # If not in last segment then re-adjust intercept to new turning
            # point
            c_val = (slope * turning_points[iloc]) + c_val
    return yval


class GeneralFunction(object):
    """
    Class (notionally abstract) for defining the properties of a fitting
    function
    """
    def __init__(self):
        """
        Instantiate
        """
        self.params = []

    def run(self, params, xval):
        """
        Executes the funtion
        :param list params:
            Functon parameters
        :param numpy.ndarray xval:
            Input data
        """
        raise NotImplementedError

    def get_string(self, output_string, input_string):
        """
        Returns a string describing the equation with its final parameters
        :param str output_string:
            Name of output parameter
        :param str input_string:
            Name of input parameter
        """
        raise NotImplementedError

    def to_conversion_rule(self, author, scale, params, sigma, start_date=None,
                           end_date=None, key=None, model_name=None):
        """
        Returns as model as a magnitude conversion rule for use with
        ISCHomogenisor
        """
        raise NotImplementedError


class PiecewiseLinear(GeneralFunction):
    """
    Implements a Piecewise linear functional form with N-segements
    """

    def run(self, params, xval):
        """
        Executes the model
        :param list params:
            Contolling parameters as
            [slope_1, slope_2, ..., slope_i, turning_point1, turning_point2,
             ..., turning_point_i-1, intercept]
        :param numpy.ndarray xval:
            Input data
        """
        self.params = []
        n_params = len(params)
        if fabs(float(n_params / 2) - float(n_params) / 2.) > 1E-7:
            raise ValueError(
                'Piecewise Function requires 2 * nsegments parameters')

        n_seg = n_params / 2

        if n_seg == 1:
            return params[1] + params[0] * xval
        gradients = params[0:n_seg]
        turning_points = params[n_seg: -1]
        c_val = params[-1]
        for iloc, slope in enumerate(gradients):
            if iloc == 0:
                yval = (slope * xval) + c_val
                self.params.append((c_val, slope, turning_points[iloc]))
            else:
                select = np.where(xval >= turning_points[iloc - 1])[0]
                # Project line back to x = 0
                c_val = c_val - turning_points[iloc - 1] * slope
                yval[select] = (slope * xval[select]) + c_val
                if iloc < (n_seg - 1):
                    self.params.append(
                        (c_val, slope, turning_points[iloc - 1]))
                else:
                    # In the last segment
                    self.params.append(
                        (c_val, slope, turning_points[iloc - 1]))
            if iloc < (n_seg - 1):
                # If not in last segment then re-adjust intercept to turning
                # turning point
                c_val = (slope * turning_points[iloc]) + c_val

        return yval

    def get_string(self, output_string, input_string):
        """
        Returns the title string
        """
        n_seg = len(self.params)
        full_string = []
        for iloc, params in enumerate(self.params):
            eq_string = "{:s} = {:.3f} {:s} {:s}".format(
                _to_latex(output_string),
                params[0],
                _set_string(params[1]),
                _to_latex(input_string))
            if iloc == 0:
                cond_string = eq_string + "    for {:s} < {:.3f}".format(
                    _to_latex(input_string),
                    params[2])
            elif iloc == (n_seg - 1):
                cond_string = eq_string + "    for {:s} $\geq$ {:.3f}".format(
                    _to_latex(input_string),
                    params[2])
            else:
                cond_string = eq_string + \
                    "    for {:.3f} $\leq$ {:s} < {:.3f}".format(
                        self.params[iloc - 1][2],
                        _to_latex(input_string),
                        params[2])
            full_string.append(cond_string)
        return "\n".join([case_string for case_string in full_string])

    def to_conversion_rule(self, author, scale, params, sigma, start_date=None,
                           end_date=None, key=None, model_name=None):
        """
        Returns as model as a magnitude conversion rule for use with
        ISCHomogenisor
        """
        # Vector piecewise linear function is working but scalar is not -
        # this is ugly but it works for now
        mean = lambda x: piecewise_linear(params, np.array([x]))[0]
        if isinstance(sigma, (list, tuple)):
            stddev = lambda x: _piecewise_linear_sigma(sigma, params, x)
        else:
            stddev = deepcopy(sigma)
        return MagnitudeConversionRule(author, scale, mean, stddev,
                                       start_date, end_date, key, model_name)


class Polynomial(GeneralFunction):
    """
    Implements a nth-order polynomial function
    """
    def run(self, params, xval):
        """
        Returns the polynomial f(xval) where the order is defined by the
        number of params, i.e.
        yval = \SUM_{i=1}^{Num Params} params[i] * (xval ** i - 1)
        """
        yval = np.zeros_like(xval)
        for iloc, param in enumerate(params):
            yval += (param * (xval ** float(iloc)))
        self.params = params
        return yval

    def get_string(self, output_string, input_string):
        """
        Returns the title string
        """
        base_string = "{:s} = ".format(_to_latex(output_string))
        for iloc, param in enumerate(self.params):
            if iloc == 0:
                base_string = base_string + "{:.3f}".format(param)
            elif iloc == 1:
                base_string = base_string + " {:s}{:s}".format(
                    _set_string(param),
                    _to_latex(input_string))
            else:
                base_string = base_string + (" %s%s$^%d$" % (
                    _set_string(param),
                    _to_latex(input_string),
                    iloc))
        return base_string

    def to_conversion_rule(self, author, scale, params, sigma, start_date=None,
                           end_date=None, key=None, model_name=None):
        """
        Returns a
        """
        mean = lambda x: np.sum([param * (x ** float(iloc))
                                 for iloc, param in enumerate(params)])
        if isinstance(sigma, float):
            stddev = lambda x: sigma
        else:
            stddev = deepcopy(sigma)
        return MagnitudeConversionRule(author, scale, mean, stddev,
                                       start_date, end_date, key, model_name)


class Exponential(GeneralFunction):
    """
    Implements an exponential function of the form y = exp(a + bX) + c
    """
    def run(self, params, xval):
        """
        Returns an exponential function
        """
        assert len(params) == 3
        self.params = params
        return np.exp(params[0] + params[1] * xval) + params[2]

    def get_string(self, output_string, input_string):
        """
        Returns the title string
        """
        base_string = "%s = e$^{(%.3f %s %s)}$ %s" % (
            _to_latex(output_string),
            self.params[0],
            _set_string(self.params[1]),
            self._to_latex(input_string),
            _set_string(self.params[2]))
        return base_string

    def _to_latex(self, string):
        """
        For a string given in the form XX(YYYY) returns the LaTeX string to
        place bracketed contents as a subscript
        :param string:
        """
        lb = string.find("(")
        ub = string.find(")")
        return string[:lb] + ("_{%s}" % string[lb+1:ub])

    def to_conversion_rule(self, author, scale, params, sigma, start_date=None,
                           end_date=None, key=None, model_name=None):
        """
        Returns an instance of :class:MagnitudeConversionRule
        """
        mean = lambda x: np.exp(params[0] + params[1] * x) + params[2]
        if isinstance(sigma, float):
            stddev = lambda x: sigma
        else:
            stddev = deepcopy(sigma)
        return MagnitudeConversionRule(author, scale, mean, stddev,
                                       start_date, end_date, key, model_name)


def _2segment_scalar(params, m, m_c):
    """
    Simple scalar function used to return the magnitude from a two-segment
    linear model with a fixed corner magnitude
    """
    if m < m_c:
        return params[0] * m + params[2]
    else:
        cval = (params[0] * m_c + params[2]) - (m_c * params[1])
        return cval + params[1] * m


class TwoSegmentLinear(GeneralFunction):
    """
    Implements a two-segement piecewise linear model with a fixed (i.e. not
    optimisable) corner magnitude
    """
    def __init__(self, corner_magnitude):
        """
        :param float corner_magnitude:
            Corner magnitude
        """
        super(TwoSegmentLinear, self).__init__()
        setattr(self, "corner_magnitude", corner_magnitude)

    def run(self, params, xval):
        """
        Runs the model
        """
        yval = params[0] * xval + params[2]
        cval = params[0] * self.corner_magnitude + params[2]
        cval -= (self.corner_magnitude * params[1])
        idx = xval > self.corner_magnitude
        yval[idx] = cval + params[1] * xval[idx]
        self.params = [[params[0], params[2]], [params[1], cval]]
        return yval

    def get_string(self, output_string, input_string):
        """
        Returns the title string
        """
        base_string = "{:s} = ".format(_to_latex(output_string))
        # Equation 1
        upper_string = base_string + \
            "{:.3f} {:s}{:s}    for {:s} < {:.2f}".format(
                self.params[0][1],
                _set_string(self.params[0][0]),
                _to_latex(input_string),
                _to_latex(input_string),
                self.corner_magnitude)
        lower_string = base_string + \
            "{:.3f} {:s}{:s}    for {:s} $\geq$ {:.2f}".format(
                self.params[1][1],
                _set_string(self.params[1][0]),
                _to_latex(input_string),
                _to_latex(input_string),
                self.corner_magnitude)
        return "\n".join([upper_string, lower_string])

    def to_conversion_rule(self, author, scale, params, sigma, start_date=None,
                           end_date=None, key=None, model_name=None):
        """
        Returns an instance of :class:MagnitudeConversionRule
        """
        mean = lambda x: _2segment_scalar(params, x, self.corner_magnitude)
        if isinstance(sigma, (list, tuple)):
            stddev = lambda x:\
                sigma[0] if x < self.corner_magnitude else sigma[1]
        else:
            stddev = deepcopy(sigma)
        return MagnitudeConversionRule(author, scale, mean, stddev,
                                       start_date, end_date, key, model_name)


function_map = {"piecewise": PiecewiseLinear,
                "polynomial": Polynomial,
                "exponential": Exponential,
                "2segment": TwoSegmentLinear}
