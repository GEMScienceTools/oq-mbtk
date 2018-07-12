#!/usr/bin/env python

import os
import glob
import numpy

from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.source import ComplexFaultSource
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.source import ComplexFaultSource
from openquake.hazardlib.const import TRT
from openquake.hazardlib.mfd import TruncatedGRMFD
from openquake.hazardlib.scalerel.strasser2010 import StrasserInterface

DEFAULTS = {'source_id': '0',
            'name': 'None',
            'tectonic_region_type': TRT.SUBDUCTION_INTERFACE,
            'mfd': TruncatedGRMFD(5.0, 6.0, 0.1, 5.0, 1.0),
            'rupture_mesh_spacing': 2,
            'magnitude_scaling_relationship': StrasserInterface(),
            'rupture_aspect_ratio': 4.,
            'temporal_occurrence_model': PoissonTOM(1.0),
            'rake': 90, }


class EdgesSet():
    """
    :param edges:
        A list of :class:`openquake.hazardlib.geo.line.Line` instances
    """

    def __init__(self, edges=[]):
        self.edges = edges

    @classmethod
    def from_files(cls, fname):
        """
        """
        lines = []
        for filename in sorted(glob.glob(os.path.join(fname, 'edge*.csv'))):
            tmp = numpy.loadtxt(filename)
            pnts = []
            for i in range(tmp.shape[0]):
                pnts.append(Point(tmp[i, 0], tmp[i, 1], tmp[i, 2]))
            lines.append(Line(pnts))
        return cls(lines)

    def get_complex_fault(self, params={}):
        """
        :param params
        """
        p = DEFAULTS
        #
        # update the default parameters
        for key in params:
            p[key] = params[key]
        #
        # create the complex fault source instance
        return ComplexFaultSource(p['source_id'],
                                  p['name'],
                                  p['tectonic_region_type'],
                                  p['mfd'],
                                  p['rupture_mesh_spacing'],
                                  p['magnitude_scaling_relationship'],
                                  p['rupture_aspect_ratio'],
                                  p['temporal_occurrence_model'],
                                  self.edges,
                                  p['rake'])
