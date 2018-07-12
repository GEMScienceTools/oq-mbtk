#!/usr/bin/env python

import os
import re
import glob
import numpy as np

from pathlib import Path
from scipy import interpolate

from openquake.hazardlib.geo import Line, Point


def _from_lines_to_array(lines):
    """
    :param lines:
        A list of :class:`openquake.hazardlib.geo.line.Line` instances
    :returns:
        A 2D :class:`numpy.ndarray` instance with 3 columns and as many rows
        as the number of points composing all the lines
    """
    out = []
    for line in lines:
        for pnt in line.points:
            x = float(pnt.longitude)
            y = float(pnt.latitude)
            z = float(pnt.depth)
            out.append([x, y, z])
    return np.array(out)


def _from_line_to_array(line):
    """
    :param list line:
        A :class:`openquake.hazardlib.geo.line.Line` instance
    :returns:
        A 2D :class:`numpy.ndarray` instance with 3 columns and as many rows
        as the number of points composing the line
    """
    assert isinstance(line, Line)
    out = np.array((len(line.points, 3)))
    for i, pnt in enumerate(line.points):
        out[:, 0] = float(pnt.longitude)
        out[:, 1] = float(pnt.latitude)
        out[:, 2] = float(pnt.depth)
    return out


class ProfileSet():
    """
    A list of :class:`openquake.hazardlib.geo.line.Line` instances
    """

    def __init__(self, profiles=[]):
        self.profiles = profiles

    @classmethod
    def from_files(cls, fname):
        """
        """
        lines = []
        for filename in sorted(glob.glob(os.path.join(fname, 'cs*.csv'))):
            tmp = np.loadtxt(filename)
            pnts = []
            for i in range(tmp.shape[0]):
                pnts.append(Point(tmp[i, 0], tmp[i, 1], tmp[i, 2]))
            lines.append(Line(pnts))
            #
            # Profile ID
            fname = Path(os.path.basename(filename)).stem
            sid = re.sub('^cs_', '', fname)
            sid = '%03d' % int(sid)
        return cls(lines)


    def smooth(self, method='linear'):

        arr = _from_lines_to_array(self.profiles)

        x1 = np.amin(arr[:, 0])
        x2 = np.amax(arr[:, 0])
        y1 = np.amin(arr[:, 1])
        y2 = np.amax(arr[:, 1])

        xv = np.linspace(x1, x2, 100)
        yv = np.linspace(y1, y2, 100)
        grd = interpolate.griddata((arr[:, 0], arr[:, 1]), arr[:, 2],
                                   (xv[None, :], yv[:, None]), method=method)

        if True:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            for pro in self.profiles:
                tmp = [[p.longitude, p.latitude, p.depth] for p in pro.points]
                tmp = np.array(tmp)
                ax.plot(tmp[:, 0], tmp[:, 1], tmp[:, 2], 'x--b', markersize=2)
            xg, yg = np.meshgrid(xv, yv)
            ax.plot(xg.flatten(), yg.flatten(), grd.flatten(), '.r', markersize=1)
            plt.show()

        return grd
