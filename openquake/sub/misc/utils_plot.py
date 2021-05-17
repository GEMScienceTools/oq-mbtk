"""
:module:`openquake.sub.tests.misc.plot_utils`
"""

import numpy as np

try:
    from mayavi import mlab
    MAYAVI = True
except:
    MAYAVI = False


def plotter(profiles, smsh):
    """
    Mesh and profiles plotter.
    :param profiles:
        A list of :class:`openquake.hazardlib.geo.line.Line` instances
    :param smsh:
        A :class:`numpy.ndarray` instance
    """
    #
    if MAYAVI:
        _ = mlab.figure(size=(600, 400))
        #
        scl = -0.01
        delta = 0.1
        wdt = 0.001
        # initialize extremes
        xmi = +1e10
        ymi = +1e10
        zmi = +1e10
        xma = -1e10
        yma = -1e10
        zma = -1e10
        # plotting profiles
        for l in profiles:
            coo = [[p.longitude, p.latitude, p.depth*scl] for p in l]
            coo = np.array(coo)
            mlab.plot3d(coo[:, 0], coo[:, 1], coo[:, 2], color=(0, 1, 0),
                        line_width=wdt, opacity=0.2)
            mlab.points3d(coo[0, 0], coo[0, 1], coo[0, 2], color=(1., 0.5, 0.),
                        line_width=wdt, scale_factor=0.1)
            xmi = min(xmi, np.amin(coo[:, 0]) - delta)
            xma = max(xma, np.amax(coo[:, 0]) + delta)
            ymi = min(ymi, np.amin(coo[:, 1]) - delta)
            yma = max(yma, np.amax(coo[:, 1]) + delta)
            zmi = min(zmi, np.amin(coo[:, 2]) - delta)
            zma = max(zma, np.amax(coo[:, 2]) + delta)
        # plotting mesh
        smsh[:, :, 2] *= scl
        for i in range(smsh.shape[0]):
            k = np.isfinite(smsh[i, :, 0])
            if np.any(k):
                _ = mlab.plot3d(smsh[i, k, 0], smsh[i, k, 1], smsh[i, k, 2],
                                color=(1, 0, 0), line_width=wdt, tube_radius=0.01)
        for i in range(smsh.shape[1]):
            k = np.isfinite(smsh[:, i, 0])
            if sum(k):
                _ = mlab.plot3d(smsh[k, i, 0], smsh[k, i, 1], smsh[k, i, 2],
                                color=(1, 0, 0), line_width=wdt, tube_radius=0.01)
        # decoration
        axes = mlab.axes(extent=[xmi, xma, ymi, yma, zmi, zma])
        axes.label_text_property.font_size = 6
        axes.axes.font_factor = 1.0
        mlab.xlabel('Longitude\n')
        mlab.ylabel('Latitude\n')
        mlab.zlabel('Depth\n [km*0.01]')
        mlab.show()
