"""
:module:`openquake.sub.tests.misc.plot_utils`
"""

import numpy as np

from vispy import app, gloo, visuals


def plotter(app.Canvas):

    self.visuals = []
    # plotting profiles
    for l in profiles:
        coo = [[p.longitude, p.latitude, p.depth*scl] for p in l]
        coo = np.array(coo)
        line = visuals.LinePlotVisual(data=coo, color='green',
                                        width=1.0)
        self.visuals.append(line)
    # plotting mesh
    smsh[:, :, 2] *= scl
    for i in range(smsh.shape[0]):
        line = visuals.LinePlotVisual(data=smsh[i, :, :], color='red',
                                        width=1.0)
        self.visuals.append(line)
    for i in range(smsh.shape[1]):
        line = visuals.LinePlotVisual(data=smsh[:, i, :], color='red',
                                        width=1.0)
        self.visuals.append(line)
    self.show()
