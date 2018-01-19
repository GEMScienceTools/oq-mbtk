import numpy as np
import unittest

from pyproj import Proj
from oqmbt.tools.smooth3d import Smoothing3D
from openquake.hazardlib.geo.mesh import Mesh
from openquake.hmtk.seismicity.catalogue import Catalogue


class Smooth3DTestCase(unittest.TestCase):
    """
    """

    def setUp(self):
        #
        #
        self.spch = 2.5
        self.spcv = 2.5
        #
        # set the projection
        self.p = Proj('+proj=lcc +lon_0={:f}'.format(10.5))
        #
        # grid limits
        xlo, ylo = tuple(t/1e3 for t in self.p(10.0, 45.0))
        xup, yup = tuple(t/1e3 for t in self.p(11.0, 46.0))
        #
        # creating a test mesh
        pnts = []
        dlt = 0.01
        for x in np.arange(xlo, xup+dlt, self.spch):
            for y in np.arange(ylo, yup+dlt, self.spch):
                for z in np.arange(0, 20+dlt, self.spcv):
                    pnts.append([x, y, z])
        pnts = np.array(pnts)
        plo, pla = self.p(pnts[:, 0]*1e3, pnts[:, 1]*1e3, inverse=True)
        pnts[:, 0] = plo
        pnts[:, 1] = pla
        self.mesh = Mesh(pnts[:, 0], pnts[:, 1], pnts[:, 2])
        #
        # create a catalogue
        keys = ['longitude', 'latitude', 'depth', 'year', 'magnitude']
        cata = np.array([[10.0, 45.0, 10.0, 2000, 5.0],
                         [10.5, 45.5, 10.0, 2000, 5.0],
                         [10.5, 45.6, 10.0, 2000, 5.0]])
        self.cat = Catalogue()
        self.cat.load_from_array(keys, cata)

    def testcase01(self):
        """
        """
        smooth = Smoothing3D(self.cat, self.mesh, self.spch, self.spcv)
        values = smooth.gaussian(20, [5, 5, 2])
        print('sum:', sum(values))

        if False:
            vsc = 0.01
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            iii = np.nonzero(values > 1e-15)[0]
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(self.mesh.lons, self.mesh.lats, self.mesh.depths*vsc, '.k',
                    alpha=0.2)
            ax.scatter(self.mesh.lons[iii], self.mesh.lats[iii],
                       self.mesh.depths[iii]*vsc, c=np.log10(values[iii]),
                       alpha=0.5)
            ax.invert_zaxis()
            plt.show()
        assert 0 == 1
