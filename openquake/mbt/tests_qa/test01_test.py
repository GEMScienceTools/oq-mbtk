

import os
import numpy
import unittest
import openquake.man.model as model
import openquake.man.tools.csv_output as csv
from openquake.mbt.tools.mfd import EEvenlyDiscretizedMFD

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'test01')


class MFDFromSESMFDFromInputTest(unittest.TestCase):
    """
    """

    def testcase01(self):
        """
        """

        # Bin width
        bwdt = 0.1
        time = 500000

        fname_ses = os.path.join(BASE_DATA_PATH, 'out', 'ruptures_786.csv')
        fname_ssm = os.path.join(BASE_DATA_PATH, 'ssm01.xml')

        # Read catalogue
        cat = csv.get_catalogue_from_ses(fname_ses, time)

        # Find minimum and maximum magnitude
        mmin = numpy.floor(min(cat.data['magnitude'])/bwdt)*bwdt
        mmax = numpy.ceil(max(cat.data['magnitude'])/bwdt)*bwdt+bwdt*0.05

        # Compute MFD from the ses catalogue
        cm_ses = numpy.arange(mmin, mmax, bwdt)
        nobs_ses, _ = numpy.histogram(cat.data['magnitude'], cm_ses)

        # Read input seismic source model
        # print(fname_ssm)
        # conv = SourceConverter(1., 5., 5., 0.1, 10.)
        # mdl = to_python(fname_ssm, conv)
        # print(mdl.src_groups)

        # Create total MFD
        ssm, _ = model.read(fname_ssm)
        cnt = 0
        occ_tot = 0
        for src in ssm:
            if cnt == 0:
                mfd = EEvenlyDiscretizedMFD.from_mfd(src.mfd, bwdt)
            else:
                mfd.stack(src.mfd)
            tmp = numpy.array(src.mfd.get_annual_occurrence_rates())
            occ_tot += sum(tmp[:, 1])
            cnt += 1

        mfdocc = numpy.array(mfd.get_annual_occurrence_rates())

        expected = occ_tot
        computed = cat.get_number_events()/time

        ratio = abs(computed/expected)
        msg = 'Total rates do not match'
        self.assertTrue(ratio < 1., msg)

        if False:
            import matplotlib.pyplot as plt
            plt.plot(mfdocc[:, 0], mfdocc[:, 1], 'x-', label='ses')
            plt.plot(cm_ses[:-1]+bwdt/2, nobs_ses/time, 'o-')
            plt.yscale('log')
            plt.grid()
            plt.legend()
            plt.show()
