"""
module test01
"""
import os
import glob
import tempfile
import subprocess
import shutil
import unittest
import numpy
import openquake.man.model as model
import openquake.man.tools.csv_output as csv
from openquake.mbt.tools.mfd import EEvenlyDiscretizedMFD

BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), 'test01')


def get_fname(folder, pattern):
    """
    Find a single file
    :param folder: the output folder
    :param pattern: the
    """
    patt = os.path.join(folder, pattern)
    lst = glob.glob(patt)
    assert len(lst) == 1
    return lst[0]


class MFDFromSESMFDFromInputTest(unittest.TestCase):
    """
    Check that the MFD computed on the source model matches the one
    obtained from the computed SES
    """

    def testcase01(self):
        """ Test case 1"""

        # Bin width
        bwdt = 0.1
        time = 500000

        folder = tempfile.mkdtemp()
        fname_ssm = os.path.join(BASE_DATA_PATH, 'ssm01.xml')
        fname_ini = os.path.join(BASE_DATA_PATH, 'job.ini')
        print(fname_ini)
        tmps = 'if [ -d "${:s}" ]; then rm -Rf $WORKING_DIR; fi'
        command = tmps.format(folder)
        subprocess.run(command, shell=True)
        command = 'oq engine --run {:s} --exports csv -p export_dir={}'.format(
            fname_ini, folder)
        subprocess.run(command, shell=True)
        fname_ses = get_fname(folder, 'ruptures_*.csv')
        print(fname_ses)

        # Read catalogue
        cat = csv.get_catalogue_from_ses(fname_ses, time)

        # Find minimum and maximum magnitude
        mmin = numpy.floor(min(cat.data['magnitude'])/bwdt)*bwdt
        mmax = numpy.ceil(max(cat.data['magnitude'])/bwdt)*bwdt+bwdt*0.05

        # Compute MFD from the ses catalogue
        cm_ses = numpy.arange(mmin, mmax, bwdt)
        nobs_ses, _ = numpy.histogram(cat.data['magnitude'], cm_ses)

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

        shutil.rmtree(folder)
