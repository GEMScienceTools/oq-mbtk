import toml
import glob
import numpy as np
from openquake.baselib import sap
from openquake.man.checking_utils.mfds_and_rates_utils import get_mags_rates


def main(xml_pattern: str, label: str, config_fname: str):
    """
    Given a pattern to a file or a set of files, the label identifying the
    source (or the set of sources) and a configuration file for characterizing
    earthquake occurrence using a workflow, check that the total MFD of the
    sources is equal to the orinal MFD in the config file.
    """

    investigation_t = 1.0

    # Parsing config
    config = toml.load(config_fname)

    # Computing the total MFD
    rates = []
    for source_model_fname in sorted(glob.glob(xml_pattern)):
        mag, rate = get_mags_rates(source_model_fname, investigation_t)
        rates.append([mag, rate])
    rates = np.array(rates)

    agr = float(config['sources'][label]['agr'])
    bgr = float(config['sources'][label]['bgr'])
    mmin = float(config['mmin'])
    binw = float(config['bin_width'])
    mmax = float(config['sources'][label]['mmax'])
    mags = np.arange(mmin, mmax+0.1*binw, binw)
    rates_gr = 10**(agr-bgr*mags[:-1]) - 10**(agr-bgr*mags[1:])

    np.testing.assert_almost_equal(rates[:, 1], rates_gr, decimal=3)


descr = 'The pattern to the .xml files to be considered'
main.labels = descr
descr = 'The label identifying the source investigated'
main.config_fname = descr
descr = 'The path to the configuration file'
main.config_fname = descr

if __name__ == '__main__':
    sap.run(main)
