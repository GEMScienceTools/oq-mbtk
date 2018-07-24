import numpy as np


def interpolate_ccumul(mfd, threshold):
    """
    Provides a value of exceedance given and MFD and a magnitude
    threshold

    :param mfd:
        An :class:'openquake.hazardlib.mfd.BaseMFD' instance
    """
    #
    # get the cumulative
    magc, occc = get_cumulative(mfd)
    magc = np.array(magc)
    occc = np.array(occc)
    #
    # no extrapolation
    if threshold < min(magc) or threshold > max(magc) + mfd.bin_width:
        msg = 'Theshold magnitude outside the mfd magnitude limits'
        raise ValueError(msg)
    #
    # find rate of exceedance
    idx = np.nonzero(abs(magc - threshold) < 1e-4)
    if len(idx[0]):
        exrate = occc[idx[0]]
    else:
        # find the index of the bin center just below the magnitude
        # threshold
        idx = max(np.nonzero(magc < threshold)[0])
        if threshold > magc[-1]:
            slope = (occc[idx] - occc[idx-1]) / mfd.bin_width
        else:
            slope = (occc[idx+1] - occc[idx]) / mfd.bin_width
        intcp = occc[idx] - slope * magc[idx]
        exrate = slope*threshold + intcp
    return exrate


def get_cumulative(mfd):
    """
    Compute a cumulative MFD from a (discrete) incremental one

    :param mfd:
        An :class:'openquake.hazardlib.mfd.BaseMFD' instance
    :returns:
        Two lists, the first one containing magnitudes values and the
        second one with annual rates of exceedance (m>m0).
     """
    mags = []
    cml = []
    occs = []
    #
    # loading information for the original MFD
    for mag, occ in mfd.get_annual_occurrence_rates():
        mags.append(mag)
        occs.append(occ)
    #
    # shifting mags of half bin
    mags = [m-mfd.bin_width/2 for m in mags]
    #
    # reverting rates
    for occ in reversed(occs):
        if len(cml):
            cml.append(occ+cml[-1])
        else:
            cml.append(occ)
    #
    return mags, cml[::-1]
