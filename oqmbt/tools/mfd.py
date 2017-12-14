"""
"""

import copy
import numpy as np

from openquake.hazardlib.mfd import (TruncatedGRMFD, EvenlyDiscretizedMFD,
                                     ArbitraryMFD)

log = True
log = False


def get_moment_from_mfd(mfd):
    """
    This computes the total scalar seismic moment released per year by a
    source

    :parameter mfd:
        An instance of openquake.hazardlib.mfd
    :returns:
        A float corresponding to the rate of scalar moment released
    """
    if isinstance(mfd, TruncatedGRMFD):
        return mfd._get_total_moment_rate()
    elif isinstance(mfd, (EvenlyDiscretizedMFD, ArbitraryMFD)):
        occ_list = mfd.get_annual_occurrence_rates()
        mo_tot = 0.0
        for occ in occ_list:
            mo_tot += occ[1] * 10.**(1.5*occ[0] + 9.05)
    else:
        raise ValueError('Unrecognised MFD type: %s' % type(mfd))
    return mo_tot


def get_evenlyDiscretizedMFD_from_truncatedGRMFD(mfd, bin_width=None):
    """
    This function converts a double truncated Gutenberg Richter distribution
    into an almost equivalent discrete representation.

    :parameter:
        A instance of :class:`~openquake.hazardlib.mfd.TruncatedGRMFD`
    :return:
        An instance of :class:`~openquake.hazardlib.mfd.EvenlyDiscretizedMFD`
    """
    assert isinstance(mfd, TruncatedGRMFD)
    agr = mfd.a_val
    bgr = mfd.b_val
    bin_width = mfd.bin_width
    left = np.arange(mfd.min_mag, mfd.max_mag, bin_width)
    rates = 10.**(agr-bgr*left)-10.**(agr-bgr*(left+bin_width))
    return EvenlyDiscretizedMFD(mfd.min_mag+bin_width/2.,
                                bin_width,
                                list(rates))


class EEvenlyDiscretizedMFD(EvenlyDiscretizedMFD):

    @classmethod
    def from_mfd(self, mfd, bin_width=None):
        """
        """
        if isinstance(mfd, EvenlyDiscretizedMFD):
            return EEvenlyDiscretizedMFD(mfd.min_mag, mfd.bin_width,
                                         mfd.occurrence_rates)
        elif isinstance(mfd, TruncatedGRMFD):
            tmfd = get_evenlyDiscretizedMFD_from_truncatedGRMFD(mfd, bin_width)
            return EEvenlyDiscretizedMFD(tmfd.min_mag, tmfd.bin_width,
                                         tmfd.occurrence_rates)
        else:
            raise ValueError('Unsupported MFD type')

    def stack(self, mfd2):
        """
        This function stacks a mfd represented as discrete histograms.

        :parameter mfd2:
            Instance of :class:`~openquake.hazardlib.mfd.EvenlyDiscretizedMFD`
        """

        if log:
            print ('===================================================')

        mfd1 = self
        bin_width = self.bin_width
        #
        # check bin width of the MFD to be added
        if (isinstance(mfd2, EvenlyDiscretizedMFD) and
                abs(mfd2.bin_width - bin_width) > 1e-10):
            if log:
                print ('resampling mfd2 - binning')
            mfd2 = mfd_resample(bin_width, mfd2)
        #
        # this is the difference between the rounded mmin and the original mmin
        dff = abs(np.floor((mfd2.min_mag+0.1*bin_width)/bin_width)*bin_width -
                  mfd2.min_mag)
        #
        #
        if dff > 1e-7:
            if log:
                print ('resampling mfd2 - homogenize mmin')
                print ('                - delta: {:.2f}'.format(dff))
                print ('                - original mmin: {:.2f}'.format(mfd2.min_mag))
            mfd2 = mfd_resample(bin_width, mfd2)
        #
        # this is the difference between the rounded mmin and the original mmin
        dff = abs(np.floor((self.min_mag+0.1*bin_width)/bin_width)*bin_width -
                  self.min_mag)
        #
        #
        if dff > 1e-7:
            if log:
                print ('resampling mfd1 - homogenize mmin')
                print ('                - delta: {:.2f}'.format(dff))
                print ('                - original mmin: {:.2f}'.format(mfd1.min_mag))
            mfd1 = mfd_resample(bin_width, mfd1)

        # mfd1 MUST be the one with the mininum minimum magnitude
        if mfd1.min_mag > mfd2.min_mag:
            if log:
                print ('SWAPPING')
            tmp = mfd2
            mfd2 = mfd1
            mfd1 = tmp

        # Find the delta index
        delta = 0
        tmpmag = mfd1.min_mag
        while abs(tmpmag - mfd2.min_mag) > 0.1*bin_width:
            delta += 1
            tmpmag += bin_width

        rates = list(np.zeros(len(mfd1.occurrence_rates)))
        mags = list(mfd1.min_mag+np.arange(len(rates))*bin_width)

        # Add to the rates list the occurrences included in the mfd with the
        # lowest minimum magnitude
        for idx, occ in enumerate(mfd1.occurrence_rates):
            rates[idx] += occ

        #  if len(mfd2.occurrence_rates)+delta >= len(rates):
        if log:
            print ('-------------')
            print ('-- mfd2')
            print (len(mfd2.occurrence_rates), '>=', len(rates))
            print (mfd2.bin_width)
            print (mfd2.min_mag)
            print (mfd2.occurrence_rates)
            print ('-- mfd1')
            print (mfd1.bin_width)
            print (mfd1.min_mag)
            print (mfd1.occurrence_rates)

        magset = set(mags)

        for idx, (mag, occ) in enumerate(mfd2.get_annual_occurrence_rates()):

            # Check that we add occurrences to the right bin
            try:
                if len(rates) > idx+delta:
                    assert abs(mag - mags[idx+delta]) < 1e-5
            except:
                print ('mag:     :', mag)
                print ('mag rates:', mags[idx+delta])
                print ('delta    :', delta)
                print ('diff     :', abs(mag - mags[idx+delta]))
                raise ValueError('Staking wrong bins')

            if log:
                print (idx, idx+delta, len(mfd2.occurrence_rates), len(rates))
                print (mag, occ)

            if len(rates) > idx+delta:
                rates[idx+delta] += occ
            else:
                if log:
                    print ('Adding mag:', mag, occ)

                tmp_mag = mags[-1] + bin_width
                while tmp_mag < mag-0.1*bin_width:
                    tmp_mag += bin_width
                    delta += 1
                    if set([tmp_mag]) not in magset:
                        rates.append(0.0)
                        mags.append(tmp_mag)
                        magset = magset | set([tmp_mag])
                    else:
                        raise ValueError('This magnitude bin is already included')

                rates.append(occ)
                mags.append(mag)

        assert (sum(mfd1.occurrence_rates) + sum(mfd2.occurrence_rates) -
                sum(rates)) < 1e-5

        if log:
            print ('Sum mfd1 :', sum(mfd1.occurrence_rates))
            print ('Sum mfd2 :', sum(mfd2.occurrence_rates))
            print ('Sum rates:', sum(rates))

        self.min_mag = mfd1.min_mag
        self.bin_width = bin_width
        self.occurrence_rates = rates


def mfd_resample(bin_width, mfd):
    tol = 1e-10
    if bin_width > mfd.bin_width+tol:
        return mfd_upsample(bin_width, mfd)
    else:
        return mfd_downsample(bin_width, mfd)


def mfd_downsample(bin_width, mfd):
    """
    :parameter float bin_width:
    :parameter mfd:
    """

    ommin = mfd.min_mag
    ommax = mfd.min_mag + len(mfd.occurrence_rates) * mfd.bin_width

    if log:
        print ('ommax     ', ommax)
        print ('bin_width ', mfd.bin_width)

    # check that the new min_mag is a multiple of the bin width
    min_mag = np.floor(ommin / bin_width) * bin_width
    # lower min mag to make sure we cover the entire magnitude range
    while min_mag-bin_width/2 > mfd.min_mag-mfd.bin_width/2:
        min_mag -= bin_width
    # preparing the list wchi will collect data
    dummy = []
    mgg = min_mag + bin_width / 2
    while mgg < (ommax + 0.51 * mfd.bin_width):
        if log:
            print (mgg, ommax + mfd.bin_width/2)
        dummy.append(mgg)
        mgg += bin_width

    # prepare the new array for occurrences
    nocc = np.zeros((len(dummy), 4))

    if log:
        print ('CHECK', len(nocc), len(dummy))
        print (dummy)

    #
    boun = np.zeros((len(mfd.occurrence_rates), 4))
    for idx, (mag, occ) in enumerate(mfd.get_annual_occurrence_rates()):
        boun[idx, 0] = mag
        boun[idx, 1] = mag-mfd.bin_width/2
        boun[idx, 2] = mag+mfd.bin_width/2
        boun[idx, 3] = occ
    # init
    for idx in range(0, len(nocc)):
        mag = min_mag+bin_width*idx
        nocc[idx, 0] = mag
        nocc[idx, 1] = mag-bin_width/2
        nocc[idx, 2] = mag+bin_width/2

    rat = bin_width/mfd.bin_width
    tol = 1e-10

    for iii, mag in enumerate(list(nocc[:, 0])):
        idx = np.nonzero(nocc[iii, 1] > (boun[:, 1]-tol))[0]
        idxa = None
        if len(idx):
            idxa = np.amax(idx)
        idx = np.nonzero(nocc[iii, 2] > boun[:, 2]-tol)[0]
        idxb = None
        if len(idx):
            idxb = np.amax(idx)

        if idxa is None and idxb is None and nocc[iii, 2] > boun[0, 1]:
            nocc[0, 3] = ((nocc[iii, 2] - boun[0, 1]) / mfd.bin_width *
                          boun[0, 3])
        elif idxa is None and idxb is None:
            pass
        elif idxa == 0 and idxb is None:
            # This is the first bin when the lower limit of the two FMDs is
            # not the same
            nocc[iii, 3] += rat * boun[idxa, 3]
        elif nocc[iii, 1] > boun[-1, 2]:
            # Empty bin
            pass
        elif idxa > idxb:
            # Bin entirely included in a bin of the original MFD
            nocc[iii, 3] += rat * boun[idxa, 3]
        else:
            dff = (boun[idxa, 2] - nocc[iii, 1])
            ra = dff / mfd.bin_width
            nocc[iii, 3] += ra * boun[idxb, 3]

            if len(boun) > 1 and nocc[iii, 1] < boun[-2, 2]:
                dff = (nocc[iii, 2] - boun[idxa, 2])
                ra = dff / mfd.bin_width
                nocc[iii, 3] += ra * boun[idxa+1, 3]

    idx0 = np.nonzero(nocc[:, 3] < 1e-20)
    idx1 = np.nonzero(nocc[:, 3] > 1e-20)
    if np.any(idx0 == 0):
        raise ValueError('Rates in the first bin are equal to 0')
    elif len(idx0):
        nocc = nocc[idx1[0], :]
    else:
        pass

    smmn = sum(nocc[:, 3])
    smmo = sum(mfd.occurrence_rates)

    if log:
        print (nocc)
        print ('SUMS:', smmn, smmo)

    assert abs(smmn-smmo) < 1e-5
    return EvenlyDiscretizedMFD(nocc[0, 0], bin_width, list(nocc[:, 3]))


def mfd_upsample(bin_width, mfd):
    ommin = mfd.min_mag
    ommax = mfd.min_mag + len(mfd.occurrence_rates) * mfd.bin_width
    # check that the new min_mag is a multiple of the bin width
    min_mag = np.floor(ommin / bin_width) * bin_width
    max_mag = np.ceil(ommax / bin_width) * bin_width
    # prepare the new array for occurrences
    nocc = np.zeros((int((max_mag-min_mag)/bin_width+1), 4))
    # set the new array
    for idx, mag in enumerate(np.arange(min_mag, max_mag, bin_width)):
        nocc[idx, 0] = mag
        nocc[idx, 1] = mag-bin_width/2
        nocc[idx, 2] = mag+bin_width/2
    # assigning occurrences
    for mag, occ in mfd.get_annual_occurrence_rates():
        idx = np.nonzero((mag-mfd.bin_width/2) > nocc[:, 1])[0]
        idxa = None
        if len(idx):
            idxa = np.amax(idx)
        idx = np.nonzero((mag+mfd.bin_width/2) > nocc[:, 2])[0]
        idxb = None
        if len(idx):
            idxb = np.amax(idx)
        if idxa > idxb:
            nocc[idxa, 3] += occ
        else:
            ra = (nocc[idxa, 2] - (mag-mfd.bin_width/2)) / mfd.bin_width
            nocc[idxa, 3] += occ*ra
            if (1.0-ra) > 1e-10:
                nocc[idxa+1, 3] += occ*(1-ra)
    smmn = sum(nocc[:, 3])
    smmo = sum(mfd.occurrence_rates)
    assert abs(smmn-smmo) < 1e-5

    idxs = set(np.arange(0, len(nocc[:,3])))
    iii = len(nocc[:,3])-1
    while nocc[iii, 3] < 1e-10:
        idxs = idxs - set([iii])
        iii -= 1

    return EvenlyDiscretizedMFD(nocc[0, 0], bin_width,
                                list(nocc[list(idxs), 3]))
