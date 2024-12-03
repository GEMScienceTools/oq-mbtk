"""
module:`openquake.mbt.tool.mfd`
"""

import scipy
import numpy as np
from scipy.stats import truncnorm

from openquake.hazardlib.mfd import (TruncatedGRMFD, EvenlyDiscretizedMFD,
                                     ArbitraryMFD)
from openquake.hazardlib.mfd.multi_mfd import MultiMFD
from openquake.hazardlib.mfd.youngs_coppersmith_1985 import (
        YoungsCoppersmith1985MFD)

#log = True
log = False


class TaperedGrMFD(object):
    """
    Implements the Tapered G-R (Pareto) MFD as described by Kagan (2002) GJI
    page 523.

    :parameter mo_t:
    :parameter mo_corner:
    :parameter b_gr:
    """

    def __init__(self, mo_t, mo_corner, b_gr):
        self.mo_t = mo_t
        self.mo_corner = mo_corner
        self.b_gr = b_gr

    def get_ccdf(self, mo):
        beta = 2./3.*self.b_gr
        ratio = self.mo_t / mo
        phi = ratio**beta * np.exp((self.mo_t-mo) / self.mo_corner)
        return phi


class GammaMFD(object):
    """
    :parameter mo_t:
        Lower moment threshold
    :parameter mo_corner:
        The corner moment controlling the decay of the distribution close
        to the larger values of magnitude admitted
    :parameter b_gr:
        Gutenberg-Richter relationship b-value
    """

    def __init__(self, mo_t, mo_corner, b_gr):
        self.mo_t = mo_t
        self.mo_corner = mo_corner
        self.b_gr = b_gr

    def get_ccdf(self, mo):
        """
        :parameter numpy.array mo:
            A 1D instance of :class:`numpy.array` moment is in [N.m]
        :returns:

        """
        beta = 2./3.*self.b_gr
        ratio = self.mo_t / self.mo_corner
        term1 = np.exp(ratio)
        term2 = scipy.special.gammainc(1.-beta, ratio)
        c = 1. - ratio**beta * term1 * term2

        term3 = c**(-1.) * (self.mo_t/mo)**beta
        term4 = np.exp((self.mo_t - mo) / (self.mo_corner))
        term5 = (mo / self.mo_corner)**beta
        term6 = np.exp(mo / self.mo_corner)
        term7 = scipy.special.gammaincc(1.-beta, mo / self.mo_corner)
        # We multiply the complemented incomplete gamma function in order
        # to reproduce the eq. 15 of Kagan (2002)
        term8 = scipy.special.gamma(1.-beta)
        phi = term3 * term4 * (1. - term5 * term6 * term7 * term8)
        return phi


def mag_to_mo(mag, c=9.05):
    """
    Scalar moment [in Nm] from moment magnitude

    :return:
        The computed scalar seismic moment
    """
    return 10**(1.5 * mag + c)


def mo_to_mag(mo, c=9.05):
    """
    From moment magnitude to scalar moment [in Nm]

    :return:
        The computed magnitude
    """
    return (np.log10(mo) - c) / 1.5


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


def get_moment_from_mfd(mfd, threshold=-1, c=9.05):
    """
    This computes the total scalar seismic moment released per year by a
    source

    :parameter mfd:
        An instance of openquake.hazardlib.mfd
    :param threshold:
        Lower threshold magnitude
    :returns:
        A float corresponding to the rate of scalar moment released
    """
    if isinstance(mfd, TruncatedGRMFD):
        return mfd._get_total_moment_rate()
    elif isinstance(mfd, (EvenlyDiscretizedMFD, ArbitraryMFD)):
        occ_list = mfd.get_annual_occurrence_rates()
        mo_tot = 0.0
        for occ in occ_list:
            if occ[0] > threshold:
                mo_tot += occ[1] * 10.**(1.5*occ[0] + c)
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


def get_evenlyDiscretizedMFD_from_multiMFD(mfd, bin_width=None):
    if mfd.kind == 'incrementalMFD':

        oc = mfd.kwargs['occurRates']
        min_mag = mfd.kwargs['min_mag']
        binw = mfd.kwargs['bin_width'][0]

        for i in range(mfd.size):
            occ = oc[0] if len(oc) == 1 else oc[i]
            min_m = min_mag[0] if len(min_mag) == 1 else min_mag[i]
            if i == 0:
                emfd = EEvenlyDiscretizedMFD(min_m, binw, occ)
                
            else:
                tmfd = EEvenlyDiscretizedMFD(min_m, binw, occ)
                emfd.stack(tmfd)
                 
    elif mfd.kind == 'truncGutenbergRichterMFD':
        aval = mfd.kwargs['a_val']
        bval = mfd.kwargs['b_val']
        min_mag = mfd.kwargs['min_mag']
        max_mag = mfd.kwargs['max_mag']
        binw = mfd.kwargs['bin_width'][0]
        # take max mag here so we don't have to rescale the FMD later
        max_m = np.max(max_mag) 
        min_m = min_mag[0] if len(min_mag) == 1 else np.min(min_mag) 

        for i in range(mfd.size):
            bgr = bval[0] if len(bval) == 1 else bval[i]
            agr = aval[0] if len(aval) == 1 else aval[i]
            
            left = np.arange(min_m, max_m, binw)
            rates = 10.**(agr-bgr*left)-10.**(agr-bgr*(left+binw))
            
            if i == 0:
                emfd = EEvenlyDiscretizedMFD(min_m+binw/2.,
                                binw,
                                list(rates))
            else:
                tmfd = EEvenlyDiscretizedMFD(min_m+binw/2.,
                                binw,
                                list(rates))
                emfd.stack(tmfd)
    
    else:
        raise ValueError('Unsupported MFD type ', mfd.kind)

    return emfd


class EEvenlyDiscretizedMFD(EvenlyDiscretizedMFD):

    @classmethod
    def from_mfd(self, mfd, bin_width=None):
        """
        :param mfd:
            An instance of :class:`openquake.hazardlib.mfd`
        """
        if isinstance(mfd, EvenlyDiscretizedMFD):
            return EEvenlyDiscretizedMFD(mfd.min_mag, mfd.bin_width,
                                         mfd.occurrence_rates)
        elif isinstance(mfd, TruncatedGRMFD):
            tmfd = get_evenlyDiscretizedMFD_from_truncatedGRMFD(mfd, bin_width)
            return EEvenlyDiscretizedMFD(tmfd.min_mag, tmfd.bin_width,
                                         tmfd.occurrence_rates)
        elif isinstance(mfd, MultiMFD):
            tmfd = get_evenlyDiscretizedMFD_from_multiMFD(mfd, bin_width)
            return tmfd
        elif isinstance(mfd, YoungsCoppersmith1985MFD):
            occ = np.array(mfd.get_annual_occurrence_rates())
            return EEvenlyDiscretizedMFD(occ[0, 0], mfd.bin_width, occ[:, 1])
        elif isinstance(mfd, ArbitraryMFD):
            occ = np.array(mfd.get_annual_occurrence_rates())
            return EEvenlyDiscretizedMFD(occ[0, 0], bin_width, occ[:, 1])
        else:
            raise ValueError('Unsupported MFD type')

    def stack(self, imfd):
        """
        This function stacks two mfds represented by discrete histograms.

        :parameter mfd2:
            Instance of :class:`~openquake.hazardlib.mfd.EvenlyDiscretizedMFD`
        """

        if isinstance(imfd, TruncatedGRMFD):
            mfd2 = get_evenlyDiscretizedMFD_from_truncatedGRMFD(imfd,
                                                                self.bin_width)
        else:
            mfd2 = imfd

        mfd1 = self
        bin_width = self.bin_width
        
        
        #
        # check bin width of the MFD to be added
        if (isinstance(mfd2, EvenlyDiscretizedMFD) and
                abs(mfd2.bin_width - bin_width) > 1e-10):
            if log:
                print('resampling mfd2 - binning')
            mfd2 = mfd_resample(bin_width, mfd2)
        # MFD2
        # this is the difference between the rounded mmin and the original mmin
        dff = abs(np.floor((mfd2.min_mag+0.1*bin_width)/bin_width)*bin_width -
                  mfd2.min_mag)
        if dff > 1e-7:
            if log:
                print('resampling mfd2 - homogenize mmin')
                print('                - delta: {:.2f}'.format(dff))
                tmps = '                - original mmin: {:.2f}'
                print(tmps.format(mfd2.min_mag))
            mfd2 = mfd_resample(bin_width, mfd2)
        # MFD1
        # this is the difference between the rounded mmin and the original mmin
        dff = abs(np.floor((self.min_mag+0.1*bin_width)/bin_width)*bin_width -
                  self.min_mag)
        #
        #
        if dff > 1e-7:
            if log:
                print('resampling mfd1 - homogenize mmin')
                print('                - delta: {:.2f}'.format(dff))
                tmps = '                - original mmin: {:.2f}'
                print(tmps.format(mfd1.min_mag))
            mfd1 = mfd_resample(bin_width, mfd1)
        #
        # mfd1 MUST be the one with the mininum minimum magnitude
        if mfd1.min_mag > mfd2.min_mag:
            print("minimum magnitudes are different")
            if log:
                print('SWAPPING')
            tmp = mfd2
            mfd2 = mfd1
            mfd1 = tmp
        #
        # Find the delta index i.e. the shift between one MFD and the other
        # one
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
            print('-------------')
            print('-- mfd2')
            print(len(mfd2.occurrence_rates), '>=', len(rates))
            print(mfd2.bin_width)
            print(mfd2.min_mag)
            print(mfd2.occurrence_rates)
            print('-- mfd1')
            print(mfd1.bin_width)
            print(mfd1.min_mag)
            print(mfd1.occurrence_rates)

        magset = set(mags)
        for idx, (mag, occ) in enumerate(mfd2.get_annual_occurrence_rates()):
            #
            # Check that we add occurrences to the right bin. Rates is the
            # list used to store the occurrences of the 'stacked' MFD
            try:
                if len(rates) > idx+delta:
                    assert abs(mag - mags[idx+delta]) < 1e-5
                    
            except:
                print('mag:     :', mag)
                print('mag rates:', mags[idx+delta])
                print('delta    :', delta)
                print('diff     :', abs(mag - mags[idx+delta]))
                raise ValueError('Stacking wrong bins')

            if log:
                print(idx, idx+delta, len(mfd2.occurrence_rates), len(rates))
                print(mag, occ)

            if len(rates) > idx+delta:
                rates[idx+delta] += occ
            else:
                if log:
                    print('Adding mag:', mag, occ)

                tmp_mag = mags[-1] + bin_width
                while tmp_mag < mag-0.1*bin_width:
                    tmp_mag += bin_width
                    delta += 1
                    if set([tmp_mag]) not in magset:
                        rates.append(0.0)
                        mags.append(tmp_mag)
                        magset = magset | set([tmp_mag])
                    else:
                        tmps = 'This magnitude bin is already included'
                        raise ValueError(tmps)

                rates.append(occ)
                mags.append(mag)
        #
        # Check that the total rate is exactly the sum of the rates in the
        # two original MFDs
        assert (sum(mfd1.occurrence_rates) + sum(mfd2.occurrence_rates) -
                sum(rates)) < 1e-5

        if log:
            print('Sum mfd1 :', sum(mfd1.occurrence_rates))
            print('Sum mfd2 :', sum(mfd2.occurrence_rates))
            print('Sum rates:', sum(rates))

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
        print('ommax     ', ommax)
        print('bin_width ', mfd.bin_width)

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
            print(mgg, ommax + mfd.bin_width/2)
        dummy.append(mgg)
        mgg += bin_width

    # prepare the new array for occurrences
    nocc = np.zeros((len(dummy), 4))

    if log:
        print('CHECK', len(nocc), len(dummy))
        print(dummy)

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
        print(nocc)
        print('SUMS:', smmn, smmo)
    assert abs(smmn-smmo) < 1e-5
    return EvenlyDiscretizedMFD(nocc[0, 0], bin_width, list(nocc[:, 3]))


def mfd_upsample(bin_width, mfd):
    """
    This is upsampling an MFD i.e. creating a new MFD with a larger
    bin width.

    :param bin_width:
    :param mfd:
    """
    #
    # computing the min and max values of magnitude
    ommin = mfd.min_mag
    ommax = mfd.min_mag + len(mfd.occurrence_rates) * mfd.bin_width
    #
    # rounding the lower and upper magnitude limits to the new
    # bin width
    min_mag = np.floor(ommin / bin_width) * bin_width
    max_mag = np.ceil(ommax / bin_width) * bin_width
    #
    # prepare the new array for occurrences
    nocc = np.zeros((int((max_mag-min_mag)/bin_width+1), 4))
    # set the new array
    for idx, mag in enumerate(np.arange(min_mag, max_mag, bin_width)):
        nocc[idx, 0] = mag
        nocc[idx, 1] = mag-bin_width/2
        nocc[idx, 2] = mag+bin_width/2
    #
    # create he arrays with magnitudes and occurrences
    """
    mago = []
    occo = []
    for mag, occ in mfd.get_annual_occurrence_rates():
        mago.append(mag)
        occo.append(occo)
    mago = np.array(mago)
    occo = np.array(occo)
    """
    #
    # assigning occurrences
    dlt = bin_width * 1e-5
    for mag, occ in mfd.get_annual_occurrence_rates():
        #
        # find indexes of lower bin limits lower than mag
        idx = np.nonzero(mag+dlt-mfd.bin_width/2 > nocc[:, 1])[0]
        idxa = None
        idxb = None
        # idxa is the index of the lower limit
        if len(idx):
            idxa = np.amax(idx)
        else:
            raise ValueError('Error in computing lower mag limit')
        # find indexes of the bin centers with magnitude larger than mag
        # idx = np.nonzero((mag+mfd.bin_width/2) > nocc[:, 2])[0]
        idx = np.nonzero(mag-dlt+mfd.bin_width/2 < nocc[:, 2])[0]
        if len(idx):
            # idxb = np.amax(idx)
            idxb = np.amin(idx)
        #
        #
        if idxb is not None and idxa == idxb:
            nocc[idxa, 3] += occ
        else:
            # ratio of occurrences in the lower bin
            ra = (nocc[idxa, 2] - (mag-mfd.bin_width/2)) / mfd.bin_width
            nocc[idxa, 3] += occ*ra
            if (1.0-ra) > 1e-10:
                nocc[idxa+1, 3] += occ*(1-ra)
        print(nocc)
    #
    # check that the the MFDs have the same total occurrence rate
    smmn = sum(nocc[:, 3])
    smmo = sum(mfd.occurrence_rates)
    print(smmn, smmo)
    #
    # check that the total number of occurrences in the original and
    # resampled MFDs are the same
    assert abs(smmn-smmo) < 1e-5

    idxs = set(np.arange(0, len(nocc[:, 3])))
    iii = len(nocc[:, 3])-1
    while nocc[iii, 3] < 1e-10:
        idxs = idxs - set([iii])
        iii -= 1

    return EvenlyDiscretizedMFD(nocc[0, 0], bin_width,
                                list(nocc[list(idxs), 3]))

def merge(mfdexp, mfdchar, magexp=None, magchar=None):
    """
    """
    mfdexp = np.array(mfdexp)
    mfdchar = np.array(mfdchar)
    tmp = np.nonzero(mfdchar > mfdexp[-len(mfdchar):])[0]
    if len(tmp):
        idx = np.min(tmp)
        idxexp = - len(mfdchar) + idx
        out = np.concatenate((mfdexp[:idxexp], mfdchar[idx:]))
        midx = len(out)
    else:
        if magexp is not None and magchar is not None:
            midx = max(np.nonzero(magexp <= max(magchar))[0])
            out = mfdexp[:midx]
    return out, midx 


def mergeinv(agr, bgr, magchar, mfdchar, mwdt):
    """
    """
    mmin = 6.0
    mupp = min(magchar) 
    # get dt mfd
    dtmfd = TruncatedGRMFD(6.0, mupp+mwdt, mwdt, agr, bgr)
    occ = dtmfd.get_annual_occurrence_rates()
    # 
    madt = numpy.array([d[0] for d in occ])
    ocdt = numpy.array([d[1] for d in occ])
    # compute moment 
    modt = sum(mag_to_mo(madt)*ocdt)
    return modt


def get_ccdf(pmf):
    cdf = np.cumsum(pmf)
    ccdf = cdf[-1] - cdf
    return ccdf


def get_dt_gaussian(mag, std, std_factor=2, mwdt=0.1):
    """
    :param mean_mag:
    :param std:
    :param std_factor:
    """
    #
    # Computing magnitude extremes
    mlow = mag - std*std_factor
    mlow = mlow - (mlow % mwdt) - mwdt / 2
    mupp = mag + std*std_factor
    mupp = mupp + (mwdt - mupp % mwdt) + mwdt / 2
    #
    # discretize the truncated normal distribution
    mags = np.arange(mlow, mupp+0.1*mwdt, mwdt)
    vlow = (mlow - mag) / std
    vupp = (mupp - mag) / std
    vals = truncnorm.pdf(mags, vlow, vupp, loc=mag, scale=std)
    vals = vals/sum(vals)
    return mags, vals


def get_dt_lognormal(mag, std, std_factor=2, mwdt=0.1):
    """
    :param mean_mag:
    :param std:
    :param std_factor:
    """
    #
    # Computing magnitude extremes
    mlow = mag - std*std_factor
    mlow = mlow - (mlow % mwdt) - mwdt / 2
    mupp = mag + std*std_factor
    mupp = mupp + (mwdt - mupp % mwdt) + mwdt / 2
    #
    # discretize the truncated normal distribution
    mags = np.arange(mlow, mupp+0.1*mwdt, mwdt)
    vlow = (mlow - mag) / std
    vupp = (mupp - mag) / std
    vals = truncnorm.pdf(mags, vlow, vupp, loc=mag, scale=std)
    vals = vals/sum(vals)
    return mags, vals
