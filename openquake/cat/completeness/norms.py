# ------------------- The OpenQuake Model Building Toolkit --------------------
# Copyright (C) 2022 GEM Foundation
#           _______  _______        __   __  _______  _______  ___   _
#          |       ||       |      |  |_|  ||  _    ||       ||   | | |
#          |   _   ||   _   | ____ |       || |_|   ||_     _||   |_| |
#          |  | |  ||  | |  ||____||       ||       |  |   |  |      _|
#          |  |_|  ||  |_|  |      |       ||  _   |   |   |  |     |_
#          |       ||      |       | ||_|| || |_|   |  |   |  |    _  |
#          |_______||____||_|      |_|   |_||_______|  |___|  |___| |_|
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Affero General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------
# vim: tabstop=4 shiftwidth=4 softtabstop=4
# coding: utf-8

import copy
import scipy
from scipy.stats import poisson
from math import log, pi
from numpy import matlib
import numpy as np
from openquake.wkf.compute_gr_params import (get_weichert_confidence_intervals)

def logfactorial(n):
    """
    Calculate the log of factorial (n) using Ramanujan's approximation
    This is necessary for cases in which n > ~200 where factorial(n) will break

    :param n:
        Array of n 

    returns
        array of log factorial for each n 

    """
    logfact_n = np.zeros(len(n))
    for i, n_i in enumerate(n):
        logfact_n[i] = n_i*np.log(n_i) - n_i + (np.log(n_i*(1+4*n_i*(1+2*n_i))))/6 + log(pi)/2                                                                                                          
    return logfact_n


def get_completeness_matrix(tcat, ctab, mbinw, ybinw):
    """
    :param tcat: catalogue
    :param ctab: Completeness table
    :param mbinw: bin width for magnitudes
    :param ybinw: bin width for years

    """

    # Flipping ctab to make it compatible with the algorithm implemented here
    ctab = np.flipud(ctab)

    # Create the array with magnitude bins
    cat_yea = tcat.data['year']
    cat_mag = tcat.data['magnitude']
    mdlt = mbinw * 0.1
    ydlt = ybinw * 0.1

    min_m = np.floor((np.min(cat_mag)+mdlt)/mbinw)*mbinw
    max_m = np.ceil((np.max(cat_mag)+mdlt)/mbinw+1)*mbinw
    mags = np.arange(min_m, max_m, mbinw)
    cmags = mags[:-1]+mbinw/2

    min_y = np.floor((np.min(cat_yea)+ydlt)/ybinw)*ybinw
    max_y = np.ceil((np.max(cat_yea)+ydlt)/ybinw+1)*ybinw
    yeas = np.arange(min_y, max_y, ybinw)
    cyeas = yeas[:-1]+ybinw/2

    his, _, _ = np.histogram2d(cat_mag, cat_yea, bins=[mags, yeas])
    oin = copy.copy(his)
    out = copy.copy(his)

    for i, yea in enumerate(cyeas):

        if yea < ctab[0, 0]:
            oin[0:, i] = -1.0
            continue

        # Get the index of the row in the completeness table
        yidx = np.digitize(yea, ctab[:, 0])-1

        # Get the indexes of the magnitude
        midx = np.digitize(ctab[yidx, 1], mags)-1
        oin[:midx, i] = -1.0
        out[midx:, i] = -1.0

    return oin, out, cmags, cyeas

def get_norm_optimize(tcat, aval, bval, ctab, cmag, n_obs, t_per, last_year, info=False):
    """
    :param aval: 
      This calculates the difference between the number of observed vs computed events in each completeness bin. 
    The differences are normalised by the number of completeness bins in order to minimise the effect of the number
    of bins. 
        
    :param aval: 
        GR a-value
    :param bval: 
        GR b-value
    :param ctab: 
        completeness table
    :param cmag:
        An array with the magnitude values at the center of each occurrence
        bins. Output from hmtk.seismicity.occurrence.utils.get_completeness_counts
    :param t_per:
        array indicating total duration (in years) of completeness
        Output from hmtk.seismicity.occurrence.utils.get_completeness_counts
    :param n_obs:
        number of events in completeness period
        Output from hmtk.seismicity.occurrence.utils.get_completeness_counts
    :param last_year:
        last year to consider in analysis
    :param info:
        boolean controlling whether to print information as the function proceeds
    :returns:
       calculated norm for input completeness. Smaller norm is better
    """
    
    occ = np.zeros((ctab.shape[0]))
    dur = np.zeros((ctab.shape[0]))
    mags = np.array(list(ctab[:, 1])+[10])

    for i, mag in enumerate(mags[:-1]):
        idx = (cmag >= mags[i]) & (cmag < mags[i+1])
        if np.any(idx):
            occ[i] = np.sum(n_obs[idx])
            dur[i] = t_per[np.min(np.nonzero(idx))]
        else:
            occ[i] = 0
            dur[i] = (last_year-ctab[i, 0])

    # Rates of occurrence in each magnitude bin from GR with a and b
    rates = (10**(-bval * mags[:-1] + aval) -
             10**(-bval * mags[1:] + aval)) * dur

    # Standard deviation of the poisson model.
    ## This probably doesn't need to be Poisson at all - just where n > 0
    # Also this is the poisson std for the GR data NOT the observed
    #stds_poisson = scipy.stats.poisson.std(rates)
    #idx = stds_poisson > 0
    idx = rates > 0
    if not np.any(idx):
        return None

    if info:
        print('stds', stds_poisson)

    # Widths of the magnitude intervals
    wdts = np.diff(mags)
    mag_diff = (mags[-1] - mags[0])

    # Difference between modelled and observed occurrences
    occ_diff = (np.abs(rates[idx] - occ[idx]))**0.5 / dur[idx]**0.5
    
    # Checking weights
    msg = '{:f} ≠ {:f}'.format(np.sum(wdts[idx] / mag_diff), 1.)
    if np.abs(np.sum(wdts[idx] / mag_diff) - 1.0) > 1e-5:
        raise ValueError(msg)

    if info:
        print('rates, occ', rates, occ)
        print('diff:', occ_diff)

    # This is the norm. It's the weighted sum of the normalised difference
    # between the number of computed and observed occurrences in each
    # completeness bin. We divide the sum by the number of completeness
    # intervals in order to use a measure that's per-interval i.e.
    # independent from the number of completeness intervals used
    norm = np.mean(occ_diff * wdts[idx]) / mag_diff
    norm /= np.sum(idx)

   
    return norm


def get_norm_optimize_a(aval, bval, ctab, binw,  cmag, n_obs, t_per, info=False):
    """
    Computes a norm based on the probability of observing n events in each
    magnitude bin relative to an exponential (GR) frequency-magnitude distribution.
    The norm is the log-likelihood of observing the number of events in each magnitude
    bin given the expected rate from the GR parameters calculated for these 
    completeness window. Larger is better.

    :param aval:
        GR a-value
    :param bval:
        GR b-value
    :param ctab:
        Completeness table
    :param binw:
        binwidth for completeness analysis, specified in toml
    :param cmag:
        An array with the magnitude values at the center of each occurrence
        bin
    :param n_obs:
        number of events in completeness period
        Output from hmtk.seismicity.occurrence.utils.get_completeness_counts
    :param t_per:
        array indicating total duration (in years) of completeness
        Output from hmtk.seismicity.occurrence.utils.get_completeness_counts
    :returns: 
        norm - log-likelihood of observing the number of events in 
        each magnitude bin given GR params from completeness bin.
        Larger is better.
    """
    # Rates of occurrence in each magnitude bin in the completeness interval
    rates = (10**(-bval * (cmag - binw/2) + aval) -
             10**(-bval * (cmag + binw/2) + aval))*t_per


    # Probability of observing n-occurrences in each magnitude bin
    # These are Poisson probabilities
    # and these probabilities are never large, even for synthetic data

    occ_prob = np.ones_like(rates) * 0.999999
    num = (rates)**n_obs * np.exp(-rates)
    occ_prob = num / scipy.special.factorial(n_obs)
    #print("occ_prob: ", occ_prob)
    
    n_obs = n_obs.astype(int)
    log_prob = np.ones_like(n_obs) * 0.999999
    for i, obs in enumerate(n_obs):
        log_prob[i] = (-rates[i]) + (n_obs[i]*np.log(rates[i])) - logfactorial(n_obs[i])

    norm = 1 - np.prod(np.exp(log_prob))
    return norm


def get_norm_optimize_b(aval, bval, ctab, tcat, mbinw, ybinw, back=5, mmin=-1,
                        mmax=10):
    """
    Computes a norm based on occurrences inside and outside of the completeness windows.
    Difference in expected vs observed occurrences in/outside of completeness are 
    normalised by the rates from a GR distribution given the completeness. 
    The norm is calculated from the ratio of events in/out the window, and should be maximised.

    :param aval:
        GR a-value
    :param bval:
        GR b-value
    :param ctab:
        Completeness table
    :param tcat:
        A catalogue instance
    :param mbinw:
        Magnitude bin width
    :param ybinw:
        Time (i.e. year) bin width
    :param back:
        Delta year
    :param mmin:
        Minimum magnitude
    :param mmax:
        Maximum magnitude
    :returns:
        norm - a ratio of the difference in event numbers within/outwith 
        completeness relative to expected GR. Larger is better
    """

    # oin and out have shape (num mags bins) x (num years bins)
    oin, out, cmags, cyeas = get_completeness_matrix(tcat, ctab, mbinw, ybinw)

    # Count the occurrences inside and outside the completeness window. The
    # rest of the matrix has a negative value of -1
    idx = (cmags >= mmin) & (cmags <= mmax)
    cmags = cmags[idx]
    oin = oin[idx, :]
    out = out[idx, :]

    # Compute the eqks in each magnitude bin using the GR parameters provided
    rates = ((10**(aval-(bval*(cmags-mbinw/2))) -
              10**(aval-(bval*(cmags+mbinw/2)))) * ybinw)

    # Assuming a Poisson process, compute the standard deviation of the rates
    # stds_poi = scipy.stats.poisson.std(rates)

    # Preparing matrices with the rates in each magnitude bins and their
    # standard deviation. The standard deviation is not used in the rest of the
    # function
    rates = matlib.repmat(np.expand_dims(rates, 1), 1, len(cyeas))
    # stds_poi = matlib.repmat(np.expand_dims(stds_poi, 1), 1, len(cyeas))

    # Compute the year from when to count the occurrences
    mag_bins = cmags-mbinw/2
    mag_bins = np.append(mag_bins, cmags[-1]-mbinw/2)
    tmp = np.digitize(ctab[:, 1], mag_bins) - 1 - back
    tmp = np.maximum(np.zeros_like(tmp), tmp)

    # Count the occurrences in the completeness window
    diff_in = np.abs(oin / ybinw - rates) / rates
    idxin = oin < 0
    diff_in[idxin] = 0

    # Count the occurrences out of the completeness window
    diff_out = np.abs(out / ybinw - rates) / rates
    idxout = out < 0
    diff_out[idxout] = 0

    # In this case we want 'count_in' to be as small as possible and
    # 'count_out' to be as big as possible. The term '1/np.sum(idxin)' is
    # used to give preference to solutions with a larger number of cells
    # considered complete

    if np.sum(diff_out) == 0 or np.sum(idxin) == 0 or np.sum(idxout) == 0:
        norm = -1E-10

    else: 
        norm = np.sum(diff_in) / np.sum(diff_out) / np.sum(idxin) * np.sum(idxout)

    return norm


def get_idx_compl(mag, compl):
    """
    Find completeness windows for a specified magnitude
    
    :param mag:
        magnitude 
    :param compl:
        completeness table 
    :returns:
        completeness window for a specified magnitude
    """
    if mag < compl[0, 1]:
        return None
    for i, com in enumerate(compl[:-1, :]):
        if mag >= com[1] and mag < compl[i+1, 1]:
            return i
    return len(compl)-1

def poiss_prob_int_time(rate, n, t, log_out = False):
    """
    Calculate poisson probability of observing n events in some time step t given rate
    In most cases, a log probability is preferred, due to issues with powers and factorials
    when large numbers of events are involved.
    
    :param rate:
        Poisson rate 
    :param n:
        Number of observed events
    :param t:
        time step, multiplied with rate to get poisson expected number
    :param log_out:
        boolean indicating if log probabilities are preferred. If n is large, this should
        be set to true to avoid instabilities.
    :returns:
        Poisson probability of observing n events in time t given poisson rate
    """
    # Should use log probabilities so this doesn't break at large n
    log_prob = -(rate*t) + n*(np.log(rate) + np.log(t)) - logfactorial([n])
    if log_out == False:
        prob = np.exp(log_prob)
    else:
        prob = log_prob
    return prob



def get_norm_optimize_c(cat, agr, bgr, compl, last_year, ref_mag, mmax=None, binw=0.1):
    """
    Variation on Poisson optimization of completeness windows that considers events
    within and outwith the completeness windows.
    Probability is calculated as total probability of observing events within the 
    completeness windows + the probability of observing the events outside of 
    the completeness, given GR from specified a, b values. 
    
    :param cat:
        catalogue 
    :param aval:
        GR a-value
    :param bval:
        GR b-value
    :param compl: 
        completeness table
    :param last_year:
        end year to consider
    :param ref_mag:
        reference magnitude at which to perform calculations
    :param mmax: 
        maximum magnitude
    :param binw:
        binwidth 
    :returns: 
        total Poisson probability of observing the events given the FMD. Larger
        is better   

    """

    mags = cat.data['magnitude']
    yeas = cat.data['year']

    mmax = max(mags) if mmax is None else mmax
    # check minimum magnitude is greater than ref mag
    mvals = np.arange(ref_mag, mmax+binw/10, binw)
    rates = list(10**(agr-bgr * mvals[:-1]) - 10**(agr - bgr * mvals[1:]))

    # If using log (and not multiplicative) set initial prob to 0
    prob = 0
    first_year = min(yeas)
    for imag, mag in enumerate(mvals[:-1]):
        tot_n = len(mags[(mags >= mag) & (mags < mvals[imag+1])])
        #print(tot_n)
        if tot_n < 1:
            continue


        idxco = get_idx_compl(mag, compl)

        # if this magnitude bin is < mc in this window, nocc will be zero
        # Rather this disgards events outwith the completeness window, as it should!
        if idxco is None:
            nocc_in = 0
            nocc_out = tot_n
            continue 

        elif mag >= compl[idxco, 1]:
            idx = (mags >= mag) & (mags < mvals[imag+1]) & (yeas >= compl[idxco, 0])
            #print(idx)
            nocc_in = sum(idx)
        #elif mag < min(cat.data['magnitude']):
        #    nocc_in = 0
        else:
            print("how did this get here?", compl[idxco, 0], mag )

            nocc_in = 0

        delta = (last_year - compl[idxco, 0])
        # events in bin before completeness
        idx = ((mags >= mag) & (mags < mvals[imag+1]) &
               (yeas < compl[idxco, 0]))
        nocc_out = sum(idx) 

        # also is this right? should yeas[idx] extend to years before compl[idxco, 0] too?
        if np.any(idx):
            ylow = np.min(yeas[idx])
        else:
            ylow = np.min([compl[idxco, 0], first_year])-1

        # Compute the duration for the completeness interval and the time
        # interval outside of completeness
        dur_out_compl = (compl[idxco, 0] - ylow)
        # I think I want to limit this to the time interval of the completeness window
        dur_compl = last_year - compl[idxco, 0]

        pmf = poiss_prob_int_time(rates[imag], nocc_in, dur_compl, log_out = True)

        std_in = poisson.std(dur_compl*rates[imag])

        pmf_out = poiss_prob_int_time(rates[imag], nocc_out, dur_out_compl, log_out = True)

        prob += pmf +  (np.log(1.0) - pmf_out)
    

    return prob

def get_norm_optimize_poisson(cat, agr, bgr, compl, last_year, mmax=None, binw=0.1):
    """
    Alternative to get_norm_optimize_c that loops over the time increments
    Probability is calculated as total probability of observing events within the 
    completeness windows + the probability of observing the events outside of 
    the completeness, given GR from specified a, b values. 
    
    :param cat:
        catalogue 
    :param aval:
        GR a-value
    :param bval:
        GR b-value
    :param compl: 
        completeness table
    :param last_year:
        end year to consider
     :param binw:
        binwidth 
    :returns: 
        total Poisson probability of observing the events given the FMD. Larger
        is better     
    """

    mags = cat.data['magnitude']
    yeas = cat.data['year']

    mmax = max(mags) if mmax is None else mmax
    # check minimum magnitude is greater than
    mvals = np.arange(min(compl[:, 1]), mmax+binw/10, binw)

    rates = list(10**(agr-bgr * mvals[:-1]) - 10**(agr - bgr * mvals[1:]))

    # Using log (and not multiplicative) set initial prob to 0
    prob = 0
    first_year = min(yeas)
    for imag, mag in enumerate(mvals[:-1]):
        idxco = get_idx_compl(mag, compl)

        
        # if this magnitude bin is < mc in this window, nocc_in will be zero
        # This disgards events outwith the completeness window, as it should!
        if mag >= compl[idxco, 0]:
            idx = (mags >= mag) & (mags < mvals[imag+1]) & (yeas >= compl[idxco, 0])
            nocc_in = sum(idx)
        
        else:
            nocc_in = 0

        delta = (last_year - compl[idxco, 0])
        idx = ((mags >= mag) & (mags < mvals[imag+1]) & (yeas < compl[idxco, 0]) & (yeas > (compl[idxco, 0] - delta)))
        nocc_out = sum(idx)

        if mag < compl[idxco, 0]:
            nocc_out = 0

        # also is this right? should yeas[idx] extend to years before compl[idxco, 0] too?
        if np.any(idx):
            ylow = np.min(yeas[idx])
        else:
            ylow = np.min([compl[idxco, 0], first_year])-1

        # Compute the duration for the completeness interval and the time
        # interval outside of completeness
        dur_out_compl = compl[idxco, 0] - ylow
        # I think I want to limit this to the time interval of the completeness window
        dur_compl = last_year - compl[idxco, 0]

        pmf = poiss_prob_int_time(rates[imag], nocc_in, dur_compl, log_out = True)

        std_in = poisson.std(dur_compl*rates[imag])
        pmf_out = poiss_prob_int_time(rates[imag], nocc_out, dur_out_compl, log_out = True)

        prob += pmf +  (np.log(1.0) - pmf_out)


    return prob

def get_norm_optimize_d(cat, agr, bgr, compl, last_year, mmax=None, binw=0.1):


    mags = cat.data['magnitude']
    yeas = cat.data['year']

    mmax = max(mags) if mmax is None else mmax
    mvals = np.arange(min(compl[:, 1]), mmax+binw/10, binw)

    rates = list(10**(agr-bgr * mvals[:-1]) - 10**(agr - bgr * mvals[1:]))

    # Using log (and not multiplicative) set initial prob to 0
    prob = 0
    first_year = min(yeas)

    llhood = 0
    for j, window in enumerate(compl):
        weichert_ll_allM = [0]*len(compl)

        if j == (len(compl) - 1):
            upper_time = last_year
        else: 
            upper_time = compl[j+1, 0]


        window_mags_idx = (mags >= window[1]) & (yeas >= window[0]) & (yeas < upper_time)
        window_mags = mags[window_mags_idx]
        dur_compl = upper_time - window[0]

        
        # Loop over all magnitude bins, calculate poiss probability for bin
        for imag, mag in enumerate(mvals[:-1]):
        
            idxco = get_idx_compl(mag, compl)

            if mag >= compl[idxco, 0]:
                idx = (window_mags >= mag) & (window_mags < mvals[imag+1])
                nocc_in = sum(idx)
            else:
                nocc_in = 0
            
            # nocc_out is events in the mag intervals in this time step
            idx = ((mags < mag) & (mags < mvals[imag+1]) & (yeas >= window[0]) & (yeas < upper_time))
            nocc_out = sum(idx)

            if np.any(idx):
                ylow = np.min(yeas[idx])
            else:
                ylow = np.min([compl[idxco, 0], first_year])-1

            # Compute the duration for the completeness interval and the time
            # interval outside of completeness
            pmf = poiss_prob_int_time(rates[imag], nocc_in, dur_compl, log_out = True)

            std_in = poisson.std(dur_compl*rates[imag])

            # Probability of events in time outwith magnitude interval
            pmf_out = poiss_prob_int_time(rates[imag], nocc_out, dur_compl, log_out = True)
        
            prob += pmf +  (np.log(1.0) - pmf_out)
        
        
        llhood = llhood + prob
        
    return llhood

def get_norm_optimize_weichert(cat, agr, bgr, compl, last_year, mmax=None, binw=0.1):
    """
    Optimize for completeness using Weichert likelihood
    NB: This is not technically correct! Weichert likelihood is calculated
        using events *within* completeness windows. This means that a lower
        completeness which keeps more events will result in a larger likelihood.
        So when we try to use this to condition, we will find that smaller Mc 
        are preffered earlier because this increases the Weichert likelihood.
        This is why we should not compare likelihoods when using a different
        number of events within the calculation (as we do here with different
        completeness windows).
        This can still be interesting, but the above needs to be considered!
        
    :param cat:
        catalogue 
    :param aval:
        GR a-value
    :param bval:
        GR b-value
    :param compl: 
        completeness table
    :param last_year:
        end year to consider
    :param mmax: 
        maximum magnitude
    :param binw:
        binwidth
    :returns:
        norm - Weichert likelihood of observing the number of events across
        all bins. Larger is better.
    """

    mags = cat.data['magnitude']
    yeas = cat.data['year']

    beta = bgr*log(10)

    mmax = max(mags) if mmax is None else mmax
    mvals = np.arange(min(compl[:, 1]), mmax+binw/10, binw)
    N = len(mags)

    # Using log (and not multiplicative) set initial prob to 0
    prob = 0
    first_year = min(yeas)
    llhood = 0

    weichert_llhood = [0]*len(compl)
    # Loop through completeness windows
    for j, window in enumerate(compl):
        weichert_ll_allM = [0]*len(compl)

        if j == (len(compl) - 1):
            upper_time = last_year
        else: 
            upper_time = compl[j+1, 0]


        window_mags_idx = (mags >= window[1]) & (yeas >= window[0]) & (yeas < upper_time)
        window_mags = mags[window_mags_idx]
        # test if this time window has exponential GR?
        dur_compl = upper_time - window[0]

        n_i = [0]*len(mvals)
        p_i = [0]*len(mvals)
        # Loop over all magnitude bins, counting events in each and calculate p_i from weichert (see eqn 5)
        for imag, mag in enumerate(mvals[:-1]):
            
            idx = (window_mags >= mag) & (window_mags < mvals[imag+1]) 
            n_i[imag] = sum(idx)
            
            p_i[imag] = dur_compl*np.exp(-beta*mag)

        log_sum_p_i = 0
        log_fact_n = logfactorial(n_i)
        log_fact_n = np.nan_to_num(log_fact_n, nan = 0.0)
         
        #calculate L(Beta|n_i, m_i, t_i)
        for i in range(0, len(mvals)):
            p_j = np.delete(p_i, [i], axis=0)
            prob_j = sum(p_j)
            prob_i = p_i[i] / prob_j
            prob_inc = n_i[i]*np.log(prob_i)
            if p_i[i] == 0:
                prob_inc = 0
            log_sum_p_i = log_sum_p_i + prob_inc
            # weichert likelihood for magnitude bin
            weichert_llhood = logfactorial([N]) - np.sum(log_fact_n) + log_sum_p_i
            weichert_ll_allM[j] = weichert_ll_allM[j] + weichert_llhood

    # Sum to get total likleihood
    weichert_per_t = sum(weichert_ll_allM)
        
       
    return weichert_per_t


def get_norm_optimize_gft(tcat, aval, bval, ctab, cmag, n_obs, t_per, last_year):
    """
    Optimize fit using a version of the goodness-of-fit completeness approach 
    (Wiemer and Wyss, 2000), using a parameter R to compare the goodness of fit.
    A returned norm of 100 implies a perfect fit between observed and expected
    events.

    :param aval: 
        GR a-value
    :param bval: 
        GR b-value
    :param ctab: 
        completeness table
    :param cmag:
        An array with the magnitude values at the center of each occurrence
        bins
    :param t_per:
        time period of completeness
    :param n_obs:
        Number of observations
    :param last_year: 
        last year to consider
    :param info:
        boolean controlling whether to print information as the function proceeds
    :returns: 
        norm calculated with the Wiemer and Wyss (2000) 'R' parameter. Larger is better.
    """
    # Select only events within 'complete' part of the catalogue
    occ = np.zeros((ctab.shape[0]))
    dur = np.zeros((ctab.shape[0]))
    mags = np.array(list(ctab[:, 1])+[10])

    gwci = get_weichert_confidence_intervals
    # calculate rates for all events
    lcl, ucl, ex_rates, ex_rates_scaled_all = gwci(
                cmag, n_obs, t_per, bval)

    # Rates for events in completeness
    lcl, ucl, ex_rates, ex_rates_scaled_comp = gwci(
                mags, occ, dur, bval)

    # Expected rates of occurrence in each magnitude bin above completeness
    # from GR with a and b
    cum_rates = 10**(aval-bval*mags)
    
    # Actually should only be for Mc < M < MMax
    norm = 100 -(np.abs((sum(ex_rates_scaled_comp - cum_rates)/sum(ex_rates_scaled_all)))*100)

    return norm



