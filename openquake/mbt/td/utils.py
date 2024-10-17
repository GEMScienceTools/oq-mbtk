import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
import math
from openquake.hazardlib.source import KiteFaultSource, CharacteristicFaultSource
from openquake.sub.utils import _read_edges
from openquake.hazardlib.geo.surface import ComplexFaultSurface
from openquake.hazardlib.mfd import EvenlyDiscretizedMFD
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.source import NonParametricSeismicSource
from openquake.hazardlib.source.rupture import BaseRupture
from openquake.hazardlib.pmf import PMF
from openquake.hazardlib.geo import Point
from openquake.hazardlib.geo.surface.gridded import GriddedSurface
from decimal import Decimal, getcontext
from openquake.hazardlib.sourceconverter import SourceGroup
from openquake.hazardlib.nrml import SourceModel

def bpt_pdf(mu, alpha, times):
    """
    alpha: aperiodicity
    mu: mean recurrence interval
    times: time series across which distribution is created

    in notation, this is f(t) the pdf
    """
    p1 = np.sqrt((mu / (2 * np.pi * alpha**2 * times**3))) 
    p2 = np.exp( - (times - mu) ** 2 / (2 * mu * alpha**2 * times) ) 
    bpt = p1 * p2
    if math.isnan(bpt[0]):
        bpt[0] = 0
    return bpt

def exp_pdf(mu, times):

    exp = [1/mu * np.exp(-1/mu * t) for t in times]
    
    return exp


def get_time_series(end, sample):
    """
    end: end time in years of time series
    sample: sample interval in years, default = 1 year
    """

    return np.arange(0, end+sample, sample)


def cdf_field(times, pdf):
    return integrate.cumtrapz(pdf, times)
    

def survivor_function(times, pdf):
    """
    times: time series across which distribution is created
    pdf: probability density function 

    returns sf: survivor function
    """

    int_t = integrate.cumtrapz(pdf, times)
    sf = [int_t[-1] - i for i in int_t]
    return sf


def hazard_function(pdf, sf):

    hf = [f/F for F, f in zip(sf, pdf)]
    return hf

def get_conditional_probabilities(p_invt, sample, sf, times):
    """
    p_invt: prediction investigation time
    sample: sample used to create time steps
    sf: survivor function
    times: time series

    returns cprobs: conditional probabilities
    """
    cprobs = [] 
    num_idx = int(p_invt / sample)
    times_cut = times[1:-num_idx]
    
    for ii, t in enumerate(times_cut):
        
        FT = sf[ii]
        FT_delT = sf[ii+num_idx]
        cprob = (FT - FT_delT) / FT
        cprobs.append(cprob)

    return cprobs, times_cut


def get_P_at_time(t_inv, cprobs, times):
    
    ind = list(times).index(t_inv)
    return cprobs[ind]
    

def time_dependent_probability(mu, alpha, sample, 
                               p_invt, t_inv, endtime,
                               pdf_shape='bpt'):
    """
    """
    if pdf_shape != 'bpt': 
        msg = 'Only BPT is implemented now!'
        raise AssertionError(msg)

    times = get_time_series(endtime, sample)
    
    bpt = bpt_pdf(mu, alpha, times)
    exp = exp_pdf(mu, times)
    sfbpt = survivor_function(times, bpt)
    sfexp = survivor_function(times, exp)
    hfunct = hazard_function(bpt, sfbpt)
    hfunctexp = hazard_function(exp, sfexp)
    cprobs, times_cut = get_conditional_probabilities(p_invt, sample, sfbpt, times)
    cprobs_exp, times_cut = get_conditional_probabilities(p_invt, sample, sfexp, times)
    cpbpt = get_P_at_time(t_inv, cprobs, times)
    cpexp = get_P_at_time(t_inv, cprobs_exp, times)
    
    return cpbpt, cpexp 

def time_dependent_probability_uk_histopen(mu, alpha, sample, 
                               p_invt, endtime, TH,
                               pdf_shape='bpt'):
    """
    adds TH - time of historic catalogue
    """
    if pdf_shape != 'bpt': 
        msg = 'Only BPT is implemented now!'
        raise AssertionError(msg)

    times = get_time_series(endtime, sample)
    taus = times

    # get pdf
    ft = bpt_pdf(mu, alpha, times)
    # get CDF,  CDF of CDF, and CDF of survivor
    Ft = integrate.cumtrapz(ft, taus)
    FFt = integrate.cumtrapz(Ft, taus[:-1])
    iFt = integrate.cumtrapz(1-Ft, times[:-1])

    # get parts of integral
    top = (p_invt - (FFt[p_invt + TH] - FFt[TH]))
    bottom = iFt[-1] - iFt[TH]

    # return probability (one value)
    fp = top/bottom
    probabilities = {0: 1-fp, 1: fp}
    return probabilities

def time_dependent_probability_uk(mu, alpha, sample, 
                               p_invt, endtime,
                               pdf_shape='bpt'):
    """
    adds TH - time of historic catalogue
    """
    if pdf_shape != 'bpt': 
        msg = 'Only BPT is implemented now!'
        raise AssertionError(msg)

    times = get_time_series(endtime, sample)
    taus = times

    # get pdf
    ft = bpt_pdf(mu, alpha, times)
    # get CDF,  CDF of CDF, and CDF of survivor
    Ft = integrate.cumtrapz(ft, taus)
    FFt = integrate.cumtrapz(Ft, taus[:-1])

    # compute probability 
    fp = (p_invt - (FFt[p_invt] - FFt[0]))/mu

    probabilities = {0: 1-fp, 1: fp}
    
    return probabilities

def characteristic_mfd_from_prob(mag, cp, t_inv, bw=0.1):
    
    rate = -np.log(1-cp)/t_inv
    mfd = EvenlyDiscretizedMFD(mag, bw, [rate])

    return mfd

def characteristic_mfd_from_rate(mag, rate, bw=0.1):
    
    return EvenlyDiscretizedMFD(mag, bw, [rate])


def make_cf_surface(edge_dir):
    """
    """
    edges = _read_edges(edge_dir)
    src_surf = ComplexFaultSurface.from_fault_data(edges, 10.)

    return src_surf

def create_source(sid, name, trt, mfd, t_inv, surf, rake=90.):

    tom = PoissonTOM(time_span=t_inv)
    source = CharacteristicFaultSource(sid, name, trt, mfd, tom, surf, rake)
    
    return source

def create_source_npss(sid, name, trt, mag, pno, t_inv, surf, rake=90., smname='SourceModel'):
    
    data = [] 
    hypo = surf.mesh.get_middle_point()
    points = [pt for pt in surf.mesh]
    srf = GriddedSurface.from_points_list(points)
    brup = BaseRupture(mag=mag, rake=rake,
                       tectonic_region_type=trt,
                       hypocenter=hypo,
                       surface=srf)
    brup.weight = None
    xxx = Decimal(f'{pno:.4f}')
    pmf = PMF(data=[((Decimal('1')-xxx), 0), (xxx, 1)])
    data.append((brup, pmf))       

    src = NonParametricSeismicSource(sid, name, trt, data=data)
    return src


def create_source_npss_pmf(sid, name, trt, mag, bpt_ivt, surf, rake=90.):
    
    data = [] 
    hypo = surf.mesh.get_middle_point()
    points = [pt for pt in surf.mesh]
    srf = GriddedSurface.from_points_list(points)
    brup = BaseRupture(mag=mag, rake=rake,
                       tectonic_region_type=trt,
                       hypocenter=hypo,
                       surface=srf)

    brup.weight = None
    tups = [(bpt_ivt[k], k) for k in bpt_ivt]
    pmf = PMF(data=tups)
    data.append((brup, pmf))       

    src = NonParametricSeismicSource(sid, name, trt, data=data)
    return src



def run_simulation_bpt(bpt, current_time, future_time, tseries, num_simulations=int(1e6)):
    
    # Initialize counters
    counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for _ in range(num_simulations):
        bpt /= bpt.sum()
        cdf = np.cumsum(bpt)
        # Generate the time until the next event, given that current_time has passed
        uniform_samples = np.random.rand(1)
        # Map uniform samples to values based on the CDF
        sample = tseries[np.searchsorted(cdf, uniform_samples)]
        next_event = sample - current_time
        
        while next_event <= 0:
            
            # Generate uniform random numbers
            uniform_samples = np.random.rand(1)
            
            # Map uniform samples to values based on the CDF
            sample = tseries[np.searchsorted(cdf, uniform_samples)]
            next_event = sample - current_time
        
        if next_event > future_time - current_time:
            counts[0] += 1
        else:
            time = current_time + next_event
            events = 1
            
            while time < future_time:                
                # Generate uniform random numbers
                uniform_samples = np.random.rand(1)
                
                # Map uniform samples to values based on the CDF
                next_event = tseries[np.searchsorted(cdf, uniform_samples)]
                time += next_event
                if time <= future_time:
                    events += 1
            
            if events > 3:
                counts[4] += 1
            else:
                counts[events] += 1
    
    # Calculate probabilities
    probabilities = {k: v / num_simulations for k, v in counts.items()}
    return probabilities

def time_dependent_probability_ivt(mu, alpha, current_time, future_time, end_time, sample=1):
    
    times = get_time_series(end_time, sample)
    
    bpt = bpt_pdf(mu, alpha, times)

    cps_ivt = run_simulation_bpt(bpt, current_time, future_time, times, num_simulations=int(1e5))
    
    return cps_ivt