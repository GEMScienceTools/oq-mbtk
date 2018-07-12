import numpy as np
from copy import deepcopy
import warnings

from openquake.hazardlib.source import SimpleFaultSource
import openquake.hazardlib as hz
import openquake.mbt as oqmbt
from openquake.mbt.tools.faults import rates_for_double_truncated_mfd


sfs_params = ('source_id', 
              'name', 
              'tectonic_region_type', 
              'mfd',
              'rupture_mesh_spacing', 
              'magnitude_scaling_relationship',
              'rupture_aspect_ratio', 
              'temporal_occurrence_model',
              'upper_seismogenic_depth', 
              'lower_seismogenic_depth',
              'fault_trace', 
              'average_dip', 
              'average_rake')

all_params = list(sfs_params)

all_params += ['slip_type', 'trace_coordinates', 'dip_dir', 'M_min', 'M_max',
               'net_slip_rate', 'strike_slip_rate', 'dip_slip_rate',
               'vert_slip_rate', 'shortening_rate']


param_map = {p:p for p in all_params}


# default parameter values
defaults = {'tectonic_region_type': hz.const.TRT.ACTIVE_SHALLOW_CRUST,
            'b_value': 1.,
            'bin_width': 0.1,
            'aseismic_coefficient': 0.9,
            'upper_seismogenic_depth': 0.,
            'lower_seismogenic_depth': 20.,
            'rupture_mesh_spacing': 2.,
            'rupture_aspect_ratio': 2.,
            'minimum_fault_length': 5.,
            'M_min': 6.0,
            'M_max': 8.0,
            'temporal_occurrence_model': hz.tom.PoissonTOM(1.0),
            'magnitude_scaling_relationship': 
                hz.scalerel.leonard2014.Leonard2014_Interplate(),
            }  


def construct_sfs_dict(fault_dict, area_method='simple', width_method='seismo_depth',
                       scaling_rel='leonard_2010', slip_class='mle',
                       mag_scaling_rel=None, M_max=None, M_min=None,
                       b_value=None, slip_rate=None, bin_width=None,
                       fault_area=None, defaults=defaults,
                       param_map=param_map, ):
    
    """ 
    Makes a dictionary containing all of the parameters needed to create a SimpleFaultSource.

    :para



    """

    sfs = {p: None for p in sfs_params}

    # source_id, name, tectonic_region_type
    sfs.update(write_metadata(fault_dict, defaults=defaults, param_map=param_map))

    # dip, rake, fault_trace, upper_seismogenic_depth, lower_seismogenic_depth
    sfs.update(write_geom(fault_dict, defaults=defaults, param_map=param_map))

    # rupture_mesh_spacing, magnitude_scaling_relationship,
    # rupture_aspect_ratio, temporal_occurrence_model
    sfs.update(write_rupture_params(fault_dict, defaults=defaults, param_map=param_map))

    # mfd
    sfs.update(
        {'mfd': calc_mfd_from_fault_params(fault_dict, area_method=area_method,
                                           width_method=width_method,
                                           scaling_rel=scaling_rel,
                                           slip_class=slip_class,
                                           mag_scaling_rel=mag_scaling_rel,
                                           M_max=M_max, M_min=M_min,
                                           b_value=b_value,
                                           slip_rate=slip_rate,
                                           bin_width=bin_width,
                                           fault_area=fault_area,
                                           defaults=defaults,
                                           param_map=param_map)
         })

    for param in sfs_params:
        if sfs[param] is None:
            err_msg = 'Missing Value: {} for id {}'.format(param, 
                                                           sfs['source_id'])
            raise ValueError(err_msg)

    return sfs


def make_fault_source(sfs_dict):
    """
    """

    try:
        src = SimpleFaultSource(*[sfs_dict[param] for param in sfs_params])
        return src
    except Exception as e:
        print(Exception)


###
# util functions
###

def fetch_param_val(fault_dict, param, defaults=defaults, param_map=param_map):
    """
    """

    try:
        val = fault_dict[param_map[param]]
        if val is None:
            val = fault_dict[defaults[param]]
    except KeyError:
        try:
            val = fault_dict[defaults[param]]
        except (KeyError, TypeError):  # not a field in the fault's attr dict
            val = defaults[param]


    return val


# tuple parsing
def tuple_to_vals(tup):
    """
    """

    tup = tup.replace('(', '').replace(')', '')
    vals = tup.split(',')
    return vals


def get_vals_from_tuple(tup):
    """
    """

    if type(tup) == str:
        # print('str')
        vals = tuple_to_vals(tup)
        vals = [np.float(v) for v in vals if v is not '']
    elif np.isscalar(tup):
        # print('scalar')
        try:
            num_check = np.float(tup)
            vals = [np.float(tup)]
        except Exception as e:
            raise ValueError
    elif type(tup) in [tuple, list, np.ndarray]:
        # print('sequence')
        vals = tup
    else:
        # print('other')
        raise ValueError

    if len(vals) == 0:
        raise Exception(ValueError)
    elif len(vals) == 1:
        return vals
    elif len(vals) > 3:
        raise ValueError

    else:
        if len(vals) == 2:
            vals = [vals[0], np.mean(vals), vals[1]]
        vals = np.sort(vals)[np.array([1, 0, 2])]
    return vals


def get_val_from_tuple(tup, requested_val='mle', _abs_sort=False):
    """
    """

    # not guaranteed to work if min and max have different sign
    if requested_val == 'suggested':
        requested_val = 'mle'

    vals = get_vals_from_tuple(tup)
    if np.isscalar(vals):
        return vals
    elif len(vals) == 1:
        return vals[0]
    else:
        if requested_val == 'min':
            if _abs_sort is True:
                return min(vals.min(), vals.max(), key=abs)
            else:
                return vals[1]
        elif requested_val == 'mle':
            return vals[0]
        elif requested_val == 'max':
            if _abs_sort is True:
                return max(vals.min(), vals.max(), key=abs)
            else:
                return vals[2]


###
# metadata
###

def write_metadata(fault_dict, defaults=defaults, param_map=param_map):
    metadata_params = ('tectonic_region_type', 'name', 'source_id')

    meta_param_d = {p: fetch_param_val(fault_dict, p, defaults=defaults,
                                       param_map=param_map)
                    for p in metadata_params}
    return meta_param_d

###
# rupture params
###


def write_rupture_params(fault_dict, defaults=defaults, param_map=param_map):

    rupture_params = ('rupture_mesh_spacing', 'magnitude_scaling_relationship',
                      'rupture_aspect_ratio', 'temporal_occurrence_model')

    rup_param_d = {p: fetch_param_val(fault_dict, p, defaults=defaults,
                                      param_map=param_map)
                   for p in rupture_params}
    return rup_param_d


###
# geometry and kinematics
###


rake_map = {'Normal': -90.,
              'Normal-Dextral': -135.,
              'Normal-Sinistral': -45.,
              'Reverse': 90.,
              'Reverse-Dextral': 135.,
              'Reverse-Sinistral': 45.,
              'Sinistral': 0.,
              'Sinistral-Normal': -45.,
              'Sinistral-Reverse': 45.,
              'Dextral': 180.,
              'Dextral-Reverse': 135.,
              'Dextral-Normal': -135.,
              'Strike-Slip': 0.,
              'Thrust': 90.,
              'Blind-Thrust': 90.,
              'Spreading-Ridge': -90.}

dip_map = {'Normal': 60.,
             'Normal-Dextral': 65.,
             'Normal-Sinistral': 65.,
             'Reverse': 40.,
             'Reverse-Dextral': 65.,
             'Reverse-Sinistral': 65.,
             'Sinistral': 90.,
             'Sinistral-Normal': 65.,
             'Sinistral-Reverse': 65.,
             'Dextral': 90.,
             'Dextral-Reverse': 65.,
             'Dextral-Normal': 65.,
             'Strike-Slip': 90.,
             'Thrust': 40.,
             'Blind-Thrust': 40.,
             'Spreading-Ridge': 60.}

# To transform literal values into numbers
direction_map = {'N': 0.,
                 'NNE': 22.5,
                 'NE': 45.,
                 'ENE': 67.5,
                 'E': 90.,
                 'ESE': 112.5,
                 'S': 180.,
                 'W': 270.,
                 'NW': 315.,
                 'SE': 135.,
                 'SW': 225.,
                 'U': 0.}

#def get_slip_type(f):
#    """
#    """
#
#    # for this we exclude Subduction-Thrusts, misnamed as they are
#    # I think this is just to calculate dip and rake if not included
#    return fetch_param_val(f, 'slip_type')
#
#
#def get_dip_azimuth(f):
#    """
#    """
#
#    pass


def trace_from_coords(fault_dict, defaults=defaults, param_map=param_map,
                      check_coord_order=True):
    # get the coords
    # check against dip direction
        # flip if need be
    # return a Line

    trace_coords = fetch_param_val(fault_dict, 'trace_coordinates', defaults=defaults,
                                   param_map=param_map)

    fault_trace = line_from_trace_coords(trace_coords)

    if check_coord_order is True:
        fault_trace = _check_trace_coord_ordering(fault_dict, fault_trace)

    return fault_trace


def line_from_trace_coords(trace_coords):

    fault_trace = hz.geo.Line([hz.geo.Point( i[0], i[1]) 
                               for i in trace_coords])

    return fault_trace


def _check_trace_coord_ordering(fault_dict, fault_trace, reverse_angle_threshold=90.):
    """
    Enforces right-hand rule with respect to fault trace coordinate ordering
    and dip direction.  If there is an inconsitency, the trace coordinates
    are reversed.

    :param fault_dict:
        Dictionary with fault parameters.

    :type fault_dict:
        dict

    :param fault_trace:
        line

    :type fault_trace:
        openquake.hazardlib.geo.line.Line

    :param reverse_angle_threshold:
        Angle difference (between calculated and given dip direction) above
        which the coordinate ordering will be reversed. Defaults to 90.

    :type reverse_angle_threshold:
        float
    """

    strike = fault_trace.average_azimuth()

    trace_dip_trend = strike + 90.

    fault_dip_dir = fetch_param_val(fault_dict, 'dip_dir', defaults=defaults,
                                    param_map=param_map)

    fault_dip_trend = direction_map[fault_dip_dir]

    trend_angle_diff = angle_difference(fault_dip_trend, trace_dip_trend)

    if abs(90 - trend_angle_diff) < 15:
        warnings.warn('Given dip direction <15 degrees of strike')

    if trend_angle_diff > reverse_angle_threshold:
        new_fault_trace = deepcopy(fault_trace)
        new_fault_trace.points.reverse()
        return new_fault_trace
    else:
        return fault_trace


def angle_difference(trend_1, trend_2, return_abs=True):
    """
    Calculates the difference between two trends or azimuths (trend_1 and 
    trend_2), in degrees.

    :param trend_1:
        Number in degrees of first trend/azimuth.

    :type trend_1:
        float

    :param trend_2:
        Number in degrees of second trend/azimuth.

    :type trend_2:
        float

    :param return_abs:
        Flag for returning the absolute value of the angular
        difference. if `return_abs` is False, the returned angle is from
        `trend_2` to `trend_1` in the coordinate convention adopted
        (clockwise for azimuth, counter-clockwise for unit circle angles).

    :type return_abs:
        bool

    """

    difference = trend_2 - trend_1

    while difference < -180.:
        difference += 360.
    while difference > 180:
        difference -= 360.

    if return_abs is True:
        difference = abs(difference)

    return difference




def write_geom(fault_dict, requested_val='mle', defaults=defaults, param_map=param_map):
    """
    """

    #TODO: ensure consistency w/ rake and dip for min/max slip_class

    geom_params = {'average_rake': get_rake(fault_dict, requested_val=requested_val,
                                            defaults=defaults,
                                            param_map=param_map),

                   'average_dip': get_dip(fault_dict, requested_val=requested_val,
                                          defaults=defaults,
                                          param_map=param_map),

                   'fault_trace': trace_from_coords(fault_dict, param_map=param_map,
                                                    defaults=defaults),

                   'upper_seismogenic_depth': fetch_param_val(fault_dict,
                                                     'upper_seismogenic_depth',
                                                              defaults=defaults,
                                                              param_map=param_map),
                        
                   'lower_seismogenic_depth': fetch_param_val(fault_dict,
                                                     'lower_seismogenic_depth',
                                                              defaults=defaults,
                                                              param_map=param_map),
                        
                    }
    
    
    return geom_params


def get_rake(fault_dict, requested_val='mle', defaults=defaults, param_map=param_map):
    """
    """

    try:
        rake_tuple = fetch_param_val(fault_dict, 'average_rake', defaults=defaults,
                                     param_map=param_map)
        rake = get_val_from_tuple(rake_tuple, requested_val=requested_val)

    except KeyError:
        try:
            slip_type = fetch_param_val(fault_dict, 'slip_type', defaults=defaults,
                                        param_map=param_map)
            rake = rake_map[slip_type]
        except KeyError as e:
            print(e)

    return rake


def get_dip(fault_dict, requested_val='mle', defaults=defaults, param_map=param_map):
    """
    Returns a value of dip from the dip tuple for each structure. If no
    dip tuple is present, a default value is returned based on the fault
    kinematics.
    """

    try:
        dip_tuple = fetch_param_val(fault_dict, 'average_dip', defaults=defaults,
                                    param_map=param_map)
        dip = get_val_from_tuple(dip_tuple, requested_val=requested_val)
    
        return dip
    except KeyError:
        try:
            slip_type = fetch_param_val(fault_dict, 'slip_type', defaults=defaults,
                                        param_map=param_map)
            dip = dip_map[slip_type]
            return dip
        except Exception as e:
            raise e


###
# slip rates and mfds
###

def fetch_slip_rate(fault_dict, rate_component, slip_class='suggested', _abs_sort=True):
    """
    """

    requested_val = 'mle' if slip_class is 'suggested' else slip_class
    slip_rate_tup = fetch_param_val(fault_dict, rate_component)
    
    return get_val_from_tuple(slip_rate_tup, requested_val, 
                              _abs_sort=_abs_sort)


def get_net_slip_rate(fault_dict, slip_class='suggested', defaults=defaults,
                      param_map=param_map):
    """
    """
    # should rename slip_class

    #TODO: Return dip and rake for min/max when ambiguity

    if slip_class == 'suggested':
        slip_class = 'mle'

    rate_comps = []
    for rate_type in ['net_slip_rate', 'strike_slip_rate',
                      'shortening_rate', 'vert_slip_rate']:
        try:
            rate = fault_dict[param_map[rate_type]]
            if rate is not None:
                rate_comps.append(rate_type)
        except KeyError:
            pass

    if 'net_slip_rate' in rate_comps:
        return fetch_slip_rate(fault_dict, 'net_slip_rate', slip_class=slip_class)
    elif rate_comps == ['strike_slip_rate']:
        return net_slip_from_strike_slip_fault_geom(fault_dict, slip_class=slip_class)
    elif rate_comps == ['vert_slip_rate']:
        return net_slip_from_vert_slip_fault_geom(fault_dict, slip_class=slip_class)
    elif rate_comps == ['shortening_rate']:
        return net_slip_from_shortening_fault_geom(fault_dict, slip_class=slip_class)
    elif set(rate_comps) == {'strike_slip_rate', 'shortening_rate'}:
        return net_slip_from_strike_slip_shortening(fault_dict, slip_class=slip_class)
    elif set(rate_comps) == {'vert_slip_rate', 'shortening_rate'}:
        return net_slip_from_vert_slip_shortening(fault_dict, slip_class=slip_class)
    elif set(rate_comps) == {'strike_slip_rate', 'vert_slip_rate'}:
        return net_slip_from_vert_strike_slip(fault_dict, slip_class=slip_class)
    elif set(rate_comps) == {'strike_slip_rate', 'shortening_rate'}:
        return net_slip_from_strike_slip_shortening(fault_dict, slip_class=slip_class)
    elif set(rate_comps) == {'strike_slip_rate', 'shortening_rate',
                             'vert_slip_rate'}:
        return net_slip_from_all_slip_comps(fault_dict, slip_class=slip_class)

    else:
        print(rate_comps)


def net_slip_from_strike_slip_fault_geom(fault_dict, slip_class='mle', _abs=True):
    """
    """

    strike_slip_rate = fetch_slip_rate(fault_dict, 'strike_slip_rate',
                                       slip_class=slip_class)
    
    rake = get_vals_from_tuple(fetch_param_val(fault_dict, 'rake'))

    net_slip_rate = strike_slip_rate / np.cos(np.radians(rake))
    if _abs is True:
        net_slip_rate = np.abs(net_slip_rate)

    if np.isscalar(rake):
        return net_slip_rate
    elif slip_class == 'mle':
        return np.sort(net_slip_rate)[1]
    elif slip_class == 'min':
        return np.min(net_slip_rate)
    elif slip_class == 'max':
        return np.max(net_slip_rate)
    else:
        raise Exception('not enough info')


def dip_slip_from_vert_slip(fault_dict, slip_class='mle', _abs=True):
    """
    """

    vert_slip_rate = fetch_slip_rate(fault_dict, 'vert_slip_rate',
                                     slip_class=slip_class)
    
    dips = get_vals_from_tuple(fetch_param_val(fault_dict, 'average_dip'))

    dip_slip_rate = vert_slip_rate / np.sin(np.radians(dips))

    if not np.isscalar(dip_slip_rate):
        if len(dip_slip_rate) == 1:
            return dip_slip_rate[0]
        elif slip_class == 'mle':
            return dip_slip_rate[0]
        elif slip_class == 'min':
            return dip_slip_rate[1]
        elif slip_class == 'max':
            return dip_slip_rate[2]
    else:
        return dip_slip_rate


def net_slip_from_vert_slip_fault_geom(fault_dict, slip_class='mle', _abs=True):
    """
    """

    vert_slip_rate = fetch_slip_rate(fault_dict, 'vert_slip_rate',
                                     slip_class=slip_class)
    dip_slip_rate = dip_slip_from_vert_slip(fault_dict, slip_class=slip_class)

    return net_slip_from_dip_slip_fault_geom(dip_slip_rate, fault_dict,
                                             slip_class=slip_class,
                                             _abs=_abs)


def net_slip_from_shortening_fault_geom(fault_dict, slip_class='mle', _abs=True):
    """
    """

    shortening_rate = fetch_slip_rate(fault_dict, 'shortening_rate',
                                      slip_class=slip_class)

    if _abs is True:
        shortening_rate = np.abs(shortening_rate)
    dips = get_vals_from_tuple(fetch_param_val(fault_dict, 'average_dip'))
    rakes = get_vals_from_tuple(fetch_param_val(fault_dict, 'rake'))

    if slip_class == 'mle':
        dip = dips[0] if not np.isscalar(dips) else dips
        rake = rakes[0] if not np.isscalar(rakes) else rakes
        apparent_dip = _apparent_dip_from_dip_rake(dip, rake)
        net_slip_rate = shortening_rate / np.cos(np.radians(apparent_dip))
        return net_slip_rate

    else:
        if np.isscalar(dips):
            dips = [dip]
        if np.isscalar(rakes):
            rakes = [rakes]

        apparent_dips = [_apparent_dip_from_dip_rake(dip, rake)
                         for dip in dips
                         for rake in rakes]
        net_slip_rates = shortening_rate / np.cos(np.radians(apparent_dips))
        if slip_class == 'max':
            return np.max(net_slip_rates)
        elif slip_class == 'min':
            return np.min(net_slip_rates)


def net_slip_from_dip_slip_fault_geom(dip_slip_rate, fault_dict, slip_class='mle',
                                      _abs=True):
    """
    """

    rakes = get_vals_from_tuple(fetch_param_val(fault_dict, 'rake'))

    if _abs:
        rakes = np.abs(rakes)

    if slip_class == 'mle':
        if not np.isscalar(rakes):
            rake = rakes[0]
        else:
            rake = rakes
        if rake in (0, 0., 180, 180., -180, -180.):
            warnings.warn(
                    "Cannot derive dip slip rate with rake {}".format(rake))
        return dip_slip_rate / np.sin(np.radians(rakes[0]))

    else:
        for rake in rakes:
            if rake in (0, 0., 180, 180., -180, -180.):
                warnings.warn(
                       "Cannot derive dip slip rate with rake {}".format(rake))

        net_slip_rates = [dip_slip_rate / np.sin(np.radians(rake))
                          for rake in rakes]
        if slip_class == 'min':
            return np.min(net_slip_rates)
        elif slip_class == 'max':
            return np.max(net_slip_rates)


def net_slip_from_vert_slip_shortening(fault_dict, slip_class='mle', _abs=True):
    """
    """

    vert_slip_rate = fetch_slip_rate(fault_dict, 'vert_slip_rate',
                                     slip_class=slip_class)
    shortening_rate = fetch_slip_rate(fault_dict, 'shortening_rate',
                                      slip_class=slip_class)
    dip_slip_rate = dip_slip_from_vert_rate_shortening(vert_slip_rate, 
                                                       shortening_rate)

    rakes = get_vals_from_tuple(fetch_param_val(fault_dict, 'rake'))
    rake_diffs = np.abs(np.pi / 2 - np.radians(rakes))

    net_slip_rates = dip_slip_rate / np.cos(rake_diffs)

    if slip_class == 'mle':
        if len(net_slip_rates) == 3:
            return net_slip_rates[0]
        elif len(net_slip_rates) == 1:
            return net_slip_rates[0]
        else:
            raise Exception('oops!')

    elif slip_class == 'min':
        return np.min(net_slip_rates)
    elif slip_class == 'max':
        return np.max(net_slip_rates)


def net_slip_from_vert_strike_slip(fault_dict, slip_class='mle', _abs=True):
    """
    """

    strike_slip_rate = fetch_slip_rate(fault_dict, 'strike_slip_rate',
                                       slip_class=slip_class)
    dip_slip_rate = dip_slip_from_vert_slip(fault_dict, slip_class=slip_class,
                                            _abs=_abs)

    return np.sqrt(dip_slip_rate ** 2 + strike_slip_rate ** 2)


def net_slip_from_strike_slip_shortening(fault_dict, slip_class='mle',
                                         _abs=True):
    """
    """

    strike_slip_rate = fetch_slip_rate(fault_dict, 'strike_slip_rate',
                                       slip_class=slip_class)
    dip_slip_rate = dip_slip_from_shortening(fault_dict, slip_class=slip_class,
                                             _abs=True)

    return np.sqrt(dip_slip_rate ** 2 + strike_slip_rate ** 2)


def dip_slip_from_shortening(fault_dict, slip_class='mle', _abs=True):
    """
    """

    short_rate = fetch_slip_rate(fault_dict, 'shortening_rate', slip_class=slip_class)
    dips = get_vals_from_tuple(fetch_param_val(fault_dict, 'average_dip'))
    if np.isscalar(dips):
        dip = dips
    elif len(dips) == 1:
        dip = dips
    elif slip_class == 'mle':
        dip = dips[0]
    elif slip_class == 'min':
        dip = dips[1]
    elif slip_class == 'max':
        dip = dips[2]

    if dip in (90., 90):
        warnings.warn(
              'Cannot calculate dip slip from shortening with vertical fault.')

    dip = np.radians(dip)

    return short_rate / np.cos(dip)


def net_slip_from_all_slip_comps(fault_dict, slip_class='mle', _abs=True):
    """
    """

    vert_slip_rate = fetch_slip_rate(fault_dict, 'vert_slip_rate',
                                     slip_class=slip_class)
    shortening_rate = fetch_slip_rate(fault_dict, 'shortening_rate',
                                      slip_class=slip_class)
    strike_slip_rate = fetch_slip_rate(fault_dict, 'strike_slip_rate',
                                       slip_class=slip_class)

    dip_slip_rate = dip_slip_from_vert_rate_shortening(vert_slip_rate,
                                                       shortening_rate)

    return np.sqrt(dip_slip_rate ** 2 + strike_slip_rate ** 2)


def _apparent_dip_from_dip_rake(dip, rake):
    """
    """

    dip = np.abs(dip)
    rake = np.abs(rake)
    return np.degrees(np.arctan(np.tan(np.radians(rake)) 
                                * np.sin(np.radians(dip))))


def true_dip_from_vert_short(vert, short):
    """
    """

    vert, short = np.abs((vert, short))
    return np.degrees(np.arctan(vert / short))


def dip_slip_from_vert_rate_shortening(vert, short):
    """
    """

    return np.sqrt(vert ** 2 + short ** 2)


WIDTH_CLASS = {'cl1': ['Normal', 'Reverse', 'Thrust', 'Normal-Dextral',
                       'Normal-Sinistral', 'Reverse-Sinistral',
                       'Reverse-Dextral', 'Spreading-Ridge',
                       'Blind-Thrust'],
               'cl2': ['Dextral', 'Sinistral', 'Strike-Slip',
                       'Dextral-Normal', 'Dextral-Reverse',
                       'Sinistral-Normal', 'Sinistral-Reverse']
               }


# make spreading ridge width very small?


def get_fault_length(fault_dict, defaults=defaults, param_map=param_map):
    """
    Returns the length of a fault.
    """

    try:
        fault_trace = fault_dict['fault_trace']
    except KeyError:
        fault_trace = trace_from_coords(fault_dict, param_map=param_map,
                                        defaults=defaults)

    return fault_trace.get_length()


def get_fault_width(fault_dict, method='length_scaling', scaling_rel='leonard_2010',
                    defaults=defaults, param_map=param_map):
    """
    Returns the width (i.e., the down-dip distance) of a fault. Two methods
    exist: One based on the fault length and a scaling relationship, and one
    based on the upper and lower seismogenic depths.

    :param fault_dict:
        desc

    :type fault_dict:
        dict

    :param method:
        Method used to calculate the width of the fault. 'length_scaling'
        implements a scaling relationship between the fault length (derived
        from the trace) and the fault width, which is calculated.
        'seismo_depth' calculates the width based on the fault's dip and the
        given values for upper and lower seismogenic depth.

    :type method:
        str

    :param scaling_rel:
        The scaling relationship between length and width. Currently,
        only the scaling relationship of Leonard (2010) BSSA is implemented,
        as 'leonard_2010'.

    :type scaling_rel:
        str
        

    """

    if method == 'length_scaling':
        width = calc_fault_width_from_length(fault_dict, scaling_rel=scaling_rel)

    elif method == 'seismo_depth':
        width = calc_fault_width_from_usd_lsd_dip(fault_dict, defaults=defaults,
                                                  param_map=param_map)
    else:
        raise ValueError('method ', method, 'not recognized')

    return width


def calc_fault_width_from_usd_lsd_dip(fault_dict, defaults=defaults,
                                      param_map=param_map):
    """
    Calculates the width (down-dip distance) of the fault from the fault's
    dip and seismogenic boundaries).

    :param fault_dict:
        desc

    :type fault_dict:
        dict

    """


    usd = fetch_param_val(fault_dict, 'upper_seismogenic_depth', defaults=defaults,
                          param_map=param_map)
    lsd = fetch_param_val(fault_dict, 'lower_seismogenic_depth', defaults=defaults,
                          param_map=param_map)
    dip = get_dip(fault_dict, defaults=defaults, param_map=param_map)

    denom = np.sin(np.radians(dip))

    if denom == 0.:
        raise ValueError("Cannot calculate down-dip width when dip is zero")

    width = (lsd - usd) / denom

    return width


def calc_fault_width_from_length(fault_dict, scaling_rel='leonard_2010', **kwargs):
    """
    """


    scale_func_dict = {'leonard_2010': leonard_width_from_length}

    #try:
    width = scale_func_dict[scaling_rel](fault_dict, **kwargs)
    return width
    #except KeyError:
    #    raise ValueError('scaling relationship ', scaling_rel, 
    #                     'not implemented.')


def leonard_width_from_length(fault_dict, const_1=1.75, const_2=1.5, dip_cutoff=45.,
                              exp=2./3., max_width_1=150., max_width_2=20.,
                              defaults=defaults, param_map=param_map):
    """
    """

    slip_type = fetch_param_val(fault_dict, 'slip_type', defaults=defaults,
                                param_map=param_map)

    fault_length = get_fault_length(fault_dict, defaults=defaults, param_map=param_map)

    if slip_type in WIDTH_CLASS['cl1']:
        width = const_1 * fault_length ** exp
        width = min((width, max_width_1))

    elif slip_type in WIDTH_CLASS['cl2']:
        if fault_length <= dip_cutoff:
            width = const_2 * fault_length ** exp
            width = min((width, max_width_2))
        else:
            width = min((width, max_width_2))
    return width


def get_fault_area(fault_dict, area_method='simple', width_method='seismo_depth',
                   scaling_rel='leonard_2010', defaults=defaults,
                   param_map=param_map): 
    """ 
    """

    if area_method == 'simple':
        fault_length = get_fault_length(fault_dict, defaults=defaults,
                                        param_map=param_map)
        fault_width = get_fault_width(fault_dict, method=width_method,
                                      defaults=defaults, param_map=param_map)
        fault_area = fault_length * fault_width

    elif area_method == 'from_surface':

        try:
            fault_trace = fault_dict['fault_trace']
        except KeyError:
            fault_trace = trace_from_coords(fault_dict, param_map=param_map,
                                            defaults=defaults)

        usd = fetch_param_val(fault_dict, 'upper_seismogenic_depth')

        if width_method == 'seismo_depth':
            lsd = fetch_param_val(fault_dict, 'lower_seismogenic_depth')

        elif width_method == 'length_scaling':

            width = calc_fault_width_from_length(fault_dict, scaling_rel=scaling_rel,
                                                 defaults=defaults,
                                                 param_map=param_map)

            dip = get_dip(fault_dict, param_map=param_map, defaults=defaults)

            lsd = width * np.sin(np.radians(dip))

        else:
            raise ValueError('width_method ', width_method, 'not recognized')

        dip = get_dip(fault_dict, defaults=defaults, param_map=param_map)
        mesh_spacing = fetch_param_val(fault_dict, 'rupture_mesh_spacing')

        fault_area = hz.geo.surface.SimpleFaultSurface.from_fault_data(
                                                                 fault_trace,
                                                                 usd, lsd, dip,
                                                                 mesh_spacing
                                                                 ).get_area()
    return fault_area



def get_M_min(fault_dict, mag_scaling_rel=None, defaults=defaults, param_map=param_map):
    """
    """

    try:
        M_min = fault_dict[param_map['M_min']]
    except KeyError:
        try:
            M_min = defaults['M_min']
        except KeyError:
            if scaling_rel is None:
                scaling_rel = defaults['msr_scaling_rel']
            pass

    return M_min


def get_M_max(fault_dict, mag_scaling_rel=None, area_method='simple',
              width_method='seismo_depth', width_scaling_rel='leonard_2010',
              defaults=defaults, param_map=param_map):
    """
    doxx
    """

    # TODO: check if fault M_max is greater than zone M_max
    try:
        M_max = fault_dict[param_map['M_max']]

    except KeyError:
        try:
            # fetch?
            if mag_scaling_rel is None:
                mag_scaling_rel = defaults['mag_scaling_rel']

            rake = get_rake(fault_dict)
            fault_area = get_fault_area(fault_dict, method=area_method,
                                        width_method=width_method,
                                        scaling_rel=width_scaling_rel,
                                        defaults=defaults, param_map=param_map)

            M_max = mag_scaling_rel.get_median_mag(fault_area, rake)

        except:
            M_max = defaults['M_max']

    return M_max



def calc_mfd_from_fault_params(fault_dict, area_method='simple',
                               width_method='seismo_depth',
                               scaling_rel='leonard_2010',
                               slip_class='mle',
                               mag_scaling_rel=None,
                               M_max=None, M_min=None,
                               b_value=None, slip_rate=None,
                               bin_width=None, fault_area=None,
                               defaults=defaults, param_map=param_map
                               ):
    """
    """

    if fault_area is None:
        fault_area = get_fault_area(fault_dict)

    if M_min is None:
        M_min = get_M_min(fault_dict, defaults=defaults, param_map=param_map)
    if M_max is None:
        M_max = get_M_max(fault_dict, defaults=defaults, param_map=param_map,
                          mag_scaling_rel=mag_scaling_rel)

    if slip_rate is None:
        slip_rate = get_net_slip_rate(fault_dict, defaults=defaults,
                                      param_map=param_map)
    if M_min > M_max:
        raise ValueError('M_min is greater than M_max')

    if b_value is None:
        b_value = fetch_param_val(fault_dict, 'b_value', defaults=defaults,
                                  param_map=param_map)
    if bin_width is None:
        bin_width = fetch_param_val(fault_dict, 'bin_width', defaults=defaults,
                                    param_map=param_map)

    bin_rates = rates_for_double_truncated_mfd( fault_area, slip_rate, M_min,
                                               M_max, b_value, bin_width)

    mfd = hz.mfd.EvenlyDiscretizedMFD(M_min + bin_width / 2., bin_width,
                                      bin_rates)

    return mfd

