import json
import unittest
from openquake.mbt.tools.fault_conversion_utils import *


class TestConversionUtils(unittest.TestCase):

    def setUp(self):

        with open('../data/tools/data_test_fault_conversion.geojson') as ff:
            self.fault_dataset = json.load(ff)

        fault_1 = {k: v for k, v in f1['properties'].items()}
        fault_1['coords'] = f1['geometry']['coordinates']

        param_map_ = {# format: module_keyword: GIS file keyword
                      'source_id': 'fid',
                      'trace_coordinates': 'coords',
                      'upper_seismogenic_depth': 'upper_seis_depth',
                      'lower_seismogenic_depth': 'lower_seis_depth',
                      }

        param_map.update(param_map_)



    # metadata tests
    def test_get_source_id():
        _src_id = 0
        src_id = get_source_id(fault_1, defaults=defaults, param_map=param_map)
        assert _src_id == src_id
    
    
    def test_get_name():
        _name = "Karakoram Fault (Gar Basin segment)"
    
        name = get_name(fault_1, defaults=defaults, param_map=param_map)
    
        assert name == _name
    
    
    def test_get_tectonic_region_type():
        _trt = hz.const.TRT.ACTIVE_SHALLOW_CRUST
        trt = get_tectonic_region_type(fault_1, defaults=defaults)
    
        assert _trt == trt
    
    
    def test_write_metadata():
        
        _src_id = 0
        _name = "Karakoram Fault (Gar Basin segment)"
        _trt = hz.const.TRT.ACTIVE_SHALLOW_CRUST
    
        _meta_dict = {'source_id': _src_id,
                      'name': _name,
                      'tectonic_region_type': _trt}
    
        meta_dict = write_metadata(fault_1, defaults=defaults, 
                                   param_map=param_map)
    
        assert meta_dict == _meta_dict

    ## geometry tests
    def test_get_dip():
        _dip = 75.
        dip = get_dip(fault_1, requested_val='mle', defaults=defaults,
                      param_map=param_map)
    
        assert dip == _dip
    
    
    def test_get_dip_from_kinematics():
        fault_ = {'slip_type': 'Dextral'}
        _dip = 90.
    
        dip = get_dip(fault_, param_map=param_map)
    
        assert dip == _dip
    
    
    def test_get_rake():
        _rake = -10.
        rake = get_rake(fault_1, requested_val='mle', defaults=defaults,
                        param_map=param_map)
    
        assert rake == _rake
    
    
    def test_get_rake_from_kinematics():
        fault_ = {'slip_type': 'Dextral'}
        _rake = 180.
    
        rake = get_rake(fault_, param_map=param_map)
    
        assert rake == _rake
    
    
    def test_check_trace_from_coords_no_coord_reversal():
    
        fault = {'coords': [[0., 0.,], [-1., 1.]],
                 'dip_dir': 'E',
                 'slip_type': 'Reverse'
                 }
    
        trace = trace_from_coords(fault, param_map=param_map, defaults=defaults,
                                  check_coord_order=True)
    
        _trace = line_from_trace_coords(fault['coords'])
    
        assert trace.points == _trace.points
    
    
    def test_check_trace_from_coords_yes_coord_reversal():
    
        fault = {'coords': [[0., 0.,], [-1., 1.]],
                 'dip_dir': 'S',
                 'slip_type': 'Reverse'
                 }
    
        trace = trace_from_coords(fault, param_map=param_map, defaults=defaults,
                                  check_coord_order=True)
    
        _trace = line_from_trace_coords(fault['coords'])
        _trace.points.reverse()
    
        assert trace.points == _trace.points
    
    
    def test_calc_fault_width_from_usd_lsd_dip():
    
        fault = {'coords': [[0.,0.], [0.,1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E'
                 }
    
        _width = 20.
    
    
        width = calc_fault_width_from_usd_lsd_dip(fault, param_map=param_map,
                                                  defaults=defaults)
    
        assert abs(_width - width) < 0.01
    

    def test_get_fault_width_seismo_depth():
    
        fault = {'coords': [[0.,0.], [0.,1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E'
                 }
    
        _width = 20.
    
        width = get_fault_width(fault, method='seismo_depth', 
                                param_map=param_map,
                                defaults=defaults)
    
        assert abs(_width - width) < 0.01
    
    
    def test_get_fault_area_simple():
    
        fault = {'coords': [[0.,0.], [0.,1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E',
                 'slip_type': 'Reverse'
                 }
        
        length = get_fault_length(fault, defaults=defaults, param_map=param_map)
        _area = length * 20.
    
        area = get_fault_area(fault, area_method='simple', 
                              width_method='seismo_depth',
                              param_map=param_map, defaults=defaults)
    
        assert abs(_area - area) < 0.01


    # rates
    def test_get_net_slip_rate():
        _nsr = 6.
    
        nsr = get_net_slip_rate(fault_1, slip_class='mle', param_map=param_map,
                                defaults=defaults)
    
        assert nsr == _nsr

    def test_net_slip_from_strike_slip_fault_geom():
        pass

    def test_net_slip_from_vert_slip_fault_geom():
        pass

    def test_net_slip_from_shortening_fault_geom():
        pass

    def test_net_slip_from_strike_slip_shortening():
        pass

    def test_net_slip_from_vert_slip_shortening():
        pass

    def test_net_slip_from_vert_strike_slip():
        pass

    def test_net_slip_from_strike_slip_shortening():
        pass

    def test_net_slip_from_all_slip_comps():
        pass


    # MFDs and final objects
    def test_calc_mfd_from_fault_params_1():
        passs 
