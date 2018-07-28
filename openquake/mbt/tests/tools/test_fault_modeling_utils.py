import json
import unittest
import os

from openquake.mbt.tools.fault_modeler.fault_modeling_utils import *


BASE_DATA_PATH = os.path.dirname(__file__)
data_dir_str = '../data/tools/'
test_data_dir = os.path.join(BASE_DATA_PATH, data_dir_str)


class TestConversionUtils(unittest.TestCase):


    def setUp(self):
        data_file_name = 'data_test_fault_conversion.geojson'

        test_file_path = os.path.join(test_data_dir, data_file_name)

        with open(test_file_path) as ff:
            self.fault_dataset = json.load(ff)
            del ff

        f1 = self.fault_dataset['features'][0]

        self.fault_1 = {k: v for k, v in f1['properties'].items()}
        self.fault_1['coords'] = f1['geometry']['coordinates']

        param_map_ = {# format: module_keyword: GIS file keyword
                      'source_id': 'fid',
                      'trace_coordinates': 'coords',
                      'upper_seismogenic_depth': 'upper_seis_depth',
                      'lower_seismogenic_depth': 'lower_seis_depth',
                      }

        self.param_map = deepcopy(param_map)
        self.param_map.update(param_map_)



    # metadata tests
    
    
    def test_write_metadata(self):
        
        _src_id = 0
        _name = "Karakoram Fault (Gar Basin segment)"
        _trt = hz.const.TRT.ACTIVE_SHALLOW_CRUST
    
        _meta_dict = {'source_id': _src_id,
                      'name': _name,
                      'tectonic_region_type': _trt}
    
        meta_dict = write_metadata(self.fault_1, defaults=defaults, 
                                   param_map=self.param_map)
    
        self.assertEqual(meta_dict, _meta_dict)

    ## geometry tests
    def test_get_dip(self):
        _dip = 75.
        dip = get_dip(self.fault_1, requested_val='mle', defaults=defaults,
                      param_map=self.param_map)
    
        self.assertEqual(dip, _dip)
    
    
    def test_get_dip_from_kinematics(self):
        fault_ = {'slip_type': 'Dextral'}
        _dip = 90.
    
        dip = get_dip(fault_, param_map=self.param_map)
    
        self.assertEqual(dip, _dip)
    
    
    def test_get_rake(self):
        _rake = -10.
        rake = get_rake(self.fault_1, requested_val='mle', defaults=defaults,
                        param_map=self.param_map)
    
        self.assertEqual(rake, _rake)
    
    
    def test_get_rake_from_kinematics(self):
        fault_ = {'slip_type': 'Dextral'}
        _rake = 180.
    
        rake = get_rake(fault_, param_map=self.param_map)
    
        self.assertEqual(rake, _rake)
    
    
    def test_check_trace_from_coords_no_coord_reversal(self):
    
        fault = {'coords': [[0., 0.,], [-1., 1.]],
                 'dip_dir': 'E',
                 'slip_type': 'Reverse'
                 }
    
        trace = trace_from_coords(fault, param_map=self.param_map, 
                                  defaults=defaults,
                                  check_coord_order=True)
    
        _trace = line_from_trace_coords(fault['coords'])
    
        self.assertEqual(trace.points, _trace.points)
    
    
    def test_check_trace_from_coords_yes_coord_reversal(self):
    
        fault = {'coords': [[0., 0.,], [-1., 1.]],
                 'dip_dir': 'S',
                 'slip_type': 'Reverse'
                 }
    
        trace = trace_from_coords(fault, param_map=self.param_map, 
                                  defaults=defaults,
                                  check_coord_order=True)
    
        _trace = line_from_trace_coords(fault['coords'])
        _trace.points.reverse()
    
        self.assertEqual(trace.points, _trace.points)
    
    
    def test_calc_fault_width_from_usd_lsd_dip(self):
    
        fault = {'coords': [[0.,0.], [0.,1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E'
                 }
    
        _width = 20.
    
    
        width = calc_fault_width_from_usd_lsd_dip(fault, 
                                                  param_map=self.param_map,
                                                  defaults=defaults)
    
        self.assertTrue( abs(_width - width) < 0.01)
    

    def test_get_fault_width_seismo_depth(self):
    
        fault = {'coords': [[0.,0.], [0.,1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E'
                 }
    
        _width = 20.
    
        width = get_fault_width(fault, method='seismo_depth', 
                                param_map=self.param_map,
                                defaults=defaults)
    
        self.assertTrue( abs(_width - width) < 0.01)
    
    
    def test_get_fault_area_simple(self):
    
        fault = {'coords': [[0.,0.], [0.,1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E',
                 'slip_type': 'Reverse'
                 }
        
        length = get_fault_length(fault, defaults=defaults, 
                                  param_map=self.param_map)
        _area = length * 20.
    
        area = get_fault_area(fault, area_method='simple', 
                              width_method='seismo_depth',
                              param_map=self.param_map, defaults=defaults)
    
        self.assertTrue( abs(_area - area) < 0.01)


    # rates
    def test_get_net_slip_rate(self):
        _nsr = 6.
    
        nsr = get_net_slip_rate(self.fault_1, slip_class='mle', 
                                param_map=self.param_map,
                                defaults=defaults)
    
        self.assertEqual( nsr, _nsr)

    def test_net_slip_from_strike_slip_fault_geom(self):
        pass

    def test_net_slip_from_vert_slip_fault_geom(self):
        pass

    def test_net_slip_from_shortening_fault_geom(self):
        pass

    def test_net_slip_from_strike_slip_shortening(self):
        pass

    def test_net_slip_from_vert_slip_shortening(self):
        pass

    def test_net_slip_from_vert_strike_slip(self):
        pass

    def test_net_slip_from_strike_slip_shortening(self):
        pass

    def test_net_slip_from_all_slip_comps(self):
        pass


    # MFDs and final objects
    def test_calc_mfd_from_fault_params_1(self):
        pass





if __name__ == "__main__":
    unittest.main()
