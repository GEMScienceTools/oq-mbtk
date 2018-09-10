import json
import unittest
import os
import numpy as np

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

        param_map_ = {  # format: module_keyword: GIS file keyword
                      'source_id': 'fid',
                      'trace_coordinates': 'coords',
                      'upper_seismogenic_depth': 'upper_seis_depth',
                      'lower_seismogenic_depth': 'lower_seis_depth'}

        self.param_map = deepcopy(param_map)
        self.param_map.update(param_map_)

    # Metadata tests
    def test_write_metadata(self):

        meta_dict = write_metadata(self.fault_1, defaults=defaults,
                                   param_map=self.param_map)

        self.assertEqual(meta_dict, {
            'source_id': 0,
            'name': "Karakoram Fault (Gar Basin segment)",
            'tectonic_region_type': hz.const.TRT.ACTIVE_SHALLOW_CRUST})

    # Geometry tests
    def test_get_dip(self):

        dip = get_dip(self.fault_1, requested_val='mle', defaults=defaults,
                      param_map=self.param_map)

        self.assertEqual(dip, 75.)

    def test_get_dip_from_kinematics(self):

        dip = get_dip({'slip_type': 'Dextral'}, param_map=self.param_map)

        self.assertEqual(dip, 90.)

    def test_get_rake(self):

        rake = get_rake(self.fault_1, requested_val='mle', defaults=defaults,
                        param_map=self.param_map)

        self.assertEqual(rake, -10.)

    def test_get_rake_from_kinematics(self):

        rake = get_rake({'slip_type': 'Dextral'}, param_map=self.param_map)

        self.assertEqual(rake, 180.)

    def test_check_trace_from_coords_no_coord_reversal(self):

        fault = {'coords': [[0., 0.], [-1., 1.]],
                 'dip_dir': 'E',
                 'slip_type': 'Reverse'}

        trace = trace_from_coords(fault, param_map=self.param_map,
                                  defaults=defaults,
                                  check_coord_order=True)

        ref_trace = line_from_trace_coords(fault['coords'])

        self.assertEqual(trace.points, ref_trace.points)

    def test_check_trace_from_coords_yes_coord_reversal(self):

        fault = {'coords': [[0., 0.], [-1., 1.]],
                 'dip_dir': 'S',
                 'slip_type': 'Reverse'}

        trace = trace_from_coords(fault, param_map=self.param_map,
                                  defaults=defaults,
                                  check_coord_order=True)

        exp_trace = line_from_trace_coords(fault['coords'])
        exp_trace.points.reverse()

        self.assertEqual(trace.points, exp_trace.points)

    def test_calc_fault_width_from_usd_lsd_dip(self):

        fault = {'coords': [[0., 0.], [0., 1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E'}

        width = calc_fault_width_from_usd_lsd_dip(fault,
                                                  param_map=self.param_map,
                                                  defaults=defaults)

        self.assertTrue(abs(20. - width) < 0.01)

    def test_get_fault_width_seismo_depth(self):

        fault = {'coords': [[0., 0.], [0., 1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E'}

        width = get_fault_width(fault, width_method='seismo_depth',
                                param_map=self.param_map,
                                defaults=defaults)

        self.assertTrue(abs(20. - width) < 0.01)

    def test_get_lsd_from_width_scaling_rel(self):
        """
        Tests lower seismogenic depth (from length scaling relationship)
        by asserting the seismogenic thickness should be half the fault
        width for a fault with 30 degree dip
        """

        fault = {'coords': [[0., 0.], [0., 1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E',
                 'slip_type': 'Reverse'}

        width = get_fault_width(fault, width_method='length_scaling',
                                defaults=defaults, param_map=self.param_map)

        lsd = get_lower_seismo_depth(fault, width_method='length_scaling',
                                     defaults=defaults,
                                     param_map=self.param_map)

        seis_thickness = lsd - fault['upper_seis_depth']

        self.assertAlmostEqual(seis_thickness * 2, width)

    def test_get_fault_area_simple(self):

        fault = {'coords': [[0., 0.], [0., 1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E',
                 'slip_type': 'Reverse'}

        length = get_fault_length(fault, defaults=defaults,
                                  param_map=self.param_map)

        area = get_fault_area(fault, area_method='simple',
                              width_method='seismo_depth',
                              param_map=self.param_map, defaults=defaults)

        self.assertTrue(abs((length * 20.) - area) < 0.01)

    # Rates
    def test_get_net_slip_rate(self):

        nsr = get_net_slip_rate(self.fault_1, slip_class='mle',
                                param_map=self.param_map,
                                defaults=defaults)

        self.assertEqual(nsr, 6.)

    def test_net_slip_from_strike_slip_fault_geom(self):
        pass

    def test_net_slip_from_vert_slip_fault_geom(self):
        pass

    def test_net_slip_from_shortening_fault_geom(self):

        fault = {'coords': [[0., 0.], [0., 1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E',
                 'slip_type': 'Reverse',
                 'shortening_rate': '({},,)'.format(np.sqrt(3.))}

        net_slip_rate = net_slip_from_shortening_fault_geom(fault)

        self.assertTrue(abs(2. - net_slip_rate) < 0.01)

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

    # Utils
    def test_apparent_dip_from_dip_rake_1(self):

        apparent_dip = apparent_dip_from_dip_rake(30., 90.)
        self.assertTrue(abs(apparent_dip - 30.) < 0.01)

if __name__ == "__main__":
    unittest.main()
