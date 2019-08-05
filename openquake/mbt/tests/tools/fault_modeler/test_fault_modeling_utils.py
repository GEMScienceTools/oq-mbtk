import json
import unittest
import os
import numpy as np
from copy import deepcopy

import openquake.mbt.tools.fault_modeler.fault_modeling_utils as fmu
import openquake.hazardlib as hz

# -----------------------------------------------------------------------------

BASE_DATA_PATH = os.path.dirname(__file__)
test_data_dir = os.path.join(BASE_DATA_PATH, '..', '..', 'data', 'tools')


# -----------------------------------------------------------------------------

class TestModelingUtils(unittest.TestCase):

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

        self.param_map = deepcopy(fmu.param_map)
        self.param_map.update(param_map_)

    # Metadata tests
    def test_write_metadata(self):

        meta_dict = fmu.write_metadata(self.fault_1, defaults=fmu.defaults,
                                       param_map=self.param_map)

        self.assertEqual(meta_dict, {
            'source_id': 0,
            'name': "Karakoram Fault (Gar Basin segment)",
            'tectonic_region_type': hz.const.TRT.ACTIVE_SHALLOW_CRUST})

    # Geometry tests
    def test_get_dip(self):

        dip = fmu.get_dip(self.fault_1, requested_val='mle',
                          defaults=fmu.defaults,
                          param_map=self.param_map)

        self.assertEqual(dip, 75.)

    def test_get_dip_from_kinematics(self):

        dip = fmu.get_dip({'slip_type': 'Dextral'}, param_map=self.param_map)

        self.assertEqual(dip, 90.)

    def test_get_rake(self):

        rake = fmu.get_rake(self.fault_1, requested_val='mle',
                            defaults=fmu.defaults, param_map=self.param_map)

        self.assertEqual(rake, -10.)

    def test_get_rake_from_kinematics(self):

        rake = fmu.get_rake({'slip_type': 'Dextral'}, param_map=self.param_map)

        self.assertEqual(rake, 180.)

    def test_check_trace_from_coords_no_coord_reversal(self):

        fault = {'coords': [[0., 0.], [-1., 1.]],
                 'dip_dir': 'E',
                 'slip_type': 'Reverse'}

        trace = fmu.trace_from_coords(fault, param_map=self.param_map,
                                      defaults=fmu.defaults,
                                      check_coord_order=True)

        ref_trace = fmu.line_from_trace_coords(fault['coords'])

        self.assertEqual(trace.points, ref_trace.points)

    def test_check_trace_from_coords_yes_coord_reversal(self):

        fault = {'coords': [[0., 0.], [-1., 1.]],
                 'dip_dir': 'S',
                 'slip_type': 'Reverse'}

        trace = fmu.trace_from_coords(fault, param_map=self.param_map,
                                      defaults=fmu.defaults,
                                      check_coord_order=True)

        exp_trace = fmu.line_from_trace_coords(fault['coords'])
        exp_trace.points.reverse()

        self.assertEqual(trace.points, exp_trace.points)

    def test_calc_fault_width_from_usd_lsd_dip(self):

        fault = {'coords': [[0., 0.], [0., 1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E'}

        width = fmu.calc_fault_width_from_usd_lsd_dip(
                    fault,
                    param_map=self.param_map,
                    defaults=fmu.defaults)

        self.assertTrue(abs(20. - width) < 0.01)

    def test_get_fault_width_seismo_depth(self):

        fault = {'coords': [[0., 0.], [0., 1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E'}

        width = fmu.get_fault_width(fault, width_method='seismo_depth',
                                    param_map=self.param_map,
                                    defaults=fmu.defaults)

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

        width = fmu.get_fault_width(fault, width_method='length_scaling',
                                    defaults=fmu.defaults,
                                    param_map=self.param_map)

        lsd = fmu.get_lower_seismo_depth(fault, width_method='length_scaling',
                                         defaults=fmu.defaults,
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

        length = fmu.get_fault_length(fault, defaults=fmu.defaults,
                                      param_map=self.param_map)

        area = fmu.get_fault_area(fault, area_method='simple',
                                  width_method='seismo_depth',
                                  param_map=self.param_map,
                                  defaults=fmu.defaults)

        self.assertTrue(abs((length * 20.) - area) < 0.01)

    # Rates
    def test_get_net_slip_rate(self):

        nsr = fmu.get_net_slip_rate(self.fault_1, slip_class='mle',
                                    param_map=self.param_map,
                                    defaults=fmu.defaults)

        self.assertEqual(nsr, 6.)

    @unittest.skip("not yet implemented")
    def test_net_slip_from_strike_slip_fault_geom(self):
        pass

    def test_net_slip_from_shortening_fault_geom(self):

        fault = {'coords': [[0., 0.], [0., 1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E',
                 'slip_type': 'Reverse',
                 'shortening_rate': '({},,)'.format(np.sqrt(3.))}

        net_slip_rate = fmu.net_slip_from_shortening_fault_geom(fault)

        self.assertTrue(abs(2. - net_slip_rate) < 0.01)

    def test_net_slip_from_vert_slip_fault_geom(self):

        fault = {'coords': [[0., 0.], [0., 1.]],
                 'upper_seis_depth': 0.,
                 'lower_seis_depth': 10.,
                 'average_dip': '(30,,)',
                 'dip_dir': 'E',
                 'slip_type': 'Reverse',
                 'vert_slip_rate': '(1.,,)'}

        net_slip_rate = fmu.net_slip_from_vert_slip_fault_geom(fault)

        net_slip_true_rate = 2.

        self.assertTrue(abs(net_slip_rate - net_slip_true_rate) < 0.01)


    @unittest.skip("not yet implemented")
    def test_net_slip_from_strike_slip_shortening(self):
        pass

    @unittest.skip("not yet implemented")
    def test_net_slip_from_vert_slip_shortening(self):
        pass

    @unittest.skip("not yet implemented")
    def test_net_slip_from_vert_strike_slip(self):
        pass

    @unittest.skip("not yet implemented")
    def test_net_slip_from_all_slip_comps(self):
        pass

    # MFDs and final objects
    def test_calc_mfd_from_fault_params_double_gr(self):

        # mfd_type should be set to 'DoubleTruncatedGR' by default;
        # this is part of the test.

        mfd, seis_rate = fmu.calc_mfd_from_fault_params(
                                                self.fault_1,
                                                param_map=self.param_map,
                                                defaults=fmu.defaults)
        # mdf_rates values were computed by hand using m_min = 4.0
        # and m_cli = 6.0 as default values
        mfd_rates = [(6.05, 0.006316366706615863),
                     (6.1499999999999995, 0.005017268415937408),
                     (6.25, 0.0039853579639694565),
                     (6.35, 0.0031656823562642173),
                     (6.45, 0.0025145908777491803),
                     (6.55, 0.0019974105329762706),
                     (6.65, 0.0015865995826787318),
                     (6.75, 0.0012602808457234777),
                     (6.85, 0.0010010766594403568),
                     (6.95, 0.0007951834557169339),
                     (7.05, 0.0006316366706616012)]

        seis_rate_ = 6.0

        mfd_rate_calc = mfd.get_annual_occurrence_rates()

        self.assertTrue(abs(seis_rate_ - seis_rate) < 0.01)

        for i, rate in enumerate(mfd_rates):
            self.assertTrue(abs(rate[1] - mfd_rate_calc[i][1]) < 0.01)

    def test_calc_mfd_from_fault_params_yc_1985(self):

        mfd, seis_rate = fmu.calc_mfd_from_fault_params(
                                            self.fault_1,
                                            mfd_type='YoungsCoppersmith1985',
                                            param_map=self.param_map,
                                            defaults=fmu.defaults)

        mfd_rates = [(6.05, 0.0005161196304084878),
                     (6.1499999999999995, 0.00040996839492892304),
                     (6.249999999999999, 0.00032564947143663876),
                     (6.349999999999999, 0.00025867256978516083),
                     (6.449999999999998, 0.00020547092572904065),
                     (6.549999999999998, 0.0001632113577215128),
                     (6.649999999999998, 0.0001296433896658826),
                     (6.749999999999997, 0.00010297940485697282),
                     (6.849999999999997, 0.0008258332824773236),
                     (6.949999999999997, 0.0008258332824773236),
                     (7.049999999999996, 0.0008258332824773236),
                     (7.149999999999996, 0.0008258332824773236),
                     (7.249999999999996, 0.0008258332824773236)]

        seis_rate_ = 6.0

        mfd_rate_calc = mfd.get_annual_occurrence_rates()

        self.assertTrue(abs(seis_rate_ - seis_rate) < 0.01)

        for i, rate in enumerate(mfd_rates):
            self.assertTrue(abs(rate[1] - mfd_rate_calc[i][1]) < 0.01)

    # Utils
    def test_apparent_dip_from_dip_rake_1(self):

        apparent_dip = fmu.apparent_dip_from_dip_rake(30., 90.)
        self.assertTrue(abs(apparent_dip - 30.) < 0.01)

    def test_apparent_dip_from_dip_rake(self):
        dips = [81., 81., 79, 79., 30, 30, 30]
        rakes = [17., 163., 50, 130, 20, 40, 60]
        ads = [17, 17., 49, 49, 10, 19, 26]

        for i, dip in enumerate(dips):
            rake = rakes[i]
            ad_true = ads[i]
            ad = fmu.apparent_dip_from_dip_rake(dip, rake)
            self.assertTrue( np.abs(ad - ad_true) < 1)

if __name__ == "__main__":
    unittest.main()
