import os
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil

from openquake.cat.converter import GallahueAbrahamson2023Model1, GallahueAbrahamson2023Model2

class TestGallahueAbrahamson2023(unittest.TestCase):

    def setUp(self):
        """Initialize temporary directory and create structured array for testing."""
        self.test_dir = tempfile.mkdtemp()
        
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            "pga": [0.07, 0.20, 0.46],       # g
            "mag": [6.0, 6.5, 7.3],
            "rhypo": [10.0, 50.0, 100.0],    # km
            "rjb": [9.0, 45.0, 95.0],        # km
            "intensity": [3.9, 5.8, 8.0]
        })
        
        self.structured_data = self.sample_data.to_records(index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if os.path.exists("output"):
            shutil.rmtree("output")

    def test_model1_eq19_success(self):
        """Testing if equation 19 calculates intensity and saves correctly."""
        model = GallahueAbrahamson2023Model1(self.structured_data)
        model.get_intensity(mode='eq19')
        
        output_path = os.path.join("output", "test_gmice.csv")
        model.save(output_path)
        
        self.assertTrue(os.path.exists(output_path))
        df_res = pd.read_csv(output_path)
        
        # Checking if the result column 'intensity' exists
        self.assertIn("intensity", df_res.columns)
        self.assertEqual(len(df_res), 3)

    def test_model1_eq20_with_epsilon(self):
        """Testing if equation 20 handles epsilon shifts correctly."""
        model = GallahueAbrahamson2023Model1(self.structured_data)

        # Calculation with epsilon=0 and epsilon=1
        mean_results = model.get_intensity(mode='eq20', epsilon=0).copy()
        eps_results = model.get_intensity(mode='eq20', epsilon=1)
        
        # Eq 20 has h4 = -0.568. If epsilon=1, intensity should be lower than the mean.
        self.assertTrue(np.all(eps_results < mean_results))

    def test_model1_missing_column_error(self):
        bad_dt = np.dtype([('pga', 'f8'), ('mag', 'f8')])
        bad_data = np.array([(0.1, 6.0)], dtype=bad_dt)
        
        model = GallahueAbrahamson2023Model1(bad_data)
        with self.assertRaises(ValueError) as cm:
            model.get_intensity(mode='eq19')
        self.assertIn("Missing required columns", str(cm.exception))

    def test_model2_eq22_success(self):
        """Testing if equation 22 calculates PGA and saves correctly."""
        model = GallahueAbrahamson2023Model2(self.structured_data)
        model.get_pga(mode='eq22')
        
        output_path = os.path.join("output", "test_igmce.csv")
        model.save(output_path)
        
        self.assertTrue(os.path.exists(output_path))
        df_res = pd.read_csv(output_path)
        self.assertIn("pga", df_res.columns)
        self.assertTrue((df_res["pga"] > 0).all())

    def test_model2_eq23_with_epsilon(self):
        """Verifies equation 23 shifts PGA when epsilon is changed."""
        model = GallahueAbrahamson2023Model2(self.structured_data)
        
        pga_mean = model.get_pga(mode='eq23', epsilon=0).copy()
        pga_eps = model.get_pga(mode='eq23', epsilon=1)
        
        # Confirming results are different
        self.assertFalse(np.array_equal(pga_mean, pga_eps))

    def test_invalid_mode_error(self):
        """Verifies that an invalid mode string raises a ValueError."""
        model = GallahueAbrahamson2023Model1(self.structured_data)
        with self.assertRaises(ValueError) as cm:
            model.get_intensity(mode='wrong_mode')
        self.assertIn("Invalid mode", str(cm.exception))

    def test_save_before_calculation_error(self):
        """Verifies that saving before calculation triggers an error."""
        model = GallahueAbrahamson2023Model2(self.structured_data)
        with self.assertRaises(ValueError) as cm:
            model.save("failure.csv")
        self.assertIn("run 'get_pga' before saving", str(cm.exception))
