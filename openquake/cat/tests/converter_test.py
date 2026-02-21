import os
import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil

from openquake.cat.converter import gmice, igmce

class TestGallahueAbrahamson2023(unittest.TestCase):

    def setUp(self):
        """Initialize a temporary directory and create dummy data for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.test_dir, 'data.csv')
        
        # Sample data for testing
        self.sample_data = pd.DataFrame({
            "PGA": [0.07, 0.20, 0.46],         # g
            "Mw": [6.0, 6.5, 7.3],
            "Rhyp": [10.0, 50.0, 100.0],       # km
            "Rjb": [9.0, 45.0, 95.0],          # km
            "Intensity": [3.9, 5.8, 8.0]
        })
        self.sample_data.to_csv(self.data_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        if os.path.exists("output"):
            shutil.rmtree("output")

    def test_gmice_eq19_success(self):
        """Testing if Equation 19 calculates Intensity_pred and saves correctly."""
        gmice(self.data_path, "eq19")
        output_path = os.path.join("output", "gmice_res.csv")
        
        self.assertTrue(os.path.exists(output_path))
        df_res = pd.read_csv(output_path)
        self.assertIn("Intensity_pred", df_res.columns)
        self.assertEqual(len(df_res), 3)

    def test_gmice_eq20_with_epsilon(self):
        """Testing if Equation 20 handles epsilon correctly."""
        gmice(self.data_path, "eq20", epsilon=0)
        df_mean = pd.read_csv(os.path.join("output", "gmice_res.csv"))
        
        # Calculate with epsilon=1
        gmice(self.data_path, "eq20", epsilon=1)
        df_eps = pd.read_csv(os.path.join("output", "gmice_res.csv"))
        
        # Eq 20 has h4 = -0.568. If epsilon=1, intensity should be lower than the mean.
        self.assertTrue((df_eps["Intensity_pred"] < df_mean["Intensity_pred"]).all())

    def test_gmice_eq19_missing_column_error(self):
        """Verifies that Equation 19 raises ValueError if required columns are not found."""
        bad_data = self.sample_data.drop(columns=["Rhyp"])
        bad_input = os.path.join(self.test_dir, 'bad_input.csv')
        bad_data.to_csv(bad_input, index=False)
        
        with self.assertRaises(ValueError) as cm:
            gmice(bad_input, "eq19")
        self.assertIn("Rhyp", str(cm.exception))

    def test_igmce_eq22_success(self):
        """Testing if Equation 22 calculates PGA successfully."""
        igmce(self.data_path, "eq22")
        output_path = os.path.join("output", "igmce_res.csv")
        
        self.assertTrue(os.path.exists(output_path))
        df_res = pd.read_csv(output_path)
        self.assertIn("PGA_pred", df_res.columns)
        self.assertTrue((df_res["PGA_pred"] > 0).all())

    def test_igmce_eq23_with_epsilon(self):
        """Testing if Equation 23 handles epsilon and changes the PGA_pred result."""
        igmce(self.data_path, "eq23", epsilon=0)
        pga_mean = pd.read_csv(os.path.join("output", "igmce_res.csv"))["PGA_pred"]
        
        igmce(self.data_path, "eq23", epsilon=1)
        pga_eps = pd.read_csv(os.path.join("output", "igmce_res.csv"))["PGA_pred"]
        
        # Eq 23 has i4 = -0.187. Check if the results are different.
        self.assertFalse(np.array_equal(pga_mean.values, pga_eps.values))

    def test_invalid_equation_id(self):
        """Testing if an incorrect equation_id triggers the appropriate error message."""
        with self.assertRaises(ValueError) as cm:
            gmice(self.data_path, "eq_wrong")
        self.assertIn("Invalid equation ID", str(cm.exception))

    def test_output_directory_creation(self):
        """Testing if the script automatically creates the 'output' directory."""
        if os.path.exists("output"):
            shutil.rmtree("output")
            
        gmice(self.data_path, "eq20")
        self.assertTrue(os.path.isdir("output"))
