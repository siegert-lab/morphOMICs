import unittest
import os
import pandas as pd
from morphomics.pipeline import Pipeline

class TestInput(unittest.TestCase):
    def setUp(self):
        # Define common variables
        self.data_location_filepath = "examples/data"
        self.extension = "corrected.swc"
        self.conditions = ["Region", "Model", "Sex", "Animal"]
        self.result_folderpath = "results/input"
        
        # Define the expected columns in a cleaner way
        self.expected_columns = ['file_path', 'file_name']
        self.expected_columns.extend(self.conditions)
        self.expected_columns.extend(['swc_array', 'cells'])
        
        # Check if data directory exists - fail early if it doesn't
        self.assertTrue(os.path.exists(self.data_location_filepath), 
                        f"Data directory {self.data_location_filepath} not found. Tests will fail without data.")
        
        # Create parameter dictionaries based on _pip_a_input.ipynb
        
        # 1. Basic Parameters
        self.params_basic = {
            'Input': {
                'data_location_filepath': self.data_location_filepath,
                'extension': self.extension,
                'conditions': self.conditions,
                'separated_by': None,
                'morphoframe_name': "input",
                'save_data': False,
                'save_folderpath': None,
                'save_filename': None
            }
        }
        
        # 2. Separated By
        self.params_separated_by = {
            'Input': {
                'data_location_filepath': self.data_location_filepath,
                'extension': self.extension,
                'conditions': self.conditions,
                'separated_by': 'Animal',
                'morphoframe_name': "input",
                'save_data': False,
                'save_folderpath': None,
                'save_filename': None
            }
        }
        
        # 3. Save Data, Default Filename
        self.params_save_default_filename = {
            'Input': {
                'data_location_filepath': self.data_location_filepath,
                'extension': self.extension,
                'conditions': self.conditions,
                'separated_by': 'Sex',
                'morphoframe_name': "input",
                'save_data': True,
                'save_folderpath': self.result_folderpath,
                'save_filename': None
            }
        }
        
        # 4. Save Data, Defined Filename
        self.params_save_defined_filename = {
            'Input': {
                'data_location_filepath': self.data_location_filepath,
                'extension': self.extension,
                'conditions': self.conditions,
                'separated_by': 'Sex',
                'morphoframe_name': "input",
                'save_data': True,
                'save_folderpath': self.result_folderpath,
                'save_filename': "defined_name"
            }
        }
        
        # 5. Default params
        self.params_default = {
            'Input': {
                'data_location_filepath': self.data_location_filepath,
                'extension': self.extension,
                'save_data': True,
                'save_filename': 'default_params',
                'save_folderpath': self.result_folderpath,
            }
        }
        
        # 6. Wrong params (for testing error handling)
        self.params_wrong = {
            'Input': {
                'data_location_filepath': self.data_location_filepath,
                'wrong_params': 0,
                'extension': self.extension,
                'save_data': False,
                'save_filename': 'wrong_params',
                'save_folderpath': self.result_folderpath,
            }
        }
    
    def _run_input_and_verify_basic(self, params, morphoframe_name="input"):
        """Helper method to run Input protocol and verify basic expectations"""
        my_pip = Pipeline(parameters=params, Parameters_ID='test')
        my_pip.Input()
        
        # Assert that the morphoframe has been created with the expected key
        self.assertIn(morphoframe_name, my_pip.morphoframe)
        # Check that the morphoframe is a pandas DataFrame
        self.assertIsInstance(my_pip.morphoframe[morphoframe_name], pd.DataFrame)
        
        # Check that the DataFrame has the expected columns
        for col in self.expected_columns:
            self.assertIn(col, my_pip.morphoframe[morphoframe_name].columns)
            
        return my_pip
    
    def _verify_file_saved(self, filename):
        """Helper method to verify that a file was saved"""
        expected_filename = os.path.join(self.result_folderpath, f"{filename}")
        # Check if the file exists - we expect data to be present
        self.assertTrue(os.path.exists(expected_filename), 
                       f"Expected file {expected_filename} was not created")
        return expected_filename

    def test_input_basic(self):
        """Test Input protocol with basic parameters"""
        self._run_input_and_verify_basic(self.params_basic)
    
    def test_input_separated_by(self):
        """Test Input protocol with separated_by parameter"""
        my_pip = self._run_input_and_verify_basic(self.params_separated_by)
        
        # Check that the 'Animal' column exists and has values
        self.assertIn('Animal', my_pip.morphoframe['input'].columns)
        self.assertTrue(len(my_pip.morphoframe['input']['Animal'].unique()) > 0)
    
    def test_input_save_default_filename(self):
        """Test Input protocol with save_data and default filename"""
        my_pip = self._run_input_and_verify_basic(self.params_save_default_filename)
        
        # Check if the file was saved with the default filename
        self._verify_file_saved('Morphomics.PID_test.Cell.pkl')
        self._verify_file_saved('Morphomics.PID_test.Cell.Sex-M.pkl')
        self._verify_file_saved('Morphomics.PID_test.Cell.Sex-F.pkl')
        self._verify_file_saved('Morphomics.PID_test.Cell-FailedFiles.txt')

    def test_input_save_defined_filename(self):
        """Test Input protocol with save_data and defined filename"""
        my_pip = self._run_input_and_verify_basic(self.params_save_defined_filename)
        
        # Check if the file was saved with the defined filename
        self._verify_file_saved('Morphomics.PID_test.defined_name.pkl')
        self._verify_file_saved('Morphomics.PID_test.defined_name.Sex-M.pkl')
        self._verify_file_saved('Morphomics.PID_test.defined_name.Sex-F.pkl')
        self._verify_file_saved('Morphomics.PID_test.defined_name-FailedFiles.txt')

    def test_input_default_params(self):
        """Test Input protocol with default parameters"""
        my_pip = self._run_input_and_verify_basic(self.params_default, morphoframe_name='microglia')
        
        # Check if the file was saved with the specified filename
        self._verify_file_saved('Morphomics.PID_test.default_params.pkl')
        self._verify_file_saved('Morphomics.PID_test.default_params-FailedFiles.txt')

    def test_input_wrong_params(self):
        """Test Input protocol with wrong parameters (should raise ValueError)"""
        my_pip = Pipeline(parameters=self.params_wrong, Parameters_ID='test')
        
        # This should raise a ValueError due to the 'wrong_params' key
        with self.assertRaises(ValueError):
            my_pip.Input()

if __name__ == '__main__':
    unittest.main() 