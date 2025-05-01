import unittest
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
import sys

# Add project root to sys.path to allow importing psg_sleepedf
# Adjust the relative path if test_psg_sleepedf.py is placed elsewhere
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Assuming psg_sleepedf.py is in the PROJECT_ROOT. If it's elsewhere, adjust the path.
# If running tests from within the project root, this might not be strictly necessary,
# but it helps make the test runner more robust.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
# Import functions from the script to be tested
# Make sure psg_sleepedf.py is importable (e.g., doesn't run main() on import directly without __main__ check)
try:
    from psg_sleepedf import (
        get_true_subject_id,
        bandpower_welch,
        extract_psd_features,
        process_file,
        FS # Import sampling frequency used in the script
        # Add other necessary imports from psg_sleepedf if needed, e.g., FEATURES_DIR for patching
    )
    # Temporarily override FEATURES_DIR for testing process_file if it's used globally
    # This assumes FEATURES_DIR is defined globally in psg_sleepedf.py
    import psg_sleepedf 
    _original_features_dir = getattr(psg_sleepedf, 'FEATURES_DIR', None)

except ImportError as e:
    print(f"Error importing from psg_sleepedf.py: {e}")
    print(f"Make sure psg_sleepedf.py is in the Python path (PROJECT_ROOT: {PROJECT_ROOT}) and has no syntax errors.")
    sys.exit(1)

class TestPsgSleepEdfProcessing(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for test outputs."""
        self.test_dir = tempfile.mkdtemp()
        # If process_file uses the global FEATURES_DIR, patch it
        if _original_features_dir is not None:
            psg_sleepedf.FEATURES_DIR = self.test_dir

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)
        # Restore original FEATURES_DIR if patched
        if _original_features_dir is not None:
            psg_sleepedf.FEATURES_DIR = _original_features_dir


    def test_get_true_subject_id(self):
        """Test subject ID extraction."""
        self.assertEqual(get_true_subject_id("SC4001E0_epochs.npz"), "SC400")
        self.assertEqual(get_true_subject_id("SC4112E0_epochs.npz"), "SC411")
        self.assertEqual(get_true_subject_id("ST7052J0_epochs.npz"), "ST705")
        self.assertEqual(get_true_subject_id("ST7132J0_epochs.npz"), "ST713")
        # Test fallback (first 6 chars) - adjust if needed based on actual non-standard names
        self.assertEqual(get_true_subject_id("MySubject123_epochs.npz"), "MySubj") 
        self.assertEqual(get_true_subject_id("ABC_epochs.npz"), "ABC") # Shorter name


    def test_bandpower_welch(self):
        """Test PSD bandpower calculation."""
        fs = FS # Use sampling frequency from the script (e.g., 100.0)
        duration = 5 # seconds
        n_samples = int(fs * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # 1. Test signal within a band (10 Hz sine wave in alpha band 8-12 Hz)
        signal_10hz = np.sin(2 * np.pi * 10 * t)
        power_alpha = bandpower_welch(signal_10hz, fs, (8, 12))
        power_delta = bandpower_welch(signal_10hz, fs, (0.5, 4))
        self.assertGreater(power_alpha, 0.01, "Power in alpha band should be significant for 10Hz sine")
        self.assertAlmostEqual(power_delta, 0.0, delta=0.01, msg="Power in delta band should be near zero for 10Hz sine")

        # 2. Test signal outside a band (25 Hz sine wave vs alpha band)
        signal_25hz = np.sin(2 * np.pi * 25 * t)
        power_alpha_25 = bandpower_welch(signal_25hz, fs, (8, 12))
        power_beta_25 = bandpower_welch(signal_25hz, fs, (12, 30))
        self.assertAlmostEqual(power_alpha_25, 0.0, delta=0.01, msg="Power in alpha band should be near zero for 25Hz sine")
        self.assertGreater(power_beta_25, 0.01, "Power in beta band should be significant for 25Hz sine")
        
        # 3. Test Nyquist limit (gamma band 30-50 Hz with fs=100)
        # The function should use effective high freq = fs/2 = 50
        power_gamma_25 = bandpower_welch(signal_25hz, fs, (30, 50))
        self.assertAlmostEqual(power_gamma_25, 0.0, delta=0.01, msg="Power in gamma band should be near zero for 25Hz sine")
        
        signal_40hz = np.sin(2 * np.pi * 40 * t)
        power_gamma_40 = bandpower_welch(signal_40hz, fs, (30, 50))
        self.assertGreater(power_gamma_40, 0.01, "Power in gamma band should be significant for 40Hz sine")

        # 4. Test empty signal
        power_empty = bandpower_welch(np.array([]), fs, (4, 8))
        self.assertEqual(power_empty, 0.0, "Power for empty signal should be 0")

        # 5. Test very short signal (less than default nperseg if static)
        signal_short = np.random.rand(int(fs // 2)) # half a second
        power_short = bandpower_welch(signal_short, fs, (4, 8))
        # Just check it runs without error and returns a float
        self.assertIsInstance(power_short, float)


    def test_extract_psd_features(self):
        """Test extraction of PSD features for multiple channels."""
        fs = FS
        n_samples = int(fs * 30) # 30 second epoch
        t = np.linspace(0, 30, n_samples, endpoint=False)

        # Create synthetic data: channel 0 (EEG) low freq, channel 1 (EOG) mid freq
        eeg_signal = np.sin(2 * np.pi * 5 * t) # 5 Hz (theta/delta boundary)
        eog_signal = np.sin(2 * np.pi * 15 * t) # 15 Hz (beta)
        data_chunk = np.array([eeg_signal, eog_signal]) # Shape (2, n_samples)

        features = extract_psd_features(data_chunk, fs)

        # Check if all expected keys are present
        expected_keys = [f"{ch}_{band}" for ch in ['eeg', 'eog'] for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']]
        self.assertCountEqual(features.keys(), expected_keys, "Mismatch in expected feature keys")

        # Check relative power (rough check)
        self.assertGreater(features['eeg_theta'], features['eeg_beta'], "EEG theta power should dominate for 5Hz input")
        self.assertGreater(features['eog_beta'], features['eog_theta'], "EOG beta power should dominate for 15Hz input")
        self.assertAlmostEqual(features['eeg_gamma'], 0.0, delta=0.01, msg="EEG gamma should be near zero for 5Hz input")
        self.assertAlmostEqual(features['eog_delta'], 0.0, delta=0.01, msg="EOG delta should be near zero for 15Hz input")


    def test_process_file_integration(self):
        """Test the process_file function with temporary files."""
        # 1. Create a dummy input _epochs.npz file
        n_epochs = 5
        n_channels = 2
        n_samples = int(FS * 30)
        dummy_epochs = np.random.rand(n_epochs, n_channels, n_samples).astype('float32')
        dummy_labels = np.random.randint(0, 5, n_epochs).astype('int8')
        dummy_input_filename = "SC4999T0_epochs.npz"
        dummy_input_path = os.path.join(self.test_dir, dummy_input_filename)
        np.savez_compressed(dummy_input_path, data=dummy_epochs, labels=dummy_labels)

        # 2. Run process_file
        # process_file uses the FEATURES_DIR which we patched in setUp
        subj_id, rec_id = process_file(dummy_input_path)

        # 3. Check outputs
        self.assertEqual(subj_id, "SC499", "Subject ID mismatch")
        self.assertEqual(rec_id, "SC4999T0", "Recording ID mismatch")

        expected_csv_path = os.path.join(self.test_dir, "SC4999T0_psd.csv")
        expected_npz_path = os.path.join(self.test_dir, "SC4999T0_psd.npz")

        self.assertTrue(os.path.exists(expected_csv_path), "Output CSV file was not created")
        self.assertTrue(os.path.exists(expected_npz_path), "Output NPZ file was not created")

        # Optional: Basic checks on created files
        try:
            df = pd.read_csv(expected_csv_path)
            self.assertEqual(len(df), n_epochs, "CSV file has incorrect number of rows (epochs)")
            # Check number of columns: 5 bands * 2 channels + 1 label = 11
            self.assertEqual(len(df.columns), 11, "CSV file has incorrect number of columns")
            self.assertIn('eeg_delta', df.columns)
            self.assertIn('eog_gamma', df.columns)
            self.assertIn('label', df.columns)
            
            with np.load(expected_npz_path) as npz_data:
                self.assertIn('features', npz_data)
                self.assertIn('labels', npz_data)
                self.assertIn('feature_names', npz_data)
                self.assertIn('subject_id', npz_data)
                self.assertIn('recording_id', npz_data)
                self.assertEqual(npz_data['features'].shape[0], n_epochs, "NPZ features array has incorrect number of rows")
                self.assertEqual(npz_data['features'].shape[1], 10, "NPZ features array has incorrect number of columns") # 5 bands * 2 channels
                self.assertEqual(len(npz_data['labels']), n_epochs, "NPZ labels array has incorrect length")
                self.assertEqual(len(npz_data['feature_names']), 10)


        except Exception as e:
            self.fail(f"Error reading or verifying output files: {e}")


if __name__ == '__main__':
    unittest.main(argv=['', '-v'], exit=False) # '-v' for verbose output 