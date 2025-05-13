import os
import numpy as np

def get_file_info(npz_file_path):
    """
    Loads an NPZ file and extracts data shape, sampling rate, and inferred channel count.

    Args:
        npz_file_path (str): The path to the .npz file.

    Returns:
        dict: A dictionary containing 'shape', 'sampling_rate', 'num_channels', 
              'channel_source', and 'error' keys. Values will be None or an 
              error message if info cannot be found.
    """
    results = {
        'data_key': None,
        'shape': None,
        'sampling_rate': None,
        'num_channels': None,
        'channel_source': None,  # How num_channels was determined
        'error': None,
        'keys_found': None,
    }

    try:
        with np.load(npz_file_path) as data:
            keys = data.files
            results['keys_found'] = keys

            # Find sampling rate
            for fs_key in ['fs', 'sampling_rate', 'sfreq']:
                if fs_key in keys:
                    try:
                        results['sampling_rate'] = data[fs_key].item() # Use .item() if it's a 0-d array
                        break
                    except Exception as e:
                         results['error'] = f"Error accessing sampling rate key '{fs_key}': {e}"
                         # Continue trying other keys

            # Find primary data array and its shape
            data_array = None
            for data_key in ['x', 'sequences', 'data', 'signals']:
                if data_key in keys:
                    try:
                        data_array = data[data_key]
                        results['data_key'] = data_key
                        results['shape'] = data_array.shape
                        break # Stop after finding the first primary data key
                    except Exception as e:
                        results['error'] = f"Error accessing shape of data key '{data_key}': {e}"
                        # Reset data_array if error occurred
                        data_array = None 
                        results['data_key'] = None
                        results['shape'] = None
                        # Continue searching for other data keys in case this one was problematic

            # Infer number of channels based on shape and filename convention
            if results['shape'] is not None:
                shape = results['shape']
                if len(shape) >= 3 and '_epochs' in npz_file_path:
                    # Assume shape (num_epochs, num_channels, num_samples)
                    results['num_channels'] = shape[1]
                    results['channel_source'] = f"Inferred from shape[1] for epochs file: {shape}"
                elif len(shape) >= 2: 
                     # Assume shape (..., num_samples, num_channels) or (num_samples, num_channels) for sequences/other
                     # or (num_channels, num_samples) - check shape[0] vs shape[1] maybe?
                     # Let's default to last dim for sequences, but second-to-last if only 2D
                     if len(shape) == 2:
                         results['num_channels'] = shape[0] # Guessing channels first for 2D
                         results['channel_source'] = f"Inferred from shape[0] for 2D array: {shape}"
                     else: # 3+ Dims, likely sequences (..., seq_len, channels)
                         results['num_channels'] = shape[-1]
                         results['channel_source'] = f"Inferred from shape[-1]: {shape}"
                else:
                    results['channel_source'] = f"Could not infer channels from shape: {shape}"

            # Fallback: check channel name list if shape inference failed
            if results['num_channels'] is None:
                for ch_key in ['ch_names', 'channels']:
                    if ch_key in keys:
                        try:
                            results['num_channels'] = len(data[ch_key])
                            results['channel_source'] = f"From length of '{ch_key}' key"
                            break
                        except Exception as e:
                            error_msg = f"Error accessing length of '{ch_key}': {e}"
                            results['error'] = f"{results.get('error', '')}; {error_msg}".lstrip('; ')


            if results['shape'] is None and results['num_channels'] is None:
                 if results['error'] is None:
                    results['error'] = f"Could not find known data keys or channel keys. Found: {keys}"


    except FileNotFoundError:
        results['error'] = "File not found."
    except Exception as e:
        results['error'] = f"Error loading file: {e}"

    return results

if __name__ == "__main__":
    # Determine the project root based on the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..')) # Go up two levels from notebooks/mesa_code
    target_dir = os.path.join(project_root, 'new_processed_mesa')
    
    print(f"Checking NPZ files in directory: {target_dir}")

    if not os.path.isdir(target_dir):
        print(f"Error: Directory not found: {target_dir}")
    else:
        for filename in sorted(os.listdir(target_dir)):
            if filename.endswith(".npz"):
                file_path = os.path.join(target_dir, filename)
                print(f"--- Processing: {filename} ---")
                
                info = get_file_info(file_path)

                print(f"  Keys found: {info['keys_found']}")
                print(f"  Data key used: {info['data_key']}")
                print(f"  Shape: {info['shape']}")
                print(f"  Sampling Rate (Hz): {info['sampling_rate']}")
                print(f"  Inferred Channels: {info['num_channels']}")
                print(f"  Channel Source: {info['channel_source']}")
                if info['error']:
                    print(f"  Error/Info: {info['error']}")
                
                print("-" * (len(filename) + 18))

    print("\nScript finished.") 