import os
import re
import glob
import logging
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
# Optional: For potential basic EDF reading in the future
# try:
#     import pyedflib
# except ImportError:
#     print("Warning: pyedflib not installed. Cannot perform advanced EDF checks.", file=sys.stderr)
#     pyedflib = None

BASE_DATA_DIR = "/Users/kimberly/Documents/STAT4830/STAT-4830-GOALZ-project/data/sleep-edf-database-expanded-1.0.0"
CASSETTE_DIR_NAME = "sleep-cassette"
TELEMETRY_DIR_NAME = "sleep-telemetry"
LOG_FILENAME = "preprocessing_full.log"
# Use fewer workers initially to avoid overwhelming system/logs if needed
MAX_WORKERS = os.cpu_count() // 2 if os.cpu_count() > 1 else 1

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(processName)s: %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.INFO)

# File Handler (overwrite log file each run)
file_handler = logging.FileHandler(LOG_FILENAME, mode='w')
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout) 
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


def find_hypnogram(psg_filepath):
    """
    Finds the corresponding hypnogram file for a given PSG file.

    Args:
        psg_filepath (str): The full path to the PSG file.

    Returns:
        str or None: The full path to the hypnogram file if found uniquely, otherwise None.
    """
    psg_filename = os.path.basename(psg_filepath)
    dir_path = os.path.dirname(psg_filepath)

    # identifier (e.g., "SC4022" or "ST7011")
    match = re.match(r"(SC|ST)(\d{4})", psg_filename)
    if not match:
        return None, f"Could not extract core ID from PSG filename: {psg_filename}"
    core_id = match.group(0) # e.g., "SC4022"

    hypnogram_pattern = f"{core_id}*-Hypnogram.edf"
    search_pattern = os.path.join(dir_path, hypnogram_pattern)

    try:
        found_hypno_files = glob.glob(search_pattern)
    except Exception as e:
        return None, f"Error during glob search for {search_pattern}: {e}"

    if not found_hypno_files:
        return None, f"No hypnogram file found matching pattern {search_pattern}"
    elif len(found_hypno_files) > 1:
        # Log details about multiple files found
        file_list = ", ".join([os.path.basename(f) for f in found_hypno_files])
        return None, f"Multiple hypnograms found matching {search_pattern}: [{file_list}]. Skipping."
    else:
        # Unique match found
        return found_hypno_files[0], None 

def process_file_pair(psg_filepath, hypno_filepath):
    """

    Args:
        psg_filepath (str): Path to the PSG file.
        hypno_filepath (str): Path to the Hypnogram file.

    Returns:
        tuple: (bool, str) indicating (success_status, message_code)
    """
    psg_filename = os.path.basename(psg_filepath)
    hypno_filename = os.path.basename(hypno_filepath)
    try:
        # Minimal action: Log successful pairing.
        logging.debug(f"Processing pair: PSG='{psg_filename}', Hypnogram='{hypno_filename}'")

        if not os.path.exists(psg_filepath):
             return False, f"PSG file disappeared: {psg_filename}"
        if not os.path.exists(hypno_filepath):
             return False, f"Hypno file disappeared: {hypno_filename}"

        # Simulate success as pairing and basic checks passed
        time.sleep(0.01) # Tiny sleep to simulate work
        success = True
        message_code = "PROCESSED_OK" # Indicates pairing/basic checks OK

    except Exception as e:
        # Catch potential errors from the replaceable section
        logging.error(f"Error during processing step for {psg_filename}: {e}", exc_info=True)
        success = False
        message_code = f"PROCESSING_ERROR: {e}"

    return success, message_code

def worker_task(psg_filepath):
    """
    Task executed by each worker process. Finds hypnogram and processes the pair.

    Args:
        psg_filepath (str): The full path to the PSG file.

    Returns:
        tuple: (str, str, str) -> (psg_filename, status_code, details_message)
    """
    psg_filename = os.path.basename(psg_filepath)
    logging.info(f"--- Starting pipeline for: {psg_filename} ---")

    hypno_filepath, find_error_msg = find_hypnogram(psg_filepath)

    if hypno_filepath is None:
        logging.warning(f"{find_error_msg} for {psg_filename}")
        return (psg_filename, "FAILED_HYPNO", find_error_msg) 

    # If hypnogram found, proceed to processing
    success, message_code = process_file_pair(psg_filepath, hypno_filepath)

    if success:
        # Log success at INFO level maybe, or just rely on final summary
        logging.debug(f"Processing step successful for {psg_filename} with code {message_code}")
        return (psg_filename, message_code, f"Successfully processed {psg_filename}")
    else:
        # Log failure at WARNING/ERROR level
        logging.warning(f"Processing step failed for {psg_filename} with code {message_code}")
        return (psg_filename, message_code, f"Failed during processing step for {psg_filename}")

def main():
    start_time = time.time()
    logging.info(f"Base Data Directory: {BASE_DATA_DIR}")
    logging.info(f"Using up to {MAX_WORKERS} workers.")

    cassette_path = os.path.join(BASE_DATA_DIR, CASSETTE_DIR_NAME)
    telemetry_path = os.path.join(BASE_DATA_DIR, TELEMETRY_DIR_NAME)

    # Find all PSG files
    try:
        psg_files_cassette = glob.glob(os.path.join(cassette_path, "SC*PSG.edf"))
        psg_files_telemetry = glob.glob(os.path.join(telemetry_path, "ST*PSG.edf"))
        all_psg_files = sorted(psg_files_cassette + psg_files_telemetry)
        total_files = len(all_psg_files)
    except Exception as e:
        logging.error(f"Fatal error finding PSG files: {e}", exc_info=True)
        return

    if total_files == 0:
        logging.warning("No PSG files found in the specified directories. Check paths:")
        logging.warning(f" - Cassette: {cassette_path}")
        logging.warning(f" - Telemetry: {telemetry_path}")
        logging.warning("Exiting.")
        return

    logging.info(f"Found {total_files} PSG files to process.")

    results = []
    processed_count = 0
    # Using specific success code from process_file_pair
    success_code = "PROCESSED_OK"
    success_count = 0
    failure_count = 0
    # Store failure details {status_code: [(filename, details_message), ...]}
    failed_files_summary = {}

    # Using ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks
        futures = {executor.submit(worker_task, psg_file): psg_file for psg_file in all_psg_files}

        for future in as_completed(futures):
            psg_filepath_orig = futures[future] # Get original path for error logging
            processed_count += 1
            try:
                # Result is (psg_filename, status_code, details_message)
                result = future.result()
                results.append(result)
                psg_filename, status_code, details_message = result

                if status_code == success_code:
                    success_count += 1
                    # Keep console less verbose on success, log file has details via worker_task
                    # logging.info(f"Job COMPLETED [{processed_count}/{total_files}] - SUCCESS ({status_code}): {psg_filename}")
                else:
                    failure_count += 1
                    logging.warning(f"Job COMPLETED [{processed_count}/{total_files}] - FAILED ({status_code}): {psg_filename}")
                    if status_code not in failed_files_summary:
                        failed_files_summary[status_code] = []
                    # Store filename and the specific error detail message
                    failed_files_summary[status_code].append((psg_filename, details_message))

            except Exception as exc:
                # Catch exceptions *from the future itself* (e.g., worker process crash)
                psg_filename = os.path.basename(psg_filepath_orig)
                failure_count += 1
                status_code = "EXCEPTION_IN_FUTURE"
                logging.error(f"Job COMPLETED [{processed_count}/{total_files}] - FAILED ({status_code}): Task for {psg_filename} generated an exception: {exc}", exc_info=True)
                results.append((psg_filename, status_code, str(exc)))
                if status_code not in failed_files_summary:
                     failed_files_summary[status_code] = []
                failed_files_summary[status_code].append((psg_filename, str(exc)))

    end_time = time.time()
    total_time = end_time - start_time

    logging.info("Preprocessing Script Finished")
    logging.info(f"Total files attempted: {total_files}")
    logging.info(f"   Successfully processed (basic checks OK): {success_count}")
    logging.info(f"   Failed (hypno find, processing error, etc.): {failure_count}")
    logging.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    if failure_count > 0:
        logging.warning(" ")
        logging.warning(f"Failures occurred ({failure_count}). Check log file ('{LOG_FILENAME}') for detailed errors and tracebacks.")
        logging.warning("Failed file summaries (status: count):")
        # Log summary counts first
        for status, files_details in failed_files_summary.items():
             logging.warning(f"   - {status}: {len(files_details)} file(s)")

        # Log details for each failure category
        max_details_to_print = 10 # Limit detailed printing per category
        for status, files_details in failed_files_summary.items():
            logging.warning(f"   --- Details for status '{status}' ---")
            for i, (filename, detail_msg) in enumerate(files_details):
                 if i < max_details_to_print:
                     # Log the specific detail message associated with the failure
                     logging.warning(f"     - {filename}: {detail_msg}")
                 elif i == max_details_to_print:
                     logging.warning(f"     - ... (further details for {status} logged in file)")
                     break # Stop printing details for this status

    logging.info("                  End of Preprocessing Run")


if __name__ == "__main__":
    main()