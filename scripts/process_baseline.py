# === ASD Directory Processor - Radiance/Digital Number Focus ===
# Complete script to parse ASD spectral data and process all .asd files in a directory tree.
# Adapted to primarily handle radiance and digital number (raw) data.

import os
import struct
import datetime
from collections import namedtuple
import numpy as np
import pandas as pd

# === Constants ===
spectra_type = ('RAW', 'REF', 'RAD', 'NOUNITS', 'IRRAD', 'QI', 'TRANS', 'UNKNOWN', 'ABS')
data_type = ('FLOAT', 'INTEGER', 'DOUBLE', 'UNKNOWN')
instrument_type = ('UNKNOWN', 'PSII', 'LSVNIR', 'FSVNIR', 'FSFR', 'FSNIR', 'CHEM', 'FSFR_UNATTENDED',)
calibration_type = ('ABSOLUTE', 'BASE', 'LAMP', 'FIBER')


# === Parsing functions ===
def parse_bstr(asd, offset):
    size = struct.unpack_from('<h', asd, offset)[0]
    offset += 2
    bstr = struct.unpack_from(f'<{size}s', asd, offset)[0]
    offset += size
    return bstr, offset


def parse_time(timestring):
    s = struct.unpack_from('9h', timestring)
    return datetime.datetime(1900 + s[5], month=s[4], day=s[3], hour=s[2], minute=s[1], second=s[0])


def parse_metadata(asd):
    asdformat = '<3s 157s 18s b b b b l b l f f b b b b b H 128s 56s L hh H H f f f f h b 4b H H H b L HHHH f f f 5b'
    asd_file_info = namedtuple('metadata', [
        'file_version', 'comment', 'save_time', 'parent_version', 'format_version', 'itime',
        'dc_corrected', 'dc_time', 'data_type', 'ref_time', 'ch1_wave', 'wave1_step',
        'data_format', 'old_dc_count', 'old_ref_count', 'old_sample_count', 'application',
        'channels', 'app_data', 'gps_data', 'intergration_time', 'fo', 'dcc', 'calibration',
        'instrument_num', 'ymin', 'ymax', 'xmin', 'xmax', 'ip_numbits', 'xmode',
        'flags1', 'flags2', 'flags3', 'flags4', 'dc_count', 'ref_count', 'sample_count',
        'instrument', 'cal_bulb_id', 'swir1_gain', 'swir2_gain', 'swir1_offset', 'swir2_offset',
        'splice1_wavelength', 'splice2_wavelength', 'smart_detector_type',
        'spare1', 'spare2', 'spare3', 'spare4', 'spare5'])
    unpacked = struct.unpack_from(asdformat, asd)
    comment = unpacked[1].strip(b'\x00')
    save_time = parse_time(unpacked[2])
    dc_time = datetime.datetime.fromtimestamp(unpacked[7])
    ref_time = datetime.datetime.fromtimestamp(unpacked[9])
    unpacked = list(unpacked)
    unpacked[1] = comment
    unpacked[2] = save_time
    unpacked[7] = dc_time
    unpacked[9] = ref_time
    return asd_file_info._make(unpacked), 484


def parse_spectra(asd, offset, channels):
    spec = np.array(struct.unpack_from(f'<{channels}d', asd, offset))
    offset += channels * 8
    return spec, offset


def parse_reference(asd, offset):
    offset += 18
    description, offset = parse_bstr(asd, offset)
    return (None, None, None, description), offset


def normalise_spectrum(spec, metadata):
    res = spec.copy()
    splice1_index = int(metadata.splice1_wavelength)
    splice2_index = int(metadata.splice2_wavelength)
    res[:splice1_index] = spec[:splice1_index] / metadata.intergration_time
    res[splice1_index:splice2_index] = spec[splice1_index:splice2_index] * metadata.swir1_gain / 2048
    res[splice2_index:] = spec[splice2_index:] * metadata.swir1_gain / 2048
    return res


# === Reader class ===
class reader:
    def __init__(self, filename):
        with open(filename, 'rb') as fh:
            self.asd = fh.read()
        self.md, offset = parse_metadata(self.asd)
        self.wavelengths = np.arange(self.md.ch1_wave, self.md.ch1_wave + self.md.channels * self.md.wave1_step,
                                     self.md.wave1_step)
        self.spec, offset = parse_spectra(self.asd, 484, self.md.channels)
        _, offset = parse_reference(self.asd, offset)
        self.reference, offset = parse_spectra(self.asd, offset, self.md.channels)

    def get_digital_numbers(self):
        """Get raw digital numbers (DN) - works for RAW data type"""
        if spectra_type[self.md.data_type] == 'RAW':
            return self.spec
        raise TypeError(f'File contains {spectra_type[self.md.data_type]}; expected RAW for digital numbers.')

    def get_radiance(self):
        """Get radiance values - works for RAD data type"""
        if spectra_type[self.md.data_type] == 'RAD':
            return self.reference * self.spec * self.md.intergration_time / (500 * 544 * np.pi)
        raise TypeError(f'File contains {spectra_type[self.md.data_type]}; expected RAD for radiance.')

    def get_reflectance(self):
        """Get reflectance values - works for REF data type"""
        if spectra_type[self.md.data_type] == 'REF':
            return normalise_spectrum(self.spec, self.md) / normalise_spectrum(self.reference, self.md)
        raise TypeError(f'File contains {spectra_type[self.md.data_type]}; expected REF for reflectance.')

    def get_white_reference(self):
        """Get white reference spectrum"""
        return normalise_spectrum(self.reference, self.md)

    def get_spectrum_data(self):
        """Automatically get the appropriate spectrum data based on file type"""
        spec_type = spectra_type[self.md.data_type]

        if spec_type == 'RAW':
            return self.get_digital_numbers(), 'digital_numbers'
        elif spec_type == 'RAD':
            return self.get_radiance(), 'radiance'
        elif spec_type == 'REF':
            return self.get_reflectance(), 'reflectance'
        else:
            # For other types, return raw spectrum data
            return self.spec, spec_type.lower()


# === Directory Scanner ===
def scan_directory_for_asd_files(root_dir):
    asd_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith('.asd'):
                asd_files.append(os.path.join(dirpath, file))
    return asd_files


def apply_illumination_correction(spectra_df, baseline_filename, min_threshold=0.01):
    """
    Apply illumination correction to spectra using a baseline spectrum

    Parameters:
    - spectra_df: DataFrame with spectra (wavelengths as index, filenames as columns)
    - baseline_filename: Name of the file to use as illumination baseline
    - min_threshold: Minimum value threshold to avoid division by very small numbers

    Returns:
    - corrected_df: DataFrame with corrected spectra
    - baseline_spectrum: The baseline spectrum used for correction
    """
    if baseline_filename not in spectra_df.columns:
        raise ValueError(f"Baseline file '{baseline_filename}' not found in spectra data")

    # Get the baseline spectrum
    baseline_spectrum = spectra_df[baseline_filename].copy()

    # Apply threshold to avoid division by very small numbers
    baseline_spectrum = np.where(baseline_spectrum < min_threshold, min_threshold, baseline_spectrum)

    # Create corrected spectra by dividing each spectrum by the baseline
    corrected_df = spectra_df.div(baseline_spectrum, axis=0)

    return corrected_df, baseline_spectrum


def find_potential_baselines(meta_df, baseline_pattern='_baseline'):
    """
    Find potential baseline files based on filename pattern

    Parameters:
    - meta_df: Metadata DataFrame
    - baseline_pattern: Pattern to search for in filenames (default: '_baseline')

    Returns:
    - List of potential baseline filenames
    """
    potential_baselines = []

    for _, row in meta_df.iterrows():
        filename_lower = row['filename'].lower()

        # Check if baseline pattern appears in filename
        if baseline_pattern.lower() in filename_lower:
            potential_baselines.append(row['filename'])

    return potential_baselines


def process_asd_directory(root_dir, prefer_data_type='auto', baseline_filename=None,
                          apply_correction=True, auto_detect_baseline=True):
    """
    Process ASD files in directory tree with illumination correction

    Parameters:
    - root_dir: Root directory to scan
    - prefer_data_type: 'auto', 'radiance', 'digital_numbers', or 'reflectance'
    - baseline_filename: Specific filename to use as illumination baseline (None for auto-detection)
    - apply_correction: Whether to apply illumination correction
    - auto_detect_baseline: Whether to automatically detect potential baseline files
    """
    asd_files = scan_directory_for_asd_files(root_dir)
    print(f"Found {len(asd_files)} ASD files.")

    meta_records = []
    spectra_matrices = {
        'digital_numbers': {},
        'radiance': {},
        'reflectance': {},
        'other': {}
    }

    # Track what data types we encounter
    data_type_counts = {}

    for path in asd_files:
        try:
            r = reader(path)
            spec_type = spectra_type[r.md.data_type]

            # Count data types
            data_type_counts[spec_type] = data_type_counts.get(spec_type, 0) + 1

            # Create metadata record
            record = {
                'filename': os.path.basename(path),
                'full_path': path,
                'comment': r.md.comment.decode('utf-8', errors='ignore'),
                'save_time': r.md.save_time,
                'integration_time': r.md.intergration_time,
                'data_type': spec_type,
                'instrument_type': instrument_type[r.md.instrument],
                'channels': r.md.channels,
                'wavelength_start': r.md.ch1_wave,
                'wavelength_step': r.md.wave1_step,
            }
            meta_records.append(record)

            # Get spectrum data based on preference or auto-detect
            try:
                if prefer_data_type == 'auto':
                    spectrum_data, data_label = r.get_spectrum_data()
                elif prefer_data_type == 'radiance' and spec_type == 'RAD':
                    spectrum_data = r.get_radiance()
                    data_label = 'radiance'
                elif prefer_data_type == 'digital_numbers' and spec_type == 'RAW':
                    spectrum_data = r.get_digital_numbers()
                    data_label = 'digital_numbers'
                elif prefer_data_type == 'reflectance' and spec_type == 'REF':
                    spectrum_data = r.get_reflectance()
                    data_label = 'reflectance'
                else:
                    # Fallback to raw spectrum
                    spectrum_data = r.spec
                    data_label = 'other'

                # Store in appropriate matrix
                if data_label in spectra_matrices:
                    spectra_matrices[data_label][record['filename']] = spectrum_data
                else:
                    spectra_matrices['other'][record['filename']] = spectrum_data

            except TypeError as e:
                print(f"Warning: Could not extract preferred data from {path}: {e}")
                # Store raw spectrum as fallback
                spectra_matrices['other'][record['filename']] = r.spec

        except Exception as e:
            print(f"Error processing {path}: {e}")

    # Print data type summary
    print("\nData type summary:")
    for dtype, count in data_type_counts.items():
        print(f"  {dtype}: {count} files")

    # Create metadata DataFrame
    meta_df = pd.DataFrame(meta_records)

    # Create spectrum DataFrames for each data type
    spectrum_dfs = {}
    corrected_spectrum_dfs = {}
    wavelengths = None
    baseline_info = {}

    for data_label, matrix in spectra_matrices.items():
        if matrix:  # Only create DataFrame if we have data
            if wavelengths is None:
                wavelengths = r.wavelengths  # Use wavelengths from last processed file

            df = pd.DataFrame(matrix, index=wavelengths)
            df.index.name = 'Wavelength (nm)'
            spectrum_dfs[data_label] = df
            print(f"Created {data_label} matrix with {len(matrix)} spectra")

            # Apply illumination correction if requested
            if apply_correction and len(df.columns) > 1:  # Need at least 2 files (sample + baseline)
                try:
                    # Determine baseline filename
                    current_baseline = baseline_filename

                    if current_baseline is None and auto_detect_baseline:
                        potential_baselines = find_potential_baselines(meta_df, baseline_pattern='_baseline')
                        if potential_baselines:
                            # Use the first potential baseline found
                            current_baseline = potential_baselines[0]
                            print(f"Auto-detected potential baseline for {data_label}: {current_baseline}")
                            if len(potential_baselines) > 1:
                                print(f"  Note: Found {len(potential_baselines)} baseline files, using first one")
                                print(f"  Other baselines found: {potential_baselines[1:]}")
                        else:
                            print(f"No baseline files with '_baseline' pattern found for {data_label}")
                            print(f"Available files (first 10):")
                            for i, filename in enumerate(df.columns[:10]):  # Show first 10
                                print(f"  {i}: {filename}")
                            if len(df.columns) > 10:
                                print(f"  ... and {len(df.columns) - 10} more files")
                            continue

                    if current_baseline and current_baseline in df.columns:
                        corrected_df, baseline_spectrum = apply_illumination_correction(df, current_baseline)
                        corrected_spectrum_dfs[data_label] = corrected_df
                        baseline_info[data_label] = {
                            'filename': current_baseline,
                            'spectrum': baseline_spectrum
                        }
                        print(f"Applied illumination correction to {data_label} using baseline: {current_baseline}")
                    else:
                        print(f"Baseline file '{current_baseline}' not found in {data_label} data")

                except Exception as e:
                    print(f"Error applying illumination correction to {data_label}: {e}")

    return meta_df, spectrum_dfs, corrected_spectrum_dfs, baseline_info


# === Change Down Here ==========
if __name__ == '__main__':
    root_dir = '/Users/kluis/PycharmProjects/SoCal_interTidal/data/data_250718'  # <- Replace this

    # Configuration options
    baseline_file = None  # Set to specific filename like 'white_tile_001.asd' or None for auto-detection

    # Process directory with illumination correction
    metadata_df, raw_spectrum_dfs, corrected_spectrum_dfs, baseline_info = process_asd_directory(
        root_dir,
        prefer_data_type='auto',
        baseline_filename=baseline_file,
        apply_correction=True,
        auto_detect_baseline=True
    )

    # Save results
    output_dir = "/Users/kluis/PycharmProjects/SoCal_interTidal/data/output/ASD"  # <- Replace this
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    # Save raw spectra
    print("\nSaving raw spectra...")
    for data_type, df in raw_spectrum_dfs.items():
        filename = f"raw_{data_type}_matrix.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath)
        print(f"Saved raw {data_type} data to {filename}")

    # Save corrected spectra
    if corrected_spectrum_dfs:
        print("\nSaving illumination-corrected spectra...")
        for data_type, df in corrected_spectrum_dfs.items():
            filename = f"corrected_{data_type}_matrix.csv"
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath)
            print(f"Saved corrected {data_type} data to {filename}")

            # Save baseline info
            if data_type in baseline_info:
                baseline_filename = f"baseline_{data_type}_spectrum.csv"
                baseline_filepath = os.path.join(output_dir, baseline_filename)
                baseline_spectrum = baseline_info[data_type]['spectrum']

                # Get wavelengths from the corresponding raw spectrum DataFrame
                wavelengths = raw_spectrum_dfs[data_type].index

                baseline_df = pd.DataFrame({
                    'Wavelength (nm)': wavelengths,
                    'Baseline_DN': baseline_spectrum.values if hasattr(baseline_spectrum,
                                                                       'values') else baseline_spectrum,
                    'Baseline_File': baseline_info[data_type]['filename']
                })
                baseline_df.to_csv(baseline_filepath, index=False)
                print(f"Saved baseline spectrum to {baseline_filename}")
    else:
        print("\nNo corrected spectra generated (no baseline found or correction disabled)")

    # Summary
    print(f"\nProcessing complete!")
    print(f"Files saved to: {output_dir}")
    if baseline_info:
        print(f"Baselines used:")
        for data_type, info in baseline_info.items():
            print(f"  {data_type}: {info['filename']}")

    # Show potential baseline files for manual selection if needed
    if not baseline_file:
        potential_baselines = find_potential_baselines(metadata_df, baseline_pattern='_baseline')
        if potential_baselines:
            print(f"\nBaseline files detected:")
            for i, filename in enumerate(potential_baselines):
                print(f"  {i + 1}: {filename}")
            if len(potential_baselines) > 1:
                print("Note: Script automatically uses the first baseline found.")
                print("To manually specify a different baseline, set baseline_file = 'your_filename.asd' in the script")
        else:
            print(f"\nNo baseline files with '_baseline' pattern found.")
            print("Available files (first 10):")
            for i, filename in enumerate(metadata_df['filename'].head(10)):
                print(f"  {filename}")
            print("To specify a baseline, set baseline_file = 'your_filename.asd' in the script")