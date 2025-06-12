# === ASD Directory Processor ===
# Complete script to parse ASD spectral data and process all .asd files in a directory tree.
# Includes extraction of reflectance, radiance, and white reference spectra.

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
        self.wavelengths = np.arange(self.md.ch1_wave, self.md.ch1_wave + self.md.channels * self.md.wave1_step, self.md.wave1_step)
        self.spec, offset = parse_spectra(self.asd, 484, self.md.channels)
        _, offset = parse_reference(self.asd, offset)
        self.reference, offset = parse_spectra(self.asd, offset, self.md.channels)

    def get_reflectance(self):
        if spectra_type[self.md.data_type] == 'REF':
            return normalise_spectrum(self.spec, self.md) / normalise_spectrum(self.reference, self.md)
        raise TypeError(f'File contains {spectra_type[self.md.data_type]}; expected REF.')

    def get_radiance(self):
        if spectra_type[self.md.data_type] == 'RAD':
            return self.reference * self.spec * self.md.intergration_time / (500 * 544 * np.pi)
        raise TypeError(f'File contains {spectra_type[self.md.data_type]}; expected RAD.')

    def get_white_reference(self):
        return normalise_spectrum(self.reference, self.md)

# === Directory Scanner ===
def scan_directory_for_asd_files(root_dir, include_dirs=None):
    asd_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if include_dirs and not any(d in dirpath for d in include_dirs):
            continue
        for file in filenames:
            if file.lower().endswith('.asd'):
                asd_files.append(os.path.join(dirpath, file))
    return asd_files

def process_asd_directory(root_dir, include_dirs=None):
    asd_files = scan_directory_for_asd_files(root_dir, include_dirs)
    print(f"Found {len(asd_files)} ASD files.")
    meta_records = []
    spectra_matrix = {}
    for path in asd_files:
        try:
            r = reader(path)
            reflectance = r.get_reflectance() if spectra_type[r.md.data_type] == 'REF' else None
            record = {
                'filename': os.path.basename(path),
                'full_path': path,
                'comment': r.md.comment.decode('utf-8', errors='ignore'),
                'save_time': r.md.save_time,
                'integration_time': r.md.intergration_time,
            }
            meta_records.append(record)
            if reflectance is not None:
                spectra_matrix[record['filename']] = reflectance
        except Exception as e:
            print(f"Error processing {path}: {e}")
    meta_df = pd.DataFrame(meta_records)
    if spectra_matrix:
        spectra_df = pd.DataFrame(spectra_matrix, index=r.wavelengths)
        spectra_df.index.name = 'Wavelength (nm)'
    else:
        spectra_df = pd.DataFrame()
    return meta_df, spectra_df

# === Change Down Here ==========
if __name__ == '__main__':
    root_dir = '/Users/kluis/PycharmProjects/SoCal_interTidal/data/ASD'  # <- Replace this
    include_dirs = ['WR', 'freshwater', 'nowater', 'saltwater']
    metadata_df, reflectance_df = process_asd_directory(root_dir, include_dirs)
    metadata_df.to_csv("/Users/kluis/PycharmProjects/SoCal_interTidal/data/output/ASD/metadata.csv", index=False)  # <- Replace this
    reflectance_df.to_csv("/Users/kluis/PycharmProjects/SoCal_interTidal/data/output/ASD/reflectance_matrix.csv")  # <- Replace this

