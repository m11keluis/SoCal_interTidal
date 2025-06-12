import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def read_sed_file(filepath, use_column=2):
    metadata = {}
    wavelengths = []
    radiances = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Extract metadata
    for line in lines:
        if line.startswith('Latitude:'):
            val = line.split(':')[1].strip()
            metadata['latitude'] = float(val) if val.lower() != 'n/a' else None
        elif line.startswith('Longitude:'):
            val = line.split(':')[1].strip()
            metadata['longitude'] = float(val) if val.lower() != 'n/a' else None
        elif line.startswith('GPS Time:'):
            metadata['gps_time'] = line.split(':', 1)[1].strip()
        elif line.strip().startswith('Data:'):
            data_start_idx = lines.index(line) + 2
            break
    else:
        raise ValueError(f"No 'Data:' section found in {filepath}")

    for line in lines[data_start_idx:]:
        if line.strip():
            parts = line.strip().split()
            if len(parts) > use_column:
                try:
                    wavelengths.append(float(parts[0]))
                    radiances.append(float(parts[use_column]))
                except ValueError:
                    continue

    return metadata, wavelengths, radiances

def process_sed_directory(directory, output_dir=None, use_column=2):
    sed_files = [f for f in os.listdir(directory) if f.lower().endswith('.sed')]
    if not sed_files:
        raise ValueError(f"No .sed files found in {directory}.")

    spectra_dict = {}
    wavelengths_master = None
    meta_records = []

    for filename in sed_files:
        filepath = os.path.join(directory, filename)
        metadata, wavelengths, radiances = read_sed_file(filepath, use_column=use_column)
        basename = os.path.splitext(filename)[0]

        if wavelengths_master is None:
            wavelengths_master = wavelengths
        elif wavelengths != wavelengths_master:
            raise ValueError(f"Wavelength mismatch in file: {filename}")

        spectra_dict[basename] = radiances

        meta_records.append({
            'filename': basename,
            'latitude': metadata.get('latitude'),
            'longitude': metadata.get('longitude'),
            'gps_time': metadata.get('gps_time')
        })

    df = pd.DataFrame(spectra_dict, index=wavelengths_master)
    df.index.name = 'wavelength'
    meta_df = pd.DataFrame(meta_records)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, 'spectra_data.csv'))
        meta_df.to_csv(os.path.join(output_dir, 'spectra_metadata.csv'), index=False)

    return df, meta_df

def extract_features(spectra_df):
    features = []
    for col in spectra_df.columns:
        spectrum = spectra_df[col]
        features.append([
            np.mean(spectrum),
            np.max(spectrum),
            spectrum.idxmax(),
            spectrum.iloc[-1] / spectrum.iloc[0] if spectrum.iloc[0] != 0 else 0,
            np.std(spectrum) / np.mean(spectrum) if np.mean(spectrum) != 0 else 0
        ])
    feature_names = ['mean', 'max', 'peak_wavelength', 'slope', 'flatness']
    return pd.DataFrame(features, index=spectra_df.columns, columns=feature_names)

def cluster_spectra(features_df, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features_df)
    features_df['cluster'] = labels
    return features_df

def plot_clusters(spectra_df, cluster_labels, output_dir=None):
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_clusters))
    fig, ax = plt.subplots(figsize=(12, 6))

    for idx, col in enumerate(spectra_df.columns):
        cluster_id = cluster_labels[idx]
        ax.plot(spectra_df.index, spectra_df[col], color=colors(cluster_id), alpha=0.5)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Radiance (W/m²/sr/nm)')
    ax.set_title('Spectra colored by cluster')
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'clustered_spectra.png'), dpi=300)
    plt.close()

def plot_cluster_means(spectra_df, cluster_labels, output_dir=None):
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_clusters))
    fig, ax = plt.subplots(figsize=(12, 6))

    for cluster_id in unique_clusters:
        spectra_in_cluster = spectra_df.iloc[:, np.where(cluster_labels == cluster_id)[0]]
        mean_spectrum = spectra_in_cluster.mean(axis=1)
        ax.plot(spectra_df.index, mean_spectrum, label=f'Cluster {cluster_id}', color=colors(cluster_id))

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Mean Radiance (W/m²/sr/nm)')
    ax.set_title('Cluster Mean Spectra')
    ax.legend()
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'cluster_means.png'), dpi=300)
    plt.close()

############################################
# Process Sed:
directory = "/Users/kluis/PycharmProjects/SoCal_interTidal/data/sedDataLCDM/" #change to your directory
output_dir = '/Users/kluis/PycharmProjects/SoCal_interTidal/data/output/'
spectra_df, metadata_df = process_sed_directory(directory, output_dir=output_dir)
