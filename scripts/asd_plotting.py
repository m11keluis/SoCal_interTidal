#!/usr/bin/env python3
"""
Plot Corrected Spectra Script
Visualizes the illumination-corrected ASD spectral data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


def load_spectral_data(data_dir, data_type='digital_numbers'):
    """
    Load raw and corrected spectral data

    Parameters:
    - data_dir: Directory containing the CSV files
    - data_type: Type of data to load ('digital_numbers', 'radiance', 'reflectance')

    Returns:
    - raw_df: Raw spectra DataFrame
    - corrected_df: Corrected spectra DataFrame
    - baseline_df: Baseline spectrum DataFrame
    - metadata_df: Metadata DataFrame
    """

    # Load files
    raw_file = os.path.join(data_dir, f"raw_{data_type}_matrix.csv")
    corrected_file = os.path.join(data_dir, f"corrected_{data_type}_matrix.csv")
    baseline_file = os.path.join(data_dir, f"baseline_{data_type}_spectrum.csv")
    metadata_file = os.path.join(data_dir, "metadata.csv")

    # Check if files exist
    files_to_check = [
        (raw_file, "raw spectra"),
        (corrected_file, "corrected spectra"),
        (metadata_file, "metadata")
    ]

    for file_path, file_type in files_to_check:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find {file_type} file: {file_path}")

    # Load data
    raw_df = pd.read_csv(raw_file, index_col=0)
    corrected_df = pd.read_csv(corrected_file, index_col=0)
    metadata_df = pd.read_csv(metadata_file)

    baseline_df = None
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)

    print(f"Loaded {data_type} data:")
    print(f"  Raw spectra: {raw_df.shape[1]} samples, {raw_df.shape[0]} wavelengths")
    print(f"  Corrected spectra: {corrected_df.shape[1]} samples, {corrected_df.shape[0]} wavelengths")
    print(f"  Wavelength range: {raw_df.index.min():.1f} - {raw_df.index.max():.1f} nm")

    return raw_df, corrected_df, baseline_df, metadata_df


def plot_raw_vs_corrected_comparison(raw_df, corrected_df, sample_names=None, n_samples=5):
    """
    Plot comparison between raw and corrected spectra for selected samples
    """
    if sample_names is None:
        # Select a few random samples (excluding baseline files)
        non_baseline_samples = [col for col in raw_df.columns if '_baseline' not in col.lower()]
        sample_names = np.random.choice(non_baseline_samples,
                                        min(n_samples, len(non_baseline_samples)),
                                        replace=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot raw spectra
    for sample in sample_names:
        if sample in raw_df.columns:
            ax1.plot(raw_df.index, raw_df[sample], alpha=0.7, label=sample[:20] + '...' if len(sample) > 20 else sample)

    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Raw Digital Numbers')
    ax1.set_title('Raw Spectra')
    ax1.set_xlim(400, 800)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot corrected spectra
    for sample in sample_names:
        if sample in corrected_df.columns:
            ax2.plot(corrected_df.index, corrected_df[sample], alpha=0.7,
                     label=sample[:20] + '...' if len(sample) > 20 else sample)

    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Corrected Ratio (Sample/Baseline)')
    ax2.set_title('Illumination-Corrected Spectra')
    ax2.set_xlim(400, 800)

    # Set y-axis limits based on data percentiles to show variability
    y_data = []
    for sample in sample_names:
        if sample in corrected_df.columns:
            y_data.extend(corrected_df[sample].loc[400:800].values)

    if y_data:
        y_min = np.percentile(y_data, 1)  # 1st percentile
        y_max = np.percentile(y_data, 99)  # 99th percentile
        y_range = y_max - y_min
        ax2.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_baseline_spectrum(baseline_df, raw_df):
    """
    Plot the baseline spectrum used for correction
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if baseline_df is not None:
        ax.plot(baseline_df['Wavelength (nm)'], baseline_df['Baseline_DN'],
                'k-', linewidth=2, label=f"Baseline: {baseline_df['Baseline_File'].iloc[0]}")
    else:
        print("No baseline data available to plot")
        return None

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Digital Numbers')
    ax.set_title('Baseline Spectrum Used for Illumination Correction')
    ax.set_xlim(400, 800)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_all_corrected_spectra(corrected_df, max_samples=50, alpha=0.3):
    """
    Plot all corrected spectra in one plot (with transparency for many samples)
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # If too many samples, plot a subset
    if corrected_df.shape[1] > max_samples:
        sample_indices = np.linspace(0, corrected_df.shape[1] - 1, max_samples, dtype=int)
        samples_to_plot = corrected_df.columns[sample_indices]
        title_suffix = f" (showing {max_samples} of {corrected_df.shape[1]} samples)"
    else:
        samples_to_plot = corrected_df.columns
        title_suffix = f" (all {corrected_df.shape[1]} samples)"

    # Exclude baseline files from the plot
    non_baseline_samples = [col for col in samples_to_plot if '_baseline' not in col.lower()]

    for sample in non_baseline_samples:
        ax.plot(corrected_df.index, corrected_df[sample], alpha=alpha, linewidth=1)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Corrected Ratio (Sample/Baseline)')
    ax.set_title(f'All Illumination-Corrected Spectra{title_suffix}')
    ax.set_xlim(400, 800)

    # Set y-axis limits based on data in the visible range
    visible_data = corrected_df.loc[400:800, non_baseline_samples]
    y_min = np.percentile(visible_data.values, 1)  # 1st percentile
    y_max = np.percentile(visible_data.values, 99)  # 99th percentile
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    ax.grid(True, alpha=0.3)

    # Add some statistics
    mean_spectrum = corrected_df[non_baseline_samples].mean(axis=1)
    std_spectrum = corrected_df[non_baseline_samples].std(axis=1)

    ax.plot(corrected_df.index, mean_spectrum, 'r-', linewidth=2, label='Mean spectrum')
    ax.fill_between(corrected_df.index,
                    mean_spectrum - std_spectrum,
                    mean_spectrum + std_spectrum,
                    alpha=0.2, color='red', label='±1 std')

    ax.legend()
    plt.tight_layout()
    return fig


def plot_spectral_statistics(corrected_df):
    """
    Plot statistical summary of corrected spectra
    """
    # Exclude baseline files
    non_baseline_samples = [col for col in corrected_df.columns if '_baseline' not in col.lower()]
    data = corrected_df[non_baseline_samples]

    # Filter data to visible range for better y-axis scaling
    visible_data = data.loc[400:800]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Mean spectrum
    mean_spectrum = visible_data.mean(axis=1)
    ax1.plot(visible_data.index, mean_spectrum, 'b-', linewidth=2)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Mean Corrected Ratio')
    ax1.set_title('Mean Corrected Spectrum')
    ax1.set_xlim(400, 800)
    ax1.grid(True, alpha=0.3)

    # Standard deviation
    std_spectrum = visible_data.std(axis=1)
    ax2.plot(visible_data.index, std_spectrum, 'r-', linewidth=2)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Standard Deviation')
    ax2.set_title('Spectral Variability (Std Dev)')
    ax2.set_xlim(400, 800)
    ax2.grid(True, alpha=0.3)

    # Coefficient of variation
    cv_spectrum = std_spectrum / mean_spectrum
    ax3.plot(visible_data.index, cv_spectrum, 'g-', linewidth=2)
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Coefficient of Variation')
    ax3.set_title('Relative Variability (CV)')
    ax3.set_xlim(400, 800)
    ax3.grid(True, alpha=0.3)

    # Histogram of values at a specific wavelength (e.g., 550 nm)
    target_wavelength = 550
    closest_wavelength = visible_data.index[np.argmin(np.abs(visible_data.index - target_wavelength))]
    values_at_wl = visible_data.loc[closest_wavelength]

    ax4.hist(values_at_wl, bins=20, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Corrected Ratio')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Distribution at {closest_wavelength:.1f} nm')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_summary_report(raw_df, corrected_df, baseline_df, metadata_df, output_dir):
    """
    Create a comprehensive summary report with multiple plots
    """
    print("Creating summary plots...")

    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Raw vs Corrected comparison
    print("  - Raw vs Corrected comparison")
    fig1 = plot_raw_vs_corrected_comparison(raw_df, corrected_df)
    fig1.savefig(os.path.join(plots_dir, "01_raw_vs_corrected_comparison.png"),
                 dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 2. Baseline spectrum
    print("  - Baseline spectrum")
    fig2 = plot_baseline_spectrum(baseline_df, raw_df)
    if fig2:
        fig2.savefig(os.path.join(plots_dir, "02_baseline_spectrum.png"),
                     dpi=300, bbox_inches='tight')
        plt.close(fig2)

    # 3. All corrected spectra
    print("  - All corrected spectra")
    fig3 = plot_all_corrected_spectra(corrected_df)
    fig3.savefig(os.path.join(plots_dir, "03_all_corrected_spectra.png"),
                 dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # 4. Statistical summary
    print("  - Statistical summary")
    fig4 = plot_spectral_statistics(corrected_df)
    fig4.savefig(os.path.join(plots_dir, "04_spectral_statistics.png"),
                 dpi=300, bbox_inches='tight')
    plt.close(fig4)

    print(f"All plots saved to: {plots_dir}")


# === Main execution ===
if __name__ == '__main__':
    # Configuration
    data_dir = "/Users/kluis/PycharmProjects/SoCal_interTidal/data/output/ASD"  # <- Replace this
    data_type = "digital_numbers"  # Change to 'radiance' or 'reflectance' if needed

    try:
        # Load data
        print("Loading spectral data...")
        raw_df, corrected_df, baseline_df, metadata_df = load_spectral_data(data_dir, data_type)

        # Create comprehensive summary report
        create_summary_report(raw_df, corrected_df, baseline_df, metadata_df, data_dir)

        # Interactive plotting (optional - uncomment to show plots interactively)
        # print("\nCreating interactive plots...")
        #
        # # Show a few individual plots
        fig1 = plot_raw_vs_corrected_comparison(raw_df, corrected_df, n_samples=3)
        plt.show()
        #
        fig2 = plot_baseline_spectrum(baseline_df, raw_df)
        if fig2:
            plt.show()
        #
        fig3 = plot_all_corrected_spectra(corrected_df, max_samples=30)
        plt.show()

        print("\nPlotting complete!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you've run the ASD processing script first to generate the data files.")
    except Exception as e:
        print(f"Unexpected error: {e}")