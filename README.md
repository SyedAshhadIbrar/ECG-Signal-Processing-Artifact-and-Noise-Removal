# ECG Signal Processing: Artifact and Noise Removal

This project involves the removal of artifacts and noise from three ECG (Electrocardiography) signals, each with a duration of 30 minutes. The processed signals are further analyzed to estimate the mean heart rate every 0.5 seconds for a two-channel ECG signal.

## Table of Contents
- [Overview](#overview)
- [Assignment Requirements](#assignment-requirements)
- [Methodology](#methodology)
- [Software Used](#software-used)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Overview
This project focuses on preprocessing ECG signals by removing common types of noise and artifacts such as powerline interference, baseline wander, EMG noise, and motion artifacts. The final signal is analyzed to compute the mean heart rate.

## Objectives
1. **Artifact and Noise Removal**: To process ECG signals to eliminate various noise components using a wavelet-based approach.
2. **Mean Heart Rate Calculation**: To estimate and report the mean heart rate every 0.5 seconds for a two-channel ECG signal.

## Methodology
The **Wavelet Method** is used for noise and artifact removal due to its effectiveness in decomposing non-stationary signals such as ECG. Key steps include:
- **Notch Filtering**: Remove 50 Hz powerline interference.
- **Wavelet Decomposition**: Break down signals into multiple levels of detail and approximation coefficients.
- **Thresholding and Reconstruction**: Suppress noise using hard and soft thresholding, followed by signal reconstruction.

### Wavelet Transform
This approach uses the Symlet 8 (sym8) wavelet to separate noise from the signal effectively, removing baseline drift and high-frequency noise while preserving essential ECG characteristics.

## Usage
Run the main script (provided as ECG.py files depending on your platform).

View the plots and results generated, which include filtered signals and calculated heart rates.

Modify parameters (e.g., sampling frequency, wavelet type) in the script as necessary.
