import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
import pywt
import matplotlib.pyplot as plt

# Read the ECG signal
f_s, ecg = wav.read('119e00m.wav')
N = len(ecg)
ti = np.arange(0, N) / f_s  # time period

# IIR Notch filter to remove powerline (50Hz)
w = 50 / (f_s / 2)
bw = w * 0.05  # Adjust bandwidth for stability
b, a = signal.iirnotch(w, bw)
sos = signal.tf2sos(b, a)  # Convert to second-order sections for stability
ecg_notch = signal.sosfiltfilt(sos, ecg, axis=0)  # Apply filter

# Plot ECG after Notch filter
plt.figure()
plt.plot(ecg[:, 0], label="Initial Signal of Channel 1")
plt.plot(ecg_notch[:, 0], 'r', label="Signal of Channel 1 after IIR Notch Filter")
plt.title("Channel 1")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Wavelet decomposition and filtering
final_signals = []
for i in range(2):
    # Select the appropriate channel
    j = ecg_notch[:, i]
    
    # Wavelet decomposition of the signal
    coeffs = pywt.wavedec(j, 'sym8', level=10)
    
    # Baseline wandering removal
    coeffs[-1] = np.zeros_like(coeffs[-1])  # Remove approx coefficients for baseline
    
    # EMG noise reduction for cd4 and cd3
    def hard_thresholding(coeff, low_thresh, up_thresh):
        sorted_coeff = np.sort(coeff)
        ecdf_values = np.cumsum(sorted_coeff / np.sum(sorted_coeff))
        coeff[(coeff > sorted_coeff[low_thresh]) & (coeff < sorted_coeff[up_thresh])] = 0
        return coeff

    coeffs[4] = hard_thresholding(coeffs[4], int(len(coeffs[4]) * 0.15), int(len(coeffs[4]) * 0.85))
    coeffs[3] = hard_thresholding(coeffs[3], int(len(coeffs[3]) * 0.10), int(len(coeffs[3]) * 0.90))
    
    # Motion artifact reduction with soft thresholding for cd9 and cd8
    def soft_thresholding(coeff):
        threshold = np.median(np.abs(coeff)) / 0.6457 * np.sqrt(2 * np.log(len(coeff)))
        return pywt.threshold(coeff, threshold, 'soft')

    coeffs[9] = soft_thresholding(coeffs[9])
    coeffs[8] = soft_thresholding(coeffs[8])
    
    # High frequencies elimination (cd1, cd2)
    coeffs[1] = np.zeros_like(coeffs[1])
    coeffs[2] = np.zeros_like(coeffs[2])

    # Signal reconstruction
    signal_rec = pywt.waverec(coeffs, 'sym8')
    final_signals.append(signal_rec)

# Concatenate final signals
final_signal = np.column_stack(final_signals)

# Plot baseline wander removal
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(final_signal)
plt.title("Baseline Wander Removal")

plt.subplot(2, 1, 2)
plt.plot(final_signal[:2000])
plt.title("Samples 1:2000")
plt.legend(["Samples 1:2000"])
plt.show()

# Plot final and initial signal
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(ecg[:2000, :])
plt.title("Raw ECG Data")
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(final_signal[:2000, :])
plt.title("Final Signal")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# Heart peak identification & mean heart rate calculation
from scipy.signal import find_peaks

def Rpeakfinder(signal, min_dist, min_height):
    locs, _ = find_peaks(signal, distance=min_dist, height=min_height)
    pks = signal[locs]
    return locs, pks

locs, pks = Rpeakfinder(final_signal[:, 0], 180, 0.11)

plt.figure()
plt.plot(final_signal[:, 0])
plt.plot(locs, pks, 'ro')
plt.title("Detected R-peaks")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()

# Mean heart rate per 30 seconds
mean_heart_rate = 30 * f_s * len(locs) / len(final_signal)
print(f'The mean heart rate per 30 seconds is {mean_heart_rate:.2f}')
