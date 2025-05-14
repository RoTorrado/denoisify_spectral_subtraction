import numpy as np
from scipy import signal

def analysis_STFT(signal_in, fft_size, hop_size):
    """
    Compute the Short-Time Fourier Transform (STFT) of an input signal.

    Parameters:
    -----------
    signal_in : ndarray
        Input time-domain signal.
    fft_size : int
        Length of the analysis window (FFT size).
    hop_size : int
        Hop size (stride) between successive frames.

    Returns:
    --------
    stft_matrix : ndarray (complex)
        STFT matrix where each column is the FFT of a windowed segment.
    """
    signal_length = signal_in.size
    num_frames = (signal_length - fft_size) // hop_size + 1
    window = signal.windows.hann(fft_size)
    stft_matrix = np.zeros((fft_size, num_frames), dtype=complex)

    for i in range(num_frames):
        start = i * hop_size
        segment = signal_in[start:start + fft_size] * window
        stft_matrix[:, i] = np.fft.fft(segment, fft_size)

    return stft_matrix

def synthesis_STFT(stft_matrix, fft_size, hop_size):
    """
    Reconstruct a time-domain signal from its Short-Time Fourier Transform (STFT).

    Parameters:
    -----------
    stft_matrix : ndarray (complex)
        STFT matrix where each column is a spectrum of a time segment.
    fft_size : int
        FFT size used during analysis.
    hop_size : int
        Hop size used during analysis.

    Returns:
    --------
    signal_out : ndarray
        Time-domain reconstructed signal.
    """
    num_frames = stft_matrix.shape[1]
    window = signal.windows.hann(fft_size)
    signal_out = np.zeros((num_frames - 1) * hop_size + fft_size)

    for i in range(num_frames):
        segment = np.fft.ifft(stft_matrix[:, i], fft_size).real
        start = i * hop_size
        signal_out[start:start + fft_size] += segment 

    # Normalize by constant overlap-add factor
    normalization_factor = np.sum(window) / hop_size
    signal_out /= normalization_factor

    return signal_out

def short_time_energy(signal_in, fft_size, hop_size):
    """
    Compute the short-time energy of a signal.

    Parameters:
    -----------
    signal_in : ndarray
        Input signal in the time domain.
    fft_size : int
        Window size in samples.
    hop_size : int
        Hop size between successive windows.

    Returns:
    --------
    energy : ndarray
        Short-time energy vector.
    """
    signal_in = np.asarray(signal_in, dtype=float)
    num_frames = (len(signal_in) - fft_size) // hop_size + 1
    energy = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_size
        window = signal_in[start:start + fft_size]
        energy[i] = np.sum(window ** 2)

    return energy

def short_time_zero_crossing_rate(signal_in, fft_size, hop_size):
    """
    Compute the short-time zero-crossing rate of a signal.

    Parameters:
    -----------
    signal_in : ndarray
        Input signal in the time domain.
    fft_size : int
        Window size in samples.
    hop_size : int
        Hop size between successive windows.

    Returns:
    --------
    zcr : ndarray
        Zero-crossing rate for each frame.
    """
    signal_in = np.asarray(signal_in, dtype=float)
    num_frames = (len(signal_in) - fft_size) // hop_size + 1
    zcr = np.zeros(num_frames)

    for i in range(num_frames):
        start = i * hop_size
        window = signal_in[start:start + fft_size]
        zero_crossings = np.sum(np.abs(np.diff(np.sign(window)))) / 2
        zcr[i] = zero_crossings / fft_size

    return zcr

def low_pass_filter_stft(magnitude_spectrogram, smoothing_factor):
    """
    Apply a low-pass filter to the magnitude spectrogram across time.

    Parameters:
    -----------
    magnitude_spectrogram : ndarray
        Magnitude of the STFT.
    smoothing_factor : float
        Smoothing factor (rho). Higher values yield smoother results.

    Returns:
    --------
    filtered_spectrogram : ndarray
        Smoothed magnitude spectrogram.
    """
    filtered = np.empty_like(magnitude_spectrogram)
    filtered[:, 0] = magnitude_spectrogram[:, 0]

    for i in range(1, magnitude_spectrogram.shape[1]):
        filtered[:, i] = (smoothing_factor * filtered[:, i - 1] +
                          (1 - smoothing_factor) * magnitude_spectrogram[:, i])

    return filtered

def spectral_subtraction(stft_matrix, alpha, beta, rho, frame_mask):
    """
    Perform spectral subtraction noise reduction.

    Parameters:
    -----------
    stft_matrix : ndarray (complex)
        Input STFT matrix.
    alpha : float
        Spectral subtraction scaling factor.
    beta : float
        psp lower bound scaling factor.
    rho : float
        Low-pass filter smoothing factor.
    frame_mask : ndarray (binary)
        Binary vector indicating which frames are noise (1 = noise).

    Returns:
    --------
    denoised_stft : ndarray (complex)
        STFT after spectral subtraction and PSP.
    """
    noise_profile = np.mean(np.abs(stft_matrix[:, frame_mask[:stft_matrix.shape[1]] == 1]), axis=1)
    
    original_phase = np.angle(stft_matrix)
    
    smoothed_magnitude = low_pass_filter_stft(np.abs(stft_matrix), rho)
    
    spectral_subtracted = smoothed_magnitude - alpha * noise_profile[:, None]
    
    psp_applied = np.maximum(spectral_subtracted, beta*smoothed_magnitude)
    
    return psp_applied * np.exp(1j * original_phase)