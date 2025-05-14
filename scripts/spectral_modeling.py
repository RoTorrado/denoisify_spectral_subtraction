from smstools.models import sineModel as SM
import numpy as np
from scipy import signal
import librosa

def sinusoidal_modeling(signal, sample_rate, fft_size, 
                        threshold, min_sine_duration, max_num_sines, 
                        freq_dev_offset, freq_dev_slope):
    """
    Apply sinusoidal modeling to a signal using analysis and synthesis stages.

    This function decomposes the input signal into a sum of sinusoids and a residual component
    by first performing peak tracking on the STFT and then resynthesizing the sinusoidal part.

    Parameters:
    -----------
    signal : ndarray
        Input time-domain signal.
    sample_rate : int
        Sampling rate of the signal.
    fft_size : int
        FFT size used for analysis and synthesis.
    threshold : float
        Magnitude threshold in dB for peak detection.
    min_sine_duration : float
        Minimum duration for a sinusoid to be retained.
    max_num_sines : int
        Maximum number of simultaneous sinusoids.
    freq_dev_offset : float
        Offset threshold for frequency deviation in peak continuation.
    freq_dev_slope : float
        Slope threshold for frequency deviation in peak continuation.

    Returns:
    --------
    sinusoidal_component : ndarray
        Reconstructed sinusoidal part of the signal.
    residual : ndarray
        Residual component after subtracting the sinusoidal model.
    """
    hop_size = fft_size // 4
    window = np.hamming(fft_size)

    # Analysis: extract sinusoidal tracks
    frequencies, magnitudes, phases = SM.sineModelAnal(
        signal, sample_rate, w=window, N=fft_size, H=hop_size, t=threshold,
        minSineDur=min_sine_duration, maxnSines=max_num_sines,
        freqDevOffset=freq_dev_offset, freqDevSlope=freq_dev_slope
    )

    # Synthesis: generate sinusoidal signal
    sinusoidal_component = SM.sineModelSynth(frequencies, magnitudes, phases, fft_size, hop_size, sample_rate)

    # Compute residual
    residual = signal[:len(sinusoidal_component)] - sinusoidal_component

    return sinusoidal_component, residual

def convert_l_sec_to_frames(L_h_sec, Fs=22050, N=1024, H=512):
    """Convert filter length parameter from seconds to frame indices

    Args:
        L_h_sec (float): Filter length (in seconds)
        Fs (scalar): Sample rate (Default value = 22050)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 512)

    Returns:
        L_h (int): Filter length (in samples)
    """
    L_h = int(np.ceil(L_h_sec * Fs / H))
    return L_h

def convert_l_hertz_to_bins(L_p_Hz, Fs=22050, N=1024, H=512):
    """Convert filter length parameter from Hertz to frequency bins

    Args:
        L_p_Hz (float): Filter length (in Hertz)
        Fs (scalar): Sample rate (Default value = 22050)
        N (int): Window size (Default value = 1024)
        H (int): Hop size (Default value = 512)

    Returns:
        L_p (int): Filter length (in frequency bins)
    """
    L_p = int(np.ceil(L_p_Hz * N / Fs))
    return L_p

def make_integer_odd(n):
    """Convert integer into odd integer

    Args:
        n (int): Integer

    Returns:
        n (int): Odd integer
    """
    if n % 2 == 0:
        n += 1
    return n

def transient_detection(x, Fs, N, H, L_h, L_p, L_unit='physical', mask='binary', eps=0.001, detail=False):
    """Harmonic-percussive separation (HPS) algorithm

    Args:
        x (np.ndarray): Input signal
        Fs (scalar): Sampling rate of x
        N (int): Frame length
        H (int): Hopsize
        L_h (float): Horizontal median filter length given in seconds or frames
        L_p (float): Percussive median filter length given in Hertz or bins
        L_unit (str): Adjusts unit, either 'pyhsical' or 'indices' (Default value = 'physical')
        mask (str): Either 'binary' or 'soft' (Default value = 'binary')
        eps (float): Parameter used in soft maskig (Default value = 0.001)
        detail (bool): Returns detailed information (Default value = False)

    Returns:
        x_h (np.ndarray): Harmonic signal
        x_p (np.ndarray): Percussive signal
        details (dict): Dictionary containing detailed information; returned if ``detail=True``
    """
    assert L_unit in ['physical', 'indices']
    assert mask in ['binary', 'soft']
    # stft
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N, window='hann', center=True, pad_mode='constant')
    # power spectrogram
    Y = np.abs(X) ** 2
    # median filtering
    if L_unit == 'physical':
        L_h = convert_l_sec_to_frames(L_h_sec=L_h, Fs=Fs, N=N, H=H)
        L_p = convert_l_hertz_to_bins(L_p_Hz=L_p, Fs=Fs, N=N, H=H)
    L_h = make_integer_odd(L_h)
    L_p = make_integer_odd(L_p)
    Y_h = signal.medfilt(Y, [1, L_h])
    Y_p = signal.medfilt(Y, [L_p, 1])

    # masking
    if mask == 'binary':
        M_h = np.int8(Y_h >= Y_p)
        M_p = np.int8(Y_h < Y_p)
    if mask == 'soft':
        eps = 0.00001
        M_h = (Y_h + eps / 2) / (Y_h + Y_p + eps)
        M_p = (Y_p + eps / 2) / (Y_h + Y_p + eps)
    X_h = X * M_h
    X_p = X * M_p

    # istft
    x_h = librosa.istft(X_h, hop_length=H, win_length=N, window='hann', center=True, length=x.size)
    x_p = librosa.istft(X_p, hop_length=H, win_length=N, window='hann', center=True, length=x.size)

    if detail:
        return x_h, x_p, dict(Y_h=Y_h, Y_p=Y_p, M_h=M_h, M_p=M_p, X_h=X_h, X_p=X_p)
    else:
        return x_h, x_p
