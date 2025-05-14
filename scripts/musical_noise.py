import numpy as np
from scripts.signal_processing import analysis_STFT, synthesis_STFT

def remove_musical_noise(signal, nfft, hop_size, threshold_db, max_event_len):
    """
    Detect and suppress musical noise artifacts based on amplitude and temporal duration.

    This function operates on the STFT magnitude of the signal to detect short-duration,
    low-energy spectral events (commonly perceived as "musical noise") and attenuates them.

    Parameters:
    -----------
    signal : ndarray
        Input time-domain signal.
    nfft : int
        FFT size used for STFT analysis and synthesis.
    hop_size : int
        Hop size for STFT.
    threshold_db : float
        Amplitude threshold in decibels. Events below this threshold are considered noise.
    max_event_len : int
        Maximum duration (in STFT frames) for a low-energy event to be considered musical noise.

    Returns:
    --------
    y : ndarray
        Output signal with musical noise attenuated.
    """

    # STFT analysis
    X = analysis_STFT(signal, nfft, hop_size)
    magnitude = np.abs(X)
    phase = np.angle(X)

    # Convert dB threshold to linear scale
    threshold_linear = 10 ** (threshold_db / 20)

    # Create binary mask of low-amplitude regions
    low_energy_mask = magnitude < threshold_linear

    # Iterate over frequency bins
    for freq_bin in range(low_energy_mask.shape[0]):

        # Detect start and end of low-energy events
        diff_mask = np.diff(np.concatenate(([0], low_energy_mask[freq_bin, :].astype(int), [0])))
        event_starts = np.where(diff_mask == 1)[0]
        event_ends = np.where(diff_mask == -1)[0]
        event_lengths = event_ends - event_starts

        # Suppress short low-energy events (musical noise)
        for start, length in zip(event_starts, event_lengths):
            if 0 < length <= max_event_len:
                magnitude[freq_bin, start:start + length] *= 0

    # Reconstruct signal from modified magnitude and original phase
    cleaned_signal = synthesis_STFT(magnitude * np.exp(1j * phase), nfft, hop_size)

    return cleaned_signal
