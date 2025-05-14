import numpy as np
from scripts.signal_processing import short_time_energy, short_time_zero_crossing_rate, analysis_STFT
from math import ceil

def noise_profile_detection(
        x, fs, nfft,  
        th_energy, th_zcr, th_hf,
        zcr_pct_hfcut,
        min_silence_len, min_sound_len, 
        start_silence_margin, end_silence_margin, 
        init_noise_frames
):
    """
    Estimates the noise profile of an audio signal using temporal and spectral features.

    This algorithm assumes the last `init_noise_frames` frames contain noise, and uses them to estimate
    thresholds for energy, zero-crossing rate (ZCR), and high-frequency content. Silence segments are detected 
    based on these thresholds, and the noise profile is computed by averaging spectral components of these segments.

    Parameters
    ----------
    x : np.ndarray
        Input time-domain audio signal.

    fs : int
        Sampling frequency of the signal.

    nfft : int
        Frame size used for short-time analysis (STFT, energy, ZCR).

    th_energy : float
        Scaling factor for energy-based silence threshold.

    th_zcr : float
        Scaling factor for ZCR-based silence threshold.

    th_hf : float
        Scaling factor for high-frequency content threshold.

    zcr_pct_hfcut : float
        Fraction (0–1) of the ZCR used to define the high-frequency cut-off.

    min_silence_len : int
        Minimum number of frames to consider a segment as silence.

    min_sound_len : int
        Minimum number of frames to separate two silence segments.

    start_silence_margin : int
        Number of frames removed from the start of each detected silence to avoid transients.

    end_silence_margin : int
        Number of frames removed from the end of each detected silence to avoid transients.

    init_noise_frames : int
        Number of ending frames assumed to be pure noise.

    Returns
    -------
    silence_mask : np.ndarray
        Binary array where 1 indicates a frame detected as silence.

    noise_profile : np.ndarray
        Averaged spectrum representing the estimated noise profile.

    ste : np.ndarray
        Short-time energy for each frame.

    zcr : np.ndarray
        Short-time zero-crossing rate for each frame.

    hf_energy : np.ndarray
        Mean high-frequency magnitude for each frame.
    """

    hop = nfft // 4
    n_frames = int(ceil((len(x) - nfft) / hop)) + 1
    
    stft_matrix = analysis_STFT(x, nfft, hop)                            
    stft_mag = np.abs(stft_matrix)

    # Feature extraction
    ste = short_time_energy(x, nfft, hop)
    zcr = short_time_zero_crossing_rate(x, nfft, hop)

    energy_noise = np.mean(ste[-init_noise_frames:])
    zcr_noise = np.mean(zcr[-init_noise_frames:])

    hf_cut = (1 - zcr_pct_hfcut) * zcr_noise * fs / 2
    hf_bin = int(hf_cut * nfft / fs)
    hf_energy = np.mean(stft_mag[hf_bin:nfft // 2, :], axis=0)
    hf_noise = np.mean(hf_energy[-init_noise_frames:])

    # Thresholds
    energy_thresh = energy_noise * (1 + th_energy)
    zcr_thresh = zcr_noise * (1 - th_zcr)
    hf_thresh = hf_noise * (1 + th_hf)

    # Output initialization
    noise_profile = np.sum(stft_mag[:, -init_noise_frames:], axis=1)
    silence_mask = np.zeros(n_frames)
    silence_mask[-init_noise_frames:] = 1
    noise_frames_count = init_noise_frames

    silence_start = None
    last_silence_end = None

    for i in range(n_frames - init_noise_frames):
        if (ste[i] < energy_thresh) and (hf_energy[i] < hf_thresh) and (zcr[i] > zcr_thresh):
            if silence_start is None:
                silence_start = i
        else:
            if silence_start is not None:
                duration = i - silence_start
                if duration > min_silence_len:
                    orig_start = silence_start
                    if silence_start != 0:
                        silence_start += start_silence_margin
                    silence_end = i - end_silence_margin

                    if (last_silence_end is not None) and (orig_start - last_silence_end < min_sound_len):
                        noise_profile += np.sum(stft_mag[:, last_silence_end:silence_end], axis=1)
                        silence_mask[last_silence_end:silence_end] = 1
                        noise_frames_count += silence_end - last_silence_end
                    else:
                        noise_profile += np.sum(stft_mag[:, silence_start:silence_end], axis=1)
                        silence_mask[silence_start:silence_end] = 1
                        noise_frames_count += silence_end - silence_start

                    last_silence_end = i - 1
                silence_start = None

    # Handle last segment
    if silence_start is not None:
        duration = n_frames - init_noise_frames - 1 - silence_start
        if duration > min_silence_len:
            orig_start = silence_start
            if silence_start != 0:
                silence_start += start_silence_margin
            silence_end = n_frames - init_noise_frames - 1

            if (last_silence_end is not None) and (orig_start - last_silence_end < min_sound_len):
                noise_profile += np.sum(stft_mag[:, last_silence_end:silence_end], axis=1)
                silence_mask[last_silence_end:silence_end] = 1
                noise_frames_count += silence_end - last_silence_end
            else:
                noise_profile += np.sum(stft_mag[:, silence_start:silence_end], axis=1)
                silence_mask[silence_start:silence_end] = 1
                noise_frames_count += silence_end - silence_start

    noise_profile /= noise_frames_count

    return silence_mask, noise_profile, ste, zcr, hf_energy