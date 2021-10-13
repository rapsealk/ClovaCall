import os

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


def main():
    path = os.path.join(os.path.dirname(__file__), 'src', 'datasets', 'aihub_dataset', 'dummy_data', 'train_data', '001', 'hobby_00000002.wav')

    tensor = tfio.audio.AudioIOTensor(path, dtype=tf.int64)
    sample_rate = tensor.rate.numpy()

    tensor = tf.cast(tensor.to_tensor(), dtype=tf.float32) / 32768  # 16-bit dequatization
    signal = tf.squeeze(tensor, axis=-1).numpy()

    # Preemphasis
    pre_emphasis_alpha = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis_alpha * signal[:-1])

    # Framing
    frame_size = 0.025  # 25ms
    frame_stride = 0.01 # 10ms
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # Hamming Window
    frames *= np.array([0.54 - 0.46 * np.cos((2 * np.pi * n) / (frame_length - 1))
                        for n in range(frame_length)])

    # Fourier Transform
    nfft = 512
    dft_frames = np.fft.rfft(frames, n=nfft)    # (num_frames * nfft / 2 + 1,)

    # Magnitude
    mag_frames = np.absolute(dft_frames)

    # Power Spectrum
    pow_frames = (1.0 / nfft) * (mag_frames ** 2)

    # Filter Banks
    # - Mel Scale Filter
    nfilter = 40
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilter + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1)) # Convert Mel to Hz
    bin_ = np.floor((nfft + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilter, int(np.floor(nfft / 2 + 1))))

    for m in range(1, nfilter+1):
        f_m_minus = int(bin_[m-1])  # left
        f_m = int(bin_[m])          # center
        f_m_plus = int(bin_[m+1])   # right
        for k in range(f_m_minus, f_m):
            fbank[m-1, k] = (k - bin_[m-1]) / (bin_[m] - bin_[m-1])
        for k in range(f_m, f_m_plus):
            fbank[m-1, k] = (bin_[m+1] - k) / (bin_[m+1] - bin_[m])

    # - Filter Banks
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)   # Numerical Stability

    # Log-Mel Spectrum
    filter_banks = 20 * np.log10(filter_banks)  # dB

    # MFFCs
    from scipy.fftpack import dct
    num_ceps = 12
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps+1)]   # Keep 2-13

    """
    Post Processing
    """
    # Lift
    (nframes, ncoeff) = mfcc.shape
    cep_lifter = 22
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc * lift

    # Mean Normalization
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)


if __name__ == "__main__":
    main()
