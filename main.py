import argparse
import os
import json

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds

import datasets.aihub_dataset   # noqa: F401
from las.tensorflow_impl.models import Listener, Speller, Sequence2Sequence

parser = argparse.ArgumentParser()
parser.add_argument('-lr', '--learning-rate', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--max-length', type=int, default=40)
args = parser.parse_args()

NUM_WORDS = 128
MAX_SENTENCE_LENGTH = args.max_length

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=NUM_WORDS, char_level=True, oov_token='@')


def postprocess_tokens(tokens):
    for i, token in enumerate(tokens):
        tokens[i] = np.append(token[:MAX_SENTENCE_LENGTH], [0] * (MAX_SENTENCE_LENGTH - len(token)))
        tokens[i] = np.asarray(tokens[i], dtype=np.uint8) - 1
    return tokens


def _generate_spectrogram(audio: np.ndarray):
    audio = tf.cast(audio, dtype=tf.float32) / 32768.0
    # Trim the noise
    #start, stop = tfio.audio.trim(audio, axis=0, epsilon=0.1)
    #audio = audio[start:stop]
    # Fade In and Fade Out
    #audio = tfio.audio.fade(audio, fade_in=1000, fade_out=2000, mode='logarithmic')
    # Spectrogram
    audio = tfio.audio.spectrogram(audio, nfft=512, window=512, stride=256)
    audio = tfio.audio.melscale(audio, rate=16000, mels=128, fmin=0, fmax=8000)
    audio = tf.expand_dims(audio, axis=2)   # Conv2D: (height, width, channel)
    audio = tf.concat([audio, audio, audio], axis=-1)
    return audio


def _generate_log_mel_filter_bank_spectrum(audio: np.ndarray, sample_rate=16000):
    tensor = tf.cast(audio, dtype=tf.float32) / 32768  # 16-bit dequatization
    signal = tf.squeeze(tensor, axis=None).numpy()

    # Preemphasis
    pre_emphasis_alpha = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis_alpha * signal[:-1])

    # Framing
    frame_size = 0.025      # 25ms
    frame_stride = 0.01     # 10ms
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

    # Mean Normalization
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)

    return filter_banks


def preprocess_dataset(audio, text):
    global tokenizer

    # audio = _generate_spectrogram(audio)
    audio = _generate_log_mel_filter_bank_spectrum(audio)

    text = tokenizer.texts_to_sequences(text.numpy().decode('utf-8'))
    text = np.asarray(text).squeeze()
    text = postprocess_tokens([text])
    text = tf.convert_to_tensor(text, dtype=tf.float32)
    text = tf.squeeze(text)

    return audio, text


def tf_preprocess_dataset(ds):
    ds['audio'], ds['text'] = tf.py_function(preprocess_dataset, (ds['audio'], ds['text']), [tf.float32, tf.float32])
    return ds


def main():
    global tokenizer
    global NUM_WORDS
    global MAX_SENTENCE_LENGTH

    # Dataset
    ds = tfds.load('aihub_dataset')

    # max_length = args.max_length
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    transcripts = ds['train'].map(lambda x: x['text'])
    transcripts = [t.numpy().decode('utf-8') for t in transcripts]
    tokenizer.fit_on_texts(['$', '#'])  # $: <sos>, #: <eos>
    tokenizer.fit_on_texts(transcripts)

    word_counts = json.loads(tokenizer.get_config()['word_counts'])
    # print(word_counts)
    MAX_SENTENCE_LENGTH = max(tuple(map(lambda x: len(x), transcripts)))
    NUM_WORDS = len(word_counts)

    print(f'MAX_SENTENCE_LENGTH: {MAX_SENTENCE_LENGTH}')
    print(f'NUM_WORDS: {NUM_WORDS}')

    """
    print(f'[Transcripts] {transcripts[:3]}')

    tokens = tokenizer.texts_to_sequences(transcripts[:3])
    tokens = postprocess_tokens(tokens)
    print(f'[Tokens] {tokens}')

    def postprocess_text(text):
        return ''.join(list(map(lambda x: x[1], filter(lambda i: i[0] % 2 == 0, enumerate(text)))))

    texts = tokenizer.sequences_to_texts(tokens)
    print(f'[Texts] {list(map(lambda x: postprocess_text(x), texts))}')
    """

    ds_train = ds['train'].map(tf_preprocess_dataset)

    def set_shapes(data, **kwargs):
        for key, value in kwargs.items():
            data[key].set_shape(value)
        return data

    for item in ds_train.take(1):
        audio_shape = item['audio'].shape
        text_shape = item['text'].shape

    ds_train = ds_train.map(lambda x: set_shapes(x, audio=audio_shape, text=text_shape))
    ds_train = ds_train.map(lambda x: (x['audio'], x['text']))

    oov_token_index = json.loads(tokenizer.get_config()['word_index'])[tokenizer.get_config()['oov_token']]
    print('oov_token_index:', oov_token_index)

    # Model
    encoder = Listener(input_shape=(batch_size,)+audio_shape)
    decoder = Speller(output_shape=NUM_WORDS)
    model = Sequence2Sequence(encoder, decoder)

    checkpoint_filepath = os.path.join(os.path.dirname(__file__), 'checkpoint')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='accuracy',
        mode='max',
        save_best_only=False)

    # ValueError: Weights for model sequential have not yet been created. Weights are created when the Model is first called on inputs or `build()` is called with an `input_shape`.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy', 'sparse_categorical_crossentropy'])

    with tf.device('/GPU:0'):
        # 338/2028 [====>.........................] - ETA: 1:24:15 - loss: nan - mse: 23260.0879
        hist = model.fit(ds_train.batch(batch_size, drop_remainder=True), epochs=args.epochs, verbose=1,
                         callbacks=[checkpoint_callback])

    print(hist.history)

    """
    with tf.device('/CPU:0'):
        output = model.predict(ds_train.take(batch_size))
        print(output)
        print(output.shape)
    """


if __name__ == "__main__":
    main()
