import argparse
import string

import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds

import datasets.aihub_dataset
from las.tensorflow_impl.models import Listener, Speller, Sequence2Sequence

# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, char_level=True, oov_token=None)


def preprocess_dataset(audio, text):
    global tokenizer

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

    text = tokenizer.texts_to_sequences(text)
    text = tf.convert_to_tensor(text, dtype=tf.float32)
    text = tf.squeeze(text)

    return audio, text


def tf_preprocess_dataset(ds):
    ds['audio'], ds['text'] = tf.py_function(preprocess_dataset, (ds['audio'], ds['text']), [tf.float32, tf.float32])
    return ds


def main():
    global tokenizer

    # Dataset
    ds = tfds.load('aihub_dataset')

    transcripts = ds['train'].map(lambda x: x['text'])
    transcripts = [t.numpy().decode('utf-8') for t in transcripts]
    tokenizer.fit_on_texts(transcripts)

    print(f'[Transcripts] {transcripts[:3]}')

    tokens = tokenizer.texts_to_sequences(transcripts[:3])
    print(f'[Tokens] {tokens}')

    def postprocess_text(text):
        return ''.join(list(map(lambda x: x[1], filter(lambda i: i[0] % 2 == 0, enumerate(text)))))

    texts = tokenizer.sequences_to_texts(tokens)
    print(f'[Texts] {list(map(lambda x: postprocess_text(x), texts))}')
    return

    # Model
    encoder = Listener()
    decoder = Speller(output_shape=len(string.ascii_lowercase))


if __name__ == "__main__":
    main()
