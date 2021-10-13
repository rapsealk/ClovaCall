"""aihub_dataset dataset."""
import os
import json
from pathlib import Path

import requests
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd

# TODO(aihub_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(aihub_dataset): BibTeX citation
_CITATION = """
"""

with open(os.path.join(os.path.dirname(__file__), '..', 'config.json'), 'r') as f:
    config = json.loads(''.join(f.readlines()))
    dataset_storage = config['storage']     # https://drive.google.com/u/0/uc?export=download&confirm=jWDd&id=1uBVr5NciHj59ZxmexYMopae7iH1ddSAO


class AihubDataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    MAX_SOUND_LENGTH = 160000

    def _info(self) -> tfds.core.DatasetInfo:
        # max_sound_length = 160000
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'audio': tfds.features.Audio(shape=(AihubDataset.MAX_SOUND_LENGTH,), file_format='wav', dtype=tf.int64, sample_rate=16000),
                'text': tfds.features.Text()    # ByteTextEncoder
            }),
            supervised_keys=('audio', 'text'),
            homepage='https://aihub.or.kr/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        try:
            path = dl_manager.download_and_extract(f'{dataset_storage}/dacon.tar.gz')
        except requests.exceptions.ConnectionError:
            path = Path(os.path.join(os.path.dirname(__file__), 'dummy_data'))
        print(f'Download path: {path} ({type(path)})')

        train_df = pd.read_csv(path / 'train.csv')
        train_df.file_name = [os.path.join(path, 'train_data', f'{name}.wav')
                              for name in train_df.file_name]

        test_path = path / 'test_data'
        test_files = [os.path.join(test_path, x) for x in tf.io.gfile.listdir(test_path)]
        test_df = pd.DataFrame({'file_name': test_files, 'text': [''] * len(test_files)})

        return {
            'train': self._generate_examples(train_df.values),
            'test': self._generate_examples(test_df.values)
        }

    def _generate_examples(self, df):
        """
        `tfds.features.Audio` accepts:
            - `np.ndarray` of shape `(length,)` or `(length, channels)`
            - a path to a `.mp3`, `.wav`, ... file
            - a file-object (e.g. `with path.open('rb') as fobj:`)
        """
        for file_name, text in df:
            data = tfio.audio.AudioIOTensor(file_name, dtype=tf.int64)
            data = tf.squeeze(data.to_tensor(), axis=-1)
            data = data.numpy()

            if data.shape[0] <= AihubDataset.MAX_SOUND_LENGTH:
                audio = np.zeros(AihubDataset.MAX_SOUND_LENGTH)
                audio[:data.shape[0]] = data
            else:
                audio = data[:AihubDataset.MAX_SOUND_LENGTH]

            yield os.path.split(file_name)[1], {
                'audio': audio,
                'text': text
            }


"""
def listdir(path, recursive=False, fmt=None):
    routes = [os.path.abspath(path)]
    while routes:
        path = routes.pop(0)
        files = tuple(map(lambda x: os.path.join(path, x), os.listdir(path)))
        routes.extend(filter(lambda x: os.path.isdir(x), files))
        files = filter(lambda x: os.path.isfile(x), files)
        if fmt:
            files = filter(lambda x: os.path.splitext(x)[1] == f'.{fmt}', files)
        for file in files:
            yield file
"""


if __name__ == "__main__":
    pass
