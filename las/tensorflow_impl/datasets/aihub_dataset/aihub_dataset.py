"""aihub_dataset dataset."""
import os
import zipfile

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio
import pandas as pd
from sklearn.model_selection import train_test_split

# TODO(aihub_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(aihub_dataset): BibTeX citation
_CITATION = """
"""


class AihubAudioDataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.'
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'audio': tfds.features.Audio(file_format='wav', shape=(None,), dtype=tf.float32, sample_rate=16000),    # 32768
                'transcript': tfds.features.Text(encoder=None)
            }),
            supervised_keys=('audio', 'transcript'),
            homepage='https://aihub.or.kr/',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = dl_manager.download_and_extract('https://drive.google.com/u/0/uc?export=download&confirm=jWDd&id=1uBVr5NciHj59ZxmexYMopae7iH1ddSAO')
        if not tf.io.gfile.isdir(path) and os.path.splitext(path) == '.zip':
            path = unzip(path, remove_after=True, recursive=True)

        csv = pd.read_csv(os.path.join(path, 'train.csv'))
        csv.file_name = [os.path.join(path, 'train_data', f'{name}.wav') for name in csv.file_name]

        train_data, test_data = train_test_split(csv.values, test_size=0.2)

        # files = tuple(listdir(os.path.join(path, 'train_data'), recursive=True, fmt='wav'))
        # train_files, test_files = train_test_split(files, test_size=0.2)
        return {
            'train': self._generate_examples(train_data),
            'test': self._generate_examples(test_data)
        }

    def _generate_examples(self, data):
        """
        `tfds.features.Audio` accepts:
            - `np.ndarray` of shape `(length,)` or `(length, channels)`
            - a path to a `.mp3`, `.wav`, ... file
            - a file-object (e.g. `with path.open('rb') as fobj:`)
        """
        for file_name, transcript in data:
            audio = tfio.audio.AudioIOTensor(file_name, dtype=tf.int8)
            audio = tf.squeeze(audio.to_tensor(), axis=-1)
            audio = tf.cast(audio, dtype=tf.float32) / 32768.0
            yield os.path.split(file_name)[1], {
                'audio': audio.numpy(),
                'transcript': transcript
            }


class AihubSpectrogramDataset(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.'
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=None, dtype=tf.uint8),
                'transcript': tfds.features.Text(encoder=None)
            }),
            supervised_keys=('image', 'transcript'),
            homepage='https://aihub.or.kr/',
            citation=_CITATION
        )

    def _split_generator(self, dl_manager: tfds.download.DownloadManager):
        # TODO(aihub_dataset): Downloads the data and defined the splits
        path = dl_manager.download_and_extract('https://drive.google.com/u/0/uc?export=download&confirm=jWDd&id=1uBVr5NciHj59ZxmexYMopae7iH1ddSAO')
        path = unzip(path, remove_after=True, recursive=True)

        # TODO(aihub_dataset): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples(path / 'train_data'),
            'test': self._generate_examples(path / 'test_data')
        }

    def _generate_examples(self, path):
        """
        `tfds.features.Image` accepts:
            - `str`: path to a {bmp, gif, jpeg, png} image.
            - `np.array`: 3d `np.uint8` array representing an image.
            - A file object containing the png or jpeg encoded image string (ex: `io.BytesIO(encoded_img_bytes)`)
        """
        for f in path.glob('*.jpeg'):
            yield 'key', {
                'image': f,
                'label': 'yes'
            }


def unzip(path, remove_after=False, recursive=False):
    zipfiles = [os.path.abspath(path)]
    while zipfiles:
        path = zipfiles.pop(0)
        dirpath = os.path.splitext(path)[0]
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dirpath)
        if remove_after:
            os.remove(path)
        if not recursive:
            break
        new_files = os.listdir(dirpath)
        # FIXME: tf.io.gfile.glob('./*.zip)
        new_dirs = tuple(filter(lambda x: os.path.isfile(x) and os.path.splitext(x) == '.zip',
                         map(lambda x: os.path.join(dirpath, x),
                         new_files)))
        zipfiles.extend(new_dirs)
    return os.path.splitext(os.path.abspath(path))[0]


"""
def listdir(path, recursive=False, fmt='wav'):
    paths = [os.path.abspath(path)]

    while paths:
        path = paths.pop(0)
        files = map(lambda x: os.path.join(path, x), os.listdir(path))
        paths.extend(filter(lambda x: os.path.isdir(x), files))
        files = filter(lambda x: os.path.isfile(x), files)
        if fmt:
            files = filter(lambda x: os.path.splitext(x)[1] == f'.{fmt}', files)
        for file in files:
            yield file
"""


if __name__ == "__main__":
    pass
