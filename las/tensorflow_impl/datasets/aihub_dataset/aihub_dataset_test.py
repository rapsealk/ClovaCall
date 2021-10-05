"""aihub_dataset dataset."""

import tensorflow_datasets as tfds

import aihub_dataset


class AihubAudioDatasetTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for aihub_dataset dataset."""
    # TODO(aihub_dataset):
    DATASET_CLASS = aihub_dataset.AihubAudioDataset
    number_of_data = 32454
    SPLITS = {
        'train': round(number_of_data * 0.8),  # Number of fake train example (25963)
        'test': round(number_of_data * 0.2)  # Number of fake test example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
    tfds.testing.test_main()
