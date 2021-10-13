import os
import sys
import json
from urllib import request
from pathlib import Path

DATASET_STORAGE = os.path.join(str(Path.home()), 'stored_datasets')


def download(url):
    _, path = os.path.split(url)
    url = request.Request(url)
    url.add_header('Authorization', '-')
    try:
        print('url:', url)
        # request.urlretrieve(url, path, reporthook=_download_reporthook(url))
        r = request.urlopen(url)
    except KeyboardInterrupt:
        if os.path.exists(path):
            os.remove(path)


def _download_reporthook(url):
    block_size = 0  # Enclosing

    def download_reporthook(blocknum, bs, size):
        nonlocal block_size
        block_size += bs
        width = os.get_terminal_size().columns
        message = '\rDownload {url}: ({rate:.2f}%)'.format(url=url, rate=block_size/size*100)

        if len(message) > width:
            message = message[:width//2-3] + '...' + message[-width//2:]
        sys.stdout.write(message)

        if block_size == size:
            sys.stdout.write('\n')

    return download_reporthook


if __name__ == "__main__":
    with open(os.path.join(os.path.dirname(__file__), 'src', 'datasets', 'config.json'), 'r') as f:
        config = json.loads(''.join(f.readlines()))
        files = config.get('dacon', [])

    for file_ in files:
        download(file_)

    sys.command('zip -s 0 dacon.zip --out dacon.merged.zip')
    sys.command('unzip dacon.merged.zip')
