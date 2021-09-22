import os
import argparse
import zipfile

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=os.getcwd())
    return parser.parse_args()


def extract_all(path):
    paths = [path]
    while paths:
        path = os.path.abspath(paths.pop(0))
        # print(f'- path: {path} ({os.path.isdir(path)})')
        if os.path.isdir(path):
            paths.extend(map(lambda x: os.path.join(path, x), os.listdir(path)))
        elif os.path.isfile(path):
            path_, ext = os.path.splitext(path)
            if ext == '.zip':
                with zipfile.ZipFile(path) as zf:
                    zf.extractall(path_)
                os.remove(path)
                paths.extend(map(lambda x: os.path.join(path_, x), os.listdir(path_)))


if __name__ == "__main__":
    args = parse_args()
    print('args.path:', args.path)
    extract_all(args.path)
