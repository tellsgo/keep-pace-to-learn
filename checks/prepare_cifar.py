"""Optional official CIFAR-10 download and conversion; never run by smoke checks."""
from pathlib import Path
import hashlib
import io
import pickle
import tarfile
import urllib.request
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def main():
    out = ROOT / 'data/cifar10'
    out.mkdir(parents=True, exist_ok=True)
    archive = out / 'cifar-10-python.tar.gz'
    if not archive.exists():
        urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', archive)
    if hashlib.md5(archive.read_bytes()).hexdigest() != 'c58f30108f718f92721af3b95e74349a':
        raise ValueError('Archive digest mismatch')
    with tarfile.open(archive) as tf:
        def batch(name):
            member = tf.extractfile('cifar-10-batches-py/' + name)
            if member is None:
                raise ValueError('Missing batch')
            obj = pickle.load(io.BytesIO(member.read()), encoding='bytes')
            return obj[b'data'].reshape(-1, 3, 32, 32), np.asarray(obj[b'labels'], dtype=np.int64)
        train = [batch(f'data_batch_{i}') for i in range(1, 6)]
        tx, ty = batch('test_batch')
    np.savez_compressed(out / 'cifar10.npz', train_x=np.concatenate([x for x,y in train]), train_y=np.concatenate([y for x,y in train]), test_x=tx, test_y=ty)
    print('Prepared data/cifar10/cifar10.npz')

if __name__ == '__main__':
    main()
