import libcoral # import the libcoral library
import time
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)

def main():
    data = get_fashion_mnist()

    coreset = libcoral.Coreset(200)
    start = time.time()
    assignment = coreset.fit_transform(data)
    assert assignment.shape[0] == data.shape[0]
    elapsed = time.time() - start
    print("elapsed time", elapsed, "seconds")

    assert coreset.weights_.sum() == data.shape[0]


def get_fashion_mnist():
    """This function just downloads the 60k training images of fashion-mnist locally, if not already available,
    and returns them as a float32 array.
    """
    import urllib.request
    import os
    import gzip
    fashion_local = "train-images-idx3-ubyte.gz"
    fashion_url = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz"
    if not os.path.isfile(fashion_local):
        print("downloading fashion-mnist file")
        urllib.request.urlretrieve(fashion_url, fashion_local)
    with gzip.open(fashion_local, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(60000, 784)
    return images.astype(np.float32)


main()
