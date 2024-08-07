import libcoral
import numpy as np
from matplotlib import pyplot as plt
import logging

logger = logging.getLogger("libcoral")
logger.setLevel(logging.DEBUG)

def load_fashion_mnist():
    """This function just downloads the 60k training images of fashion-mnist locally, if not already available,
    and returns them as a float32 array.
    """
    import urllib.request
    import os
    import gzip
    import umap

    fashion_local = "train-images-idx3-ubyte.gz"
    fashion_url = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz"
    fashion_labels_local = "train-labels-idx1-ubyte.gz"
    fashion_labels_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz"
    if not os.path.isfile(fashion_local):
        print("downloading fashion-mnist file")
        urllib.request.urlretrieve(fashion_url, fashion_local)
    if not os.path.isfile(fashion_labels_local):
        print("downloading fashion-mnist file")
        urllib.request.urlretrieve(fashion_labels_url, fashion_labels_local)
    with gzip.open(fashion_local, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(60000, 784)
    with gzip.open(fashion_labels_local, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    data = images.astype(np.float32)

    umap_file = "fashion-mnist-umap.npy"
    if not os.path.isfile(umap_file):
        embed = umap.UMAP()
        embedding = embed.fit_transform(data)
        np.save(umap_file, embedding)
    embedding = np.load(umap_file)

    return data, labels.astype(np.uint32), embedding

partition_matroid = libcoral.MatroidDescription([1] * 10)

data, categories, embedding = load_fashion_mnist()
print(data.shape, categories.shape)

palette = np.array(plt.get_cmap('tab10').colors)
colors = palette[categories]

diversity = libcoral.DiversityMaximization(
    10,
    "remote-clique",
    coreset_size=2000,
    matroid=partition_matroid,
    epsilon=0.0001
)
selected = diversity.solve(embedding, categories)
cost = diversity.cost(embedding[selected])
print(categories[selected])

plt.figure(figsize=(10,10))
plt.scatter(
    embedding[:,0],
    embedding[:,1],
    c=colors,
    cmap="tab10",
    s=0.01
)
plt.scatter(
    embedding[selected,0],
    embedding[selected,1],
    c=colors[selected,:],
    edgecolors="black",
    cmap="tab10",
    s=100
)

plt.savefig("check.png")


