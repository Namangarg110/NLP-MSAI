from train import *
import numpy as np
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from scipy.spatial import procrustes




def main():

    gpt_embeddings = get_embeddings()

    dict_size = len(gpt_embeddings[0])
    num_samples = 100
    embbed_samples = []
    samples = np.random.randint(0, dict_size, size=num_samples)
    for embed in gpt_embeddings:
        embbed_samples.append([embed[i, :] for i in samples])

    mtx1, mtx2, disparity = procrustes(*embbed_samples)
    combined = np.stack([mtx1, mtx2], axis=0).reshape(num_samples*2, 512)
    colors = np.zeros((num_samples*2, 3))
    colors[num_samples:] = (1,0,0)
    colors[0:num_samples] = (0,1,0)
    manifold_embedding = Isomap(n_neighbors=10, n_components=2)
    z = manifold_embedding.fit_transform(combined)
    fig, ax = plt.subplots()
    ax.scatter(z[:,0], z[:,1], c=colors)
    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    main()

