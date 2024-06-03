from train import *
import rnn_train as rnntr
import numpy as np
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import directed_hausdorff
from scipy.stats import wasserstein_distance_nd

def get_embedded_samples(embeddings, num_samples, sample_func):
    def default_sample_func(embeds, i: int):
        return embeds[i, :]

    if sample_func is None:
        sample_func = default_sample_func

    dict_size = len(embeddings)
    samples = []
    sample_indices = np.random.randint(0, dict_size, size=num_samples)
    for si in sample_indices:
        samples.append(sample_func(embeddings, si))
    return samples


def plot_embedding_similarity(embeddings, title, num_samples, sample_func=None):
    np.random.seed(1337)
    embeddings = np.stack(embeddings)
    embbed_samples = [get_embedded_samples(embeddings[i], num_samples, sample_func) for i in range(len(embeddings))]
    mtx1, mtx2, disparity = procrustes(*embbed_samples)

    combined = np.stack([mtx1, mtx2], axis=0).reshape(num_samples * 2, mtx2.shape[-1])
    colors = np.zeros((num_samples * 2, 3))
    colors[num_samples:] = (1, 0, 0)
    colors[0:num_samples] = (0, 1, 0)

    manifold_embedding = Isomap(n_neighbors=5, n_components=2)
    z = manifold_embedding.fit_transform(combined)
    pl_x = z[:, 0]
    pl_y = z[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(pl_x, pl_y, c=colors)
    sample_func_name = sample_func.__name__ if sample_func is not None else 'Default'
    plt.title(f'{title}\nDisparity: {disparity:.4f}, SampleFunc: {sample_func_name}, Samples: {num_samples}')
    plt.show()


def main():
    rnn_embeddings = rnntr.get_embeddings(['./RNN/model1.pth', './RNN/model2.pth'])
    gpt_embeddings = get_embeddings()
    sample_count = 500

    def delta_sample(embeddings, index):
        temp = index - 1 if index > 0 else index + 1
        i2 = max(temp, index)
        i1 = min(temp, index)
        p1 = embeddings[i1, :]
        p2 = embeddings[i2, :]
        delta = p1 - p2
        return delta

    def hausdorff_distance(embeddings, index):
        temp = index - 1 if index > 0 else index + 1
        i2 = max(temp, index)
        i1 = min(temp, index)
        p1 = embeddings[i1, :]
        p2 = embeddings[i2, :]




    #plot_embedding_similarity(rnn_embeddings, 'RNN', sample_count)
    #plot_embedding_similarity(rnn_embeddings, 'RNN', sample_count, sample_func=delta_sample)
    plot_embedding_similarity(gpt_embeddings, 'GPT2', sample_count)
    #plot_embedding_similarity(gpt_embeddings, 'GPT2', sample_count, sample_func=delta_sample)


if __name__ == '__main__':
    main()
