import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from sklearn.neighbors import NearestNeighbors as SKNeighbors


class NearestNeighbors:

    def __init__(self, n_neighbors=5):
        """ Unsupervised learner for implementing neighbor searches.

        Parameters
        ----------
        n_neighbors : int, optional (default=5)
            Number of neighbors to use by default for kneighbors queries.

        Examples
        --------
        >>> from dislib.cluster import NearestNeighbors
        >>> from dislib.data import load_data
        >>> x = np.random.random((100, 5))
        >>> data = load_data(x, subset_size=25)
        >>> knn = NearestNeighbors(n_neighbors=10)
        >>> knn.fit(data)
        >>> distances, indices = knn.kneighbors(data)
        """
        self._n_neighbors = n_neighbors
        self._fit_dataset = None

    def fit(self, dataset):
        """ Fit the model using dataset as training data.

        Parameters
        ----------
        dataset : Dataset
            Training data.
        """
        self._fit_dataset = dataset

    def kneighbors(self, dataset, n_neighbors=None, return_distance=True):
        """ Finds the K nearest neighbors of the samples in dataset. Returns
        indices and distances to the neighbors of each sample.

        Parameters
        ----------
        dataset : Dataset
            The query samples.
        n_neighbors: int, optional (default=None)
            Number of neighbors to get. If None, the value passed in the
            constructor is employed.

        Returns
        -------
        dist : array
            Array representing the lengths to points, only present if
            return_distance=True.
        ind : array
            Indices of the nearest samples in the fitted data.
        """
        if n_neighbors is None:
            n_neighbors = self._n_neighbors

        distances = []
        indices = []

        for subset in dataset:
            dist, ind = _get_neighbors(subset, n_neighbors, *self._fit_dataset)
            distances.append(dist)
            indices.append(ind)

        final_indices = _merge_arrays(*indices)
        final_distances = _merge_arrays(*distances)

        final_indices = compss_wait_on(final_indices)
        query = final_indices

        if return_distance:
            final_distances = compss_wait_on(final_distances)
            query = final_distances, final_indices

        return query


@task(returns=2)
def _get_neighbors(subset, n_neighbors, *dataset):
    # find the k nearest neighbors of subset in dataset by finding the k
    # nearest neighbors of subset in every subset in dataset
    knn = SKNeighbors(n_neighbors=n_neighbors)
    samples = subset.samples
    n_samples = samples.shape[0]
    ind_offset = dataset[0].samples.shape[0]

    knn.fit(X=dataset[0].samples)
    final_dist, final_ind = knn.kneighbors(X=samples)

    for subset2 in dataset[1:]:
        knn.fit(X=subset2.samples)
        dist, ind = knn.kneighbors(X=samples)

        ind += ind_offset
        ind_offset += subset2.samples.shape[0]

        # keep the indices of the samples that are at minimum distance
        m_ind = _min_indices(final_dist, dist)
        comb_ind = np.hstack((final_ind, ind))
        final_ind = np.array([comb_ind[i][m_ind[i]] for i in range(n_samples)])

        # keep the minimum distances
        final_dist = _min_distances(final_dist, dist)

    return final_dist, final_ind


def _min_distances(d1, d2):
    size, num = d1.shape
    d = [np.sort(np.hstack((d1[i], d2[i])))[:num] for i in range(size)]
    return np.array(d)


def _min_indices(d1, d2):
    size, num = d1.shape
    d = [np.argsort(np.hstack((d1[i], d2[i])))[:num] for i in range(size)]
    return np.array(d)


@task(returns=np.array)
def _merge_arrays(*arrays):
    return np.vstack(arrays)
