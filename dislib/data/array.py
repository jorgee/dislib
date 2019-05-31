import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import COLLECTION_IN, Depth, Type
from pycompss.api.parameter import COLLECTION_INOUT
from pycompss.api.task import task
from scipy import sparse as sp
from scipy.sparse import issparse


def array(x, block_size):
    # TODO: documentation ain't true
    """
    Loads data into a Distributed Array.

    Parameters
    ----------
    x : ndarray, shape=[n_samples, n_features]
        Array of samples.
    block_size : (int, int)
        Block sizes in number of samples.

    Returns
    -------
    darray : Array
        A distributed representation of the data divided in blocks.
    """
    x_size, y_size = block_size

    blocks = []
    for i in range(0, x.shape[0], x_size):
        row = [x[i: i + x_size, j: j + y_size] for j in
               range(0, x.shape[1], y_size)]
        blocks.append(row)

    sparse = issparse(x)
    darray = Array(blocks=blocks, block_size=block_size, sparse=sparse)

    return darray


class Array(object):
    # """ A dataset containing samples and, optionally, labels that can be
    # stored in a distributed manner.
    #
    # Dataset works as a list of Subset instances, which can be future objects
    # stored remotely. Accessing Dataset.labels and Dataset.samples runs
    # collect() and transfers all the data to the local machine.
    #
    # Parameters
    # ----------
    # n_features : int
    #     Number of features of the samples.
    # sparse : boolean, optional (default=False)
    #     Whether this dataset uses sparse data structures.
    #
    # Attributes
    # ----------
    # n_features : int
    #     Number of features of the samples.
    # _samples : ndarray
    #     Samples of the dataset.
    # _labels : ndarray
    #     Labels of the samples.
    # sparse: boolean
    #     True if this dataset uses sparse data structures.
    # """

    # TODO: after implementing more constructors decide a better way to avoid
    # having to synchornize bot_right block size to compute shape
    def __init__(self, blocks, block_size, sparse):
        self._validate_blocks(blocks)

        self._blocks = blocks
        self._block_size = block_size
        self._blocks_shape = (len(blocks), len(blocks[0]))
        self._shape = None
        # self._sizes = list()
        # self._max_features = None
        # self._min_features = None
        # self._samples = None
        # self._labels = None
        self._sparse = sparse

    @staticmethod
    def _validate_blocks(blocks):
        if len(blocks) == 0 or len(blocks[0]) == 0:
            raise AttributeError('Blocks must a list of lists, with at least'
                                 ' an empty numpy/scipy matrix.')
        row_length = len(blocks[0])
        for i in range(1, len(blocks)):
            if len(blocks[i]) != row_length:
                raise AttributeError(
                    'All rows must contain the same number of blocks.')

    @staticmethod
    def _merge_blocks(blocks):
        b0 = blocks[0][0]
        if sp.issparse(b0):
            ret = sp.bmat(blocks, format=b0.getformat(), dtype=b0.dtype)
        else:
            ret = np.block(blocks)

        return ret

    @staticmethod
    def _get_out_blocks(x, y):
        return [[object() for _ in range(y)] for _ in range(x)]

    # def __getitem__(self, item):
    #     return self._subsets.__getitem__(item)
    #
    # def __len__(self):
    #     return len(self._subsets)
    #
    # def __iter__(self):
    #     for j in range(len(self._blocks)):
    #         yield _collect_blocks(self._blocks[j], axis=1)


    @property
    def shape(self):
        if self._shape is None:
            # last blocks may be of different size so we get bot right corner
            # to add the correct number of elements to the total
            # TODO try avoid this with 2 params like: top_left_shape &
            # bot_right_shape
            bot_right_shape = compss_wait_on(_get_shape(self._blocks[-1][-1]))

            blocks_x, blocks_y = self._blocks_shape
            size_x, size_y = self._block_size

            x = (blocks_x - 1) * size_x + bot_right_shape[0]
            y = (blocks_y - 1) * size_y + bot_right_shape[1]

            self._shape = (x, y)

        return self._shape

    def iterator(self, axis=0):
        # iterate through rows
        if axis == 0 or axis == 'rows':
            for row in self._blocks:
                yield Array(blocks=[row], block_size=self._block_size,
                            sparse=self._sparse)

        # iterate through columns
        elif axis == 1 or axis == 'columns':
            for j in range(self._blocks_shape[1]):
                col_blocks = [[self._blocks[i][j]] for i in
                              range(self._blocks_shape[0])]
                yield Array(blocks=col_blocks, block_size=self._block_size,
                            sparse=self._sparse)

        else:
            raise Exception(
                "Axis must be [0|'rows'] or [1|'columns']. Got: %s" % axis)

    def mean(self, axis=0, count_zero=False):
        """
        Compute the means iterating along the specified axis. With 0, iterates
         along the x-axis computing the means of the columns.
        """

        if axis == 1 or axis == 'rows':
            out_blocks = self._get_out_blocks(self._blocks_shape[0], 1)
            _mean(self._blocks, out_blocks, axis=1, count_zero=count_zero)
            block_size = (self._block_size[0], 1)
        elif axis == 0 or axis == 'columns':
            out_blocks = self._get_out_blocks(1, self._blocks_shape[1])
            _mean(self._blocks, out_blocks, axis=0, count_zero=count_zero)
            block_size = (1, self._block_size[1])
        else:
            raise AssertionError(
                "Axis must be [0|'columns'] or [1|'rows'], got %s" % str(axis))
        return Array(out_blocks, block_size=block_size, sparse=self._sparse)

    def transpose(self, mode='auto'):

        if mode == 'all':
            n, m = self._blocks_shape[0], self._blocks_shape[1]
            out_blocks = self._get_out_blocks(n, m)
            _tranpose(self._blocks, out_blocks)
        elif mode == 'rows':
            out_blocks = []
            for r in self.iterator(axis=0):
                _blocks = self._get_out_blocks(*r._blocks_shape)
                _tranpose(r._blocks, _blocks)
                out_blocks.append(_blocks[0])
        elif mode == 'columns':
            out_blocks = [[] for _ in range(self._blocks_shape[0])]
            for i, c in enumerate(self.iterator(axis=1)):
                _blocks = self._get_out_blocks(*c._blocks_shape)
                _tranpose(c._blocks, _blocks)
                for i2 in range(len(_blocks)):
                    out_blocks[i2].append(_blocks[i2][0])
        else:
            raise Exception(
                "Unknown tranpose mode'%s'. Options are: [all|rows|columns]" % mode)

        blocks_t = list(map(list, zip(*out_blocks)))

        # notice block_size is transposed
        bn, bm = self._block_size[1], self._block_size[0]

        return Array(blocks_t, block_size=(bm, bn), sparse=self._sparse)

    def collect(self):
        blocks = compss_wait_on(self._blocks)

        return self._merge_blocks(blocks)


@task(returns=1)
def _get_shape(block):
    return block.shape


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _mean(blocks, out_blocks, axis, count_zero):
    def _sparse_mean():
        if axis == 1:  # row
            for i, r in enumerate(blocks):
                rows = Array._merge_blocks([r])
                if count_zero:
                    out_blocks[i][0] = rows.mean(axis=axis).reshape(
                        rows.shape[0], 1)
                else:
                    out_blocks[i][0] = (
                        rows.sum(axis=1) / (rows != 0).toarray().sum(
                            axis=1).reshape(-1, 1))

        else:  # cols
            for j in range(len(blocks[0])):
                c = np.block([[blocks[i][j]] for i in range(len(blocks))])
                cols = Array._merge_blocks(c)
                if count_zero:
                    out_blocks[0][j] = cols.mean(axis=axis)
                else:
                    out_blocks[0][j] = cols.sum(axis=0) / (
                        cols != 0).toarray().sum(axis=0)

        return out_blocks

    def _dense_mean():
        if axis == 1:  # row
            for i, r in enumerate(blocks):
                rows = Array._merge_blocks([r])
                out_blocks[i][0] = rows.mean(axis=axis).reshape(len(rows), 1)
        else:  # cols
            for j in range(len(blocks[0])):
                c = np.block([[blocks[i][j]] for i in range(len(blocks))])
                cols = Array._merge_blocks(c)
                out_blocks[0][j] = cols.mean(axis=axis)
        return out_blocks

    # Depending on whether the data is sparse call appropriate nested function
    if issparse(blocks[0][0]):
        _sparse_mean()
    else:
        _dense_mean()


@task(blocks={Type: COLLECTION_IN, Depth: 2},
      out_blocks={Type: COLLECTION_INOUT, Depth: 2})
def _tranpose(blocks, out_blocks):
    for i in range(len(blocks)):
        for j in range(len(blocks[i])):
            out_blocks[i][j] = blocks[i][j].transpose()

# def transpose(self, n_subsets=None):
#     """ Transposes the Dataset.
#
#     Parameters
#     ----------
#     n_subsets : int, optional (default=None)
#         Number of subsets in the transposed dataset. If none, defaults to
#         the original number of subsets
#
#     Returns
#     -------
#     dataset_t: Dataset
#         Transposed dataset divided by rows.
#     """
#
#     if n_subsets is None:
#         n_subsets = len(self._subsets)
#
#     subsets_t = []
#     for i in range(n_subsets):
#         subsets_i = [_get_spli_i(s, i, n_subsets) for s in self._subsets]
#         new_subset = _merge_split_subsets(self.sparse, *subsets_i)
#         subsets_t.append(new_subset)
#
#     n_rows = np.sum(self.subsets_sizes())
#
#     dataset_t = Dataset(n_features=n_rows, sparse=self._sparse)
#
#     dataset_t.extend(subsets_t)
#
#     return dataset_t
#

#
#
# def min_features(self):
#     """ Returns the minimum value of each feature in the dataset. This
#     method might compute the minimum and perform a synchronization.
#
#     Returns
#     -------
#     min_features : array, shape = [n_features,]
#         Array representing the minimum value that each feature takes in
#         the dataset.
#     """
#     if self._min_features is None:
#         self._compute_min_max()
#
#     return self._min_features
#
# def max_features(self):
#     """ Returns the maximum value of each feature in the dataset. This
#     method might compute the maximum and perform a synchronization.
#
#     Returns
#     -------
#     max_features : array, shape = [n_features,]
#         Array representing the maximum value that each feature takes in
#         the dataset.
#     """
#     if self._max_features is None:
#         self._compute_min_max()
#
#     return self._max_features
#
# def collect(self):
#     self._blocks = compss_wait_on(self._blocks)
#
# @property
# def sparse(self):
#     return self._sparse
#
# def _reset_attributes(self):
#     self._max_features = None
#     self._min_features = None
#     self._samples = None
#     self._labels = None
#
# def _compute_min_max(self):
#     minmax = []
#
#     for subset in self._subsets:
#         minmax.append(_get_min_max(subset))
#
#     minmax = compss_wait_on(minmax)
#     self._min_features = np.nanmin(minmax, axis=0)[0]
#     self._max_features = np.nanmax(minmax, axis=0)[1]
#
# def _update_labels(self):
#     self.collect()
#     labels_list = []
#
#     for subset in self._subsets:
#         if subset.labels is not None:
#             labels_list.append(subset.labels)
#
#     if len(labels_list) > 0:
#         self._labels = np.concatenate(labels_list)
#
# def _update_samples(self):
#     self.collect()
#     if len(self._subsets) > 0:
#         # use the first subset to init to keep the subset's dtype
#         self._samples = self._subsets[0].samples
#
#         concat_f = sp.vstack if self._sparse else np.concatenate
#
#         for subset in self._subsets[1:]:
#             self._samples = concat_f((self._samples, subset.samples))

#
# @task(returns=object)
# def _subset_apply(subset, f, return_subset=False):
#     samples = [f(sample) for sample in subset.samples]
#     s = np.array(samples).reshape(len(samples), -1)
#
#     if return_subset:
#         s = Subset(samples=s)
#
#     return s
#
# @task(returns=np.array)
# def _get_min_max(subset):
#     mn = np.min(subset.samples, axis=0)
#     mx = np.max(subset.samples, axis=0)
#
#     if issparse(subset.samples):
#         mn = mn.toarray()[0]
#         mx = mx.toarray()[0]
#
#     return np.array([mn, mx])
