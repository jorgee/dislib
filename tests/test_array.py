# import os
import unittest
from math import ceil

import numpy as np
from scipy import sparse as sp
from scipy.sparse import issparse

import dislib as ds


def equal(arr1, arr2):
    if issparse(arr1) and not issparse(arr2):
        raise AttributeError('Arrays are of different type: %s != %s' % (
            type(arr1), type(arr2)))

    if issparse(arr1):
        equal = (arr1 != arr2).nnz == 0
    else:
        equal = not (arr1 != arr2).any()

    return equal


class DataLoadingTest(unittest.TestCase):
    def test_array_constructor(self):
        """ Tests load_data
        """
        x = np.random.randint(0, 10, size=(6, 10))
        darray = ds.array(x=x, block_size=(4, 3)).collect()

        self.assertTrue(equal(darray, x))
        self.assertEqual(type(darray), type(x))
        self.assertEqual(type(darray[0, 0]), type(x[0, 0]))

        x = sp.csr_matrix(x)
        darray = ds.array(x=x, block_size=(4, 3)).collect()

        self.assertTrue(equal(darray, x))
        self.assertEqual(type(darray[0, 0]), type(x[0, 0]))
        self.assertEqual(type(darray), type(x))


# def test_load_data_without_labels(self):
#         """ Tests load_data with an unlabeled sparse and dense dataset.
#         """
#         x = np.random.random((100, 2))
#         dataset = load_data(x=x, subset_size=10)
#
#         self.assertTrue(np.array_equal(dataset.samples, x))
#         self.assertEqual(len(dataset), 10)
#         self.assertFalse(dataset.sparse)
#
#         x = csr_matrix(x)
#         dataset = load_data(x=x, subset_size=10)
#
#         self.assertTrue(np.array_equal(dataset.samples.toarray(),
# x.toarray()))
#         self.assertEqual(len(dataset), 10)
#         self.assertTrue(dataset.sparse)
#
#     def test_load_libsvm_file_sparse(self):
#         """ Tests loading a LibSVM file in sparse mode.
#         """
#         file_ = "tests/files/libsvm/2"
#
#         data = load_libsvm_file(file_, 10, 780)
#         data.collect()
#         x, y = load_svmlight_file(file_, n_features=780)
#
#         read_x = np.empty((0, x.shape[1]))
#         read_y = np.empty(0)
#
#         for subset in data:
#             read_x = np.concatenate((read_x, subset.samples.toarray()))
#             read_y = np.concatenate((read_y, subset.labels))
#
#         self.assertTrue((read_x == x.toarray()).all())
#         self.assertTrue((read_y == y).all())
#         self.assertEqual(len(data), 6)
#
#     def test_load_libsvm_file_dense(self):
#         """ Tests loading a LibSVM file in dense mode.
#         """
#         file_ = "tests/files/libsvm/1"
#
#         data = load_libsvm_file(file_, 20, 780, False)
#         data.collect()
#         x, y = load_svmlight_file(file_, n_features=780)
#
#         read_x = np.empty((0, x.shape[1]))
#         read_y = np.empty(0)
#
#         for subset in data:
#             read_x = np.concatenate((read_x, subset.samples))
#             read_y = np.concatenate((read_y, subset.labels))
#
#         self.assertTrue((read_x == x.toarray()).all())
#         self.assertTrue((read_y == y).all())
#         self.assertEqual(len(data), 4)
#
#     def test_load_libsvm_files_sparse(self):
#         """ Tests loading multiple LibSVM files in sparse mode.
#         """
#         dir_ = "tests/files/libsvm"
#
#         file_list = os.listdir(dir_)
#         data = load_libsvm_files(dir_, 780)
#         data.collect()
#
#         for i, subset in enumerate(data):
#             samples = subset.samples.toarray()
#             file_ = os.path.join(dir_, file_list[i])
#             x, y = load_svmlight_file(file_, n_features=780)
#
#             self.assertTrue((samples == x).all())
#             self.assertTrue((subset.labels == y).all())
#
#         self.assertEqual(len(data), 3)
#
#     def test_load_libsvm_files_dense(self):
#         """ Tests loading multiple LibSVM files in dense mode.
#         """
#         dir_ = "tests/files/libsvm"
#
#         file_list = os.listdir(dir_)
#         data = load_libsvm_files(dir_, 780, False)
#         data.collect()
#
#         for i, subset in enumerate(data):
#             samples = subset.samples
#             file_ = os.path.join(dir_, file_list[i])
#             x, y = load_svmlight_file(file_, n_features=780)
#
#             self.assertTrue((samples == x).all())
#             self.assertTrue((subset.labels == y).all())
#
#         self.assertEqual(len(data), 3)
#
#     def test_load_csv_file(self):
#         """ Tests loading a CSV file.
#         """
#         csv_file = "tests/files/csv/1"
#
#         data = load_txt_file(csv_file, subset_size=300, n_features=122)
#         data.collect()
#         csv = np.loadtxt(csv_file, delimiter=",")
#
#         read_x = np.empty((0, csv.shape[1]))
#
#         for subset in data:
#             read_x = np.concatenate((read_x, subset.samples))
#
#         self.assertTrue((read_x == csv).all())
#         self.assertEqual(len(data), 15)
#         self.assertIsNone(subset.labels)
#
#     def test_load_csv_file_labels_last(self):
#         """ Tests loading a CSV file with labels at the last column.
#         """
#         csv_file = "tests/files/csv/1"
#
#         data = load_txt_file(csv_file, subset_size=1000, n_features=121,
#                              label_col="last")
#         data.collect()
#         csv = np.loadtxt(csv_file, delimiter=",")
#
#         read_x = np.empty((0, csv.shape[1] - 1))
#         read_y = np.empty(0)
#
#         for subset in data:
#             read_x = np.concatenate((read_x, subset.samples))
#             read_y = np.concatenate((read_y, subset.labels))
#
#         self.assertTrue((read_x == csv[:, :-1]).all())
#         self.assertTrue((read_y == csv[:, -1]).all())
#         self.assertEqual(len(data), 5)
#
#     def test_load_csv_file_labels_first(self):
#         """ Tests loading a CSV file with labels at the first column.
#         """
#         csv_file = "tests/files/csv/2"
#
#         data = load_txt_file(csv_file, subset_size=100, n_features=121,
#                              label_col="first")
#         data.collect()
#         csv = np.loadtxt(csv_file, delimiter=",")
#
#         read_x = np.empty((0, csv.shape[1] - 1))
#         read_y = np.empty(0)
#
#         for subset in data:
#             read_x = np.concatenate((read_x, subset.samples))
#             read_y = np.concatenate((read_y, subset.labels))
#
#         self.assertTrue((read_x == csv[:, 1:]).all())
#         self.assertTrue((read_y == csv[:, 0]).all())
#         self.assertEqual(len(data), 44)
#
#     def test_load_csv_files(self):
#         """ Tests loading multiple CSV files.
#         """
#         csv_dir = "tests/files/csv"
#
#         file_list = os.listdir(csv_dir)
#         data = load_txt_files(csv_dir, n_features=122)
#         data.collect()
#
#         for i, subset in enumerate(data):
#             csv_file = os.path.join(csv_dir, file_list[i])
#             csv = np.loadtxt(csv_file, delimiter=",")
#
#             self.assertTrue((subset.samples == csv).all())
#
#         self.assertEqual(len(data), 3)
#
#     def test_load_csv_files_labels_last(self):
#         """ Tests loading multiple CSV files with labels at the last column.
#         """
#         csv_dir = "tests/files/csv"
#
#         file_list = os.listdir(csv_dir)
#         data = load_txt_files(csv_dir, n_features=122, label_col="last")
#         data.collect()
#
#         for i, subset in enumerate(data):
#             csv_file = os.path.join(csv_dir, file_list[i])
#             csv = np.loadtxt(csv_file, delimiter=",")
#
#             self.assertTrue((subset.samples == csv[:, :-1]).all())
#             self.assertTrue((subset.labels == csv[:, -1]).all())
#
#         self.assertEqual(len(data), 3)
#
#     def test_load_csv_files_labels_first(self):
#         """ Tests loading multiple CSV files with labels at the first column.
#         """
#         csv_dir = "tests/files/csv"
#
#         file_list = os.listdir(csv_dir)
#         data = load_txt_files(csv_dir, n_features=122, label_col="first")
#         data.collect()
#
#         for i, subset in enumerate(data):
#             csv_file = os.path.join(csv_dir, file_list[i])
#             csv = np.loadtxt(csv_file, delimiter=",")
#
#             self.assertTrue((subset.samples == csv[:, 1:]).all())
#             self.assertTrue((subset.labels == csv[:, 0]).all())
#
#         self.assertEqual(len(data), 3)
#
#     def test_load_txt_delimiter(self):
#         """ Tests load_txt_file with a custom delimiter """
#         path_ = "tests/files/other/4"
#         data = load_txt_file(path_, n_features=122, subset_size=1000,
#                              delimiter=" ")
#         csv = np.loadtxt(path_, delimiter=" ")
#
#         self.assertTrue(np.array_equal(data.samples, csv))
#         self.assertEqual(len(data), 5)
#         self.assertIsNone(data.labels)
#
#     def test_load_txt_files_delimiter(self):
#         """ Tests loading multiple files with a custom delimiter"""
#         path_ = "tests/files/other"
#
#         file_list = os.listdir(path_)
#         data = load_txt_files(path_, n_features=122, delimiter=" ")
#         data.collect()
#
#         for i, subset in enumerate(data):
#             file_ = os.path.join(path_, file_list[i])
#             read_data = np.loadtxt(file_, delimiter=" ")
#
#             self.assertTrue(np.array_equal(subset.samples, read_data))
#
#         self.assertEqual(len(data), 2)
#
#


class ArrayTest(unittest.TestCase):
    #     def test_get_item(self):
    #         """ Tests Dataset item getter. """
    #         arr = np.array((range(10), range(10, 20)))
    #         dataset = load_data(arr, subset_size=2)
    #         samples = dataset[0].samples
    #
    #         self.assertTrue((samples[0] == arr[0]).all())
    #
    def test_sizes(self):
        """ Tests sizes consistency. """

        x_size, y_size = 40, 25
        bx_size, by_size = 9, 11
        x = np.random.randint(10, size=(x_size, y_size))
        darray = ds.array(x=x, block_size=(bx_size, by_size))

        self.assertEqual(darray.shape, (x_size, y_size))

        self.assertEqual(darray._blocks_shape,
                         (ceil(x_size / bx_size), ceil(y_size / by_size)))
        self.assertEqual(darray._block_size, (bx_size, by_size))

        x = sp.csr_matrix(x)
        darray = ds.array(x=x, block_size=(bx_size, by_size))

        self.assertEqual(darray.shape, (x_size, y_size))
        self.assertEqual(darray._blocks_shape,
                         (ceil(x_size / bx_size), ceil(y_size / by_size)))
        self.assertEqual(darray._block_size, (bx_size, by_size))

    def test_iterate_rows(self):
        """ Testing the row iterator of the ds.array
        """
        x_size = 2
        # Dense
        x = np.random.randint(10, size=(10, 10))
        data = ds.array(x=x, block_size=(x_size, 2))
        for i, r in enumerate(data.iterator(axis='rows')):
            r_data = r.collect()
            r_x = x[i * x_size:(i + 1) * x_size]
            self.assertTrue(equal(r_data, r_x))

        # Sparse
        x = sp.csr_matrix(x)
        data = ds.array(x=x, block_size=(x_size, 2))
        for i, r in enumerate(data.iterator(axis='rows')):
            r_data = r.collect()
            r_x = x[i * x_size:(i + 1) * x_size]
            self.assertTrue(equal(r_data, r_x))

    def test_iterate_cols(self):
        """ Tests iterating through the rows of the ds.array
        """
        y_size = 2
        # Dense
        x = np.random.randint(10, size=(10, 10))
        data = ds.array(x=x, block_size=(2, y_size))

        for i, c in enumerate(data.iterator(axis='columns')):
            c_data = c.collect()
            c_x = x[:, i * y_size:(i + 1) * y_size]
            self.assertTrue(equal(c_data, c_x))

        # Sparse
        x = sp.csr_matrix(x)
        data = ds.array(x=x, block_size=(2, y_size))

        for i, c in enumerate(data.iterator(axis='columns')):
            c_data = c.collect()
            c_x = x[:, i * y_size:(i + 1) * y_size]
            self.assertTrue(equal(c_data, c_x))

    def test_mean(self):
        y_size = 2

        # Dense
        x = np.random.randint(10, size=(10, 10))
        data = ds.array(x=x, block_size=(2, y_size))

        self.assertTrue(equal(data.mean(axis=0).collect(), x.mean(axis=0)))
        self.assertTrue(
            equal(data.mean(axis=1).collect().flatten(), x.mean(axis=1)))

        # Sparse
        x = sp.csr_matrix(x)
        data = ds.array(x=x, block_size=(2, y_size))

        # Compute the mean counting empty positions as 0
        data_mean = data.mean(axis=0, count_zero=True).collect()
        self.assertTrue(equal(data_mean, x.mean(axis=0)))

        data_mean = data.mean(axis=1, count_zero=True).collect()
        self.assertTrue(equal(data_mean, x.mean(axis=1)))

        # Compute the mean without considering empty positions
        data_mean = data.mean(axis=0, count_zero=False).collect()
        x_mean = x.sum(axis=0) / (x != 0).toarray().sum(axis=0)
        self.assertTrue(equal(data_mean, x_mean))

        data_mean = data.mean(axis=1, count_zero=False).collect()
        x_mean = x.sum(axis=1) / (x != 0).toarray().sum(axis=1).reshape(-1, 1)
        self.assertTrue(equal(data_mean, x_mean))

    def test_transpose(self):
        """ Tests array transpose."""

        x_size, y_size = 4, 6
        bx_size, by_size = 2, 3

        x = np.random.randint(10, size=(x_size, y_size))
        darray = ds.array(x=x, block_size=(bx_size, by_size))

        darray_t = darray.transpose(mode='all')
        self.assertTrue(equal(darray_t.collect(), x.transpose()))
        # ensure that original data was not modified
        self.assertTrue(equal(darray.collect(), x))

        darray_t = darray.transpose(mode='rows')
        self.assertTrue(equal(darray_t.collect(), x.transpose()))

        darray_t = darray.transpose(mode='columns')
        self.assertTrue(equal(darray_t.collect(), x.transpose()))

        self.assertRaises(Exception, darray.transpose, 'invalid')


# def test_min_max_features(self):
#         """ Tests that min_features and max_features correctly return min
#         and max values in a toy dataset.
#         """
#         s1 = Subset(samples=np.array([[1, 2], [4, 5], [2, 2], [6, 6]]),
#                     labels=np.array([0, 1, 1, 1]))
#         s2 = Subset(samples=np.array([[7, 8], [9, 8], [0, 4]]),
#                     labels=np.array([0, 1, 1]))
#         s3 = Subset(samples=np.array([[3, 9], [0, 7], [6, 1], [0, 8]]),
#                     labels=np.array([0, 1, 1, 1]))
#         dataset = Dataset(n_features=2)
#         dataset.append(s1)
#         dataset.append(s2)
#         dataset.append(s3)
#
#         min_ = dataset.min_features()
#         max_ = dataset.max_features()
#
#         self.assertTrue(np.array_equal(min_, np.array([0, 1])))
#         self.assertTrue(np.array_equal(max_, np.array([9, 9])))
#
#     def test_min_max_features_sparse(self):
#         """ Tests that min_features and max_features correctly return min
#         and max values with sparse dataset. """
#
#         file_ = "tests/files/libsvm/1"
#         sparse = load_libsvm_file(file_, 10, 780)
#         dense = load_libsvm_file(file_, 10, 780, store_sparse=False)
#
#         max_sp = sparse.max_features()
#         max_d = dense.max_features()
#         min_sp = sparse.min_features()
#         min_d = sparse.min_features()
#
#         self.assertTrue(np.array_equal(max_sp, max_d))
#         self.assertTrue(np.array_equal(min_sp, min_d))
#


def main():
    unittest.main()


if __name__ == '__main__':
    main()
