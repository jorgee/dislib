import os
import numpy as np
from numpy.lib import format
from pycompss.api.parameter import COLLECTION_OUT, Type, Depth, \
    COLLECTION_FILE_IN
from pycompss.api.task import task

from dislib.data.array import Array
from math import ceil


def load_svmlight_file(path, block_size, n_features, store_sparse):
    """ Loads a SVMLight file into a distributed array.

    Parameters
    ----------
    path : string
        File path.
    block_size : tuple (int, int)
        Size of the blocks for the output ds-array.
    n_features : int
        Number of features.
    store_sparse : boolean
        Whether to use scipy.sparse data structures to store data. If False,
        numpy.array is used instead.

    Returns
    -------
    x, y : (ds-array, ds-array)
        A distributed representation (ds-array) of the X and y.
    """
    n, m = block_size
    lines = []
    x_blocks, y_blocks = [], []

    n_rows = 0
    with open(path, "r") as f:
        for line in f:
            n_rows += 1
            lines.append(line.encode())

            if len(lines) == n:
                # line 0 -> X, line 1 -> y
                out_blocks = Array._get_out_blocks((1, ceil(n_features / m)))
                out_blocks.append([object()])
                # out_blocks.append([])
                _read_svmlight(lines, out_blocks, col_size=m,
                               n_features=n_features,
                               store_sparse=store_sparse)
                # we append only the list forming the row (out_blocks depth=2)
                x_blocks.append(out_blocks[0])
                y_blocks.append(out_blocks[1])
                lines = []

    if lines:
        out_blocks = Array._get_out_blocks((1, ceil(n_features / m)))
        out_blocks.append([object()])
        _read_svmlight(lines, out_blocks, col_size=m,
                       n_features=n_features, store_sparse=store_sparse)
        # we append only the list forming the row (out_blocks depth=2)
        x_blocks.append(out_blocks[0])
        y_blocks.append(out_blocks[1])

    x = Array(x_blocks, top_left_shape=block_size, reg_shape=block_size,
              shape=(n_rows, n_features), sparse=store_sparse)

    # y has only a single line but it's treated as a 'column'
    y = Array(y_blocks, top_left_shape=(n, 1), reg_shape=(n, 1),
              shape=(n_rows, 1), sparse=False)

    return x, y


def load_txt_file(path, block_size, delimiter=","):
    """ Loads a text file into a distributed array.

    Parameters
    ----------
    path : string
        File path.
    block_size : tuple (int, int)
        Size of the blocks of the array.
    delimiter : string, optional (default=",")
        String that separates columns in the file.

    Returns
    -------
    x : ds-array
        A distributed representation of the data divided in blocks.
    """

    with open(path, "r") as f:
        first_line = f.readline().strip()
        n_cols = len(first_line.split(delimiter))

    n_blocks = ceil(n_cols / block_size[1])
    blocks = []
    lines = []
    n_lines = 0

    with open(path, "r") as f:
        for line in f:
            n_lines += 1
            lines.append(line.encode())

            if len(lines) == block_size[0]:
                out_blocks = [object() for _ in range(n_blocks)]
                _read_lines(lines, block_size[1], delimiter, out_blocks)
                blocks.append(out_blocks)
                lines = []

    if lines:
        out_blocks = [object() for _ in range(n_blocks)]
        _read_lines(lines, block_size[1], delimiter, out_blocks)
        blocks.append(out_blocks)

    return Array(blocks, top_left_shape=block_size, reg_shape=block_size,
                 shape=(n_lines, n_cols), sparse=False)


def load_npy_file(path, block_size):
    """ Loads a file in npy format (must be 2-dimensional).

    Parameters
    ----------
    path : str
        Path to the npy file.
    block_size : tuple (int, int)
        Block size of the resulting ds-array.

    Returns
    -------
    x : ds-array
    """
    try:
        fid = open(path, "rb")
        version = format.read_magic(fid)
        format._check_version(version)
        shape, fortran_order, dtype = format._read_array_header(fid, version)

        if fortran_order:
            raise ValueError("Fortran order not supported for npy files")

        if len(shape) != 2:
            raise ValueError("Array is not 2-dimensional")

        if block_size[0] > shape[0] or block_size[1] > shape[1]:
            raise ValueError("Block size is larger than the array")

        blocks = []
        n_blocks = int(ceil(shape[1] / block_size[1]))

        for i in range(0, shape[0], block_size[0]):
            read_count = min(block_size[0], shape[0] - i)
            read_size = int(read_count * shape[1] * dtype.itemsize)
            data = fid.read(read_size)
            out_blocks = [object() for _ in range(n_blocks)]
            _read_from_buffer(data, dtype, shape[1], block_size[1], out_blocks)
            blocks.append(out_blocks)

        return Array(blocks=blocks, top_left_shape=block_size,
                     reg_shape=block_size, shape=shape, sparse=False)
    finally:
        fid.close()


def load_hstack_npy_files(path, cols_per_block=None):
    """ Loads the .npy files in a directory into a ds-array, stacking them
    horizontally, like (A|B|C). The order of concatenation is arbitrary.

    At least 1 valid .npy file must exist in the directory, and every .npy file
    must contain a valid array. Every array must have the same dtype, order,
    and number of rows.

    The blocks of the returned ds-array will have the same number of rows as
    the input arrays, and cols_per_block columns, which defaults to the number
    of columns of the first array.

    Parameters
    ----------
    path : string
        Folder path.
    cols_per_block : tuple (int, int)
        Number of columns of the blocks for the output ds-array. If None, the
        number of columns of the first array is used.

    Returns
    -------
    x : ds-array
        A distributed representation (ds-array) of the stacked arrays.
    """
    folder_paths = [os.path.join(path, name) for name in os.listdir(path)]
    # Full path of .npy files in the folder
    files = [pth for pth in folder_paths
             if os.path.isfile(pth) and pth[-4:] == '.npy']
    # Read the header of the first file to get shape, order, and dtype
    with open(files[0], "rb") as fid:
        version = format.read_magic(fid)
        format._check_version(version)
        shape0, order0, dtype0 = format._read_array_header(fid, version)
    rows = shape0[0]
    if cols_per_block is None:
        cols_per_block = shape0[1]
    # Check that all files have the same number of rows, order and datatype,
    # and store the number of columns for each file.
    files_cols = [shape0[1]]
    for filename in files[1:]:
        with open(filename, "rb") as fid:
            version = format.read_magic(fid)
            format._check_version(version)
            shape, order, dtype = format._read_array_header(fid, version)
            if shape[0] != shape0[0] or order0 != order or dtype0 != dtype:
                raise AssertionError()
            files_cols.append(shape[1])

    # Compute the parameters block_files, start_col and end_col for each block,
    # and call the task _load_hstack_npy_block() to generate each block.
    blocks = []
    file_idx = 0
    start_col = 0
    while file_idx < len(files):
        block_files = [files[file_idx]]
        cols = files_cols[file_idx] - start_col
        while cols < cols_per_block:  # while block not completed
            if file_idx + 1 == len(files):  # last file
                break
            file_idx += 1
            block_files.append(files[file_idx])
            cols += files_cols[file_idx]
        # Compute end_col of last file in block (last block may be smaller)
        end_col = files_cols[file_idx] - max(0, (cols - cols_per_block))
        blocks.append(_load_hstack_npy_block(block_files, start_col, end_col))
        if end_col == files_cols[file_idx]:  # file completed
            file_idx += 1
            start_col = 0
        else:  # file uncompleted
            start_col = end_col

    return Array(blocks=[blocks], top_left_shape=(rows, cols_per_block),
                 reg_shape=(rows, cols_per_block),
                 shape=(rows, sum(files_cols)),
                 sparse=False)


@task(out_blocks=COLLECTION_OUT)
def _read_from_buffer(data, dtype, shape, block_size, out_blocks):
    arr = np.frombuffer(data, dtype=dtype)
    arr = arr.reshape((-1, shape))

    for i in range(len(out_blocks)):
        out_blocks[i] = arr[:, i * block_size:(i + 1) * block_size]


@task(out_blocks=COLLECTION_OUT)
def _read_lines(lines, block_size, delimiter, out_blocks):
    samples = np.genfromtxt(lines, delimiter=delimiter)

    if len(samples.shape) == 1:
        samples = samples.reshape(1, -1)

    for i, j in enumerate(range(0, samples.shape[1], block_size)):
        out_blocks[i] = samples[:, j:j + block_size]


@task(out_blocks={Type: COLLECTION_OUT, Depth: 2})
def _read_svmlight(lines, out_blocks, col_size, n_features, store_sparse):
    from tempfile import SpooledTemporaryFile
    from sklearn.datasets import load_svmlight_file

    # Creating a tmp file to use load_svmlight_file method should be more
    # efficient than parsing the lines manually
    tmp_file = SpooledTemporaryFile(mode="wb+", max_size=2e8)
    tmp_file.writelines(lines)
    tmp_file.seek(0)

    x, y = load_svmlight_file(tmp_file, n_features)
    if not store_sparse:
        x = x.toarray()

    # tried also converting to csc/ndarray first for faster splitting but it's
    # not worth. Position 0 contains the X
    for i in range(ceil(n_features / col_size)):
        out_blocks[0][i] = x[:, i * col_size:(i + 1) * col_size]

    # Position 1 contains the y block
    out_blocks[1][0] = y.reshape(-1, 1)


@task(block_files=COLLECTION_FILE_IN)
def _load_hstack_npy_block(block_files, start_col, end_col):
    if len(block_files) == 1:
        return np.load(block_files[0])[:, start_col:end_col]
    arrays = [np.load(block_files[0])[:, start_col:]]
    for file in block_files[1:-1]:
        arrays.append(np.load(file))
    arrays.append(np.load(block_files[-1])[:, :end_col])
    return np.concatenate(arrays, axis=1)
