import dislib as ds
from dislib.decomposition import PCA
import numpy as np
import time
import unittest


class MyPCATest(unittest.TestCase):

    def setUp(self):
        self._started_at = time.time()

    def tearDown(self):
        elapsed = time.time() - self._started_at
        print('{} ({}s)'.format(self.id(), round(elapsed, 2)))

    def test_fit_svd_old(self):
        """Tests PCA.fit()"""
        np.random.seed(8)
        n_samples = 150
        n_features = 5
        n_blobs = 3

        # Create normal clusters along a diagonal line
        data = []
        cov = np.eye(n_features)
        size = n_samples // n_blobs
        for i in range(n_blobs):
            mu = [i] * n_features
            data.append(np.random.multivariate_normal(mu, cov, size))
        data = np.vstack(data)
        bn, bm = 25, 2
        dataset = ds.array(x=data, block_size=(bn, bm))

        pca = PCA(method='svd')
        pca.fit(dataset)

        expected_eigvec = np.array([
            [0.46266375, 0.40183351, 0.50259945, 0.41394469, 0.44778976],
            [0.6548706, -0.63369901, 0.27008578, -0.15049713, -0.27198225],
            [-0.32039683, 0.24776797, 0.72090966, -0.18219517, -0.53202545],
            [-0.27511718, -0.36710543, 0.04871129, 0.85622265, -0.23249543],
            [-0.42278027, -0.4907138, 0.39033823, -0.19921873, 0.62322129]
        ])
        expected_eigval = np.array(
            [4.97145301, 1.3622795, 1.24328201, 0.96016518, 0.81758398]
        )

        comp = pca.components_.collect()
        var = pca.explained_variance_.collect()

        # use abs because signs do not really matter
        self.assertTrue(np.allclose(abs(comp), abs(expected_eigvec)))
        self.assertTrue(np.allclose(abs(var), abs(expected_eigval)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
