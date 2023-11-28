"""Code to test fmri_nuisance_effects"""

import copy
from absl.testing import absltest
import numpy as np
import nibabel as nib
from . import NuisanceEffects
from . import model_utils


class NuisanceEffectsTest(absltest.TestCase):
    """Test cases for NuisanceEffects."""

    def test_compare_estimator_with_ols(self):
        """Compare estimates with OLS"""
        def matrix_include_intercept_terms(X):
            return np.c_[np.ones(X.shape[0]), copy.deepcopy(X)]

        np.random.seed(1)

        for num_obs in [100, 1000]:
            for num_voxels in [2, 100, 1000]:
                # Generate some data
                X_train = np.random.randn(num_obs, 20)*100 + 100
                Y_train = np.random.randn(num_obs, num_voxels)*100 + 10000
                X_val = np.random.randn(num_obs, 20)*100 + 100

                pred = (
                    model_utils.ridge_regression_estimator(X_train,
                                                           Y_train,
                                                           X_val, 0.)
                )

                D_train = matrix_include_intercept_terms(X_train)
                D_val = matrix_include_intercept_terms(X_val)
                betas = np.linalg.lstsq(D_train, Y_train, rcond=None)[0]

                np.testing.assert_almost_equal(D_val@betas, pred)

    def test_dct_means(self):
        """DCT coefficients"""

        for n_volumes in [100, 500, 1000]:
            for cutoff_hz in [1/100, 1/128, 1/256]:
                for sampling_rate in [1/2, 1/3]:

                    D = model_utils.dct_coefficients(
                        n_volumes, cutoff_hz, sampling_rate)
                    mu_target = np.zeros(D.shape[1])
                    mu_target[0] = 1./np.sqrt(n_volumes)
                    np.testing.assert_almost_equal(D.mean(axis=0), mu_target)

        target = np.array(
            [0.1, 0.002221350117087,
             -0.141351573335010,
             -0.006661858146884,
             0.141142293495411])
        np.testing.assert_almost_equal(
            model_utils.dct_coefficients(100, 1/100, 2.)[49, :], target)

        target = np.array(
            [0.1,
             -0.122843046842317,
             0.071989327396378,
             -0.002221350117087,
             -0.068130257963771])
        np.testing.assert_almost_equal(
            model_utils.dct_coefficients(100, 1/100, 2.)[83, :], target)

    def test_model_fit(self):
        """Fit model and check that output voxels match expectations."""
        np.random.seed(1)
        tr = 2
        n_vols = 200
        time = np.arange(n_vols)*tr
        X = np.sin(2*np.pi*time*1/50)
        Y = np.random.randn(10, n_vols)*0.1
        Y[-1] += X

        v_img = nib.Nifti1Image(Y, affine=np.eye(4))

        nuef = NuisanceEffects(tr=tr, utilize_ols=False,
                               replace_nans_with_zeros=True)

        v_estimate = nuef.fit(v_img, X[:, None])

        np.testing.assert_array_less(v_estimate.get_fdata()[:-1], 0.05)
        np.testing.assert_array_less(0.95, v_estimate.get_fdata()[-1])

        nuef = NuisanceEffects(tr=tr, utilize_ols=True,
                               replace_nans_with_zeros=True)

        v_estimate = nuef.fit(v_img, X[:, None])

        np.testing.assert_array_less(v_estimate.get_fdata()[:-1], 0.05)
        np.testing.assert_array_less(0.95, v_estimate.get_fdata()[-1])

    def test_ndims(self):
        """Test output shapes"""
        np.random.seed(1)

        for ndim in [0, 1, 100, 100]:
            Y = np.random.randn(ndim, 100)

            if Y.ndim == 0:
                Y = Y[None, :]

            v_img = nib.Nifti1Image(Y, affine=np.eye(4))
            nuef = NuisanceEffects(tr=2, utilize_ols=True,
                                   replace_nans_with_zeros=True)

            v_estimate = nuef.fit(v_img, np.random.randn(100, 10))
            np.testing.assert_array_equal(v_estimate.shape[0], ndim)

    def test_nans(self):
        """Explore what happens with all-zero voxels time courses."""

        def check_nans(v, mask):
            """Check that there are no nans or zeros where mask==0."""
            np.testing.assert_array_equal(
                np.any(v.get_fdata()[mask == 0] == 0), False
            )
            np.testing.assert_array_equal(
                np.any(np.isnan(v.get_fdata()[mask == 0])), False
            )

        np.random.seed(1)

        for ndim in [1, 100, 100]:
            Y = np.random.randn(ndim, 2, 2, 2, 100)
            X = np.random.randn(100, 10)

            # Inject zeros
            Y[0, 1, 1, 1, :] = 0

            M = np.zeros((ndim, 2, 2, 2))
            M[0, 1, 1, 1] = 1

            v_img = nib.Nifti1Image(Y, affine=np.eye(4))
            nuef = NuisanceEffects(tr=2, utilize_ols=True,
                                   replace_nans_with_zeros=True)
            v_estimate = nuef.fit(v_img, X)
            check_nans(v_estimate, M)
            np.testing.assert_approx_equal(
                v_estimate.get_fdata()[0, 1, 1, 1], 0.)

            nuef = NuisanceEffects(tr=2, utilize_ols=False,
                                   replace_nans_with_zeros=True)

            v_estimate = nuef.fit(v_img, X)
            check_nans(v_estimate, M)
            np.testing.assert_approx_equal(
                v_estimate.get_fdata()[0, 1, 1, 1], 0.)

            nuef = NuisanceEffects(tr=2, utilize_ols=False,
                                   replace_nans_with_zeros=False)

            v_estimate = nuef.fit(v_img, X)
            check_nans(v_estimate, M)
            assert np.isnan(v_estimate.get_fdata()[0, 1, 1, 1])

            nuef = NuisanceEffects(tr=2, utilize_ols=True,
                                   replace_nans_with_zeros=False)

            v_estimate = nuef.fit(v_img, X)
            check_nans(v_estimate, M)
            assert np.isnan(v_estimate.get_fdata()[0, 1, 1, 1])


if __name__ == '__main__':
    absltest.main()
