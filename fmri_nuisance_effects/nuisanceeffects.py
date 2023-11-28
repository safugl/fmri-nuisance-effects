"""NuisanceEffects class."""
import os
from typing import List, Union
import numpy as np
import nibabel as nib
from . import model_utils


class NuisanceEffects:
    """Using cross-validated regression for exploring nuisance effects."""

    def __init__(self,
                 tr: float,
                 cutoff_hz: float = 1/128,
                 num_splits: int = 5,
                 utilize_ols: bool = False,
                 replace_nans_with_zeros: bool = False
                 ) -> None:
        r"""Initialize NuisanceEffects.

        Parameters
        ----------
        tr : float
            Repetition time in seconds.
        cutoff_hz : float, optional
            Approximate cutoff of high pass filter in Hertz.
            The default is 1/128 .
        num_splits : int, optional
            Number of splits considered for the cross validation procedures.
            The default is 5.
        replace_nans_with_zeros : bool, optional
            How to deal with voxels with standard deviation equal to zero. When
            setting replace_nans_with_zeros = True, the estimated R2 maps will
            be set to zero at these voxels. Otherwise, they will be set to nan.
            The default is False.
        utilize_ols : bool, optional
            Specify if the ordinary least squares should be used rather than
            Ridge regression. When ``utilize_ols = True``, ordinary least
            squares is utilized and it is subsequently not possible to update
            the alphas parameter. When ``utilize_ols = False``, Ridge
            regression is used. The default is True

        Examples
        --------
        Use Nilearn to fetch a preprocessed data set. Import data from a single
        subject and define a single confound regressor CSF. Use cross-validated
        ordinary least squares to estimate R2 maps and visualize using Nilearn
        plotting utilities. Notice that mat_nuisance has to have two dimensions

            >>> import pandas as pd
            >>> import nilearn
            >>> import nilearn.datasets
            >>> import nilearn.plotting
            >>> from fmri_nuisance_effects import NuisanceEffects

            >>> data = nilearn.datasets.fetch_development_fmri(n_subjects=1)
            >>> file_epi = data.func[0]
            >>> df = pd.read_csv(data.confounds[0], sep='\t')
            >>> mat_nuisance = df['csf'].to_numpy()[:, None]

            >>> nuef = NuisanceEffects(tr=2., utilize_ols=True,
            >>>                        replace_nans_with_zeros=True)
            >>> v_estimate = nuef.fit(file_epi, mat_nuisance)
            >>> nilearn.plotting.plot_img(v_estimate,
            >>>                           threshold=0.,
            >>>                           vmin=0,
            >>>                           colorbar=True)



        Use data from the same example and now focus on a set of motion
        regressors as specified with the keys list. Use Ridge regression to
        estimate the model. This is done by setting utilize_ols = False

            >>> import pandas as pd
            >>> import nilearn
            >>> import nilearn.datasets
            >>> import nilearn.plotting
            >>> from fmri_nuisance_effects import NuisanceEffects

            >>> data = nilearn.datasets.fetch_development_fmri(n_subjects=1)
            >>> file_epi = data.func[0]
            >>> df = pd.read_csv(data.confounds[0], sep='\t')
            >>> keys = ['trans_x', 'trans_y', 'trans_z',
            >>>         'rot_x', 'rot_y', 'rot_z']
            >>> mat_nuisance = df[keys].to_numpy()

            >>> nuef = NuisanceEffects(tr=2., utilize_ols=False,
            >>>                        replace_nans_with_zeros=True)
            >>> v_estimate = nuef.fit(file_epi, mat_nuisance)
            >>> nilearn.plotting.plot_img(v_estimate,
            >>>                           threshold=0.,
            >>>                           vmin=0,
            >>>                           colorbar=True)

        When focusing on Ridge regression estimators, the default
        hyperparameters are between 10**-5 and 10**5. It is also possible to
        define another set of alpha parameters:

            >>> import numpy as np
            >>> nuef = NuisanceEffects(tr=2.,
            >>>                        utilize_ols=False,
            >>>                        replace_nans_with_zeros=True)
            >>> nuef.set_alphas(list(np.logspace(-10,10,50)))
            >>> v_estimate = nuef.fit(file_epi, mat_nuisance)
            >>> nilearn.plotting.plot_img(v_estimate,
            >>>                           threshold=0.,
            >>>                           vmin=0,
            >>>                           colorbar=True)

        """
        self.tr = tr
        self.cutoff_hz = cutoff_hz
        self.num_splits = num_splits
        self.replace_nans_with_zeros = replace_nans_with_zeros
        self.utilize_ols = utilize_ols

        if utilize_ols is True:
            self.alphas = [0.]
        else:
            self.alphas = list(10**np.arange(-5, 5, dtype=np.float64))

        self.D = None

    def fit(self,
            epi_img: Union[str, nib.Nifti1Image],
            mat_nuisance: np.ndarray,
            ) -> nib.Nifti1Image:
        """Fit and evaluate cross-validated Ridge regression model.


        Parameters
        ----------
        epi_img : Union[str,nib.Nifti1Image]
            Image to be loaded. The last dimension in the image
            should correspond to time, i.e., (..., num_volumes). This can
            either be a string pointing to an existing file or a valid nibabel
            image class.
        mat_nuisance : np.ndarray
            An 2D array containing nuisance features of interest. Each row
            should correspond to a given volume and each column should
            correspond to a given nuisance feature.

        Raises
        ------
        TypeError
            Raises TypeError if <epi_img> points to a non-existing file.
            Raises TypeError if <epi_img> is not a string.
            Raises TypeError if the last dimension in the image does not have
            the same number of rows as the matrix mat_nuisance.

        Returns
        -------
        Nifti1Image
            An image of cross-validated R2 for each voxel.

        """

        v = None

        if isinstance(epi_img, str):
            if not os.path.exists(epi_img):
                raise TypeError(f'{epi_img} is not existing.')

            # Load the epi file
            v = nib.load(epi_img)

        if isinstance(epi_img, nib.Nifti1Image):
            # Assume that epi_img is a valid image type
            v = epi_img
            # Consider checking for nib.all_image_classes

        if not v:
            raise TypeError(f'{epi_img} has not been properly imported.')

        v_dim = v.shape

        num_volumes = v.shape[-1]

        if num_volumes != mat_nuisance.shape[0]:
            raise TypeError(f'Nuisance features should {v.shape[-1]} rows.')

        # Import data
        Y = v.get_fdata(dtype=np.float64)

        # Flatten and transpose
        Y = Y.reshape(-1, v_dim[-1]).T

        # Prepare DCT coefficients
        self.D = (
            model_utils.dct_coefficients(num_volumes,
                                         self.cutoff_hz,
                                         self.tr)
        )

        # Correct for slow drifts
        Y = model_utils.ols_deflation(Y, self.D)
        X = model_utils.ols_deflation(mat_nuisance, self.D)

        R2 = (
            model_utils.cross_validate_ridge_model(X,
                                                   Y,
                                                   self.alphas,
                                                   self.num_splits))

        if self.replace_nans_with_zeros is True:
            R2 = np.nan_to_num(R2, copy=True, nan=0.0,
                               posinf=None, neginf=None)

        return nib.Nifti1Image(R2.reshape(v_dim[:-1]), v.affine)

    def set_alphas(self, val: List[float]) -> None:
        """Set initial alpha regularization hyperparameters"""
        if not isinstance(val, list):
            raise ValueError(f'{val} should be a list.')

        _check_elements_are_positive_floats(val)

        if self.utilize_ols is True:
            raise TypeError(
                'utilize_ols must be set to False when specifiying alphas')

        self.alphas = val


def _check_elements_are_positive_floats(vals: List) -> None:
    """Check if elements in a list are positive floats"""
    for v in vals:
        assert isinstance(v, float), f'{vals} should be floats.'
        assert v >= 0, f'{vals} should be positive.'
        assert v != np.inf, f'{vals} should be positive.'
