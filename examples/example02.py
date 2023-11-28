"""Illustrate potential usage."""
# pylint: skip-file
import nilearn
import pandas as pd
from nilearn import datasets, image, plotting
import fmri_nuisance_effects
from fmri_nuisance_effects import NuisanceEffects
import nilearn.image
import nibabel as nib
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--subid", "-s", help="Subject identifier", default=0, type=int)
parser.add_argument("--outdir", "-o", help="Output directory")
args = parser.parse_args()

# Define subject id
subject_id = int(args.subid)

if args.outdir is None:
    raise TypeError('Output directory must be specified')


# Output data will be stored in this directory
output_directory = args.outdir
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Import the dataset
data = nilearn.datasets.fetch_adhd(n_subjects=30)


# Load the data from the particular participant
file_epi = data.func[subject_id]
df = pd.read_csv(data.confounds[subject_id], sep='\t')


"""
Model 1
--------------------
"""
np.random.seed(subject_id)
mat_nuisance = np.random.randn(nib.load(file_epi).shape[-1], 15)
nuef1 = NuisanceEffects(tr=2., utilize_ols=False,
                        replace_nans_with_zeros=False)
v_model1 = nuef1.fit(file_epi, mat_nuisance)
v_model1.to_filename(os.path.join(
    output_directory, 'sub-%0.3i_model1.nii.gz' % subject_id))
del nuef1, v_model1, mat_nuisance

"""
Model 2
--------------------
"""
mat_nuisance = df['csf'].to_numpy()[:, None]
nuef2 = NuisanceEffects(tr=2., utilize_ols=True,
                        replace_nans_with_zeros=False)
v_model2 = nuef2.fit(file_epi, mat_nuisance)
v_model2.to_filename(os.path.join(
    output_directory, 'sub-%0.3i_model2.nii.gz' % subject_id))
del nuef2, v_model2, mat_nuisance

"""
Model 3
--------------------
"""
keys = ['motion-pitch',
       'motion-roll', 'motion-yaw', 'motion-x', 'motion-y', 'motion-z']

mat_nuisance = df[keys].to_numpy()
nuef3 = NuisanceEffects(tr=2., utilize_ols=False,
                        replace_nans_with_zeros=False)
v_model3 = nuef3.fit(file_epi, mat_nuisance)
v_model3.to_filename(os.path.join(
    output_directory, 'sub-%0.3i_model3.nii.gz' % subject_id))
del nuef3, v_model3, mat_nuisance

"""
Model 4
--------------------
"""
keys = ['csf', 'wm', 'global', 'motion-pitch',
       'motion-roll', 'motion-yaw', 'motion-x', 'motion-y', 'motion-z', 'gm',
       'compcor1', 'compcor2', 'compcor3', 'compcor4', 'compcor5']

mat_nuisance = df[keys].to_numpy()
nuef4 = NuisanceEffects(tr=2., utilize_ols=False,
                        replace_nans_with_zeros=False)
v_model4 = nuef4.fit(file_epi, mat_nuisance)
v_model4.to_filename(os.path.join(
    output_directory, 'sub-%0.3i_model4.nii.gz' % subject_id))

del nuef4, v_model4, mat_nuisance
