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
_, urls = nilearn.datasets.fetch_ds000030_urls()

files_rest = nilearn.datasets.select_from_index(
    urls, inclusion_filters=['*task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz'])
files_confounds = nilearn.datasets.select_from_index(
    urls, inclusion_filters=['*task-rest_bold_confounds.tsv'])

data_dir, downloaded_files = nilearn.datasets.fetch_openneuro_dataset(
    urls=files_rest+files_confounds)

files_nii = [f for f in downloaded_files if f.find(
    'task-rest_bold_space-MNI152NLin2009cAsym_preproc.nii.gz') > 0]

# Load the data from the particular participant
file_epi = files_nii[subject_id]
df = pd.read_csv(files_nii[subject_id].replace(
    'space-MNI152NLin2009cAsym_preproc.nii.gz', 'confounds.tsv'), sep='\t')

bids_id = os.path.basename(file_epi).split('_')[0]

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
    output_directory, '%s_model1.nii.gz' % bids_id))
del nuef1, v_model1, mat_nuisance

"""
Model 2
--------------------
"""
keys = ['tCompCor00', 'tCompCor01',
        'tCompCor02', 'tCompCor03', 'tCompCor04', 'tCompCor05']
mat_nuisance = df[keys].to_numpy()

nuef2 = NuisanceEffects(tr=2., utilize_ols=True,
                        replace_nans_with_zeros=False)
v_model2 = nuef2.fit(file_epi, mat_nuisance)
v_model2.to_filename(os.path.join(
    output_directory, '%s_model2.nii.gz' % bids_id))
del nuef2, v_model2, mat_nuisance

"""
Model 3
--------------------
"""
keys = ['X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']

mat_nuisance = df[keys].to_numpy()
nuef3 = NuisanceEffects(tr=2., utilize_ols=False,
                        replace_nans_with_zeros=False)
v_model3 = nuef3.fit(file_epi, mat_nuisance)
v_model3.to_filename(os.path.join(
    output_directory, '%s_model3.nii.gz' % bids_id))
del nuef3, v_model3, mat_nuisance

"""
Model 4
--------------------
"""
keys = ['WhiteMatter', 'GlobalSignal', 'stdDVARS', 'non-stdDVARS',
        'vx-wisestdDVARS', 'FramewiseDisplacement', 'tCompCor00', 'tCompCor01',
        'tCompCor02', 'tCompCor03', 'tCompCor04', 'tCompCor05', 'aCompCor00',
        'aCompCor01', 'aCompCor02', 'aCompCor03', 'aCompCor04', 'aCompCor05',
        'X', 'Y', 'Z', 'RotX', 'RotY', 'RotZ']

mat_nuisance = df[keys].to_numpy()
# The first two volumes will be discarded during cross-validation (the total
# number of volumes cannot be divided into 5 equal sized chunks and the first
# two volumes are subsequently replaced). So we just replace nans with zero 
# here. One may consider discarding volumes instead.
mat_nuisance[0, np.where(np.isnan(mat_nuisance[0, :]))] = 0
nuef4 = NuisanceEffects(tr=2., utilize_ols=False,
                        replace_nans_with_zeros=False)
v_model4 = nuef4.fit(file_epi, mat_nuisance)
v_model4.to_filename(os.path.join(
    output_directory, '%s_model4.nii.gz' % bids_id))

del nuef4, v_model4, mat_nuisance
