[![Python application](https://github.com/safugl/fmri-nuisance-effects/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/safugl/fmri-nuisance-effects/actions/workflows/python-app.yml)

# fmri-nuisance-effects
Most fMRI preprocessing pipelines outputs confound signals 
that can be incorporated in subsequent denoising steps or in general linear
models. This repository contains a tool that can be useful for exploring associations between voxel time courses and the confounds signals.

#  Procedure
The algorithm consists of the following steps:

1. Import confound signals and functional data.
2. Apply high-pass filters to confound signals and to functional data.
3. Divide data into non-overlapping, but connected chunks.
4. Fit regression models that treat the confound signals as predictors and each voxel time course as target variable. Fit models to data from on all-but-one chunk and use such models to predict voxel time courses in held-out chunks. This is repeated for all chunks.
5. Evaluate goodness-of-fit in terms of coefficient of determination (R2)

Step 4 can be realized with ordinary least squares or with Ridge regression. The latter can be useful when there are several (potentially correlated) confound signals. It is possible to specify the hyperparameters of the Ridge regression models. When multiple hyperparameters are specified, the algorithm will simply output the highest variance explained in each voxel. 


# Example 01
Data used for this example was obtained from the OpenNeuro database. Its accession number is ds000228.
Preprocessed data and confound signals are extracted from 155 participants using [Nilearn](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_development_fmri.html). Different sets of 
confound signals are incorporated into the described framework. The four following models are considered:

* Model 1: Including only random noise signals as regressors in a Ridge regression model. This serves as a control and the results should show no clear trends.
* Model 2: Including CSF confound signal as single regressor in a OLS regression model.
* Model 3: Including 6 motion regressors in a Ridge regression model.
* Model 4: Including 6 motion regressors, one framewise displacement coefficient, 6 aCompCor coefficients, CSF, and WM confounds in a Ridge regression model.

Models are evaluated using 5-fold cross-validation procedures. The models incorporate a high-pass filter
with an approximate cut-off of 1/128 Hz. R2 is extracted from each model and averaged across all 155 participants. The results from the analysis are shown below. Code is available [here](./examples/example01.py)

<img title="example01" alt="example_results02" src="./examples/example01.png">

# Example 02
This example uses data available on http://fcon_1000.projects.nitrc.org/indi/adhd200/index.html.
Preprocessed data and confound signals are extracted from 30 participants using [Nilearn](https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_adhd.html). Different sets of confound signals are incorporated into the described framework. The four following models are considered:

* Model 1: Including only random noise signals as regressors in a Ridge regression model. This serves as a control and the results should show no clear trends.
* Model 2: Including CSF confound signal as single regressor in a OLS regression model.
* Model 3: Including 6 motion regressors in a Ridge regression model.
* Model 4: Including 6 motion regressors, 6 CompCor coefficients, CSF, GM, WM and a global signal as confounds in a Ridge regression model.

Models are evaluated using 5-fold cross-validation procedures. The models incorporate a high-pass filter
with an approximate cut-off of 1/128 Hz. R2 is extracted from each model and averaged across all 30 participants. The results from the analysis are shown below. Code is available [here](./examples/example02.py)

<img title="example02" alt="example_results02" src="./examples/example02.png">

# Example 03
Data used for this example was obtained from the OpenNeuro database. Its accession number is ds000030. Preprocessed data and confound signals are extracted from 261 participants using [Nilearn](https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_ds000030_urls.html#nilearn.datasets.fetch_ds000030_urls). Different sets of confound signals are incorporated into the described framework. The four following models are considered:

* Model 1: Including only random noise signals as regressors in a Ridge regression model. This serves as a control and the results should show no clear trends.
* Model 2: Including 6 tCompCor coefficients in a Ridge regression model.
* Model 3: Including 6 motion regressors in a Ridge regression model.
* Model 4: Including 6 motion regressors, 6 aCompCor coefficients, 6 tCompCor coefficients, WM, global signal, stdVars, non-stdDVARS, vx-wisestdDVARS and a framewise displacement coefficient as confounds in a Ridge regression model.

Models are evaluated using 5-fold cross-validation procedures. The models incorporate a high-pass filter
with an approximate cut-off of 1/128 Hz. R2 is extracted from each model and averaged across all 261 participants. The results from the analysis are shown below. Code is available [here](./examples/example03.py)

<img title="example03" alt="example_results03" src="./examples/example03.png">

# Installation
Install fmri-nuisance-effects using one of the following approaches:

1. Install directly from git: `pip install git+https://github.com/safugl/fmri-nuisance-effects.git`
2. Download the repository and run `pip install <local project path>`

# Authors
Søren A. Fuglsang, Jens Hjortkjær and Hartwig R. Siebner

# References
Richardson, H., Lisandrelli, G., Riobueno-Naylor, A., & Saxe, R. (2018). Development of the social brain from age three to twelve years. Nature communications, 9(1), 1027. https://www.nature.com/articles/s41467-018-03399-2

Gorgolewski KJ, Durnez J and Poldrack RA. Preprocessed Consortium for Neuropsychiatric Phenomics dataset. F1000Research 2017, 6:1262
https://doi.org/10.12688/f1000research.11964.2

R.A. Poldrack, E. Congdon, W. Triplett, K.J. Gorgolewski, K.H. Karlsgodt, J.A. Mumford, F.W. Sabb, N.B. Freimer, E.D. London, T.D. Cannon, and R.M. Bilder. A phenome-wide examination of neural and cognitive function. Scientific Data, 3(1):160110, December 2016. URL: https://doi.org/10.1038/sdata.2016.110, doi:10.1038/sdata.2016.110.

Pierre Bellec, Carlton Chu, François Chouinard-Decorte, Yassine Benhajali, Daniel S. Margulies, R. Cameron Craddock (2017). The Neuro Bureau ADHD-200 Preprocessed repository. NeuroImage, 144, Part B, pp. 275 - 286. doi:10.1016/j.neuroimage.2016.06.034

Nitrc adhd resting-state dataset. ftp://www.nitrc.org/fcon_1000/htdocs/indi/adhd200/sites/ADHD200_40sub_preprocessed.tgz. Accessed: 2021-05-19.