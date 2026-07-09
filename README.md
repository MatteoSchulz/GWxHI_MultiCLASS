# GWHICcUP: Gravitational Wave x Neutral Hydrogen Cross-correlation Utility Pipeline

This repository contains the data generation pipeline for cross-correlating Neutral Hydrogen 21cm Intensity Mapping (HI) with Gravitational Wave (GW) distributions.
The outputs are designed to be ingested by a custom MontePython likelihood for Markov Chain Monte Carlo (MCMC) parameter estimation.

## Overview
This pipeline calculates theoretical angular power spectra ($C_\ell$) and covariance matrices using a modified version of MultiCLASS, and then adds the noise contributions to compute the correct simulated observed angula power spectra.
 It is fully vectorized and supports modular configurations for telescope parameters and GW network sensitivities (e.g., Einstein Telescope, Cosmic Explorer).

## Installation


### 1. Python Environment
To ensure reproducibility and avoid dependency conflicts, clone this repository and recreate the Conda environment used for this pipeline:

```bash
git clone https://github.com/MatteoSchulz/GWxHI_MultiCLASS.git
cd GWxHI_MultiCLASS
conda env create -f environment.yaml
conda activate  xC_multiCLASS
```


### 2. Patched MultiCLASS Installation
This pipeline requires a slight modification to the MultiCLASS Python wrappers (cclassy.pxd and classy.pyx) to correctly compute the array sizes for multi-tracer cross-correlations (selection_multitracing).

To keep this repository lightweight, we provide a patch file. Clone the base MultiCLASS repository into this folder, apply our patch, and compile:

```bash
git clone https://github.com/erikdelahaye/Multi_CLASS.git class_public
cd class_public

# Apply the custom multitracer patch
patch -p1 < ../patches/multiclass_multitracer.patch

# Compile CLASS and the Python wrapper
make clean
make -j
cd ..
```


## Repository Structure
```text
GWxHI_MultiCLASS/
│
├── data/                       
│   ├── chains/                 # Location for MontePython MCMC output chains
│   ├── plots/                  # Generated GetDist corner/triangle plots
│   └── ...                     # Generated .pkl matrices
├── notebooks/
│   └── dataset_creation.ipynb  # Main pipeline and execution notebook
│   └── mcmc_analysis.ipynb     # GetDist chain analysis and plotting notebook
├── patches/
│   └── multiclass_multitracer.patch # Required C/Cython modifications
├── environment.yaml            # Conda dependencies
├── .gitignore                  
└── README.md                   
```


## Pipeline Description
The notebook follows this workflow:
1) Generates the theoretical $dN/dz$ temperature brightness distributions for HI and source distributions for GWs.
2) Injects the computed theoretical distributions into MultiCLASS to compute theoretical $C_\ell$ spectra.
3) Calculates the physical foregrounds, instrumental thermal noise, and shot noise related to the different tracers and add them to the theoretical spectra to get the simulated observed power specta.
4) Computes the full Covariance and Inverse Covariance matrices required for the likelihood evaluation.
5) Saves .pkl files ready to be loaded directly into a MontePython likelihood for MCMC analysis.
6) A secondary notebook (mcmc_analysis.ipynb) uses GetDist to read the resulting MontePython chains, verify convergence, plot 1D/2D posterior contours of parameters.


## Output Files
The pipeline generates a suite of .pkl files containing the final matrices (e.g., _Cl.pkl, _Cov.pkl, and _InvCov.pkl).

*Note:* Output matrices and MCMC chains can be large and are explicitly excluded from version control via .gitignore.
You must run the notebook locally to generate the data files for your MCMC analysis.


## Acknowledgements
* Built using [MultiCLASS](https://github.com/erikdelahaye/Multi_CLASS)
* Data outputs are optimized for integration with the [MontePython](https://github.com/brinckmann/montepython_public) MCMC sampler.