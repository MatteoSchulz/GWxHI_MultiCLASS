# Gravitational Wave x Neutral Hydrogen Intensity Mapping Cross-correlation Pipeline

This repository contains the data generation and Fisher forecasting pipeline for cross-correlating 21cm Intensity Mapping (HI) with Gravitational Wave (GW) distributions.

## Overview
This pipeline calculates theoretical angular power spectra ($C_\ell$) and covariance matrices using a modified version of MultiCLASS.
It is fully vectorized and supports modular configurations for telescope parameters and GW network sensitivities (e.g., ET, CE).

## Installation

### 1. Python Environment
To ensure reproducibility, clone this repository and recreate the Conda environment:
```bash
git clone [https://github.com/YourUsername/21cm-GW-pipeline.git](https://github.com/YourUsername/21cm-GW-pipeline.git)
cd 21cm-GW-pipeline
conda env create -f environment.yml
conda activate your_env_name