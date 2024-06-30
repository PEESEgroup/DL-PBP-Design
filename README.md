# DL-PBP-Design
This repository contains deep learning and optimization code to generate plastic binding peptides (PBPs) for article "Discovering Plastic-Binding Peptides with Favorable Affinity, Water Solubility, and Binding Specificity Through Deep Learning and Biophysical Modeling".

## Overview
- `camsol_calculation/` contains a script to compute the CamSol value for a given peptide
- `data/` contains PepBD datasets for PE- and PS-binding that are used to train the deep learning models as well as a small sample dataset with 500 peptides designed by MCTS and their predicted PepBD scores for demonstration purpose
- `examples/` contains jupyter notebooks to demonstrate how to run our model to generate peptides and how to perform SHAP analysis based on the trained score predictors
- `peptide_generators/` contains the MCTS generators described in the paper
- `score_predictors/` contains both the trained LSTM models for PE- and PS-binding prediction

## System Requirements
### Operating System
This repository has been developed and tested on Windows 11, ensuring full compatibility with this operating system. However, the provided scripts and tools are designed to be platform-independent and should work seamlessly on other versions of Windows, as well as on Linux and macOS.

### Hardware Requirements
The light-weight models does not require any non-standard hardware and can be run on a typical desktop computer. The code has been tested on a system with the following specifications: 

- Intel i7-11800H CPU
- NVIDIA 3070 GPU
- 16GB of RAM

### Package Requirements
- python 3.8.2
- pandas 1.4.1
- shap 0.41.0
- scikit-learn 1.2.2
- numpy 1.20.1
- tensorflow 2.6.0

## Installation Guide
The source code is not distributed as a package, therefore, no installation is required. It is highly recommended to use [Anaconda](https://www.anaconda.com/) to manage the Python environment to run the code. To get started, please follow these steps:

1. **Clone the repository:** Clone the repository to a local directory using the following command:

```sh
git clone https://github.com/PEESEgroup/DL-PBP-Design.git
```

2. **Create a Conda environment:** Navigate to the cloned directory and create a Conda environment with the required packages as specified in the [Package Requirements](#package-requirements) section. 

```sh
conda create --name <environment_name>
# Install the packages listed in Package Requirements
```

3. **Activate the Conda environment and run the code:** With the Conda environment activated, you can now run the code as needed following the instructions in the [Demo](#demo) section.

```sh
conda activate <environment_name>
# Run the code
```

## Demo

Detailed examples on how to use our model to generate PBPs with high affinity to plastics (PE/PS), high solubility in water, and favorable binding specificity can be found in the `run_mcts.ipynb` notebook located in the `examples/` directory.

The notebook includes an example of generating PBPs that bind to PE without the "three tryptophan constraint" including a CamSol term with a scaling factor (SF) of 1.0. 10 PBPs with a PepBD score of around -50 are expected to be generated. The SF can be tuned to generate peptides with various solubility in water.

**To generate peptides with the "three tryptophan constraint":** Load `mcts_tryptophan_limits.py` in `peptide_generators/` instead of `mcts_camsol.py`

```
from mcts_tryptophan_limits import Node, mcts
```

**To generate peptides that preferentially bind to one plastic over another:** Load `mcts_competing_design.py` in `peptide_generators/` instead of `mcts_camsol.py`

```
from mcts_competing_design import Node, mcts
```

**To generate PBPs that bind to PS:** Replace the line

```
surrogate_model = keras.models.load_model('../score_predictors/pe/trained_model/')
```

with:

```
surrogate_model = keras.models.load_model('../score_predictors/ps/trained_model/')
```

Note: To design the peptides using a competing strategy as described in the paper, both models for PE and PS need to be loaded.

Detailed example on how to perform SHAP analysis on peptides can be found in the `shap_explainer.ipynb` notebook located in the `examples/` directory. This example is run on a small sample dataset with 500 PE-binding peptides designed by our algorithm. This dataset can be found in `data/`.

## Instructions for Use

To reproduce the results in our paper, please refer to the jupyter notebooks in the `examples/` directory and follow the instructions in the [Demo](#demo) section for any variants introduced in the paper. 

Please note that due to the stochastic nature of the MCTS algorithm, it is impossible to generate exactly identical peptides in multiple runs, but the same distributions of peptide properties (predicted PepBD score, CamSol score, sequence patterns,  and SHAP values of the amino acids) in the paper are expected to be observed with the same design parameters.

The time for generating one peptide with the default setting (2,000 MCTS iterations) with our [testing system](#hardware-requirements) is approximately 80 seconds. Our models can also run on CPU but the inference time would be longer.

## Citation

```
@article{,
author = {Tianhong Tan, Michael T. Bergman, Carol K. Hall, and Fengqi You},
title = {Discovering Plastic-Binding Peptides with Favorable Affinity, Water Solubility, and Binding Specificity Through Deep Learning and Biophysical Modeling},
journal = {},
volume = {},
number = {},
pages = {},
doi = {},
abstract = {}}
```
