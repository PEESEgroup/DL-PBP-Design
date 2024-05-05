# DL-PBP-Design
Deep learning and optimization code to generate plastic binding peptides (PBPs)
## Overview
camsol_calculation/ contains a script to compute the CamSol value for a given peptide
data/ contains PepBD datasets for PE- and PS-binding used to train the deep learning model as well as a small sample dataset with 500 peptides and their predicted PepBD scores
examples/ contains jupyter notebooks to demonstrate how to run our model to generate peptides and how to perform SHAP analysis based on the trained score predictor
peptide_generators/ contains the MCTS generators described in the paper
score_predictors/ contains both trained LSTM models for PE and PS binding prediction
## Package Requirements
python 3.8.2
pandas 1.4.1
shap 0.41.0
scikit-learn 1.2.2
numpy 1.20.1
tensorflow 2.6.0

