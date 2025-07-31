# Internship
This repository contains codes that I developed during my summer internship.

This repository is structured as followed:
```
|-- Synthetic_Dataset
|   |-- Synthetic_Dataset_Exponentially_Sinusoidal.ipynb
|   |-- Synthetic_Dataset_Nonlinear_Dynamical_System.ipynb
|   |-- Synthetic_Dataset_HyperparameterTuning.ipynb
|
|-- Real_Dataset
    |-- Real_Dataset_Model1_Bayesian.ipynb 
    |-- Real_Dataset_Model1_Grid.ipynb 
    |-- Real_Dataset_Model2_Bayesian.ipynb 
    |-- Real_Dataset_Model2_Grid.ipynb 
    |-- utils_model1.py
    |-- utils_model2.py
```

The repository includes two experiments — synthetic data and real data forecasting — each stored in its own folder named Synthetic_Dataset and Real_Dataset respectively. The real dataset used experiments is not available in this repository due to data privacy.

- ``Synthetic_Dataset`` contains all experiment releated to synthetic dataset. 
    - ``Synthetic_Dataset_Exponentially_Sinusoidal.ipynb`` contains modeling with Exponentially Sinusoidal Dataset.
    - ``Synthetic_Dataset_Nonlinear_Dynamical_System.ipynb`` contains modeling with Nonlinear Dynamical System Dataset.
    - ``Synthetic_Dataset_HyperparameterTuning.ipynb`` contains Pytorch modelling with hyperparameter optimization by using sklearn and skopts.

- ``Real_Dataset`` contains all experiment releated to real dataset. 
    - ``Real_Dataset_Model1_Bayesian.ipynb `` contains model 1 with hyperparameters tuning with Bayesian optimization using Optuna.
    - ``Real_Dataset_Model1_Grid.ipynb `` contains model 1 with hyperparameters tuning with Grid seach using Optuna.
    - ``Real_Dataset_Model2_Bayesian.ipynb `` contains model 2 with hyperparameters tuning with Bayesian optimization using Optuna.
    - ``Real_Dataset_Model2_Grid.ipynb `` contains model 2 with hyperparameters tuning with Grid seach using Optuna.
    - ``utils_model1.py`` contains functions used in Model 1.
    - ``utils_model2.py`` contains functions used in Model 2.

