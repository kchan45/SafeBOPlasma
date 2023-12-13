# Simulation Data Logging

This file provides details/notes on the significance of each data folder within
this subdirectory.

The data collected in this folder represent simulation data collected from
running Bayesian optimization (BO) to automatically tune a model predictive
controller (MPC) for the thermal dose delivery problem using the atmospheric
pressure plasma jet (APPJ).


## Data saved for paper
* 2023_03_22_12h45m59s: contains the data saved from a purely safe BO approach, with one feasible (saved in `initial_parameters.npy`) and one infeasible (saved in `initial_parameters_infeasiblex1.npy`) initial set of policy parameters 
* 2023_03_22_12h50m21s: contains the data saved from a purely relaxed safe (using UCB instead of LCB) BO approach, with one feasible (saved in `initial_parameters.npy`) and one infeasible (saved in `initial_parameters_infeasiblex1.npy`) initial set of policy parameters 
* 2023_03_24_17h50m07s: contains the data saved from a our method (SEBO), with one feasible (saved in `initial_parameters.npy`) and one infeasible (saved in `initial_parameters_infeasiblex1.npy`) initial set of policy parameters 

## Initial policy parameters for simulations
* `initial_parameters_feasible.npy` - initial set of feasible policy parameters
* `initial_parameters_infeasiblex1.npy` - initial set of infeasible policy parameters