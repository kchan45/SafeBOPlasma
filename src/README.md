# Source Code
This is considered the main working directory of this project.
The main file to run **simulations** is `run_sim.py`.

`run_sim.py` has been written to have the capability to run separate components of this work by modification of particular settings.

## Folder Descriptions
This section briefly describes the purpose of each of the *folders* nested within this directory.
* `config` - contains files that determine the configuration of the system, important for describing the problem formulation of the case study
* `external` - contains copies of the files of the Ax package (source code) that were modified to implement SEBO
* `figures` - contins the output figures generated for the paper, presentation, and a poster
* `KCutils` - contains the bulk of the helper code for this project; the main focus of the work is on a policy search routine using Bayesian optimization (BO), so the ancillary portions of the work (e.g., nonlinear MPC formulation, etc.) are placed in helper files.
* `trials` - contains folders that consist of the saved data after running `run_sim.py` and files that contain the initial policy parameterizations to initialize BO

## Other Files
This section briefly describes other code files that are located within this main working directory.
* `plot4paper.ipynb` - Jupyter notebook used to generate figures from the saved data for the paper
* `plot4poster.ipynb` - Jupyter notebook used to generate figures from the saved data for a poster and/or presentations
