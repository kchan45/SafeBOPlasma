# Safe BO for Plasma
**Safe Explorative Bayesian Optimization -- Towards Personalized Treatments in Plasma Medicine**

*Note: This repository is still undergoing verification and reproducibility tests*

*Note: The below are to be added upon publication and release of the proceedings online*
Please view our full paper in the 62nd IEEE Conference on Decision and Control (CDC 2023) Proceedings [link]()

If you find this repository or our work helpful, please cite as:
``` bibtex
\inproceedings{chan2023sebo,

}
```

## Abstract
This paper considers the problem of Bayesian optimization (BO) for systems with safety-critical constraints. Recent work has shown that a theoretically consistent way to account for constraints in BO is to relax the constraint functions such that the feasible region has a high probability of containing the global solution. However, by construction, these approaches are unable to ensure safe/feasible operation at every query, which is unacceptable in safety-critical applications. Alternatively, safe BO methods force the query points to remain in the interior of a partially-revealed safety region, which may result in unacceptable (and unquantified) performance losses. This paper presents a new safe BO method that avoids these performance losses by systematically incorporating potential performance gains from enlargement of the safety region. The proposed method avoids getting stuck at suboptimal points based on a potentially small initial safety region due to limited initial exploration of the safety boundary. The performance of the proposed method is demonstrated for safe control of a cold atmospheric plasma jet towards personalized plasma medicine.


## Implementation
To run this code on your own device, it is **required** to work within a virtual environment and modify the source code of the Ax package. Files that were modified are included within this repository under `src/external`.

First, you should create your own virtual environment and then install the required Python packages by using the command `pip3 install -r requirements.txt`.

Then, you should navigate to the virtual environment to find the source code for Ax. Typically, this is found via the following path `VENV/lib/PYTHON-VERSION/site-packages/ax`, where `VENV` is the name/path of the virtual environment and `PYTHON-VERSION` is the version of Python for the virtual environment. The Python version used in the original implementation is Python 3.8.10. There are 4 files that were modified to implement our proposed algorithm. Changes made to these files are detailed within the files using the following commented markers:
``` python
##################### BEGIN EDITS BY K.J. CHAN #####################
<modifications added here>
###################### END EDITS BY K.J. CHAN ######################
```
The 4 modified files with their original paths (copies of our modifications are located in `src/external`) are:
* `ax/core/experiment.py`
* `ax/modelbridge/base.py`
* `ax/modelbridge/torch.py`
* `ax/models/torch/botorch_modular/model.py`
It is important to replace the files exactly as how the paths are given, as there are several files within the Ax package that have the same name.

## TODO: Potential Future Feature
A potential future feature may modify the code to override the classes that were changed in the Ax repository. This planned feature will remove the need to replace the files given above.

## Test Case Replication
The main file to run simulations with Bayesian optimization is `src/run_sim.py`.

Additional details may be found within the `src` folder as well as in commentary within the files. Input-output data from our atmospheric pressure plasma jet (APPJ) and linear time-invariant (LTI) models identified from the data are given in the `model` folder. `model/data_view.ipynb` is Jupyter notebook that may be used to view the existing data. `model/systemID.m` is a MATLAB script to identify LTI models from our input-output data; it requires MATLAB (R2021a was used in this work) and the `n4sid` function.

## Rights and Collaborators
(c) 2023 Mesbah Lab

in collaboration with Joel A. Paulson and Ali Mesbah.

Questions regarding this code may be directed to kchan45 (at) berkeley.edu
