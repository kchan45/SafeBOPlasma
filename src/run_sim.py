'''
main script to run safe Bayesian optimization for tuning controllers for 
atmospheric pressure plasma jet (APPJ) system in silico

Requirements:
* Python 3
* CasADi [https://web.casadi.org]
* PyTorch [https://pytorch.org]
* BoTorch [https://botorch.org] and Ax [https://ax.dev]
* Matplotlib [https://matplotlib.org] (for data visualization)

Copyright (c) 2023 Mesbah Lab. All Rights Reserved.

Author(s): Kimberly J Chan

This file is under the MIT License. A copy of this license is included in the
download of the entire code package (within the root folder of the package).
'''

# import Python packages
import sys
sys.dont_write_bytecode = True
import os
from datetime import datetime
# import 3rd party packages
import numpy as np
# import casadi as cas
import matplotlib.pyplot as plt
import torch
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.models import FixedNoiseGP
from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models
# from ax.modelbridge.factory import get_MOO_EHVI
# from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.models.torch.botorch_modular.list_surrogate import ListSurrogate
# import custom packages
# from config.reference_signal import myRef_CEM
from KCutils.controller import EconomicMPC
from KCutils.simulation import Simulation
from KCutils.bayesian_optimization import qUpperConfidenceBoundWithLogBarrier, qUpperConfidenceBoundWithLogBarrierRelaxed
import KCutils.plot_utils as pu

### user inputs/options
ts = 0.5 # sampling time, value given by model identification from APPJ (overwritten below)
Nsim = int(2*60/ts)   # Simulation horizon (for CEM dose delivery, this is the maximum simulation steps to consider)

## problem options
population_K = 0.5 # assumed "population" value for K
patient_K = 0.6 # individiual patient value for K
population_Tmax = 45.0 # assumed "population" temperature constraint
patient_Tmax = 45.0 # individual patient temperature constraint

## controller options
Np = 5 # prediction horizon of MPC
mpc_type = 'economic' # MPC type, 'economic' for standard economic MPC with no uncertainty estimation

## options related to auto-tuning using Bayesian optimization
only_A = False # option to only use the model parameters for the state evolution matrix, A, in the Bayesian optimization
A_plus_K = True
constrained_obj = False # option to use a constrained objective (outcome constrained) instead of unconstrained optimization
auto_tune = True     # option to perform auto-tuning (True to perform policy search, False otherwise (if False, script stops after initial closed loop simulation has been tested))
random_search = False # option to use Random Search to tune (True for random search, False for Bayesian optimization)
MOO = True # option to perform a multi-objective (Bayesian) optimization routine (True for multi-objective, False for single objective)
n_objs = 2
af = 'UpperConfidenceBound' # acqusition function to use if not custom nor multi-objective (options: {"ExpectedImprovement", "UpperConfidenceBound"})
custom_af = True # option to use custom acquisition function(s) defined in bayesian_optimization.py (True to use custom, False otherwise)
ucb_beta = 1e-1 # beta value (confidence interval weight) for the general UCB/LCB acquisition function
tau = 1e3 # barrier weight if using qUpperConfidenceBoundWithLogBarrier acquisition function
barrier_beta = 1e-1 # beta value (confidence interval weight) for the barrier term of the acquisition function (assumes all beta values are equal)
n_mc = 1 # number of Monte Carlo runs of the full Bayesian optimization routine
n_random = 0 # number of random SOBOL parameterizations to try (for initializing the BO routine)
n_bo_iter = 30 # number of iterations to update the policy
Nreps_per_iteration = 3 # number of repetitions of closed-loop runs using the same BO-suggested parameters per iteration
load_addl_data = True
addl_data_file = './trials/initial_parameters_infeasiblex1.npy'
save_intermediate = False

## options related to visualization of the simulations, note that the entire script will run before displaying the figures
plot_initial_mpc = True # option to plot the trajectories of just the MPC with no mismatch between the true and assumed dose parameter (True to plot, False otherwise)
plot_individual = True # option to plot all of the individual trajectories encountered through the BO
fig_objs = {}
Fontsize = 14 # default font size for plots
Lwidth = 3 # default line width for plots

### SETUP: do not edit below this line, otherwise the reproducibility of the results is not guaranteed
## setup for establishing plotting defaults
if only_A and A_plus_K:
    print('[WARNING] `only_A` and `A_plus_K` were both specified, proceeding with the `only_A` option!')
lines = {'linewidth' : Lwidth}
plt.rc('lines', **lines)
font = {'family' : 'serif',
        'serif'  : 'Times',
        'size'   : Fontsize}
plt.rc('font', **font)  # pass in the font dict as kwargs

## setup for establishing PyTorch defaults
tkwargs = {
    'dtype': torch.double,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

################################################################################
# SIMULATION SETUP
################################################################################
if mpc_type == 'economic':
    from config.economic import get_prob_info, modify_prob_info
else:
    print('Unsupported controller type!')

date = datetime.now().strftime('%Y_%m_%d_%H'+'h%M'+'m%S'+'s')

# get problem information. Problem information is loaded from the
# config.multistage file. This file provides problem-specific information, which
# may include system size and bounds, functions for evaluating the physical
# system, controller parameters, etc.
prob_info = get_prob_info(Kcem=population_K, Tmax=population_Tmax, Np=Np)
# a second problem info is created to establish a mismatch between the assumed
# dose parameter value (derived from population data) and the "true" dose
# parameter value of a particular "patient"
prob_info2 = get_prob_info(Kcem=patient_K, Tmax=patient_Tmax, Np=Np, no_mismatch=True)

# extracting some data for preliminary plots after running simulation
ts = prob_info['ts']
xss = prob_info['xss']
uss = prob_info['uss']
xssp = prob_info['xssp']
ussp = prob_info['ussp']
x_max = prob_info['x_max']
u_min = prob_info['u_min']
u_max = prob_info['u_max']

############################ create  mpc #######################################
multistage = False
if mpc_type == 'economic':
    c = EconomicMPC(prob_info) # controller designed with plant-model mismatch
    c1 = EconomicMPC(prob_info2) # controller designed without plant-model mismatch
else:
    print('Unsupported MPC type!')
    quit()

_, _, _ = c.get_mpc()
_, _, _ = c1.get_mpc()

# run an open loop simulation to test
res, feas = c.solve_mpc()
# print(res)
# print(feas)

## run closed loop simulation using MPC
sim = Simulation(Nsim)
sim.load_prob_info(prob_info2)
sim_data = sim.run_closed_loop(c, CEM=True, multistage=multistage)
print('Simulation Data Keys: ', [*sim_data])
sim_data1 = sim.run_closed_loop(c1, CEM=True, multistage=multistage)

Yrefsim = np.ravel(sim_data['Yrefsim'])
CEMsim = np.ravel(sim_data['CEMsim'])
st = sim_data['CEM_stop_time']
print('Stop Time: ', st*ts)
ctime = sim_data['ctime'][:st]
Ysim = sim_data['Ysim']
print('Total Runtime: ', np.sum(ctime))
print('Average Runtime: ', np.mean(ctime))

CEMplot = CEMsim[:st+1]
Yrefplot = Yrefsim[:st+1]
Tplot = Ysim[0,:st+1] + xssp[0]

Yrefsim1 = np.ravel(sim_data1['Yrefsim'])
CEMsim1 = np.ravel(sim_data1['CEMsim'])
st1 = sim_data1['CEM_stop_time']
Ysim1 = sim_data1['Ysim']
CEMplot1 = CEMsim1[:st1+1]
Yrefplot1 = Yrefsim1[:st1+1]
Tplot1 = Ysim1[0,:st1+1] + xssp[0]

if plot_initial_mpc:
    ## plot outputs
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(221)
    ax.axhline(Yrefplot[0], color='k', linestyle='--')
    ax.plot(np.arange(len(CEMplot))*ts, CEMplot, label='Mismatch')
    ax.plot(np.arange(len(CEMplot1))*ts, CEMplot1, label='No Mismatch')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CEM (min)')
    ax.legend(loc='lower right')

    ax = fig.add_subplot(222)
    ax.axhline(x_max[0]+xss[0], color='r', linestyle='--')
    ax.plot(np.arange(len(Tplot))*ts, Tplot, label='Mismatch')
    ax.plot(np.arange(len(Tplot1))*ts, Tplot1, label='No Mismatch')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Surface Temperature (^\circ C)')

    ## plot inputs
    ax = fig.add_subplot(223)
    ax.step(np.arange(st-1)*ts, sim_data['Usim'][0,:(st-1)]+ussp[0])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')

    ax = fig.add_subplot(224)
    ax.step(np.arange(st-1)*ts, sim_data['Usim'][1,:(st-1)]+ussp[1])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flow Rate (SLM)')
    plt.draw()

# user input to continue the script. The purpose of this user input is to ensure
# the initial approximate policy is sufficient to continue to optimization.
continue_with_bo = input('Continue with BO? [Y/n]\n')
if continue_with_bo in ['Y', 'y']:
    ################################################################################
    # CONTROLLER TUNING with BAYESIAN OPTIMIZATION
    ################################################################################
    def evaluate(parameters, prob_info=None, save_prefix='', Nreps=1, no_noise=False, iter=0):
        # define a function which encapsulates a closed-loop run of the system;
        # new candidate parameter(s) are the necessary inputs to this function,
        # while all other options are specific to the problem

        global only_A, A_plus_K, n_objs, prob_info2
        # for controller tuning, we define the parameter(s) to be tuned as the
        # prediction horizon and/or model parameters (including parameters for
        # the CEM model)
        nx = prob_info['nx']
        nu = prob_info['nu']
        if only_A:
            Np = prob_info['Np']
            _,B,_ = prob_info['model']
            Kcem = prob_info['Kcem']
            A = np.asarray([parameters[f'A{i+1}{j+1}'] for i in range(nx) for j in range(nx)]).reshape(nx,nx)
        elif A_plus_K:
            Np = prob_info['Np']
            _,B,_ = prob_info['model']
            A = np.asarray([parameters[f'A{i+1}{j+1}'] for i in range(nx) for j in range(nx)]).reshape(nx,nx)
            Kcem = parameters['Kcem']
        else:
            Kcem = prob_info['Kcem']
            Np = parameters['Np']
            A = np.asarray([parameters[f'A{i+1}{j+1}'] for i in range(nx) for j in range(nx)]).reshape(nx,nx)
            B = np.asarray([parameters[f'B{i+1}{j+1}'] for i in range(nx) for j in range(nu)]).reshape(nx,nu)

        prob_info = modify_prob_info(prob_info, Np=Np, A=A, B=B, Kcem=Kcem, save_file=f'./trials/{date}/parameters/{save_prefix}_prob_info.npy')

        # prob_info has been updated with new parameters suggested by BO, thus
        # a new controller should be created using these new parameters
        if mpc_type == 'economic':
            c_new = EconomicMPC(prob_info)
        c_new.get_mpc()

        sim = Simulation(Nsim)
        sim.load_prob_info(prob_info2)

        # repeat the simulation Nreps
        obj_arrays = [np.zeros((Nreps,)) for _ in range(n_objs)]
        d = {}
        for i in range(Nreps):
            sim_data = sim.run_closed_loop(c_new, CEM=True, rand_seed2=i*987)

            # extract closed loop data and compute objectives/constraints
            ts = prob_info2['ts']
            st = sim_data['CEM_stop_time']
            CEM = np.ravel(sim_data['CEMsim'][:,:st])
            Ref = np.ravel(sim_data['Yrefsim'][:,:st])
            T = np.ravel(sim_data['Ysim'][0,:st])
            Tmax = prob_info2['x_max'][0]

            CEMobj = -np.sum((CEM-Ref)**2)
            Tobj = -np.sum((T[T>Tmax] - Tmax)**2)
            t_k = np.arange(st)*ts # time vector for discrete sampling time of the data
            chkpt = np.arange(0.25, 1.25, 0.25) # checkpoint percentages to check the path constraint
            T_chkpt = np.interp(chkpt*(st*ts), t_k, T)

            Tobj25 = Tmax - T_chkpt[0]
            Tobj50 = Tmax - T_chkpt[1]
            Tobj75 = Tmax - T_chkpt[2]
            Tobj100 = Tmax - T_chkpt[3]

            s = {}
            s['sim_data'] = sim_data
            if only_A:
                s['params'] = [A]
            elif A_plus_K:
                s['params'] = [A, Kcem]
            else:
                s['params'] = [Np, A, B]
            s['obj_val'] = {'CEMobj': CEMobj,
                            'Tobj': Tobj,
                            'Tobj25': Tobj25,
                            'Tobj50': Tobj50,
                            'Tobj75': Tobj75,
                            'Tobj100': Tobj100}
            d[f'rep{i}'] = s

            if n_objs == 1:
                obj_arrays[0][i] = CEMobj
            elif n_objs == 2:
                obj_arrays[0][i] = CEMobj
                obj_arrays[1][i] = Tobj
            else:
                obj_arrays[0][i] = CEMobj
                obj_arrays[1][i] = Tobj25
                obj_arrays[2][i] = Tobj50
                obj_arrays[3][i] = Tobj75
                obj_arrays[4][i] = Tobj100

            if plot_individual:
                global fig_objs, n_bo_iter, n_add
                fig_objs = pu.plot_data_from_dict_sim(sim_data, prob_info, CEM=True, fig_objs=fig_objs, label=f'iter{iter}rep{i}', alpha=(iter+1)/(n_bo_iter+n_add+2))
        # closed loop data is saved as a dictionary of dictionaries (keys are
        # 'rep{#}'). The nested dictionary contains the sim_data, the parameters
        # by BO, and the objective value(s).
        np.save(f'./trials/{date}/cl_data/{save_prefix}_sim_data.npy', d, allow_pickle=True)

        # gather the data and return the appropriate dictionary of outputs expected by Ax
        obj_names = [f'obj{i+1}' for i in range(n_objs)]
        if Nreps > 1:
            # compute stats for multiple repetitions
            obj_means = [np.mean(obj_arrays[i]) for i in range(n_objs)]
            obj_stes = [np.std(obj_arrays[i])/np.sqrt(Nreps) for i in range(n_objs)]

            objs = [torch.tensor(obj_means[i], **tkwargs) for i in range(n_objs)]
            objs_ste = [torch.tensor(obj_stes[i], **tkwargs) for i in range(n_objs)]

            obj_tuples = [(objs[i].item(), objs_ste[i].item()) for i in range(n_objs)]
            out = dict(zip(obj_names, obj_tuples))

        elif Nreps == 1 and no_noise:
            # no noise (if function is deterministic)
            objs = [torch.tensor(np.ravel(obj_arrays[i]), **tkwargs) for i in range(n_objs)]
            obj_tuples = [(objs[i].item(), 0.0) for i in range(n_objs)]
            out = dict(zip(obj_names, obj_tuples))

        else:
            # otherwise ask BO algorithm to estimate the noise
            objs = [torch.tensor(np.ravel(obj_arrays[i]), **tkwargs) for i in range(n_objs)]
            obj_tuples = [(objs[i].item(), None) for i in range(n_objs)]
            out = dict(zip(obj_names, obj_tuples))

        return out

    nx = prob_info['nx']
    nu = prob_info['nu']
    if only_A:
        n_params = nx*nx
    elif A_plus_K:
        n_params = nx*nx + 1
    else:
        n_params = 1 + nx*nx + nx*nu

    print(f'Number of Parameters: {n_params}')
    # tuning parameters: Np, A, B, Kcem
    A,B,_ = prob_info['model']
    Np = np.asarray(prob_info['Np'], dtype=np.intc)
    Kcem = np.asarray(prob_info['Kcem'], dtype=np.float64)
    if only_A:
        initial_parameters = A.flatten()
    elif A_plus_K:
        initial_parameters = np.concatenate((A.flatten(), Kcem.reshape(-1,)))
    else:
        initial_parameters = np.concatenate((Np.reshape(-1,), A.flatten(), B.flatten()))
    assert len(initial_parameters) == n_params
    # bounds for parameters are
    # Np: [3,20]
    # model parameters for A, B: geometric bounds
    # Kcem: geometric bounds
    r = 2.0
    if only_A:
        lower_bounds = [*[1/r * initial_parameters[i] if initial_parameters[i]>0 else r * initial_parameters[i] for i in range(n_params)]]
        upper_bounds = [*[r * initial_parameters[i] if initial_parameters[i]>0 else 1/r * initial_parameters[i] for i in range(n_params)]]

        param_keys = [*[f'A{i+1}{j+1}' for i in range(nx) for j in range(nx)]]
        initial_parameters = [*[float(initial_parameters[i]) for i in range(n_params)]]
    elif A_plus_K:
        lower_bounds = [*[1/r * initial_parameters[i] if initial_parameters[i]>0 else r * initial_parameters[i] for i in range(n_params)]]
        upper_bounds = [*[r * initial_parameters[i] if initial_parameters[i]>0 else 1/r * initial_parameters[i] for i in range(n_params)]]

        param_keys = [*[f'A{i+1}{j+1}' for i in range(nx) for j in range(nx)],
                      'Kcem']
        initial_parameters = [*[float(initial_parameters[i]) for i in range(n_params)]]
    else:
        lower_bounds = [3, *[1/r * initial_parameters[i] if initial_parameters[i]>0 else r * initial_parameters[i] for i in range(1,n_params)]]
        upper_bounds = [20, *[r * initial_parameters[i] if initial_parameters[i]>0 else 1/r * initial_parameters[i] for i in range(1,n_params)]]

        param_keys = ['Np',
                      *[f'A{i+1}{j+1}' for i in range(nx) for j in range(nx)],
                      *[f'B{i+1}{j+1}' for i in range(nx) for j in range(nu)],
        ]
        initial_parameters = [int(initial_parameters[0]), *[float(initial_parameters[i]) for i in range(1,n_params)]]
    initial_parameters = dict(zip(param_keys, initial_parameters))

    if auto_tune:
        # create directories for saving files
        os.makedirs(f'./trials/{date}', exist_ok=True)
        os.makedirs(f'./trials/{date}/parameters', exist_ok=True)
        os.makedirs(f'./trials/{date}/cl_data', exist_ok=True)

        ## use Bayesian optimization to adjust policy
        hv_mc = []
        for n in range(n_mc):
            ## setup experiments for Bayesian optimization
            bo_save_filepath = f'./trials/{date}/ax_client_snapshot{n}.json'

            # set a random seed/state for repeatability
            rs = int(42*(n+1234))

            if random_search:
                gs = GenerationStrategy(steps=[GenerationStep(Models.SOBOL, num_trials=-1)])
            else:
                model_kwargs = {}
                if custom_af:
                    # For 'BOTORCH_MODULAR', we pass in kwargs to specify what
                    # surrogate or acquisition function to use.
                    # 'acquisition_options' specifies the set of additional
                    # arguments to pass into the input constructor.
                    bo_model = Models.BOTORCH_MODULAR
                    if n_objs == 2:
                        surrogate = ListSurrogate(botorch_submodel_class_per_outcome={
                                                    'obj1': FixedNoiseGP,
                                                    'obj2': FixedNoiseGP,
                                                },
                                                submodel_options={},
                                            )
                    elif n_objs == 5:
                        obj_names = [f'obj{i+1}' for i in range(n_objs)]
                        submodels = dict(zip(obj_names, [FixedNoiseGP for _ in range(n_objs)]))
                        surrogate = ListSurrogate(botorch_submodel_class_per_outcome=submodels, submodel_options={})

                    model_kwargs = {'surrogate': surrogate,
                                    'botorch_acqf_class': qUpperConfidenceBoundWithLogBarrierRelaxed,
                                    'acquisition_options': {'ucb_beta': ucb_beta, 'tau': tau, 'barrier_beta': barrier_beta},
                                    # 'secondary_botorch_acqf_class':qUpperConfidenceBoundWithLogBarrier,
                                    # 'secondary_acquisition_options': {'ucb_beta': ucb_beta, 'tau': tau, 'barrier_beta': barrier_beta},
                                    'safety_heuristic': 0.5,
                    }
                    print('Creating Ax Model with the following kwargs: ', model_kwargs)
                elif MOO:
                    bo_model = Models.MOO
                else:
                    if af == 'ExpectedImprovement':
                        bo_model = Models.GPEI
                    elif af == 'UpperConfidenceBound':
                        bo_model = Models.BOTORCH_MODULAR
                        model_kwargs = {'botorch_acqf_class': qUpperConfidenceBound}

                rand_gen_step = []
                if n_random > 0:
                    # 1. Initialization step (does not require pre-exiting data and
                    # is well-suited for initial sampling of the search space)
                    rand_gen_step = [GenerationStep(
                                        model = Models.SOBOL,
                                        num_trials = n_random,
                                        max_parallelism = 5,
                                        min_trials_observed = 1,
                                        model_kwargs = {'seed': rs},
                                        model_gen_kwargs = {},
                                    )]
                gs = GenerationStrategy(
                    steps = [*rand_gen_step,
                    # 2. Bayesian optimization step (requires data obtained from
                    # previous phase and learns from all data available at the time
                    # of each new candidate generation call)
                    GenerationStep(
                        model = bo_model,
                        num_trials = -1,
                        model_kwargs = model_kwargs,
                    ),
                    ]
                )

            outcome_constraint = []
            tracking_metric_names = []
            if MOO:
                if n_objs == 2:
                    objectives = {
                            'obj1': ObjectiveProperties(minimize=False, threshold=120.0),
                            'obj2': ObjectiveProperties(minimize=False, threshold=20.0),
                            }
                elif n_objs == 5:
                    obj_names = [f'obj{i+1}' for i in range(n_objs)]
                    # obj_thresholds = [-120.0, 1.0, 1.0, 1.0, 1.0]
                    # assert len(obj_thresholds) == n_objs
                    objectives = dict(zip(obj_names, [ObjectiveProperties(minimize=False) for _ in range(n_objs)]))#, threshold=thresh) for thresh in obj_thresholds]))
                else:
                    print('[Error] Invalid number of objectives! Exiting program ...\n')
                    quit()
            elif constrained_obj:
                objectives = {'obj1': ObjectiveProperties(minimize=False)}
                outcome_constraint = ['obj2 >= 0.0']
                tracking_metric_names = ['obj2']
            else:
                objectives = {'obj1': ObjectiveProperties(minimize=False)}
                tracking_metric_names = ['obj2']

            if only_A:
                parametersA = [
                    {'name': f'A{i+1}{j+1}',
                     'type': 'range',
                     'bounds': [float(lower_bounds[i*nx+j]), float(upper_bounds[i*nx+j])],
                     'value_type': 'float',
                    } for i in range(nx) for j in range(nx)
                ]
                parameters = [*parametersA]
            elif A_plus_K:
                parametersA = [
                    {'name': f'A{i+1}{j+1}',
                     'type': 'range',
                     'bounds': [float(lower_bounds[i*nx+j]), float(upper_bounds[i*nx+j])],
                     'value_type': 'float',
                    } for i in range(nx) for j in range(nx)
                ]
                parameterK = [
                    {'name': 'Kcem',
                     'type': 'range',
                     'bounds': [lower_bounds[-1], upper_bounds[-1]],
                     'value_type': 'float',
                     }
                ]
                parameters = [*parametersA, *parameterK]
            else:
                parameterNp = [
                    {'name': 'Np',
                     'type': 'range',
                     'bounds': [lower_bounds[0], upper_bounds[0]],
                     'value_type': 'int',
                     }
                ]
                parametersA = [
                    {'name': f'A{i+1}{j+1}',
                     'type': 'range',
                     'bounds': [float(lower_bounds[i*nx+j+1]), float(upper_bounds[i*nx+j+1])],
                     'value_type': 'float',
                    } for i in range(nx) for j in range(nx)
                ]
                parametersB = [
                    {'name': f'B{i+1}{j+1}',
                     'type': 'range',
                     'bounds': [float(lower_bounds[i*nu+j+nx*nx+1]), float(upper_bounds[i*nu+j+nx*nx+1])],
                     'value_type': 'float',
                    } for i in range(nx) for j in range(nu)
                ]
                parameters = [*parameterNp, *parametersA, *parametersB]

            ax_client = AxClient(random_seed=42, generation_strategy=gs)
            ax_client.create_experiment(
                name = f'bo_auto_tuning_trial{n}',
                parameters = parameters,
                objectives = objectives,
                outcome_constraints = outcome_constraint,
                overwrite_existing_experiment = True,
                is_test = False,
                tracking_metric_names = tracking_metric_names,
            )

            # attach initial trial/data
            n_add = 0
            parameters, trial_index = ax_client.attach_trial(parameters=initial_parameters)
            raw_data = evaluate(parameters,
                                prob_info=prob_info,
                                Nreps=Nreps_per_iteration,
                                save_prefix=f'trial{n}_iter{trial_index}')
            ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
            ax_client.save_to_json_file(bo_save_filepath)
            if save_intermediate:
                inter_save_filepath = f'./trials/{date}/ax_client_snapshot{n}_iter{trial_index}.json'
                ax_client.save_to_json_file(inter_save_filepath)
            print(f'@@@-----Iteration {trial_index} complete...-----@@@\n\n')

            if load_addl_data:
                addl_data = np.load(addl_data_file)
                n_addl_data = addl_data.shape[0]
                for n_add in range(n_addl_data):
                    parameters = [*[float(addl_data[n_add,i]) for i in range(n_params)]]
                    parameters = dict(zip(param_keys, parameters))
                    parameters, trial_index = ax_client.attach_trial(parameters=parameters)
                    raw_data = evaluate(parameters,
                                        prob_info=prob_info,
                                        Nreps=Nreps_per_iteration,
                                        save_prefix=f'trial{n}_iter{trial_index}')
                    ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
                    ax_client.save_to_json_file(bo_save_filepath)
                    if save_intermediate:
                        inter_save_filepath = f'./trials/{date}/ax_client_snapshot{n}_iter{trial_index}.json'
                        ax_client.save_to_json_file(inter_save_filepath)
                    print(f'@@@-----Iteration {trial_index} complete...-----@@@\n\n')

            # main BO loop
            for i in range(n_add, n_bo_iter+n_add):
                parameters, trial_index = ax_client.get_next_trial()

                raw_data = evaluate(parameters,
                                    prob_info=prob_info,
                                    Nreps=Nreps_per_iteration,
                                    save_prefix=f'trial{n}_iter{trial_index}', iter=trial_index)
                ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
                ax_client.save_to_json_file(bo_save_filepath)
                if save_intermediate:
                    inter_save_filepath = f'./trials/{date}/ax_client_snapshot{n}_iter{trial_index}.json'
                    ax_client.save_to_json_file(inter_save_filepath)
                print(f'@@@-----Iteration {trial_index} complete...-----@@@\n\n')

            if not MOO:
                best_parameters, values = ax_client.get_best_parameters()
                print(best_parameters)

                if only_A:
                    _,B,_ = prob_info['model']
                    Np = prob_info['Np']
                    Kcem = prob_info['Kcem']
                    A = np.asarray([best_parameters[f'A{i+1}{j+1}'] for i in range(nx) for j in range(nx)]).reshape(nx,nx)
                elif A_plus_K:
                    _,B,_ = prob_info['model']
                    Np = prob_info['Np']
                    A = np.asarray([best_parameters[f'A{i+1}{j+1}'] for i in range(nx) for j in range(nx)]).reshape(nx,nx)
                    Kcem = best_parameters['Kcem']
                else:
                    Np = best_parameters['Np']
                    nx = prob_info['nx']
                    nu = prob_info['nu']
                    Kcem = prob_info['Kcem']
                    A = np.asarray([best_parameters[f'A{i+1}{j+1}'] for i in range(nx) for j in range(nx)]).reshape(nx,nx)
                    B = np.asarray([best_parameters[f'B{i+1}{j+1}'] for i in range(nx) for j in range(nu)]).reshape(nx,nu)
                prob_info = modify_prob_info(prob_info, Np=Np, A=A, B=B, Kcem=Kcem, save_file=f'./trials/{date}/parameters/best_prob_info.npy')

                # prob_info has been updated with new parameters suggested by BO
                if mpc_type == 'economic':
                    c_new = EconomicMPC(prob_info)
                c_new.get_mpc()

                # patient has different Kcem value compared to what was trained initially
                sim = Simulation(Nsim)
                sim.load_prob_info(prob_info2)

                sim_data2 = sim.run_closed_loop(c_new, CEM=True)
                st2 = sim_data2['CEM_stop_time']
                CEMplot2 = np.ravel(sim_data2['CEMsim'][:,:st2])
                Tplot2 = sim_data2['Ysim'][0,:st2] + xssp[0]

                len_list = [st, st2]

                fig = plt.figure(figsize=(10,5))
                ax = fig.add_subplot(221)
                ax.axhline(Yrefplot[0], color='r', linestyle='--', label='Target Reference')
                ax.plot(np.arange(len(CEMplot))*ts, CEMplot, label='Nominal Tuning')
                # ax.plot(np.arange(len(CEMplot1))*ts, CEMplot1, '--', label='DNN (before BO)')
                ax.plot(np.arange(len(CEMplot2))*ts, CEMplot2, ':', label='Tuning after BO')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('CEM')
                ax.legend(loc='lower right')

                ax = fig.add_subplot(222)
                ax.axhline(x_max[0]+xssp[0], color='r', linestyle='--', label='Constraint')
                ax.plot(np.arange(len(Tplot))*ts, Tplot, label='Nominal Tuning')
                # ax.plot(np.arange(len(Tplot1))*ts, Tplot1, '--', label='DNN (before BO)')
                ax.plot(np.arange(len(Tplot2))*ts, Tplot2, '--', label='Tuning after BO')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Surface Temperature ($^\circ$C)')

                ax = fig.add_subplot(223)
                ax.axhline(u_max[0]+uss[0], color='r', linestyle='-', label='Maximum')
                ax.axhline(u_min[0]+uss[0], color='r', linestyle='-', label='Minimum')
                ax.step(np.arange(len_list[0]-1)*ts, sim_data['Usim'][0,:(st-1)]+uss[0], label='Nominal Tuning')
                # ax.step(np.arange(len_list[1]-1)*ts, sim_data1['Usim'][0,:(st1-1)]+uss[0], label='DNN (before BO)')
                ax.step(np.arange(len_list[1]-1)*ts, sim_data2['Usim'][0,:(st2-1)]+uss[0], label='Tuning after BO')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Power (W)')

                ax = fig.add_subplot(224)
                ax.axhline(u_max[1]+uss[1], color='r', linestyle='-', label='Maximum')
                ax.axhline(u_min[1]+uss[1], color='r', linestyle='-', label='Minimum')
                ax.step(np.arange(len_list[0]-1)*ts, sim_data['Usim'][1,:(st-1)]+uss[1], label='Nominal Tuning')
                # ax.step(np.arange(len_list[1]-1)*ts, sim_data1['Usim'][1,:(st1-1)]+uss[1], label='DNN (before BO)')
                ax.step(np.arange(len_list[1]-1)*ts, sim_data2['Usim'][1,:(st2-1)]+uss[1], label='Tuning after BO')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Flow Rate (SLM)')
                plt.draw()

                s = {}
                s['sim_data'] = sim_data
                # s['sim_data1'] = sim_data1
                s['sim_data2'] = sim_data2
                np.save(f'./trials/{date}/trial{n}_profiles_compare.npy',s,allow_pickle=True)

else:
    print('Did not perform Bayesian optimization.')

print('--------------------------- COMPLETED SIMULATIONS ---------------------------------')
plt.show()
