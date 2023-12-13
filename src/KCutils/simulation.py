# simulation functions
#
# This file defines a Simulation class to be used for closed loop simulations of
# model predictive controllers (MPCs) generated via the MPC subclasses or via
# deep neural network approximations of MPC controllers.
#
# Requirements:
# * Python 3
#
# Copyright (c) 2021 Mesbah Lab. All Rights Reserved.
# Kimberly Chan
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).

import sys
sys.dont_write_bytecode = True
import numpy as np
import time
from numpy.random import default_rng

import KCutils

class Simulation():
    """
    The Simulation class is used to create a simulation 'environment' defined by
    given problem information.
    """

    def __init__(self, Nsim):
        """
        Instantiation of the Simulation class requires the input argument Nsim,
        which denotes the length of the simulation horizon.
        """
        super(Simulation, self).__init__()
        self.Nsim = Nsim
        self.prob_info = None
        self.rand_seed = None

    def load_prob_info(self, prob_info):
        """
        This method loads the relevant problem information for simulation and
        assigns them as attributes of the class from the prob_info dict used by
        other classes included in this package.
        """
        # extract and save relevant problem information
        self.prob_info = prob_info

        # system sizes
        self.nu = prob_info['nu']
        self.nx = prob_info['nx']
        self.ny = prob_info['ny']
        self.nyc = prob_info['nyc']
        self.nv = prob_info['nv']
        self.nw = prob_info['nw']
        self.nd = prob_info['nd']

        # disturbance/noise minimums and maximums
        if 'v_mu' in prob_info.keys() and 'v_sigma' in prob_info.keys():
            self.v_mu = prob_info['v_mu']
            self.v_sigma = prob_info['v_sigma']
            self.v_noise_generation = 'normal'
        elif 'v_max' in prob_info.keys() and 'v_min' in prob_info.keys():
            self.v_min = prob_info['v_min']
            self.v_max = prob_info['v_max']
            self.v_noise_generation = 'uniform'
        else:
            print('No noise bounds/parameters given. Assuming no measurement noise...')
            self.v_min = np.zeros((self.nv,))
            self.v_max = np.zeros((self.nv,))
            self.v_noise_generation = 'uniform'

        if 'w_mu' in prob_info.keys() and 'w_sigma' in prob_info.keys():
            self.v_mu = prob_info['w_mu']
            self.v_sigma = prob_info['w_sigma']
            self.v_noise_generation = 'normal'
        elif 'w_max' in prob_info.keys() and 'w_min' in prob_info.keys():
            self.w_min = prob_info['w_min']
            self.w_max = prob_info['w_max']
            self.w_noise_generation = 'uniform'
        else:
            print('No noise bounds/parameters given. Assuming no process noise...')
            self.w_min = np.zeros((self.nw,))
            self.w_max = np.zeros((self.nw,))
            self.w_noise_generation = 'uniform'

        self.x0 = prob_info['x0'] # initial state/point
        self.hp = prob_info['hp'] # output equation for the 'real' system (plant)
        self.fp = prob_info['fp'] # dynamics equation for the plant
        self.myref = prob_info['myref'] # reference function for the controlled output
        self.ts = prob_info['ts'] # simulation sampling time
        self.rand_seed = prob_info['rand_seed'] # random seed

    def run_closed_loop(self, c,
                        observer=None,
                        offset=False,
                        CEM=False,
                        multistage=False,
                        rand_seed2=0):
        """
        This method runs a closed-loop simulation of the system using
        information derived from loading problem information and a controller
        (implicit MPC or approximate control). The problem information must be
        loaded before a closed-loop simulation can occur. The argument c is a
        controller object created using one of the classes defined in
        controller.py (for an MPC) or neural_network.py (for a DNN approximation
        to a MPC law).
        """
        # check to ensure problem data has been loaded
        if self.prob_info is None:
            print('Problem data not loaded. Please load the appropriate problem data by running the load_prob_info method.')
            raise

        # check controller type
        mpc_controller = False
        if isinstance(c, KCutils.controller.MPC):
            mpc_controller = True
            print('Using a MPC.')
        elif isinstance(c, KCutils.neural_network.DNN) or isinstance(c, KCutils.neural_network.SimpleDNN):
            print('Using an approximate controller.')
        else:
            print('Unsupported controller type.')
            raise

        if multistage:
            Wset = self.prob_info['Wset']

        # create a random number generator (RNG) to use for generating
        # noise/disturbances; use the same seed for consistent simulations
        if self.rand_seed is None:
            rng = default_rng()
        else:
            rng = default_rng(self.rand_seed+rand_seed2)

        if self.v_noise_generation == 'uniform':
            Vsim = rng.random(size=(self.nv,self.Nsim+1)) * (self.v_max-self.v_min)[:,None] + self.v_min[:,None]
        elif self.v_noise_generation == 'normal':
            Vsim = rng.normal(self.v_mu, self.v_sigma, size=(self.nv,self.Nsim+1))
        else:
            print('Unknown/unsupported form of noise generation!')
            raise
        if self.w_noise_generation == 'uniform':
            Wsim = rng.random(size=(self.nw,self.Nsim)) * (self.w_max-self.w_min)[:,None] + self.w_min[:,None]
        elif self.w_noise_generation == 'normal':
            Wsim = rng.normal(self.w_mu, self.w_sigma, size=(self.nw,self.Nsim+1))
        else:
            print('Unknown/unsupported form of noise generation!')
            raise

        # instantiate container variables for storing simulation data
        Xsim = np.zeros((self.nx,self.Nsim+1)) # state trajectories (plant)
        Ysim = np.zeros((self.ny,self.Nsim+1)) # output trajectories (plant)
        Usim = np.zeros((self.nu,self.Nsim))   # input trajectories (plant)
        Xhat = np.zeros_like(Xsim)  # state estimation
        Dhat = np.zeros((self.nd,self.Nsim+1))
        if multistage:
            Ypred = [0 for i in range(self.Nsim)]
            Wpred = [0 for i in range(self.Nsim)]
        else:
            Ypred = np.zeros((self.ny,self.prob_info['Np'],self.Nsim))

        if offset:
            Xss_sim = np.zeros_like(Xsim)   # steady state trajectory (controller)
            Yss_sim = np.zeros_like(Ysim)   # steady state output trajectory (controller)
            Uss_sim = np.zeros_like(Usim)   # steady state input trajectory (controller)

        ctime = np.zeros(self.Nsim)   # computation time
        Jsim = np.zeros(self.Nsim)    # cost/optimal objective value (controller)
        Yrefsim = np.zeros((self.nyc,self.Nsim+1))  # output reference/target (as sent to controller)
        Feas = np.zeros(self.Nsim)
        if CEM:
            CEMsim = np.zeros((1,self.Nsim+1))  # CEM trajectory
            CEM_stop_time = self.Nsim+1

        # set initial states and reset controller for consistency
        Xsim[:,0] = np.ravel(self.x0)
        Xhat[:,0] = Xsim[:,0]
        if observer is not None:
            observer.xhat = Xhat[:,0]
            observer.dhat = Dhat[:,0]
        Ysim[:,0] = np.ravel(self.hp(Xsim[:,0],Vsim[:,0]).full())

        if mpc_controller:
            c.reset_initial_guesses()

        # loop over simulation time
        CEM_stop_trigger = False
        for k in range(self.Nsim):
            startTime = time.time()

            Yrefsim[:,k] = self.myref(k*self.ts)
            if mpc_controller:
                if CEM:
                    if multistage:
                        c.set_parameters([Xhat[:,k], Yrefsim[:,k], CEMsim[:,k], Wset])
                    else:
                        c.set_parameters([Xhat[:,k], Yrefsim[:,k], CEMsim[:,k]])
                else:
                    if offset:
                        c.set_parameters([Xhat[:,k], Yrefsim[:,k], Dhat[:,k]])
                    else:
                        c.set_parameters([Xhat[:,k], Yrefsim[:,k]])
                res, feas = c.solve_mpc(warm_start=self.prob_info['warm_start'])
                Uopt = res['U']
                Jopt = res['J']
                if multistage:
                    Ypred[k] = res['Y'] # todo: add functionality to other controllers, then move this out of conditional statement
                    Wpred[k] = res['wPred']
            else:
                Jopt = np.nan

                if CEM:
                    in_vec = np.concatenate((Xhat[:,k], CEMsim[:,k]))
                else:
                    in_vec = np.concatenate((Xhat[:,k], Yrefsim[:,k]))
                Uopt = np.ravel((c.netca(in_vec)).full())
                Uopt = np.maximum(np.minimum(Uopt, self.prob_info['u_max']), self.prob_info['u_min'])

            ctime[k] = time.time() - startTime
            if mpc_controller:
                if not feas:
                    print('Was not feasible on iteration {} of simulation'.format(k))

            if offset:
                Xss_sim[:,k] = res['Xss']
                Uss_sim[:,k] = res['Uss']
                Yss_sim[:,k] = res['Yss']

            Usim[:,k] = np.ravel(Uopt)
            Jsim[k] = Jopt

            # send optimal input to plant/real system
            Xsim[:,k+1] = np.ravel(self.fp(Xsim[:,k],Usim[:,k],Wsim[:,k]).full())
            Ysim[:,k+1] = np.ravel(self.hp(Xsim[:,k+1],Vsim[:,k+1]).full())
            if CEM:
                CEMsim[:,k+1] = CEMsim[:,k] + np.ravel(self.prob_info['CEMadd'](Ysim[:,k+1]).full())
                if not CEM_stop_trigger and CEMsim[:,k+1] > Yrefsim[:,k]:
                    CEM_stop_time = k+1
                    CEM_stop_trigger = True
                if CEM_stop_trigger:
                    break

            # get estimates
            if observer is not None:
                xhat, dhat = observer.update_observer(Usim[:,k], Ysim[:,k+1])
                Xhat[:,k+1] = np.ravel(xhat)
                Dhat[:,k+1] = np.ravel(dhat)
            else:
                Xhat[:,k+1] = Xsim[:,k+1]

        Yrefsim[:,k+1] = self.myref((k+1)*self.ts)
        # create dictionary of simulation data
        sim_data = {}
        sim_data['Xsim'] = Xsim
        sim_data['Ysim'] = Ysim
        sim_data['Usim'] = Usim
        sim_data['Jsim'] = Jsim
        sim_data['Wsim'] = Wsim
        sim_data['Vsim'] = Vsim
        sim_data['Yrefsim'] = Yrefsim
        sim_data['ctime'] = ctime
        sim_data['Xhat'] = Xhat
        sim_data['Dhat'] = Dhat
        sim_data['Ypred'] = Ypred
        sim_data['feas'] = Feas
        if offset:
            sim_data['Xss_sim'] = Xss_sim
            sim_data['Uss_sim'] = Uss_sim
            sim_data['Yss_sim'] = Yss_sim
        if multistage:
            sim_data['wPred'] = Wpred
        if CEM:
            sim_data['CEMsim'] = CEMsim
            sim_data['CEM_stop_time'] = CEM_stop_time

        return sim_data
