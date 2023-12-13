# multistage mpc problem details
#
#
# Requirements:
# * Python 3
# * CasADi [https://web.casadi.org]
#
# Copyright (c) 2021 Mesbah Lab. All Rights Reserved.
# Contributor(s): Kimberly Chan
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).

import numpy as np
import scipy.linalg
from scipy import io
import casadi as cas

from config.reference_signal import myRef_CEM

def get_prob_info(Kcem=0.5, Tmax=45.0):

    ts = 0.5 # sampling time (0.5 for 2021 data)
    rand_seed = 520
    Tref = 43.0

    Np = 5      # Prediction horizon (ensure it is the same as DNN)
    N_robust = 2 # Robust horizon
    case_idx = [-1, 0, 1]   # multiplier of the additive uncertainty

    ## load system matrices from Data model ID
    modelp = io.loadmat('../model/APPJmodel_2021_06_08_15h57m55s_n4sid_50split.mat') # 2021 data (n4sid)
    model = io.loadmat('../model/APPJmodel_2021_06_08_15h57m55s_n4sid_alldata.mat') # 2021 data (n4sid)

    A = model['A']
    B = model['B']
    C = model['C']
    xss = np.ravel(model['yss']) # [Ts; I]
    uss = np.ravel(model['uss']) # [P; q]
    print('Linear Model to be used for CONTROL:')
    print('A: ', A)
    print('B: ', B)
    print('C: ', C)
    print('xss: ', xss)
    print()

    Ap = modelp['A']
    Bp = modelp['B']
    Cp = modelp['C']
    xssp = np.ravel(modelp['yss']) # [Ts; I]
    ussp = np.ravel(modelp['uss']) # [P; q]
    print('Linear Model to be used for the PLANT:')
    print('A: ', Ap)
    print('B: ', Bp)
    print('C: ', Cp)
    print('xss: ', xssp)

    x0 = np.zeros((2,))#np.array([36-xssp[0],0]) # initial state
    myref = lambda t: myRef_CEM(t, ts) # reference signal

    nx = A.shape[1] # number of states
    nu = B.shape[1] # number of inputs (q, P)
    ny = C.shape[0] # number of outputs (Ts, I)
    nyc = 1         # number of controlled outputs
    nd = 0        # offset-free disturbances
    nw = nx         # process noise
    nv = ny         # measurement noise

    ## load/set MPC info
    # constraint bounds
    u_min = np.array([1.5, 1.5]) - uss
    u_max = np.array([5,5]) - uss
    x_min = np.array([25,0]) - xss
    x_max = np.array([Tmax,80]) - xss
    y_min = x_min
    y_max = x_max
    # v_min = 0*-0.01*np.ones(nv)
    # v_max = 0*0.01*np.ones(nv)
    v_mu = 0
    v_sigma = 0.1
    w_min = -1*np.ones(nw)
    w_max = 1*np.ones(nw)
    Wset = np.hstack((w_min, np.zeros(nw), w_max)).reshape(-1,3,order='F')

    # initial variable guesses
    u_init = (u_min+u_max)/2
    x_init = (x_min+x_max)/2
    y_init = (y_min+y_max)/2

    ## create casadi functions for problem
    # casadi symbols
    x = cas.SX.sym('x', nx)
    u = cas.SX.sym('u', nu)
    w = cas.SX.sym('w', nw)
    wp = cas.SX.sym('wp', nw) # predicted uncertainty
    v = cas.SX.sym('v', nv)
    yref = cas.SX.sym('yref', nyc)

    # dynamics function (prediction model)
    xnext = A@x + B@u + wp
    f = cas.Function('f', [x,u,wp], [xnext])

    # output equation (for control model)
    y = C@x
    h = cas.Function('h', [x], [y])

    # controlled output equation
    ymeas = cas.SX.sym('ymeas', ny)
    yc = ymeas[0]
    r = cas.Function('r', [ymeas], [yc])

    # plant model
    xnextp = A@x + B@u + w
    fp = cas.Function('fp', [x,u,w], [xnextp])

    # output equation (for plant)
    yp = C@x + v
    hp = cas.Function('hp', [x,v], [yp])

    # CEM output
    CEM = Kcem**(Tref-(x[0]+xss[0]))*ts/60
    CEMadd = cas.Function('CEMadd', [x], [CEM])

    # stage cost (nonlinear CEM cost)
    lstg = Kcem**(Tref-(x[0]+w[0]+xss[0]))*ts/60
    lstage = cas.Function('lstage', [x,w], [lstg])

    warm_start = True

    ## pack away problem info
    prob_info = {}
    prob_info['Np'] = Np
    prob_info['myref'] = myref
    prob_info['N_robust'] = N_robust
    prob_info['case_idx'] = case_idx
    prob_info['Wset'] = Wset

    prob_info['ts'] = ts
    prob_info['x0'] = x0
    prob_info['rand_seed'] = rand_seed

    prob_info['nu'] = nu
    prob_info['nx'] = nx
    prob_info['ny'] = ny
    prob_info['nyc'] = nyc
    prob_info['nv'] = nv
    prob_info['nw'] = nw
    prob_info['nd'] = nd

    prob_info['u_min'] = u_min
    prob_info['u_max'] = u_max
    prob_info['x_min'] = x_min
    prob_info['x_max'] = x_max
    prob_info['y_min'] = y_min
    prob_info['y_max'] = y_max
    # prob_info['v_min'] = v_min
    # prob_info['v_max'] = v_max
    prob_info['v_mu'] = v_mu
    prob_info['v_sigma'] = v_sigma
    prob_info['w_min'] = w_min
    prob_info['w_max'] = w_max

    prob_info['u_init'] = u_init
    prob_info['x_init'] = x_init
    prob_info['y_init'] = y_init

    prob_info['f'] = f
    prob_info['h'] = h
    prob_info['r'] = r
    prob_info['fp'] = fp
    prob_info['hp'] = hp
    prob_info['CEMadd'] = CEMadd
    prob_info['lstage'] = lstage
    prob_info['warm_start'] = warm_start

    prob_info['xssp'] = xssp
    prob_info['ussp'] = ussp
    prob_info['xss'] = xss
    prob_info['uss'] = uss

    return prob_info

def get_prob_info_exp(Kcem=0.5, Tmax=45.0, model_file=None, ts=0.5):

    rand_seed = 520
    Tref = 43.0

    Np = 2      # Prediction horizon (ensure it is the same as DNN)
    N_robust = 2 # Robust horizon
    case_idx = [-1, 0, 1]   # multiplier of the additive uncertainty

    ## load system matrices from Data model ID
    modelp = io.loadmat(model_file)
    model = io.loadmat(model_file)

    A = model['A']
    B = model['B']
    C = model['C']
    xss = np.ravel(model['yss']) # [Ts; I]
    uss = np.ravel(model['uss']) # [P; q]
    print('Linear Model to be used for CONTROL:')
    print('A: ', A)
    print('B: ', B)
    print('C: ', C)
    print('xss: ', xss)
    print()

    Ap = modelp['A']
    Bp = modelp['B']
    Cp = modelp['C']
    xssp = np.ravel(modelp['yss']) # [Ts; I]
    ussp = np.ravel(modelp['uss']) # [P; q]
    print('Linear Model to be used for the PLANT:')
    print('A: ', Ap)
    print('B: ', Bp)
    print('C: ', Cp)
    print('xss: ', xssp)

    x0 = np.array([30-xssp[0],0]) # initial state
    myref = lambda t: myRef_CEM(t, ts, CEMref=1.5) # reference signal

    nx = A.shape[1] # number of states
    nu = B.shape[1] # number of inputs (q, P)
    ny = C.shape[0] # number of outputs (Ts, I)
    nyc = 1         # number of controlled outputs
    nd = 0        # offset-free disturbances
    nw = nx         # process noise
    nv = ny         # measurement noise

    ## load/set MPC info
    # constraint bounds
    u_min = np.array([1.5,3.5]) - uss
    u_max = np.array([3.5,7.5]) - uss
    du_min = -0.5*np.ones(nu)
    du_max = 0.5*np.ones(nu)
    x_min = np.array([25,0]) - xss
    x_max = np.array([Tmax,80]) - xss
    y_min = x_min
    y_max = x_max
    # v_min = 0*-0.01*np.ones(nv)
    # v_max = 0*0.01*np.ones(nv)
    v_mu = 0
    v_sigma = 0.1
    w_min = 0.1*np.ravel([model['minErrors'][0,0],0])#-1*np.ones(nw)
    w_max = 0.1*np.ravel([model['maxErrors'][0,0],0])#1*np.ones(nw)
    Wset = np.hstack((w_min, np.zeros(nw), w_max)).reshape(-1,3,order='F')


    # initial variable guesses
    u_init = (u_min+u_max)/2
    x_init = (x_min+x_max)/2
    y_init = (y_min+y_max)/2

    ## create casadi functions for problem
    # casadi symbols
    x = cas.SX.sym('x', nx)
    u = cas.SX.sym('u', nu)
    u1 = cas.SX.sym('u1', nu)
    w = cas.SX.sym('w', nw)
    wp = cas.SX.sym('wp', nw) # predicted uncertainty
    v = cas.SX.sym('v', nv)
    yref = cas.SX.sym('yref', nyc)
    Ts = cas.SX.sym('Ts', nyc)

    # dynamics function (prediction model)
    xnext = A@x + B@u + wp
    f = cas.Function('f', [x,u,wp], [xnext])

    # output equation (for control model)
    y = C@x
    h = cas.Function('h', [x], [y])

    # controlled output equation
    ymeas = cas.SX.sym('ymeas', ny)
    yc = ymeas[0]
    r = cas.Function('r', [ymeas], [yc])

    # plant model
    xnextp = A@x + B@u + w
    fp = cas.Function('fp', [x,u,w], [xnextp])

    # output equation (for plant)
    yp = C@x + v
    hp = cas.Function('hp', [x,v], [yp])

    # CEM output
    CEM = Kcem**(Tref-Ts)*ts/60
    CEMadd = cas.Function('CEMadd', [Ts], [CEM])

    # input change cost
    R = 1e-4*np.eye(nu)
    du = (u1-u).T@R@(u1-u)
    ustage = cas.Function('ustage', [u1,u], [du])

    # stage cost (nonlinear CEM cost)
    lstg = Kcem**(Tref-(x[0]+w[0]+xss[0]))*ts/60
    lstage = cas.Function('lstage', [x,w], [lstg])

    warm_start = True

    ## pack away problem info
    prob_info = {}
    prob_info['Np'] = Np
    prob_info['myref'] = myref
    prob_info['N_robust'] = N_robust
    prob_info['case_idx'] = case_idx
    prob_info['Wset'] = Wset

    prob_info['ts'] = ts
    prob_info['x0'] = x0
    prob_info['rand_seed'] = rand_seed

    prob_info['nu'] = nu
    prob_info['nx'] = nx
    prob_info['ny'] = ny
    prob_info['nyc'] = nyc
    prob_info['nv'] = nv
    prob_info['nw'] = nw
    prob_info['nd'] = nd

    prob_info['u_min'] = u_min
    prob_info['u_max'] = u_max
    # prob_info['du_min'] = du_min
    # prob_info['du_max'] = du_max
    prob_info['x_min'] = x_min
    prob_info['x_max'] = x_max
    prob_info['y_min'] = y_min
    prob_info['y_max'] = y_max
    # prob_info['v_min'] = v_min
    # prob_info['v_max'] = v_max
    prob_info['v_mu'] = v_mu
    prob_info['v_sigma'] = v_sigma
    prob_info['w_min'] = w_min
    prob_info['w_max'] = w_max

    prob_info['u_init'] = u_init
    prob_info['x_init'] = x_init
    prob_info['y_init'] = y_init

    prob_info['f'] = f
    prob_info['h'] = h
    prob_info['r'] = r
    prob_info['fp'] = fp
    prob_info['hp'] = hp
    prob_info['CEMadd'] = CEMadd
    prob_info['lstage'] = lstage
    # prob_info['ustage'] = ustage
    prob_info['warm_start'] = warm_start

    prob_info['xssp'] = xssp
    prob_info['ussp'] = ussp
    prob_info['xss'] = xss
    prob_info['uss'] = uss

    return prob_info
