# economic mpc problem details
#
#
# Requirements:
# * Python 3
# * CasADi [https://web.casadi.org]
#
# Copyright (c) 2023 Mesbah Lab. All Rights Reserved.
# Contributor(s): Kimberly Chan
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).

import numpy as np
from scipy import io
import casadi as cas

from config.reference_signal import myRef_CEM

def get_prob_info(Kcem=0.5, Tmax=45.0, Np=5, no_mismatch=False):

    ts = 0.5 # sampling time (0.5 for 2021 data)
    rand_seed = 520
    Tref = 43.0

    ## load system matrices from Data model ID
    modelp = io.loadmat('../model/APPJmodel_2021_06_08_15h57m55s_n4sid_50split.mat') # 2021 data (n4sid)
    if no_mismatch:
        model = io.loadmat('../model/APPJmodel_2021_06_08_15h57m55s_n4sid_50split.mat') # 2021 data (n4sid)
    else:
        model = io.loadmat('../model/APPJmodel_2021_06_08_15h57m55s_n4sid_alldata.mat') # 2021 data (n4sid)

    A = model['A']
    if not no_mismatch:
        A[0,0] = A[0,0]+0.1
        A[1,1] = A[1,1]+0.1
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
    v_mu = 0.0
    v_sigma = 0.1
    w_min = -0.0*np.ones(nw)
    w_max = 0.0*np.ones(nw)

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
    xnextp = Ap@x + Bp@u + w
    fp = cas.Function('fp', [x,u,w], [xnextp])

    # output equation (for plant)
    yp = Cp@x + v
    hp = cas.Function('hp', [x,v], [yp])

    # CEM output (for plant simulation)
    CEM = Kcem**(Tref-(x[0]+xss[0]))*ts/60
    CEMadd = cas.Function('CEMadd', [x], [CEM])

    # stage cost (nonlinear CEM cost, for controller objective/model)
    lstg = Kcem**(Tref-(x[0]+wp[0]+xss[0]))*ts/60
    lstage = cas.Function('lstage', [x,wp], [lstg])

    warm_start = True

    ## pack away problem info
    prob_info = {}
    prob_info['Np'] = Np
    prob_info['myref'] = myref
    prob_info['Tref'] = Tref
    prob_info['Kcem'] = Kcem

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

    prob_info['model'] = (A,B,C)
    prob_info['plant'] = (Ap,Bp,Cp)
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

def modify_prob_info(prob_info, Np=None, A=None, B=None, Kcem=None, save_file=None):

    nx = prob_info['nx']
    nu = prob_info['nu']
    nw = prob_info['nw']

    if Np is not None:
        prob_info['Np'] = int(Np)

    if A is not None:
        rA = A.shape[0] == nx
        cA = A.shape[1] == nx
    else:
        rA = False
        cA = False

    if B is not None:
        rB = B.shape[0] == nx
        cB = B.shape[1] == nu
    else:
        rB = False
        cB = False

    if all([rA, cA, rB, cB]):
        ## create casadi functions for problem
        # casadi symbols
        x = cas.SX.sym('x', nx)
        u = cas.SX.sym('u', nu)
        wp = cas.SX.sym('wp', nw) # predicted uncertainty

        # dynamics function (prediction model)
        xnext = A@x + B@u + wp
        f = cas.Function('f', [x,u,wp], [xnext])
        prob_info['f'] = f
        _, _, Cold = prob_info['model']
        prob_info['model'] = (A, B, Cold)

    elif all([rA, cA]):
        ## create casadi functions for problem
        # casadi symbols
        x = cas.SX.sym('x', nx)
        u = cas.SX.sym('u', nu)
        wp = cas.SX.sym('wp', nw) # predicted uncertainty

        # dynamics function (prediction model)
        xnext = A@x + B@u + wp
        f = cas.Function('f', [x,u,wp], [xnext])
        prob_info['f'] = f
        _, Bold, Cold = prob_info['model']
        prob_info['model'] = (A, Bold, Cold)

    if Kcem is not None:
        x = cas.SX.sym('x', nx)
        wp = cas.SX.sym('wp', nw) # predicted uncertainty

        Tref = prob_info['Tref']
        xss = prob_info['xss']
        ts = prob_info['ts']
        # stage cost (nonlinear CEM cost)
        lstg = Kcem**(Tref-(x[0]+wp[0]+xss[0]))*ts/60
        lstage = cas.Function('lstage', [x,wp], [lstg])
        prob_info['lstage'] = lstage

    if save_file:
        prob_info_save = prob_info.copy() # make a copy to save the dictionary of problem information
        del prob_info_save['myref'] # delete the lambda function defined for the CEM reference since np.save cannot pickle the lambda function
        np.save(save_file, prob_info_save, allow_pickle=True)

    return prob_info
