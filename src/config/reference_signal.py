# reference signal functions
#
#
# Requirements:
# * Python 3
#
# Copyright (c) 2021 Mesbah Lab. All Rights Reserved.
# Contributor(s): Kimberly Chan
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).

import sys
sys.dont_write_bytecode = True
from math import ceil

def myRef_lmpc_2019(t, ts):
    Nstep = 60

    if ceil(t/ts) < Nstep:
        yref = 32.0
    elif ceil(t/ts) < 2*Nstep:
        yref = 41.0
    elif ceil(t/ts) < 3*Nstep:
        yref = 37.0
    elif ceil(t/ts) < 4*Nstep:
        yref = 34.0
    else:
        yref = 37.0

    return yref

def myRef_lmpc_2021(t, ts):
    Nstep = 1*60/ts

    if ceil(t/ts) < Nstep:
        yref = 40.0
    elif ceil(t/ts) < 2*Nstep:
        yref = 38.0
    elif ceil(t/ts) < 3*Nstep:
        yref = 42.0
    elif ceil(t/ts) < 4*Nstep:
        yref = 36.0
    else:
        yref = 40.0

    return yref

def myRef_CEM(t, ts, CEMref=1.5):
    return CEMref
