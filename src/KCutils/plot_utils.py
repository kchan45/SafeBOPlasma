# plotting utility functions
#
# This file defines several functions that act as plotting utilities for
# experimental data.
#
# Requirements:
# * Python 3
# * NumPy
# * Matplotlib
#
# Copyright (c) 2022 Mesbah Lab. All Rights Reserved.
# Kimberly Chan
#
# This file is under the MIT License. A copy of this license is included in the
# download of the entire code package (within the root folder of the package).
import numpy as np
import matplotlib.pyplot as plt

Fontsize = 12 # default font size for plots
Lwidth = 2 # default line width for plots

lines = {'linewidth' : Lwidth}
plt.rc('lines', **lines)
font = {'family' : 'serif',
        'serif'  : 'Times',
        'size'   : Fontsize}
plt.rc('font', **font)  # pass in the font dict as kwargs

def plot_core(prob_info, CEM=False, fig_objs={}):
    '''
    function to plot the "core" elements of a closed-loop trajectory

    For a plot that includes CEM, these elements may include the CEM reference,
    the maximum temperature constraint, and the bounds of the inputs. Additionally,
    the axis labels are set with the appropriate labels
    '''
    if CEM:
        xssp = prob_info['xssp']
        uss = prob_info['uss']
        x_max = prob_info['x_max']
        u_max = prob_info['u_max']
        u_min = prob_info['u_min']
        ref = prob_info['myref'](0)

        if bool(fig_objs):
            print('resused figure objects')
            fig = fig_objs['fig']
            (ax11, ax12, ax21, ax22) = fig_objs['axes']
        else:
            fig_objs = {}
            fig, ((ax11, ax12), (ax21, ax22)) = plt.subplots(2,2,figsize=(12,6),dpi=100)

        ax11.axhline(ref, color='r', linestyle='--', label='Target Reference')
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('CEM')
        ax11.legend(fontsize='small', loc='lower right')


        ax12.axhline(x_max[0]+xssp[0], color='r', linestyle='--', label='Constraint')
        ax12.set_xlabel('Time (s)')
        ax12.set_ylabel('Surface Temperature ($^\circ$C)')

        ax21.axhline(u_max[0]+uss[0], color='r', linestyle='-', label='Maximum')
        ax21.axhline(u_min[0]+uss[0], color='r', linestyle='-', label='Minimum')
        ax21.set_xlabel('Time (s)')
        ax21.set_ylabel('Power (W)')


        ax22.axhline(u_max[1]+uss[1], color='r', linestyle='-', label='Maximum')
        ax22.axhline(u_min[1]+uss[1], color='r', linestyle='-', label='Minimum')
        ax22.set_xlabel('Time (s)')
        ax22.set_ylabel('Flow Rate (SLM)')
        plt.draw()

        fig_objs['fig'] = fig
        fig_objs['axes'] = (ax11, ax12, ax21, ax22)

    return fig_objs


def plot_data_from_dict(exp_data, prob_info, open_loop=False, CEM=False, fig_objs={}, label=''):
    '''
    A function to plot the data of an experimental run.

    Input(s):
    --------
    exp_data        a dictionary of data obtained from running an experiment on the APPJ
    prob_info       a dictionary of the problem information/setup
    open_loop       a boolean describing whether or not the data is open loop or closed loop
    CEM             a boolean describing whether or not CEM (thermal dose delivery) is part of the data
    fig_objs        a dictionary of figure objects in the event that it is desired to plot more than one set of data on one figure
    label           a string for the label of the data plot (useful if multiple plots in one figure are desired)

    Output(s):
    ---------
    fig_objs        an updated dictionary of figure objects
    '''
    if not bool(fig_objs):
        fig_objs = {}

    if open_loop:
        print("Not implemented yet.")
        return
    else:
        if CEM:
            # extract data
            ts = prob_info['ts']
            xssp = prob_info['xssp']
            uss = prob_info['uss']
            x_max = prob_info['x_max']
            u_max = prob_info['u_max']
            u_min = prob_info['u_min']

            Refdata = np.ravel(exp_data['Yrefsim'])
            CEMsim = np.ravel(exp_data['CEMsim'])
            st = exp_data['CEM_stop_time']
            ctime = exp_data['ctime'][:st]
            Tdata = exp_data['Tsave']
            print('Total Runtime: ', np.sum(ctime))
            print('Average Runtime: ', np.mean(ctime))

            CEMplot = CEMsim[:st+1]
            Refplot = Refdata[:st+1]
            Tplot = Tdata[:st+1]

            # plot data
            if not bool(fig_objs):
                fig_objs = plot_core(prob_info, CEM=CEM)
            fig = fig_objs['fig']
            (ax11, ax12, ax21, ax22) = fig_objs['axes']

            ax11.plot(np.arange(len(CEMplot))*ts, CEMplot, ':', label=label)

            ax12.plot(np.arange(len(Tplot))*ts, Tplot, '--', label=label)

            ax21.step(np.arange(st-1)*ts, exp_data['Psave'][:(st-1)], label=label)

            ax22.step(np.arange(st-1)*ts, exp_data['qSave'][:(st-1)], label=label)

            plt.draw()

            fig_objs['fig'] = fig
            fig_objs['axes'] = (ax11, ax12, ax21, ax22)
        else:
            print("Not implemented yet.")
            return fig_objs

    plt.draw()
    return fig_objs


def plot_data_from_file(datafile, prob_info, open_loop=False, CEM=False, fig_objs={}, label=''):
    '''
    A wrapper function to plot the data if given a filename

    see inputs and outputs of plot_data_from_dict for more information
    '''
    exp_data = np.load(datafile, allow_pickle=True).item()

    fig_objs = plot_data_from_dict(exp_data, prob_info, open_loop=open_loop, CEM=CEM, fig_objs=fig_objs, label=label)
    return fig_objs


def plot_data_from_dict_sim(sim_data, prob_info, CEM=False, fig_objs={}, label='', alpha=1.0):
    '''
    A function to plot the data of a simulation run.

    Input(s):
    --------
    sim_data        a dictionary of data obtained from running an experiment on the APPJ
    prob_info       a dictionary of the problem information/setup
    CEM             a boolean describing whether or not CEM (thermal dose delivery) is part of the data
    fig_objs        a dictionary of figure objects in the event that it is desired to plot more than one set of data on one figure
    label           a string for the label of the data plot (useful if multiple plots in one figure are desired)

    Output(s):
    ---------
    fig_objs        an updated dictionary of figure objects
    '''
    if not bool(fig_objs):
        fig_objs = {}

    if CEM:
        # extract data
        ts = prob_info['ts']
        xssp = prob_info['xssp']
        uss = prob_info['uss']
        x_max = prob_info['x_max']
        u_max = prob_info['u_max']
        u_min = prob_info['u_min']

        Refdata = np.ravel(sim_data['Yrefsim'])
        CEMsim = np.ravel(sim_data['CEMsim'])
        st = sim_data['CEM_stop_time']
        ctime = sim_data['ctime'][:st]
        Tdata = sim_data['Ysim'][0,:] + xssp[0]
        print('Total Runtime: ', np.sum(ctime))
        print('Average Runtime: ', np.mean(ctime))

        CEMplot = CEMsim[:st]
        Refplot = Refdata[:st]
        Tplot = Tdata[:st]

        # plot data
        if not bool(fig_objs):
            fig_objs = plot_core(prob_info, CEM=CEM)
        fig = fig_objs['fig']
        (ax11, ax12, ax21, ax22) = fig_objs['axes']

        ax11.plot(np.arange(len(CEMplot))*ts, CEMplot, '-', label=label, alpha=alpha)
        ax11.set_xlim((-1,121))

        ax12.plot(np.arange(len(Tplot))*ts, Tplot, '-', label=label, alpha=alpha)
        ax12.set_xlim((-1,121))

        ax21.step(np.arange(st-1)*ts, sim_data['Usim'][0,:(st-1)]+uss[0], label=label, alpha=alpha)
        ax21.set_xlim((-1,121))

        ax22.step(np.arange(st-1)*ts, sim_data['Usim'][1,:(st-1)]+uss[1], label=label, alpha=alpha)
        ax22.set_xlim((-1,121))

        plt.draw()

        fig_objs['fig'] = fig
        fig_objs['axes'] = (ax11, ax12, ax21, ax22)
    else:
        print("Not implemented yet.")
        return fig_objs

    plt.draw()
    return fig_objs
