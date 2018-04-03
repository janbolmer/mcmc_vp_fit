#! /usr/bin/python

"""
MCMC sampler for fitting X-shooter spectra with voigt profiles
=========================================================================
e.g.: velop_mcmc.py -f spectra/GRB120815A_OB1UVB.txt -z 2.358
    -line 1347.2396 -e Cl -vr 400 -w 12 -min 3 -max 4 -it 200
    -bi 80 -res 28 -par velo_para.csv
=========================================================================
-f 		path to spectrum data file
-line 		line in AA, eg. 1808.0129
-e 		Name of the line/element / e.g.: FeII, SiII
-z 		redshift for centering
-vr 		velocity range (-vr to +vr)
-min 		minimum number of voigt profiles to fit
-max 		maximum number of voigt profiles to fit
-res 		spectral resolution km/s (fwhm)
-par 		Parameter file with velocity components
-it 		number of iterations
-bi 		burn-in
-w 		walength range (in AA) to be extracted from spectrum
-plr 		it True: plot traces and posterior distributions
=========================================================================
ClI     1347.2396  0.15300000
SiII    1526.7070  0.13300000 
SiII    1808.0129  0.00208000
FeII    1608.4509  0.05399224
FeII    2344.2129  0.12523167
=========================================================================
"""

__author__ = "Jan Bolmer"
__copyright__ = "Copyright 2018"
__version__ = "0.9"
__maintainer__ = "Jan Bolmer"
__email__ = "jbolmer@eso.org"
__status__ = "stable"

import pymc
import math
import time
import sys
import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns

from numba import jit

import matplotlib as plt
import matplotlib.pyplot as pyplot
from pylab import *
from scipy.special import wofz

from astropy.convolution import Gaussian1DKernel, convolve

sys.path.append('bin/')

from spec_functions import *  # spec_functions.py
from syn_spec import *  # syn_spec.py
from sns_plots import *  # sns_plots.py
from read_fits import *

# constants to calculate the cloumn density
e = 4.8032E-10  # cm 3/2 g 1/2 s-1
c = 2.998e10  # speed of light cm/s
m_e = 9.10938291e-28  # electron mass g

colors = ["#a6cee3", "#1f78b4",
          "#b2df8a", "#33a02c", "#fb9a99",
          "#e31a1c", "#fdbf6f", "#ff7f00",
          "#cab2d6", "#6a3d9a", "#a6cee3",
          "#1f78b4", "#b2df8a", "#33a02c",
          "#fb9a99", "#e31a1c", "#fdbf6f",
          "#ff7f00", "#cab2d6", "#6a3d9a"]

# https://www.pantone.com/color-of-the-year-2017
pt_analogous = ["#86af49", "#817397", "#b88bac",
                "#d57f70", "#dcb967", "#ac9897",
                "#ac898d", "#f0e1ce", "#86af49",
                "#817397", "#b88bac", "#d57f70",
                "#dcb967"]

#========================================================================
#========================================================================


def get_results(para_file):
    '''
    Reads the Results from the .csv file created from PyMC
    '''

    par_dic = {}
    par = pd.read_csv(para_file, delimiter=', ', engine='python')
    for i in np.arange(0, len(par), 1):
        par_dic[par['Parameter'][i]] = [par['Mean'][i], par['SD'][i]]

    return par_dic


def print_results(res_file, redshift):
    '''
    Prints out the velocity components and corresponding redshift
    '''
    comp_str = ""
    red_str = ""
    with open(res_file, "r") as f:
        for line in f:
            s = line.split(",")
            if s[0].startswith("v"):
                comp_str += str(s[1]) + " "
                nz = v_to_dz(float(s[1]), redshift)
                red_str += str(nz) + " "
    print comp_str
    print red_str

#========================================================================
#========================================================================


def gauss(x, mu, sig):
    '''
    Normal distribution used to create prior probability distributions
    '''
    return np.exp(-np.power(x - mu, 2.0) / (2.0 * np.power(sig, 2.0)))

#========================================================================
#========================================================================


@jit
def add_abs_velo(v, N, b, gamma, f, l0):
    '''
    Add an absorption line l0 in velocity space v, given the oscillator
    strength f, the damping constant gamma, column density N, and
    broadening parameter b
    '''
    A = (((np.pi * e**2) / (m_e * c)) * f * l0 * 1E-13) * (10**N)
    tau = A * voigt(v, b / np.sqrt(2.0), gamma)

    return np.exp(-tau)

#========================================================================
#========================================================================


def power_lst(my_list, exp):
    '''
    x**exp for each x in my_list
    '''
    return [x**exp for x in my_list]

#========================================================================
#========================================================================


def mult_voigts(velocity, fluxv, fluxv_err, f, gamma, l0, nvoigts, RES,
                CSV_LST, velo_range, para_dic):
    '''
    Fitting a number of Voigt profiles to a spectrum in velocity space,
    given the restframe wavelenth l0 (Angstrom), the oscillator
    strength f, damping constant gamma (km/s), and spectral resolution
    RES (km/s)
    '''

    #low_b = 2  
    low_b = round(0.354*RES/(2*np.sqrt(np.log(2))),2)

    print "\n Components with ~ b >", low_b, \
        "km/s can be resolved \n"

    #if vandels == False:
    tau = 1.0 / np.array(fluxv_err)**2.0
    #if vandels == True:
    #    tau = 1.0 / np.array(fluxv_err)

    #@pymc.stochastic(dtype=float)
    # def a(value=1.0, mu=1.0, sig=0.1, doc="B"):
    #	pp = 0.0
    #	#if 0.85 <= value < 1.15:
    #	pp = gauss(value, mu, sig)
    #	#else:
    #	#	pp = -np.inf
    #	return pp

    # Continuum model (up to quadratic polinomial)
    mu_bg = sum(fluxv) / len(fluxv)


    @pymc.stochastic(dtype=float)
    def a(value=mu_bg, mu=mu_bg, sig=0.5 * mu_bg, doc="a"):
        if mu_bg / 10 < value < mu_bg * 10:
            pp = gauss(value, mu, sig)
        else:
            pp = -np.inf
        return pp

    @pymc.stochastic(dtype=float)
    def a1(value=0.0, mu=0.0, sig=0.5, doc="a1"):
        if -0.3 < value < 0.3:
            pp = gauss(value, mu, sig)
        else:
            pp = -np.inf
        return pp

    @pymc.stochastic(dtype=float)
    def a2(value=0.0, mu=0.0, sig=0.5, doc="a2"):
        if -0.3 < value < 0.3:
            pp = gauss(value, mu, sig)
        else:
            pp = -np.inf
        return pp

    vars_dic = {}

    for i in range(1, nvoigts + 1):

        if not "v0" + str(i) in para_dic:
            v0 = pymc.Uniform('v0' + str(i), lower=-velo_range,
                              upper=velo_range, doc='v0' + str(i))
        else:
            if para_dic["v0" + str(i)][0] == 0:
                print "v0" + str(i) + " set to:", \
                    para_dic["v0" + str(i)][2], "to", \
                    para_dic["v0" + str(i)][3]
                v0 = pymc.Uniform('v0' + str(i),
                                  lower=para_dic["v0" + str(i)][2],
                                  upper=para_dic["v0" + str(i)][3],
                                  doc='v0' + str(i))

            if para_dic["v0" + str(i)][0] == 1:
                print "v0" + str(i) + " fixed to:", \
                    para_dic["v0" + str(i)][1], "+/- 0.5"
                v0 = pymc.Uniform('v0' + str(i),
                                  lower=para_dic["v0" + str(i)][1] - .5,
                                  upper=para_dic["v0" + str(i)][1] + .5,
                                  doc='v0' + str(i))

        if not "b" + str(i) in para_dic:
            print 'starting with b =', low_b+20
            b = pymc.DiscreteUniform('b' + str(i), lower=round(low_b, 0),
                             upper=80, value=low_b+20, doc='b' + str(i))

        else:
            if para_dic["b" + str(i)][0] == 0:
                print "b" + str(i) + " prior set to Gaussian Dist. around:", \
                    para_dic["b" + str(i)][1]
                b = pymc.Normal('b' + str(i), mu=para_dic["b" + str(i)][1],
                                tau=1.0 /
                                (para_dic["b" + str(i)][3] -
                                 para_dic["b" + str(i)][1])**2,
                                doc='b' + str(i))

            if para_dic["b" + str(i)][0] == 1:
                print "b" + str(i) + " prior set to Unifrom Distr. from:", \
                    para_dic["b" + str(i)][2], "to", para_dic["b" + str(i)][3]
                b = pymc.Uniform('b' + str(i), lower=para_dic["b" + str(i)][2],
                                 upper=para_dic["b" + str(i)][3], doc='b' + str(i))

        N = pymc.Exponential('N' + str(i), beta=0.1,
                             value=15.0, doc='N' + str(i))
        #if vandels == True:
        #   N  = pymc.Uniform('N' + str(i), lower=0,
        #                       upper=15.0, doc='N' + str(i))


        CSV_LST.extend(('v0' + str(i), 'b' + str(i), 'N' + str(i)))

        vars_dic['v0' + str(i)] = v0
        vars_dic['b' + str(i)] = b
        vars_dic['N' + str(i)] = N

    print "\n Starting MCMC " + '(pymc version:', pymc.__version__, ")"
    print "\n This might take a while ..."

    @pymc.deterministic(plot=False)
    def multVoigt(vv=velocity, a=a, a1=a1, a2=a2, f=f, gamma=gamma, l0=l0,
                  nvoigts=nvoigts, vars_dic=vars_dic):

        conv_val = RES / (2 * np.sqrt(2 * np.log(2)) * transform)

        gauss_k = Gaussian1DKernel(stddev=conv_val,
                                   mode="oversample")

        flux = np.ones(len(vv)) * (a + a1 * vv + a2 * (power_lst(vv, 2)))

        for i in range(1, nvoigts + 1):
            v = vv - vars_dic["v0" + str(i)]
            flux *= add_abs_velo(v, vars_dic["N" + str(i)],
                                 vars_dic["b" + str(i)], gamma, f, l0)

        return np.convolve(flux, gauss_k, mode='same')

    y_val = pymc.Normal('y_val', mu=multVoigt, tau=tau,
                        value=fluxv, observed=True)

    return locals()


def do_mcmc(grb, redshift, velocity, fluxv, fluxv_err, grb_name, f, gamma,
            l0, nvoigts, iterations, burn_in, RES, velo_range, para_dic):
    '''
    MCMC sample 
    Reading and writing Results
    '''

    CSV_LST = ["a", "a1", "a2"]

    pymc.np.random.seed(1)

    MDL = pymc.MCMC(mult_voigts(velocity, fluxv, fluxv_err,
                                f, gamma, l0, nvoigts, RES,
                                CSV_LST, velo_range, para_dic),
                    db='pickle', dbname='velo_fit.pickle')

    MDL.db
    MDL.use_step_method(pymc.AdaptiveMetropolis, [MDL.N, MDL.b],
                        scales={MDL.N: 1.0, MDL.b: 1.0}, delay=10000)
    MDL.sample(iterations, burn_in)
    MDL.db.close()

    # Autocorrelation Plots
    # pymc.Matplot.autocorrelation(MDL)

    y_min = MDL.stats()['multVoigt']['quantiles'][2.5]
    y_max = MDL.stats()['multVoigt']['quantiles'][97.5]
    y_min2 = MDL.stats()['multVoigt']['quantiles'][25]
    y_max2 = MDL.stats()['multVoigt']['quantiles'][75]
    y_fit = MDL.stats()['multVoigt']['mean']

    MDL.write_csv(grb_name + "_" + str(nvoigts) + "_" + str(l0) +
                  "_voigt_res.csv", variables=CSV_LST)

    csv_f = open(grb_name + "_" + str(nvoigts) + "_" + str(l0) +
                 "_voigt_res.csv", "a")
    csv_f.write("Osc, " + str(f) + "\n")
    csv_f.write("GAMMA, " + str(gamma_line))
    csv_f.close()

    return y_min, y_max, y_min2, y_max2, y_fit

#========================================================================
#========================================================================


def plot_results(grb, redshift, velocity, fluxv, fluxv_err, velocity_pl,
                 fluxv_pl, fluxv_err_pl, y_min, y_max, y_min2, y_max2,
                 y_fit, res_file, f, gamma, l0, nvoigts, velo_range,
                 grb_name, RES, ignore_lst, element="SiII"):
    '''
    Plotting the Spectrum including the individual Voigt Profiles
    '''
    sns.set_style("white")
    par_dic = get_results(res_file)

    velo_arrays = [[]]
    y_fit_arrays = []
    y_min_arrays = []
    y_max_arrays = []
    y_min2_arrays = []
    y_max2_arrays = []

    ign_s = sorted(ignore_lst)

    if not len(ign_s) == 0:
        ign_count = 0
        for count, val in enumerate(velocity_pl, 0):

            if ign_count < len(ign_s):

                if not ign_s[ign_count][0] < val < ign_s[ign_count][1] and ign_count == 0:
                    velo_arrays[ign_count].append(val)

                elif not ign_s[ign_count][0] < val < ign_s[ign_count][1] and 0 < ign_count < len(ign_s) and val > ign_s[ign_count - 1][1]:
                    velo_arrays[ign_count].append(val)

                elif not ign_s[ign_count][0] < val < ign_s[ign_count][1] and 0 < ign_count < len(ign_s) and val < ign_s[ign_count - 1][1]:
                    pass

                else:
                    if ign_count < len(ign_s):
                        velo_arrays.append([])
                        ign_count += 1
                    else:
                        ign_count += 0

            elif ign_count == len(ign_s) and val > ign_s[ign_count - 1][1]:
                velo_arrays[ign_count].append(val)
            elif ign_count == len(ign_s) and val < ign_s[ign_count - 1][1]:
                pass

    i_len = 0
    for interval in velo_arrays:
        y_fit_arrays.append(y_fit[i_len:len(interval) + i_len])
        y_min_arrays.append(y_min[i_len:len(interval) + i_len])
        y_max_arrays.append(y_max[i_len:len(interval) + i_len])
        y_min2_arrays.append(y_min2[i_len:len(interval) + i_len])
        y_max2_arrays.append(y_max2[i_len:len(interval) + i_len])
        i_len += len(interval)

    fig = figure(figsize=(10, 6))
    ax = fig.add_axes([0.13, 0.15, 0.85, 0.78])

    gauss_k = Gaussian1DKernel(stddev=RES / (2 * np.sqrt(2 * np.log(2)) * transform),
                               mode="oversample")

    N_all, N_all_err = [], []
    b_all, b_all_err = [], []
    v0_all, v0_all_err = [], []

    for i in range(1, nvoigts + 1):

        ff = np.ones(len(velocity)) * (par_dic["a"][0] +
                                       par_dic["a1"][0] * np.array(velocity) +
                                       par_dic["a2"][0] * np.array(power_lst(velocity, 2)))

        v = velocity - par_dic["v0" + str(i)][0]

        ff *= np.convolve(add_abs_velo(v, par_dic["N" + str(i)][0],
                                       par_dic["b" + str(i)][0], gamma,
                                       f, l0), gauss_k, mode='same')

        b_C = round(par_dic["b" + str(i)][0], 2)
        b_Cerr = round(par_dic["b" + str(i)][1], 2)
        N_C = round(par_dic["N" + str(i)][0], 2)
        N_Cerr = round(par_dic["N" + str(i)][1], 2)

        N_all.append(N_C)
        N_all_err.append(N_Cerr)
        b_all.append(b_C)
        b_all_err.append(b_Cerr)
        v0_all.append(round(par_dic["v0" + str(i)][0], 2))
        v0_all_err.append(round(par_dic["v0" + str(i)][1], 2))

        print "Component", i, ":", "b:", b_C, "+/-", b_Cerr, "N:", N_C, "+/-", N_Cerr, 'v0:', \
            round(par_dic["v0" + str(i)][0],
                  2), "+/-", round(par_dic["v0" + str(i)][1], 2)

        ax.axvline(par_dic["v0" + str(i)][0], linestyle="dashed", color="black",
                   linewidth=1.2)

        norm_fac_lst = par_dic["a"][0] + par_dic["a1"][0] * np.array(velocity) \
            + par_dic["a2"][0] * np.array(power_lst(velocity, 2))
        div_ff = [ai / bi for ai, bi in zip(ff[4:-4], norm_fac_lst[4:-4])]

        ax.plot(velocity[4:-4], div_ff, label='Comp. ' + str(i),
                color=pt_analogous[i - 1], linewidth=2)

        ax.text(par_dic["v0" + str(i)][0], 1.45, "b = " + str(b_C) + "+/-" + str(b_Cerr),
                rotation=55, color=pt_analogous[i - 1])
        ax.text(par_dic["v0" + str(i)][0], 1.65, "N = " + str(N_C) + "+/-" + str(N_Cerr),
                rotation=55, color=pt_analogous[i - 1])

    N_total = np.log10(sum([10**i for i in N_all]))
    N_total_err_tmp = np.log10(
        sum([10**(i + j) for i, j in zip(N_all, N_all_err)]))
    N_total_err = N_total_err_tmp - N_total
    print "total column density:", round(N_total, 2), "+/-", round(N_total_err, 2)

    for v_rng in ignore_lst:
        ax.axvspan(v_rng[0], v_rng[1], facecolor='black', alpha=0.25)

    print "Background a =", par_dic["a"][0]
    print "Background a1 =", par_dic["a1"][0]
    print "Background a2 =", par_dic["a2"][0]

    ylim([-0.5, 1.75])
    xlim([-velo_range, velo_range])

    ax.axhline(0.0, xmin=0.0, xmax=1.0, linewidth=2,
               linestyle="dotted", color="black")
    ax.axhline(1.0, xmin=0.0, xmax=1.0, linewidth=2,
               linestyle="-", color="black")

    # Calculate normalization factors
    norm_fac_lst = par_dic["a"][0] + par_dic["a1"][0] * np.array(velocity) + \
        par_dic["a2"][0] * np.array(power_lst(velocity, 2))

    norm_fac_lst_pl = par_dic["a"][0] + par_dic["a1"][0] * np.array(velocity_pl) + \
        par_dic["a2"][0] * np.array(power_lst(velocity_pl, 2))

    div_fluxv = [ai / bi for ai, bi in zip(fluxv, norm_fac_lst)]
    div_fluxv_pl = [ai / bi for ai, bi in zip(fluxv_pl, norm_fac_lst_pl)]

    err_tmp1 = [x1 + x2 for (x1, x2) in zip(fluxv, fluxv_err)]
    err_tmp2 = [ai / bi for ai, bi in zip(err_tmp1, norm_fac_lst)]
    div_fluxv_err = [x1 - x2 for (x1, x2) in zip(err_tmp2, div_fluxv)]

    err_tmp1_pl = [x1 + x2 for (x1, x2) in zip(fluxv_pl, fluxv_err_pl)]
    err_tmp2_pl = [ai / bi for ai, bi in zip(err_tmp1_pl, norm_fac_lst_pl)]
    div_fluxv_err_pl = [x1 - x2 for (x1, x2) in zip(err_tmp2_pl, div_fluxv_pl)]

    div_y_fit = [ai / bi for ai, bi in zip(y_fit, norm_fac_lst)]
    div_y_min = [ai / bi for ai, bi in zip(y_min, norm_fac_lst)]
    div_y_max = [ai / bi for ai, bi in zip(y_max, norm_fac_lst)]
    div_y_min2 = [ai / bi for ai, bi in zip(y_min2, norm_fac_lst)]
    div_y_max2 = [ai / bi for ai, bi in zip(y_max2, norm_fac_lst)]

    fit_file2 = open(grb_name + "_" + str(nvoigts) + "_" + str(l0)
                     + ".dat", "w")
    fit_file2.write('element ' + str(element) + '\n')
    fit_file2.write('redshift ' + str(redshift) + '\n')
    fit_file2.write('l0 ' + str(l0) + '\n')
    fit_file2.write('gamma ' + str(gamma) + '\n')
    fit_file2.write('f ' + str(f) + '\n')
    fit_file2.write('transform ' + str(transform) + '\n')
    fit_file2.write('ign_s ' + str(ign_s) + '\n')
    fit_file2.write('RES ' + str(RES) + '\n')
    fit_file2.write('N ' + str(N_all) + '\n')
    fit_file2.write('N_err ' + str(N_all_err) + '\n')
    fit_file2.write('b ' + str(b_all) + '\n')
    fit_file2.write('b_err ' + str(b_all_err) + '\n')
    fit_file2.write('v0 ' + str(v0_all) + '\n')
    fit_file2.write('v0_err ' + str(v0_all_err) + '\n')
    fit_file2.write('####### MODEL ######' + '\n')
    fit_file2.write("v m m_err_min m_err_max m_err_min2 m_err_max2" + "\n")

    ax.errorbar(velocity, div_fluxv_err, linestyle='dotted', color="black",
                linewidth=0.5, drawstyle='steps-mid', label="Error Spectrum", alpha=0.6)

    ax.errorbar(velocity_pl, div_fluxv_pl, yerr=fluxv_err_pl,
                color='gray', marker='o',
                ls='None', label='Observed')

    ax.plot(velocity_pl, div_fluxv_pl, drawstyle='steps-mid',
            color='gray', alpha=0.66)

    if len(ign_s) > 0:
        for i in range(len(velo_arrays)):

            norm_fac_lst = par_dic["a"][0] + par_dic["a1"][0] * np.array(velo_arrays[i]) + \
                par_dic["a2"][0] * np.array(power_lst(velo_arrays[i], 2))

            div_y_fit = [ai / bi for ai,
                         bi in zip(y_fit_arrays[i], norm_fac_lst)]
            div_y_min = [ai / bi for ai,
                         bi in zip(y_min_arrays[i], norm_fac_lst)]
            div_y_max = [ai / bi for ai,
                         bi in zip(y_max_arrays[i], norm_fac_lst)]
            div_y_min2 = [ai / bi for ai,
                          bi in zip(y_min2_arrays[i], norm_fac_lst)]
            div_y_max2 = [ai / bi for ai,
                          bi in zip(y_max2_arrays[i], norm_fac_lst)]

            ax.plot(velo_arrays[i], div_y_fit, label='Fit',
                    color="black", linewidth=1.5, linestyle="dashed")
            ax.fill_between(velo_arrays[i], div_y_min, div_y_max,
                            color='black', alpha=0.3)
            ax.fill_between(velo_arrays[i], div_y_min2, div_y_max2,
                            color='black', alpha=0.5)

            for j in np.arange(0, len(velo_arrays[i]), 1):
                fit_file2.write(str(velo_arrays[i][j]) +
                                " " + str(div_y_fit[j]) +
                                " " + str(div_y_min[j]) + " " + str(div_y_max[j]) +
                                " " + str(div_y_min2[j]) + " " + str(div_y_max2[j]) + "\n")

            fit_file2.write('#########' + '\n')
    else:
        ax.plot(velocity, div_y_fit, label='Fit', color="black",
                linewidth=1.5,
                linestyle="dashed")
        ax.fill_between(velocity, div_y_min, div_y_max,
                        color='black', alpha=0.3)
        ax.fill_between(velocity, div_y_min2, div_y_max2,
                        color='black', alpha=0.5)

        for i in np.arange(0, len(velocity), 1):
            fit_file2.write(str(velocity[i]) +
                            " " + str(div_y_fit[i]) +
                            " " + str(div_y_min[i]) + " " + str(div_y_max[i]) +
                            " " + str(div_y_min2[i]) + " " + str(div_y_max2[i]) + "\n")

    fit_file2.write('#########' + '\n')
    fit_file2.write('v f f_err' + '\n')
    fit_file2.write('####### DATA ######' + '\n')
    for i in np.arange(0, len(velocity_pl), 1):
        fit_file2.write(str(velocity_pl[i]) +
                        " " + str(div_fluxv_pl[i]) +
                        " " + str(div_fluxv_err_pl[i]) + '\n')
    fit_file2.close()

    lg = legend(numpoints=1, fontsize=12, loc=3, ncol=2)
    lg.get_frame().set_edgecolor("white")
    lg.get_frame().set_facecolor('#f0f0f0')

    plt.title(str(grb) + " " + element + " " + str(l0) +
              " at z = " + str(redshift), fontsize=24)

    ax.set_xlabel(r"$\sf Velocity\, (km/s)$", fontsize=24)
    ax.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=24)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(which='major', length=8, width=2)
    ax.tick_params(which='minor', length=4, width=1.5)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    # show()
    fig.savefig(grb_name + "_" + element + "_" +
                str(l0) + "_" + str(nvoigts) + ".pdf")

    return N_all, b_all, v0_all


def plt_nv_chi2(chi2_list, min_n, max_n, grb_name):
    '''
    Plot reduced Chi^2 vs. number of components 
    '''

    fig = figure(figsize=(12, 6))
    ax = fig.add_axes([0.10, 0.14, 0.86, 0.85])

    ax.errorbar(range(min_n, max_n + 1), chi2_list, linewidth=5)
    ax.errorbar(range(min_n, max_n + 1), chi2_list, fmt="o", color="black",
                markersize=15)
    ax.set_xlabel(r"Number of Components", fontsize=24)
    ax.set_ylabel(r"${\chi}^2_{red}$", fontsize=24)
    ax.set_yscale("log")
    ax.set_ylim([0.1, 600])
    ax.set_xlim([min_n - 0.5, max_n + 0.5])
    ax.set_xticks(range(min_n, max_n + 1))
    ax.set_yticks([0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 50, 100, 200, 500])
    ax.set_yticklabels(["0.2", "0.5", "1.0", "2.0", "5.0", "10",
                        "20", "50", "100", "200", "500"])
    ax.axhline(1, xmin=0.0, xmax=1.0, linewidth=2, linestyle="dashed",
               color="black")

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(which='major', length=8, width=2)
    ax.tick_params(which='minor', length=4, width=1.5)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    fig.savefig(grb_name + "_Chi2red.pdf")

if __name__ == "__main__":

    writecmd("velo_cmd_hist.dat")

    start = time.time()
    print "\n Parsing Arguments \n"

    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('-f', '--file', dest="file",
                        default=None, type=str)
    parser.add_argument('-z', '--z', dest="z", default=None, type=float)
    parser.add_argument('-e', '--element', dest="element",
                        default="FeII", type=str)
    parser.add_argument('-line', '--line', dest="line",
                        default=1608.4509, type=float)
    parser.add_argument('-w', '--wav_range', dest="wav_range",
                        default=10.0, type=float)
    parser.add_argument('-res', '--res', dest="resolution",
                        default=30, type=float)
    parser.add_argument('-it', '--it', dest="it", default=10000, type=int)
    parser.add_argument('-bi', '--bi', dest="bi", default=5000, type=int)
    parser.add_argument('-min', '--min', dest="min", default=1, type=int)
    parser.add_argument('-max', '--max', dest="max", default=1, type=int)
    parser.add_argument('-vr', '--velo_range', dest="velo_range",
                        default=410.0, type=float)
    parser.add_argument('-par', '--par', dest="par", default=None, type=str)
    parser.add_argument('-plr', '--plr', dest="plr", default=False, type=bool)
    parser.add_argument('-ign', '--ignore', dest="ignore",
                        nargs='+', default=[])
    parser.add_argument('-vand', '--vandels', dest="vandels",
                        default=False, type=bool)


    args = parser.parse_args()

    spec_file = args.file
    element = args.element
    redshift = args.z
    wav_range = args.wav_range
    l0 = args.line
    iterations = args.it
    burn_in = args.bi
    min_n = args.min
    max_n = args.max
    velo_range = args.velo_range
    para_file = args.par
    RES = args.resolution
    plr = args.plr
    ignore = args.ignore
    vandels = args.vandels

    # Read Oscillator strength f and decay rate gamma
    # for given line
    f, gamma_line = get_osc_gamma(l0)

    # converting gamma A_u to km/s
    gamma = (gamma_line * l0 * 10e-13) / (2 * math.pi)

    ignore_lst = []
    for itrvl in ignore:
        tmp_lst = []
        s = itrvl.split(",")
        if float(s[0]) > float(s[1]):
            tmp_lst.extend((-float(s[0]), -float(s[1])))
            ignore_lst.append(tmp_lst)
        else:
            tmp_lst.extend((float(s[0]), float(s[1])))
            ignore_lst.append(tmp_lst)

    print "\n Fitting", element, l0, "with", iterations, \
        "iterations and a burn-in of", burn_in

    print "\n ignore are:", ignore_lst


    if vandels == False:
        wav_aa, flux_notell, flux_err_notell, qual, cont, \
            cont_err, flux, flux_err, \
            norm_flux, norm_flux_err, \
            n_flux_pl, n_flux_err_pl = \
            get_data2(spec_file, redshift, wl_range=False)

        grb_name = spec_file.strip('spectra/')[0:-10]

    if vandels == True:
        wav_aa, flux, flux_err = easy_open(spec_file)
        #wav_aa, flux, flux_err = get_data_vandels(spec_file)
        grb_name = '180325A'

    if not min(wav_aa) < l0 * (1 + redshift) < max(wav_aa):
        print "Line is at:", l0 * (1 + redshift), "Spectrum covers: ", \
            min(wav_aa), "to", max(wav_aa)
        sys.exit("ERROR: Chosen line must fall within the wavelength \
			range of the given file")

    if burn_in >= iterations:
        sys.exit("ERROR: Burn-In cannot be bigger than Iterations")

    if redshift == None:
        sys.exit("ERROR: please specify input redshift: e.g. -z 2.358")

    para_dic = {}

    if para_file != None:
        para_dic = get_paras_velo(para_file)
        print "\n Using parameters given in:", para_file

    velocity_pl, fluxv_pl, fluxv_err_pl = aa_to_velo(wav_aa, flux,
                                                     flux_err, l0, redshift, wav_range)

    velocity, fluxv, fluxv_err = aa_to_velo_ign(wav_aa, flux,
                                                flux_err, l0, redshift, ignore_lst, wav_range)


    #for i in range(len(velocity)):
    #   print velocity[i], fluxv[i], fluxv_err[i]

    transform = np.median(np.diff(velocity_pl))


    chi2_list = []

    for nvoigts in range(min_n, max_n + 1):

        print"\n Using", nvoigts, \
            "Voigt Profile(s) convolved with R =", RES, "km/s"

        y_min, y_max, y_min2, y_max2, y_fit = do_mcmc(grb_name,
                                                      redshift, velocity, fluxv, fluxv_err, grb_name, f, gamma,
                                                      l0, nvoigts, iterations, burn_in, RES,
                                                      velo_range, para_dic)

        chi2 = 0
        for i in range(4, len(y_fit) - 4, 1):
            chi2_tmp = (fluxv[i] - y_fit[i])**2 / (fluxv_err[i])**2
            chi2 += (chi2_tmp / (len(fluxv) + (4 * nvoigts)))
        chi2_list.append(chi2)

        print "\n Chi^2_red:", chi2, "\n"

        time.sleep(0.5)

        res_file = grb_name + "_" + \
            str(nvoigts) + "_" + str(l0) + "_voigt_res.csv"

        N_all, b_all, v0_all = plot_results(grb_name + str(nvoigts), redshift,
                                            velocity, fluxv, fluxv_err, velocity_pl, fluxv_pl, fluxv_err_pl,
                                            y_min, y_max, y_min2, y_max2, y_fit, res_file, f, gamma, l0, nvoigts,
                                            velo_range, grb_name, RES, ignore_lst, element)

        print "Components:", print_results(res_file, redshift)

        if plr == True:
            sns_velo_trace_plot(grb_name, l0, file='velo_fit.pickle',
                                nvoigts=nvoigts)
            sns_velo_pair_plot(grb_name, l0, file='velo_fit.pickle',
                               nvoigts=nvoigts)

    print "\n "
    print "restframe wavlength: ", l0
    print "gamma in km/s: ", gamma
    print "f: ", f
    print "wavlength range: ", wav_range
    print "transform ", transform
    print "resoluton in km/s: ", RES

    dur = str(round((time.time() - start) / 60, 1))
    sys.exit("\n Script finished after " + dur + " minutes")

#========================================================================
#========================================================================
