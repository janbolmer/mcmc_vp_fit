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

def power_lst(my_list, exp):
    '''
    x**exp for each x in my_list
    '''
    return [x**exp for x in my_list]

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

def gauss(x, mu, sig):
    '''
    Normal distribution used to create prior probability distributions
    '''
    return np.exp(-np.power(x - mu, 2.0) / (2.0 * np.power(sig, 2.0)))

def mult_voigts(velocity, fluxv, fluxv_err, f, gamma, l0, nvoigts, RES, velo_range):
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

    tau_s = []   
    for i in [0, 1]:
        tau_s.append(1 / np.array(fluxv_err[i])**2)

    #@pymc.stochastic(dtype=float)
    # def a(value=1.0, mu=1.0, sig=0.1, doc="B"):
    # pp = 0.0
    # #if 0.85 <= value < 1.15:
    # pp = gauss(value, mu, sig)
    # #else:
    # # pp = -np.inf
    # return pp

    # Continuum model (up to quadratic polinomial)
    mu_bg_1 = np.nansum(fluxv[0]) / (len(fluxv[0]) - fluxv[0].count(np.nan))
    mu_bg_2 = np.nansum(fluxv[1]) / (len(fluxv[1]) - fluxv[1].count(np.nan))

    print mu_bg_1, mu_bg_2

    @pymc.stochastic(dtype=float)
    def a_1(value=mu_bg_1, mu=mu_bg_1, sig=0.5 * mu_bg_1, doc="a"):
        if mu_bg_1 / 2.0 < value < mu_bg_1 * 2.0:
            pp = gauss(value, mu, sig)
        else:
            pp = -np.inf
        return pp

    @pymc.stochastic(dtype=float)
    def a1_1(value=0.0, mu=0.0, sig=0.5, doc="a1"):
        if -0.3 < value < 0.3:
            pp = gauss(value, mu, sig)
        else:
            pp = -np.inf
        return pp

    @pymc.stochastic(dtype=float)
    def a2_1(value=0.0, mu=0.0, sig=0.5, doc="a2"):
        if -0.3 < value < 0.3:
            pp = gauss(value, mu, sig)
        else:
            pp = -np.inf
        return pp


    @pymc.stochastic(dtype=float)
    def a_2(value=mu_bg_2, mu=mu_bg_2, sig=0.5 * mu_bg_2, doc="a"):
        if mu_bg_2 / 2.0 < value < mu_bg_2 * 2.0:
            pp = gauss(value, mu, sig)
        else:
            pp = -np.inf
        return pp

    @pymc.stochastic(dtype=float)
    def a1_2(value=0.0, mu=0.0, sig=0.5, doc="a1"):
        if -0.3 < value < 0.3:
            pp = gauss(value, mu, sig)
        else:
            pp = -np.inf
        return pp

    @pymc.stochastic(dtype=float)
    def a2_2(value=0.0, mu=0.0, sig=0.5, doc="a2"):
        if -0.3 < value < 0.3:
            pp = gauss(value, mu, sig)
        else:
            pp = -np.inf
        return pp


    vars_dic = {}

    for i in range(1, nvoigts + 1):

        v0 = pymc.Uniform('v0' + str(i), lower=-velo_range, upper=velo_range, doc='v0' + str(i))
        b = pymc.DiscreteUniform('b' + str(i), lower=round(low_b, 0), upper=30, value=low_b+20, doc='b' + str(i))
        N = pymc.Uniform('N' + str(i), lower=0.0, upper=20, value=15, doc='N' + str(i))

        vars_dic['v0' + str(i)] = v0
        vars_dic['b' + str(i)] = b
        vars_dic['N' + str(i)] = N

    print "\n Starting MCMC " + '(pymc version:', pymc.__version__, ")"
    print "\n This might take a while ..."

    @pymc.deterministic(plot=False)
    def multVoigt(vv=velocity, a_1=a_1, a1_1=a1_1, a2_1=a2_1, a_2=a_2, a1_2=a1_2, a2_2=a2_2,
                  f=f, gamma=gamma, l0=l0,
                  nvoigts=nvoigts, vars_dic=vars_dic):

        model_matrix = []

        for i in [0, 1]:
          conv_val = RES / (2 * np.sqrt(2 * np.log(2)) * tf[i])
          gauss_k = Gaussian1DKernel(stddev=conv_val, mode="oversample")
  
          if i == 0:
              flux = np.ones(len(vv[i])) * a_1 #(a_1 + a1_1 * vv[i] + a2_1 * (power_lst(vv[i], 2)))
          if i == 1:
              flux = np.ones(len(vv[i])) * a_2 # (a_2 + a1_2 * vv[i] + a2_2 * (power_lst(vv[i], 2)))

          for j in range(1, nvoigts + 1):
              v = vv[i] - vars_dic["v0" + str(j)]
              flux *= add_abs_velo(v, vars_dic["N" + str(j)],
                                   vars_dic["b" + str(j)], gamma[i], f[i], l0[i])
  
          #model_matrix.append(flux)
          model_matrix.append(np.convolve(flux, gauss_k, mode='same'))


        #print a_1, a1_1, a2_1, a_2, a1_2, a2_2
        #print vars_dic
        #print model_matrix
        return model_matrix

    y_val = pymc.Normal('y_val', mu=multVoigt, tau=tau_s, value=fluxv, observed=True)

    return locals()

def do_mcmc(grb, redshift, velocity, fluxv, fluxv_err, grb_name, f, gamma,
            l0, nvoigts, iterations, burn_in, RES, velo_range):
    '''
    MCMC sample 
    Reading and writing Results
    '''

    CSV_LST = ["b1", "N1", "v01", "b2", "N2", "v02", "b3", "N3", "v03"]

    pymc.np.random.seed(1)

    MDL = pymc.MCMC(mult_voigts(velocity, fluxv, fluxv_err,
                                f, gamma, l0, nvoigts, RES, velo_range),
                    db='pickle', dbname='velo_fit.pickle')

    MDL.db
    #MDL.use_step_method(pymc.AdaptiveMetropolis, [MDL.N, MDL.b],
    #                   scales={MDL.N: 1.0, MDL.b: 1.0}, delay=10000)
    MDL.sample(iterations, burn_in)
    MDL.db.close()

    # Autocorrelation Plots
    # pymc.Matplot.autocorrelation(MDL)

    y_min = MDL.stats()['multVoigt']['quantiles'][2.5]
    y_max = MDL.stats()['multVoigt']['quantiles'][97.5]
    y_min2 = MDL.stats()['multVoigt']['quantiles'][25]
    y_max2 = MDL.stats()['multVoigt']['quantiles'][75]
    y_fit = MDL.stats()['multVoigt']['mean']

    MDL.write_csv("result.csv", variables=CSV_LST)
    #
    #csv_f = open(grb_name + "_" + str(nvoigts) + "_" + str(l0) +
    #             "_voigt_res.csv", "a")
    #csv_f.write("Osc, " + str(f) + "\n")
    #csv_f.write("GAMMA, " + str(gamma_line))
    #csv_f.close()

    return y_min, y_max, y_min2, y_max2, y_fit

redshift = 2.710
igns1 = [[400, 800]]
igns2 = []
ignore_lst = [igns1, igns2]
RES = 29.33
vr = 290
wr = [20, 20]
nv = 3

it = 10000
bi = 8000

wav_aa, flux_notell, flux_err_notell, qual, cont, \
  cont_err, flux, flux_err, \
  norm_flux, norm_flux_err, \
  n_flux_pl, n_flux_err_pl = \
get_data2('spectra/GRB161023A_OB1VIS.fits', redshift, wl_range=False)


velo_pl, flv_pl, flv_err_pl = [], [], []
velo, flv, flv_err = [], [], []

tf = []

f_s = []
gamma_s = []
l0s = [1608.4509, 2374.4604]



for i in range(len(l0s)):
    velocity_pl, fluxv_pl, fluxv_err_pl = aa_to_velo(wav_aa, flux, flux_err, l0s[i], redshift, wr[i])
    velocity, fluxv, fluxv_err = aa_to_velo_ign_nan(wav_aa, flux, flux_err, l0s[i], redshift, ignore_lst[i], wr[i])

    velo_pl.append(velocity_pl)
    flv_pl.append(fluxv_pl)
    flv_err_pl.append(fluxv_err_pl)
    velo.append(velocity)
    flv.append(fluxv)
    flv_err.append(fluxv_err)

    tf.append(np.median(np.diff(velocity_pl)))

    f, gamma_line = get_osc_gamma(l0s[i])
    gamma = (gamma_line * l0s[i] * 10e-13) / (2 * math.pi)
    f_s.append(f)
    gamma_s.append(gamma)

print tf
print len(velo[0]), len(velo[1])

#if len(velo[0]) != len(velo[1]):
#
#	diff_val = abs(len(velo[0])-len(velo[1]))
#	if len(velo[0]) > len(velo[1]):
#		velo[1].extend([velo[1][0] for i in range(diff_val)])
#		flv[1].extend([flv[1][0] for i in range(diff_val)])
#		flv_err[1].extend([0.0 for i in range(diff_val)])
#	else:
#		velo[0].extend([velo[0][0] for i in range(diff_val)])
#		flv[0].extend([flv[0][0] for i in range(diff_val)])
#		flv_err[0].extend([0.0 for i in range(diff_val)])	


y_min, y_max, y_min2, y_max2, y_fit = do_mcmc('120815A', redshift, velo, flv, flv_err,
                                              '120815A', f_s, gamma_s, l0s, nv, it, bi, RES, vr)


sns.set_style("white", {'legend.frameon': True})
fig = figure(figsize=(6, 6))

#ax1.set_ylabel("Normalized Flux", fontsize=20, labelpad=10)
#ax1.set_xlabel("Relative Velocity (km/s)", fontsize=20, labelpad=20)

ax1 = fig.add_axes([0.10, 0.10, 0.80, 0.35])
ax2 = fig.add_axes([0.10, 0.55, 0.80, 0.35])

ax1.plot(velo[0], y_fit[0], color='black')
ax1.errorbar(velo[0], flv[0], yerr=flv_err[0], fmt='o', color='grey')
ax1.fill_between(velo[0], y_min[0], y_max[0], color='black', alpha=0.3)


ax1.set_xlim([-550, 550])
ax1.set_ylim([min(flv[0])-0.2*min(flv[0]), max(flv[0])]+0.1*max(flv[0]))

ax2.plot(velo[1], y_fit[1], color='black')
ax2.errorbar(velo[1], flv[1], yerr=flv_err[1], fmt='o', color='grey')
ax2.fill_between(velo[1], y_min[1], y_max[1], color='black', alpha=0.3)
ax2.set_xlim([-550, 550])
ax2.set_ylim([min(flv[1])-0.2*min(flv[1]), max(flv[1])]+0.1*max(flv[1]))

show()












































