#! /usr/bin/python

__author__ = "Jan Bolmer"
__copyright__ = "Copyright 2017"
__version__ = "0.1"
__maintainer__ = "Jan Bolmer"
__email__ = "jbolmer@eso.org"
__status__ = "Production"

import math, argparse
import os, sys, time

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as plt
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gs
from pylab import *

sys.path.append('bin/')

from read_fits import *
from spec_functions import aa_to_velo, writecmd

def get_h2_df(min_wav=900):

	h2df = pd.read_csv('atoms/h2.csv', sep=', ',
		index_col=None, engine='python')
	
	J0 = h2df['line'].str.contains('J0')
	J1 = h2df['line'].str.contains('J1')
	
	Jmw = h2df['wav'] > min_wav

	h2j0df = h2df[J0 & Jmw]
	h2j0df = h2j0df.loc[~h2j0df.wav.between(1010, 1030)]
	h2j0df = h2j0df.sort_values('osz', axis=0, ascending=False)
	
	h2j1df = h2df[J1 & Jmw]
	h2j1df = h2j1df.loc[~h2j1df.wav.between(1010, 1030)]
	h2j1df = h2j1df.sort_values('osz', axis=0, ascending=False)

	return h2j0df, h2j1df

##################################################

def h2_velo_plot(h2j0df, h2j1df, data_file, redshift, 
	comps, vr, t_name, data_file_line, l0_line):

	# Get the data for the H2 lines
	wav_aa, flux_notell, flux_err_notell, qual, cont, \
		cont_err, flux, flux_err, \
		norm_flux, norm_flux_err, \
		n_flux_pl, n_flux_err_pl = \
		get_data2(data_file, redshift, wl_range=False)

	# Get the data for the reference line
	wav_aa2, flux_notell2, flux_err_notell2, qual2, cont2, \
		cont_err2, flux2, flux_err2, \
		norm_flux2, norm_flux_err2, \
		n_flux_pl2, n_flux_err_pl2 = \
		get_data2(data_file_line, redshift, wl_range=False)

	######################### H2J0 #########################
	# Create the Plot for H2J0 lines:
	######################### H2J0 #########################
	f1, f1_axes = plt.subplots(ncols=1, nrows=8+1,
		sharex=True, sharey=True)
	f1.set_figheight(8)
	f1.set_figwidth(4)

	for i in range(len(h2j0df)):
		if i < 8:
			f1_axes[i].axhline(1, linestyle="dotted", linewidth=1, color="black")
			f1_axes[i].axhline(0, linestyle="dashed", linewidth=0.5, color="black")
	
			velocity, fluxv, fluxv_err = aa_to_velo(wav_aa,
				n_flux_pl, n_flux_err_pl, np.array(h2j0df['wav'])[i],
				redshift, 40)
			f1_axes[i].errorbar(velocity, fluxv, linestyle='-', color="#08306b",
				linewidth=1, drawstyle='steps-mid')

			f1_axes[i].fill_between(velocity, fluxv,
				np.array(fluxv)+np.array(fluxv_err), color='black', alpha=0.15)
			f1_axes[i].fill_between(velocity, fluxv,
				np.array(fluxv)-np.array(fluxv_err), color='black', alpha=0.15)


			for v in comps:
				f1_axes[i].axvline(v, linestyle="dashed", color="#d7301f")
			
			str1 = str(round(np.array(h2j0df['wav'])[i],1))
			str2 = str(np.array(h2j0df['LW'])[i])+str(np.array(h2j0df['PR'])[i])
			f1_axes[i].text(-vr+20,1.6,'J0 '+str1+str2, fontsize=8,
				bbox=dict(boxstyle="round", ec=None, fc='#f0f0f0', lw=0.2))


	f1_axes[8].axhline(1, linestyle="dotted", linewidth=1, color="black")
	f1_axes[8].axhline(0, linestyle="dashed", linewidth=0.5, color="black")
	velocity2, fluxv2, fluxv_err2 = aa_to_velo(wav_aa2, n_flux_pl2,
		n_flux_err_pl2, l0_line, redshift, 40)

	f1_axes[8].errorbar(velocity2, fluxv2, linestyle='-', color="#08306b",
		linewidth=1, drawstyle='steps-mid')
	for v in comps:
		f1_axes[8].axvline(v, linestyle="dashed", color="#d7301f")
	f1_axes[8].text(-vr+20,1.5,str(l0_line),fontsize=8)

	ylim(-0.2, 2.05)
	for ax in f1_axes:
		ax.set_yticks([0, 1, 2])
	xlim(-vr, vr)
	xlabel(r"$\sf Velocity\, (km/s)$", fontsize=16)
	f1_axes[4].set_ylabel(r"$\sf Normalized\, Flux$", fontsize=16)
	f1.subplots_adjust(wspace=0, hspace=0.2, top=0.99, bottom=0.08, right=0.98)
	plt.savefig(t_name+'_H2J0.pdf')


	######################### H2J1 #########################
	# Create the Plot for H2J1 lines:
	######################### H2J1 #########################
	f2, f2_axes = plt.subplots(ncols=1, nrows=8+1,
		sharex=True, sharey=True)

	f2.set_figheight(8)
	f2.set_figwidth(4)

	for i in range(len(h2j1df)):

		if i < 8:
			f2_axes[i].axhline(1, linestyle="dotted", linewidth=1, color="black")
			f2_axes[i].axhline(0, linestyle="dashed", linewidth=0.5, color="black")
	
			velocity, fluxv, fluxv_err = aa_to_velo(wav_aa,
				n_flux_pl, n_flux_err_pl, np.array(h2j1df['wav'])[i],
				redshift, 40)
			f2_axes[i].errorbar(velocity, fluxv, linestyle='-', color="#08306b",
				linewidth=1, drawstyle='steps-mid')

			f2_axes[i].fill_between(velocity, fluxv,
				np.array(fluxv)+np.array(fluxv_err), color='black', alpha=0.15)
			f2_axes[i].fill_between(velocity, fluxv,
				np.array(fluxv)-np.array(fluxv_err), color='black', alpha=0.15)

			for v in comps:
				f2_axes[i].axvline(v, linestyle="dashed", color="#d7301f")
	
			str1 = str(round(np.array(h2j1df['wav'])[i],1))
			str2 = str(np.array(h2j1df['LW'])[i])+str(np.array(h2j1df['PR'])[i])
			f2_axes[i].text(-vr+20,1.6,'J0 '+str1+str2, fontsize=8,
				bbox=dict(boxstyle="round", ec=None, fc='#f0f0f0', lw=0.2))

	f2_axes[8].axhline(1, linestyle="dotted", linewidth=1, color="black")
	f2_axes[8].axhline(0, linestyle="dashed", linewidth=0.5, color="black")
	velocity2, fluxv2, fluxv_err2 = aa_to_velo(wav_aa2, n_flux_pl2,
		n_flux_err_pl2, l0_line, redshift, 40)

	f2_axes[8].errorbar(velocity2, fluxv2, linestyle='-', color="#08306b",
		linewidth=1, drawstyle='steps-mid')
	for v in comps:
		f2_axes[8].axvline(v, linestyle="dashed", color="#d7301f")

	f2_axes[8].text(-vr+20,1.5,str(l0_line),fontsize=8)

	ylim(-0.2, 2.05)
	for ax in f2_axes:
		ax.set_yticks([0, 1, 2])
	xlim(-vr, vr)

	xlabel(r"$\sf Velocity\, (km/s)$", fontsize=16)

	f2_axes[4].set_ylabel(r"$\sf Normalized\, Flux$", fontsize=16)
	f2.subplots_adjust(wspace=0, hspace=0.2, top=0.99, bottom=0.08, right=0.98)

	plt.savefig(t_name+'_H2J1.pdf')


##################################################

if __name__ == "__main__":

    writecmd("vinspec_old.dat")

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest="file",
                        default="spectra/GRB130408A_OB1UVB.fits", type=str)
    parser.add_argument('-fl', '--file_line', dest="file_line",
                        default="spectra/GRB130408A_OB1VIS.fits", type=str)
    parser.add_argument('-t', '--target_name', dest="target_name",
                        default="GRB", type=str)
    parser.add_argument('-z', '--z', dest="z",
                        default=3.7579, type=float)
    parser.add_argument('-l0', '--l0_line', dest="l0_line",
                        default=1808.0129, type=float)
    parser.add_argument('-vr', '--velo_range', dest="velo_range",
                        default=210.0, type=float)
    parser.add_argument('-c', '--components', dest="components", nargs='+',
                        default=[], type=float)
    parser.add_argument('-mw', '--min_wav', dest="min_wav", type=float,
                        default=980)   
    args = parser.parse_args()

    data_file = args.file
    data_file_line = args.file_line
    t_name = args.target_name
    redshift = args.z
    vr = args.velo_range
    comps = args.components
    mw = args.min_wav
    l0_line = args.l0_line

    #print comps

    #comps = [2.49, 72.23] # for testing

    h2j0df, h2j1df = get_h2_df(min_wav=mw)

    #print h2j0df
    #print h2j1df

    print 'plotting the strongest 8 J0 and J1 lines in velocity space'
    print 'for lines at >', mw, '(Angstrom)'

    x = h2_velo_plot(h2j0df, h2j1df, data_file, redshift,
    	comps, vr, t_name, data_file_line, l0_line)









