#! /usr/bin/python

import pymc, math, argparse, os, sys, time
import numpy as np
import seaborn as sns

sys.path.append('bin/')
from read_fits import *
from syn_spec import *
from spec_functions import get_lines


def plot_syn_h2(wav_aa, n_flux, n_flux_err, target,
				bh2, nh2, th2, redshift, redshifts, res, indv,
				nh2j0, nh2j1, nh2j2, bsh2):

	if indv == False:
		print bh2, nh2, th2
		spec 	= np.ones(len(wav_aa))
		synspec = SynSpec(wav_aa, redshift, res, [])
		spec 	= synspec.add_H2(spec, bh2, nh2, th2, 0.0, NROT=np.arange(0, 7, 1))

	else:
		spec 	= np.ones(len(wav_aa))
		synspec = SynSpec(wav_aa, redshift, res, [])
		for i in range(len(redshifts)):
			print float(nh2j0[i]), float(nh2j1[i]), float(nh2j2[i]), float(bsh2[i]), float(redshifts[i])
			spec 	= synspec.add_line(spec, 'H2J0', float(nh2j0[i]), float(bsh2[i]), float(redshifts[i]))
			spec 	= synspec.add_line(spec, 'H2J1', float(nh2j1[i]), float(bsh2[i]), float(redshifts[i]))
			spec 	= synspec.add_line(spec, 'H2J2', float(nh2j2[i]), float(bsh2[i]), float(redshifts[i]))

	a_name, a_wav, ai_name, ai_wav, aex_name, \
	aex_wav, h2_name, h2_wav, co_name, co_wav = get_lines(redshift)

	sns.set_style("white", {'legend.frameon': True})
	wav_range = (max(wav_aa)-min(wav_aa))/6.0

	fig = figure(figsize=(10, 12))
	ax1 = fig.add_axes([0.08, 0.08, 0.90, 0.11])
	ax2 = fig.add_axes([0.08, 0.25, 0.90, 0.11])
	ax3 = fig.add_axes([0.08, 0.41, 0.90, 0.11])
	ax4 = fig.add_axes([0.08, 0.58, 0.90, 0.11])
	ax5 = fig.add_axes([0.08, 0.73, 0.90, 0.11])
	ax6 = fig.add_axes([0.08, 0.88, 0.90, 0.11])

	for axis in [ax1, ax2, ax3, ax4, ax5, ax6]:
		axis.errorbar(wav_aa, n_flux, linestyle='-', color="black",
			linewidth=0.5, drawstyle='steps-mid', label=r"$\sf Spec.$")

		#if axis != ax1:
		#	axis.fill_between(wav_aa, 1, n_flux, color='#fec44f', alpha=0.5)

		#if axis != ax1:
		axis.fill_between(wav_aa, n_flux, n_flux+n_flux_err, color='black', alpha=0.15)
		axis.fill_between(wav_aa, n_flux, n_flux-n_flux_err, color='black', alpha=0.15)

		axis.plot(wav_aa, spec, label=r"$\sf Model$", color="#2171b5", linewidth=1.5, alpha=0.9)

		axis.set_ylim([-0.85, 1.55])
		axis.axhline(0.0, linestyle="dashed", color="black", linewidth=2)

		for side in ['top','bottom','left','right']:
		  	axis.spines[side].set_linewidth(2)
		axis.tick_params(which='major', length=8, width=2)
		axis.tick_params(which='minor', length=6, width=1)
		for tick in axis.xaxis.get_major_ticks():
			tick.label.set_fontsize(18)
		for tick in axis.yaxis.get_major_ticks():
			tick.label.set_fontsize(18)


	for i in np.arange(0, len(h2_name), 1):
			if min(wav_aa) < h2_wav[i] < (max(wav_aa)-wav_range*5):
				if i%2 == 0:
					ax6.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
					ax6.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
				else:
					ax6.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
					ax6.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
	
			if (max(wav_aa)-wav_range*5) < h2_wav[i] < (max(wav_aa)-wav_range*4):
				if i%2 == 0:
					ax5.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
					ax5.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
				else:
					ax5.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
					ax5.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
	
			if (max(wav_aa)-wav_range*4) < h2_wav[i] < (max(wav_aa)-wav_range*3):
				if i%2 == 0:
					ax4.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
					ax4.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
				else:
					ax4.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
					ax4.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
	
			if (max(wav_aa)-wav_range*3) < h2_wav[i] < (max(wav_aa)-wav_range*2):
				if i%2 == 0:
					ax3.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
					ax3.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
				else:
					ax3.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
					ax3.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
			if (max(wav_aa)-wav_range*2) < h2_wav[i] < (max(wav_aa)-wav_range*1):
				if i%2 == 0:
					ax2.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
					ax2.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
				else:
					ax2.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
					ax2.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
			if (max(wav_aa)-wav_range*1) < h2_wav[i] < max(wav_aa):
				if i%2 == 0:
					ax1.text(h2_wav[i]+0.2, -0.5, h2_name[i], fontsize=6,color="#a50f15")
					ax1.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)
				else:
					ax1.text(h2_wav[i]+0.2, -0.8, h2_name[i], fontsize=6,color="#a50f15")
					ax1.axvline(h2_wav[i], ymin=0.0, ymax=0.2, linestyle="-", color="#a50f15", linewidth=0.8)

	ax1.set_xlabel(r"$\sf Observed\, Wavelength (\AA)$", fontsize=24)
	ax3.set_ylabel(r"$\sf Normalized\, Flux$", fontsize=24)

	ax1.axhline(1, color="#2171b5", linewidth=1.0)
	ax6.set_xlim([min(wav_aa)+0.25, max(wav_aa)-wav_range*5-0.25])
	ax5.set_xlim([max(wav_aa)-wav_range*5, max(wav_aa)-wav_range*4])
	ax4.set_xlim([max(wav_aa)-wav_range*4, max(wav_aa)-wav_range*3])
	ax3.set_xlim([max(wav_aa)-wav_range*3, max(wav_aa)-wav_range*2])
	ax2.set_xlim([max(wav_aa)-wav_range*2, max(wav_aa)-wav_range*1])
	ax1.set_xlim([max(wav_aa)-wav_range*1+0.25, max(wav_aa)-0.25])

	show()
	fig.savefig(target+"_H2_inspec.pdf")



def main():

	parser = argparse.ArgumentParser(usage=__doc__)
	# Target name:
	parser.add_argument('-t','--target',
		dest="target", default="GRB", type=str)
	# Data/Spectrum file:
	parser.add_argument('-f','--file',
		dest="file", default=None, type=str)
	# Redshift(s) of absorption components:
	parser.add_argument('-z','--redshifts',
		dest="redshifts", default=[], nargs='+')
	# Wavelength start (rest frame):
	parser.add_argument('-w1','--w1',
		dest="w1", default=980., type=float)
	# Wavelength end (rest frame):
	parser.add_argument('-w2','--w2',
		dest="w2", default=1120., type=float)
	# instrumental resolution R
	parser.add_argument('-res','--resolution',
		dest="resolution", default=None, type=float)
	parser.add_argument('-bh2','--bh2',
		dest="bh2", default=8.0,type=float)
	parser.add_argument('-nh2','--nh2',
		dest="nh2", default=18.0, type=float)
	parser.add_argument('-th2','--th2',
		dest="th2", default=100.0, type=float)

	# Plot H2 lines for individual column densities
	parser.add_argument('-indv','--indv',
		dest="indv", default=False, type=bool)
	parser.add_argument('-nh2j0','--nh2j0',
		dest="nh2j0", default=[], nargs='+')
	parser.add_argument('-nh2j1','--nh2j1',
		dest="nh2j1", default=[], nargs='+')
	parser.add_argument('-nh2j2','--nh2j2',
		dest="nh2j2", default=[], nargs='+')
	parser.add_argument('-bsh2','--bsh2',
		dest="bsh2", default=[], nargs='+')

	args = parser.parse_args()

	target_name = args.target
	data_file 	= args.file
	redshift 	= float(args.redshifts[0])
	redshifts 	= args.redshifts
	w1 			= args.w1
	w2 			= args.w2
	res 		= args.resolution
	bh2 		= args.bh2
	nh2 		= args.nh2
	th2 		= args.th2
	indv 		= args.indv
	nh2j0 		= args.nh2j0
	nh2j1		= args.nh2j1
	nh2j2 		= args.nh2j2
	bsh2		= args.bsh2

	if not len(bsh2) == len(nh2j2) == len(nh2j1) == len(nh2j0) == len(redshifts):
		sys.exit('nh2j0, nh2j1, nh2j2, bsh2, and redshifts have to have the same length')


	wav_aa_pl, flux, flux_err, qual, cont, \
	cont_err, flux_corr, flux_err_corr, \
	norm_flux, norm_flux_err, \
	n_flux_pl, n_flux_err_pl = \
	get_data2(data_file, redshift, wl_range=True, wl1=w1, wl2=w2)

	plot_syn_h2(wav_aa_pl, n_flux_pl, n_flux_err_pl, target_name,
		bh2, nh2, th2, redshift, redshifts, res, indv, nh2j0, nh2j1,
		nh2j2, bsh2)

if __name__ == "__main__":
	
	main()



