#! /usr/bin/python


import numpy as np
from astropy.io import fits

wav_scale = 1.0 #change to 10 for spectra with wav in nm


def get_data2(file, z, wl_range=False, wl1=3300, wl2=5000):

	ff = fits.open(file)
	hdr = ff[0].header
	hdr2 = ff[1].header

	try:
		print 'RESOLUTION =', 299792./hdr2['RESOLUTION']
		print 'RESOLUTION =', hdr2['RESOLUTION'], 'km/s'
	except:
		print 'no RESOLUTION value found'

	dat = ff[1].data

	# Observed wavelength in vacuum,
	# corrected for barycentric motion
	# and drifts in the wavelength solution
	wav_aa 		= dat.field('WAVE').flatten() * wav_scale
	
	# Observed flux density ergs-1 cm-2 A-1
	flux 		= dat.field('FLUX').flatten()
	
	# Associated flux density error ergs-1 cm-2 A-1
	flux_err 	= dat.field('ERR').flatten()
	
	# Bad pixel map
	qual 		= dat.field('QUAL').flatten()
	
	# Continuum estimate based
	try:
		cont		= dat.field('CONTINUUM').flatten()
	except KeyError:
		cont 		= np.ones(len(flux))
	
	# Relative error on the continuum estimate
	try:
		cont_err	= dat.field('CONTINUUM_ERR').flatten()
	except KeyError:
		cont_err 	= np.ones(len(flux))
	
	# Inverse transmission spectrum. Multiply
	# flux and flux_err column with this column
	# to correct for telluric absorption.
	try:
		tell_corr	= dat.field('TELL_CORR').flatten()
	except KeyError:
		tell_corr 	= np.ones(len(flux))

	ff.close()

	flux_corr 		= flux*tell_corr
	flux_err_corr 	= flux_err*tell_corr

	norm_flux 		= []
	norm_flux_err 	= []

	norm_flux_corr 		= []
	norm_flux_err_corr 	= []

	for i in range(len(wav_aa)):
		norm_flux.append(flux[i]/cont[i])
		norm_flux_err.append((flux[i]+flux_err[i])/cont[i] - flux[i]/cont[i])

		norm_flux_corr.append(flux_corr[i]/cont[i])
		norm_flux_err_corr.append((flux_corr[i]+flux_err_corr[i])/cont[i] - flux_corr[i]/cont[i])
	
	if wl_range == False:

		return wav_aa, flux, flux_err, qual, cont, \
				cont_err, flux_corr, flux_err_corr, \
				norm_flux, norm_flux_err, \
				norm_flux_corr, norm_flux_err_corr

	if wl_range == True:

		wav_aa2, flux2, flux_err2 	= [], [], []
		qual2, cont2, cont_err2 	= [], [], []
		flux_corr2, flux_err_corr2 	= [], []
		norm_flux2, norm_flux_err2 	= [], []
		norm_flux_corr2, norm_flux_err_corr2 = [], []

		for i in range(len(wav_aa)):
			if (wl1*(1+z)) <= wav_aa[i] <= (wl2*(1+z)):
				wav_aa2 		= np.append(wav_aa2, wav_aa[i]) 
				flux2 			= np.append(flux2, flux[i])
				flux_err2 		= np.append(flux_err2, flux_err[i])
				qual2 			= np.append(qual2, qual[i])
				cont2 			= np.append(cont2, cont[i])
				cont_err2 		= np.append(cont_err2, cont_err[i])
				flux_corr2 		= np.append(flux_corr2, flux_corr[i])
				flux_err_corr2 	= np.append(flux_err_corr2, flux_err_corr[i])
				norm_flux2 		= np.append(norm_flux2, norm_flux[i])
				norm_flux_err2	= np.append(norm_flux_err2, norm_flux_err[i])
				norm_flux_corr2 = np.append(norm_flux_corr2, norm_flux_corr[i])
				norm_flux_err_corr2	= np.append(norm_flux_err_corr2, norm_flux_err_corr[i])

		return wav_aa2, flux2, flux_err2, qual2, cont2, \
				cont_err2, flux_corr2, flux_err_corr2, \
				norm_flux2, norm_flux_err2, \
				norm_flux_corr2, norm_flux_err_corr2

def get_data2_ign(file, z, ign_lst, wl1=3300, wl2=5000):

	ff = fits.open(file)
	hdr = ff[0].header
	dat = ff[1].data

	wav_aa = dat.field('WAVE').flatten() * wav_scale
	flux = dat.field('FLUX').flatten()
	flux_err = dat.field('ERR').flatten()
	qual = dat.field('QUAL').flatten()
	cont = dat.field('CONTINUUM').flatten()
	cont_err = dat.field('CONTINUUM_ERR').flatten()

	try:
		tell_corr	= dat.field('TELL_CORR').flatten()
	except KeyError:
		tell_corr 	= np.ones(len(flux))

	flux_corr = flux*tell_corr
	flux_err_corr = flux_err*tell_corr
	norm_flux = []
	norm_flux_err = []
	norm_flux_corr 	= []
	norm_flux_err_corr = []
	ff.close()

	for i in range(len(wav_aa)):
		norm_flux.append(flux[i]/cont[i])
		norm_flux_err.append((flux[i]+flux_err[i])/cont[i] - flux[i]/cont[i])
		norm_flux_corr.append(flux_corr[i]/cont[i])
		norm_flux_err_corr.append((flux_corr[i]+flux_err_corr[i])/cont[i] - flux_corr[i]/cont[i])

	wl1 = wl1*(1+z)
	wl2 = wl2*(1+z)

	wl_low 	= []
	wl_up	= []

	for wav_rng in ign_lst:
		wl_low.append(wav_rng[0]*(1+z))
		wl_up.append(wav_rng[1]*(1+z))

	wav_aa2, n_flux2, n_flux_err2 = [], [], []

	for i in range(len(wav_aa)):
		if wl1 <= wav_aa[i] <= wl2:
			tester = 0.0
			for j in np.arange(0, len(wl_low), 1):
				if wl_low[j] < wav_aa[i] < wl_up[j]:
					#wav_aa2 = np.append(wav_aa2, wav_aa[i])
					#n_flux2 = np.append(n_flux2, 1.0)
					#n_flux_err2 = np.append(n_flux_err2, 0.01)
					tester += 1.0
			if tester == 0.0:
				wav_aa2 = np.append(wav_aa2, wav_aa[i])
				n_flux2 = np.append(n_flux2, norm_flux_corr[i])
				n_flux_err2 = np.append(n_flux_err2, norm_flux_err_corr[i])

	return wav_aa2, n_flux2, n_flux_err2

def get_data_vandels(file):

	ff = fits.open(file)
	hdr = ff[0].header
	dat = ff[1].data

	wav_aa = dat.field('WAVE').flatten()
	flux = dat.field('FLUX').flatten()
	flux_err = dat.field('ERR').flatten()

	return wav_aa, flux, flux_err













































