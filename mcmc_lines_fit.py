#! /usr/bin/python

import pymc, math, argparse, os, sys, time


sys.path.append('bin/')

from sns_plots import corner_ele_plot
from syn_spec import *
from spec_functions import * 
from read_fits import *


# to have free z parameter:
high_ion = ['AlIII', 'CIV', 'NV', 'OVI', 'PV',
			'SIV', 'SVI', 'SiIV', 'SiI', 'OI',
			'OIa', 'OIb', 'MgI', 'ArI', 'ZnII', 'CrII']

def gauss(x, mu, sig):
	'''
	Normal distribution used to create prior probability distributions
	'''
	return np.exp(-np.power(x-mu,2.0)/(2.0*np.power(sig,2.0)))

def exp_prior_co(x, beta, cut_off):
	'''
	Exponential distribution with cut-off used to create prior
	probability distributions
	'''
	if 0 < value < cut_off:
		return beta * np.exp(-beta * x) 
	else:
		return -np.inf

def fit_model(wav_aa, flux, flux_err, fb, redshifts, lines,
	ign_lst, par_dic, res, nh, bg, fixed_b, fzH2, fzCO):

	# sigma error -> tau error 
	err_tau = 1. / (np.array(flux_err)**2)

	b_lmin = 0.25*299792./res/(2*np.sqrt(np.log(2)))

	print("\n Components with ~ b >", round(b_lmin, 1), \
			"km/s can be resolved \n")

	vars_dic = {}
	H2bdz_dic = {}
	CObdz_dic = {}

	for z in redshifts:
		if fixed_b == None:
			b_H2 = pymc.Uniform(
				'b_H2_'+z,
				lower=1.,
				upper=25.0,
				value=8.0,
				doc='b_H2_'+z)
				#pymc.Exponential(
				#'b_H2_'+z,
				#beta=0.05,
				#value=4.0,
				#doc='b_H2_'+z)
		else:
			b_H2 = fixed_b

		if fzH2 == False:	
			dz_H2 = pymc.Uniform(
				'dz_H2_'+z,
				lower = -80.,
				upper = +80.,
				value = 0.0,
				doc ='dz_H2_'+z)
		else:
			dz_H2 = pymc.Uniform(
				'dz_H2_'+z,
				lower = -10.,
				upper = +10.,
				value = 0.0,
				doc ='dz_H2_'+z)			

		H2bdz_dic['H2'+z] = b_H2, dz_H2

	for z in redshifts:

		if fixed_b == None:
			b_CO = pymc.Uniform(
				'b_CO_'+z,
				lower=0.,
				upper=25.0,
				value=4.0,
				doc='b_CO_'+z)
		else:
			b_CO = fixed_b

		if fzCO == False:
			dz_CO = pymc.Uniform(
				'dz_CO_'+z,
				lower = -80.,
				upper = +80.,
				value = 0.0,
				doc ='dz_CO_'+z)
		else:
			dz_CO = 0.0 #pymc.Uniform(
				#'dz_CO_'+z,
				#lower = -5.,
				#upper = +5.,
				#value = 0.0,
				#doc ='dz_CO_'+z)

		CObdz_dic['CO'+z] = b_CO, dz_CO


	for l in lines:
		for z in redshifts:

			if l.startswith('HI') and (float(z) == float(redshifts[0])):
					N = pymc.Uniform(
						'N_'+l,
						lower=float(nh[0])-float(nh[1]),
						upper=float(nh[0])+float(nh[1]),
						value=float(nh[0]),
						doc='N_'+l)
					b = pymc.Uniform(
						'b_'+l,
						lower=0.,
						upper=200.0,
						value=80.0,
						doc='b_'+l)
					dz = pymc.Uniform(
						'dz_'+l+'_'+z,
						lower = -250.,
						upper = +250.,
						value = 0.0,
						doc ='dz_'+l+'_'+z)

					vars_dic[l+z] = l, N, b, float(z), dz						


			elif l.startswith('H2J'):
				#N = pymc.Uniform(
				#	'N_'+l+'_'+z,
				#	lower=0.,
				#	upper=21.0,
				#	value=5.0,
				#	doc='N_'+l+'_'+z) 
				N = pymc.Exponential(
					'N_'+l+'_'+z,
					beta=0.05,
					value=4.0,
					doc='N_'+l+'_'+z)
				b  = H2bdz_dic['H2'+z][0]
				dz = H2bdz_dic['H2'+z][1]

				vars_dic[l+z] = l, N, b, float(z), dz


			elif l.startswith('CO'):
				N = pymc.Exponential(
					'N_'+l+'_'+z,
					beta=0.05,
					value=4.0,
					doc='N_'+l+'_'+z)
				b  = CObdz_dic['CO'+z][0]
				dz = CObdz_dic['CO'+z][1]

				vars_dic[l+z] = l, N, b, float(z), dz


			elif l.startswith('HD'):
				pass

			elif l.startswith(('AlIII', 'CIV', 'NV', 'ArI',
				'OVI', 'PV', 'SIV', 'SVI', 'SiIV', 'SiI', 'OI',
				'OIa', 'OIb', 'MgI', 'ZnII', 'CrII')):

				if l+z in par_dic:
					N = par_dic[l+z][1] 
					b = par_dic[l+z][4]
					print(str(l+z), "b set to:", b, "N set to:", N)

				else:
					N = pymc.Uniform(
						'N_'+l+'_'+z,
						lower=0.,
						upper=18.0,
						value=13.0,
						doc='N_'+l+'_'+z) 
					b = pymc.Uniform(
						'b_'+l+'_'+z,
						lower=b_lmin,
						upper=40.0,
						value=b_lmin+5,
						doc='b_'+l+'_'+z)
				dz = pymc.Uniform(
					'dz_'+l+'_'+z,
					lower = -50.,
					upper = +50.,
					value = 0.0,
					doc ='dz_'+l+'_'+z)
				vars_dic[l+z] = l, N, b, float(z), dz

			elif not l.startswith('HI'):
				if l+z in par_dic:
					N = par_dic[l+z][1] 
					b = par_dic[l+z][4]
					print(str(l+z), "b set to:", b, "N set to:", N)
				else:
					N = pymc.Uniform(
						'N_'+l+'_'+z,
						lower=0.,
						upper=18.0,
						value=13.0,
						doc='N_'+l+'_'+z) 
					b = pymc.Uniform(
						'b_'+l+'_'+z,
						lower=b_lmin,
						upper=40.0,
						value=b_lmin+5,
						doc='b_'+l+'_'+z)	

				vars_dic[l+z] = l, N, b, float(z)

	@pymc.deterministic(plot=False)
	def add_lines(wav_aa=wav_aa, vars_dic=vars_dic): #BG=BG):

		spec 	= np.ones(len(wav_aa))*bg
		synspec = SynSpec(wav_aa, float(redshifts[0]), res, ign_lst)

		for key in vars_dic:
			if len(vars_dic[key]) == 4:
				spec = synspec.add_line(spec,
					l=vars_dic[key][0],
					N=vars_dic[key][1],
					b=vars_dic[key][2],
					z=vars_dic[key][3])
			else:
				spec = synspec.add_line(spec,
					l=vars_dic[key][0],
					N=vars_dic[key][1],
					b=vars_dic[key][2],
					z=vars_dic[key][3] + (vars_dic[key][4]/100000.))
		return spec

	# Likelihood of the data:
	y_val = pymc.Normal('y_val', mu=add_lines, tau=err_tau, value=flux,
		observed=True)

	return locals()


def sMCMC(wav_aa, flux, flux_err, iterations, burn_in, thin,
	fb, redshifts, lines, ign_lst, t_name, par_dic, res, nh,
	csv_lst, bg, fixed_b, fzH2, fzCO):
	'''
	Starting the MCMC
	'''

	MDL = pymc.MCMC(fit_model(wav_aa, flux, flux_err, fb,
		redshifts, lines, ign_lst, par_dic, res, nh, bg,
		fixed_b, fzH2, fzCO), db='pickle', dbname=t_name+'_lines_fit.pickle')
	
	MDL.db
	MDL.sample(iterations, burn_in, thin)
	MDL.db.close()

	MDL.write_csv(t_name+"_lines_fit_results.csv", variables=csv_lst)

	y_fit 	= MDL.stats()['add_lines']['mean']
	y_min 	= MDL.stats()['add_lines']['quantiles'][2.5]
	y_max 	= MDL.stats()['add_lines']['quantiles'][97.5]
	y_min2 	= MDL.stats()['add_lines']['quantiles'][25]
	y_max2 	= MDL.stats()['add_lines']['quantiles'][75]


	return y_fit, y_min, y_max, y_min2, y_max2

def main():

	writecmd("wav_cmd_hist.dat")

	start = time.time()
	print("\n Parsing Arguments \n")
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
	# List if lines to be included in the fit:
	parser.add_argument('-l','--lines',
		dest="lines", default=[], nargs='+')
	# Wavelength start (rest frame):
	parser.add_argument('-w1','--w1',
		dest="w1", default=980., type=float)
	# Wavelength end (rest frame):
	parser.add_argument('-w2','--w2',
		dest="w2", default=1120., type=float)
	# List of regions to be ignored (rest frame):
	parser.add_argument('-ign','--ignore',
		dest="ignore", default=[], nargs='+')
	# instrumental resolution R
	parser.add_argument('-res','--resolution',
		dest="resolution", default=None, type=float)
	# Number of iterations:
	parser.add_argument('-it','--iterations',
		dest="iterations", default=1000, type=int)
	# Burn-in
	parser.add_argument('-bi','--burn_in',
		dest="burn_in", default=100, type=int)
	# Boolean to save the pickle file
	parser.add_argument('-sp','--save_pickle',
		dest="save_pickle", default=False, type=bool)
	# Parameter file:
	parser.add_argument('-par','--par',
		dest="par", default=None,type=str)
	# Fix b paramter of H2 lines
	parser.add_argument('-fb','--fixed_b',
		dest="fixed_b", default=None,type=float)
	# Redshift(s) of intervening systems
	parser.add_argument('-intv','--intv',
		dest="intv", default=[], nargs='+')
	parser.add_argument('-nh','--nh',
		dest="nh", default=[], nargs='+')
	parser.add_argument('-bg','--bg',
		dest="bg", default=1.0, type=float)
	parser.add_argument('-fzH2','--fzH2',
		dest='fzH2', default=False, type=bool)
	parser.add_argument('-fzCO','--fzCO',
		dest='fzCO', default=False, type=bool)

	args = parser.parse_args()

	target_name = args.target
	data_file 	= args.file
	redshifts 	= args.redshifts
	lines 		= args.lines
	w1 			= args.w1
	w2 			= args.w2
	ignore 		= args.ignore
	res 		= args.resolution
	iterations 	= args.iterations
	burn_in 	= args.burn_in
	save_pickle = args.save_pickle
	para_file 	= args.par
	fixed_b 	= args.fixed_b
	intv 		= args.intv
	nh 			= args.nh
	bg 			= args.bg
	fzH2 		= args.fzH2
	fzCO 		= args.fzCO

	# Creating a proper list for the intervals to be ignored
	ignore_lst 	= []
	for itrvl in ignore:
		tmp_lst = []
		s = itrvl.split(",")
		tmp_lst.extend((float(s[0]), float(s[1])))
		ignore_lst.append(tmp_lst)

	if 'HI' in lines and nh == []:
		sys.exit("ERROR: Please provide N_HI (+err) with -nh")

	# Check if a data file is given
	if data_file == None:
		sys.exit("ERROR: Please provide a data file with -f")

	# Check if burn-in > iterations
	if burn_in >= iterations:
		sys.exit("ERROR: Burn-In cannot be bigger than Iterations")

	# Check if resolution is given
	if res == None:
		sys.exit("ERROR: Specify the instrumental resolution with -res")

	par_dic = {}

	if para_file != None:
		par_dic = get_paras(para_file)
		print("\n Using parameters given in:", para_file)

	time.sleep(1.0)
	print("\n Fitting", target_name, "at redshift", redshifts[0], \
		"with a spectral resolution of R =", res, "with", lines)
	time.sleep(1.0)

	if fixed_b != None:
		print("\n b for H2 lines fixed to", fixed_b)

	print("\n Starting MCMC " + '(pymc version:',pymc.__version__,")")
	print("\n This might take a while ...")

	csv_lst = [] #['BG']
	corner_lines = []

	for l in lines:
		for z in redshifts:
			if not l + z in par_dic:
				if not l in corner_lines:
					corner_lines.append(l)

	for l in lines:
		for z in redshifts:
			if not l + z in par_dic:
				if l == 'HI' and z == redshifts[0]:
					csv_lst.append('N_'+l)
					csv_lst.append('b_'+l)
					csv_lst.append('dz_'+l+'_'+z)
				elif not l == 'HI' and not l.startswith(('H2', 'CO')) and not l in high_ion:
					csv_lst.append('N_'+l+'_'+z)
					csv_lst.append('b_'+l+'_'+z)
				elif l.startswith(('H2', 'CO')):
					csv_lst.append('N_'+l+'_'+z)
				elif l in high_ion:
					csv_lst.append('N_'+l+'_'+z)
					csv_lst.append('b_'+l+'_'+z)
					csv_lst.append('dz_'+l+'_'+z)
			else:
				print(l + z, 'fixed')

	if 'H2J0' in lines:
		for z in redshifts:
			if fixed_b == None:
				csv_lst.append('b_H2_'+z)
			if fzH2 == False:
				csv_lst.append('dz_H2_'+z)

	if 'COJ0' in lines:
		for z in redshifts:
			if fixed_b == None:
				csv_lst.append('b_CO_'+z)
			if fzCO == False:
				csv_lst.append('dz_CO_'+z)

	wav_aa_pl, flux, flux_err, qual, cont, \
	cont_err, flux_corr, flux_err_corr, \
	norm_flux, norm_flux_err, \
	n_flux_pl, n_flux_err_pl = \
	get_data2(data_file, float(redshifts[0]), wl_range=True, wl1=w1, wl2=w2)
	
	wav_aa, n_flux, n_flux_err = \
	get_data2_ign(data_file, float(redshifts[0]), ignore_lst, wl1=w1, wl2=w2)


	y_fit, y_min, y_max, y_min2, y_max2 = sMCMC(wav_aa, n_flux,
		n_flux_err, iterations, burn_in, 1, fixed_b, redshifts,
		lines, ignore_lst, target_name, par_dic, res, nh, csv_lst,
		bg, fixed_b, fzH2, fzCO)


	n_flux_pl = np.array(n_flux_pl)/bg
	y_fit = np.array(y_fit)/bg
	y_min = np.array(y_min)/bg
	y_max = np.array(y_max)/bg
	y_min2 = np.array(y_min2)/bg
	y_max2 = np.array(y_max2)/bg

	plot_spec(wav_aa_pl, wav_aa, n_flux_pl, n_flux_err_pl,
			y_min, y_max, y_min2, y_max2, y_fit, redshifts,
			ignore_lst, target_name, fixed_b, intv, lines)


	if iterations > 100:
		corner_ele_plot(target_name, corner_lines,
						file = target_name+'_lines_fit.pickle')

	if save_pickle == False:
		os.system("rm -r *.pickle")
		print("\n Pickle Files Deleted")
	if save_pickle == True:
		print("\n Pickle Files Saved")

if __name__ == "__main__":
	
	main()






