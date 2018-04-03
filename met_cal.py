#! /usr/bin/python

import numpy as np
import pandas as pd

import argparse, sys


def load_data():
	'''
	Loading solar abundances into a Pandas DataFrame.
	Keywords are 'element', 'photosphere', and 'meteorite'
	for the name of the element, e.g. Zn for Zink, the
	the abundances in the present day solar photosphere, and
	those drived from meteorties
	'''
	
	el_df = pd.read_csv('atoms/solar_abundances.csv', sep=', ',
		header=7, index_col=None, engine='python')

	return el_df


def cal_met(nh, ne, nh_sol, ne_sol):
	'''
	Calculating metallicity
	'''

	A = np.log10((10**ne[0])/(10**nh[0]))
	A_high = np.log10((10**(ne[0]+ne[1]))/(10**(nh[0]-nh[1])))
	A_low = np.log10((10**(ne[0]-ne[1]))/(10**(nh[0]+nh[1])))

	B = np.log10((10**ne_sol[0])/(10**nh_sol[0]))

	met = A - B
	met_high = A_high - B
	met_low = A_low - B

	met_error = (abs(met_high-met) + abs(met-met_low))/2.0

	return met, met_error


if __name__ == "__main__":

    #writecmd("vinspec_old.dat")

    '''
	Parsing commend line arguments:
	-ne: column density of the element with error
	-nh: Hydrogen column denisty with error
	-e: element, e.g. -e Zn
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', '--ne', dest='ne', nargs='+',
                        default=[], type=float)
    parser.add_argument('-nh', '--nh', dest='nh', nargs='+',
                        default=[], type=float)
    parser.add_argument('-e', '--element', dest='element',
                        default=None, type=str)
    args = parser.parse_args()

    nh = np.array(args.nh)
    ne = np.array(args.ne)
    element = args.element

    #element = 'Zn'

    if element == None:
    	sys.exit('ERROR: please indicate which \
    		elemtent you are using')

    el_df = load_data() #loading solar abundances data

    h_abundance = el_df.loc[el_df['element'] == 'H'] 
    el_abundance = el_df.loc[el_df['element'] == element]

    nh_sol = np.array(h_abundance['photosphere'])[0].split(' ')
    ne_sol = np.array(el_abundance['photosphere'])[0].split(' ')

    nh_sol = np.asfarray(np.array(nh_sol), float)
    ne_sol = np.asfarray(np.array(ne_sol), float)

    print '\n Solar abundance for', element, ':', ne_sol[0], '+/-', ne_sol[1]
    print '\n Measured column density for', element, ':', ne[0], '+/-', ne[1]

    met, met_error = cal_met(nh, ne, nh_sol, ne_sol)

    print '\n metallicity [' +element+'/H]:', met, '+/-', met_error, '\n'




