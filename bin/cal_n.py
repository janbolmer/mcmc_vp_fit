#! /usr/bin/python

import numpy as np
import argparse

parser = argparse.ArgumentParser(usage=__doc__)
parser.add_argument('-v','--vals',dest='vals',default=[],nargs='+')

args = parser.parse_args()
vals = args.vals

N, N_err = [], []

for i in range(0, len(vals), 1):
	if i%2 == 0:
		N.append(float(vals[i]))
	else:
		N_err.append(float(vals[i]))

print N, N_err

print np.log10(sum([10**i for i in N])), np.log10(sum([10**(i+j) \
	for i,j in zip(N, N_err)]))-np.log10(sum([10**i for i in N]))
