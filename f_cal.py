#! /usr/bin/python

import numpy as np

nh2 = 16.50
nh2_err = 1.20
nh = 22.15
nh_err = 0.03

def f_cal(nh2, nh2_err, nh, nh_err):

	f1 = 2 * 10**nh2
	f2 = 2 * 10**nh2 + 10**nh
	f = f1/f2

	f1_err = 2 * 10**(nh2-nh2_err)
	f2_err = 2 * 10**(nh2-nh2_err) + 10**(nh+nh_err)
	f_err = f1_err/f2_err

	f, f_err = np.log10(f), abs(np.log10(f_err))
	return f, f_err

f, f_err = f_cal(nh2, nh2_err, nh, nh_err)

print f, f_err



