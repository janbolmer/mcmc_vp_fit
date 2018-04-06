#! /usr/bin/python

import numpy as np
import argparse, sys


def f_cal(nh2, nh2_err, nh, nh_err, logbool):
    '''
    calculate molecular hydrogen fraction and error
    '''

    f1 = 2 * 10**nh2
    f2 = 2 * 10**nh2 + 10**nh
    f = f1/f2

    f1_err = 2 * 10**(nh2-nh2_err)
    f2_err = 2 * 10**(nh2-nh2_err) + 10**(nh+nh_err)
    f_err = f1_err/f2_err

    if logbool == True:
        f, f_err = np.log10(f), abs(np.log10(f_err))
    else:
        pass

    return f, f_err


if __name__ == "__main__":

    #writecmd("vinspec_old.dat")

    '''
    Parsing commend line arguments:
    -nh2: Molecular Hydrogen column denisty with error
    -nh: Hydrogen column denisty with error
    -log: return logarythmic value, default: True
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-nh2', '--nh2', dest='nh2', nargs='+',
                        default=[], type=float)
    parser.add_argument('-nh', '--nh', dest='nh', nargs='+',
                        default=[], type=float)
    parser.add_argument('-log', '--logbool', dest='logbool',
                        default=True, type=bool)
    args = parser.parse_args()

    nh2 = np.array(args.nh2)
    nh = np.array(args.nh)
    logbool = args.logbool

    f, f_err = f_cal(nh2[0], nh2[1], nh[0], nh[1], logbool)

    print f, f_err




