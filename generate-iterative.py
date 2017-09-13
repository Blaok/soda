#!/usr/bin/python3
import json
import os
import sys
from fractions import Fraction
from functools import reduce

config = json.loads(sys.stdin.read())
A = config['Aorg']
num_iter = config['iter']
coeff = [Fraction(x) for x in config['coeff']]

if len(A) != len(coeff):
    sys.stderr.write('A and coeff must have the same length\n')
    sys.exit(1)

A = {str(a):(a, c) for a, c in zip(A, coeff)}

def GetA(A,  num_iter):
    if num_iter < 1:
        return None
    if num_iter == 1:
        return A
    A1 = GetA(A, num_iter-1)
    A2 = {}
    for a1 in A1.values():
        for a in A.values():
            a2 = [o1+o2 for o1, o2 in zip(a[0], a1[0])]
            A2[str(a2)] = (a2, A2.get(str(a2), (a2, 0))[1]+a[1]*a1[1])
    return A2

def gcd(*numbers):
    from fractions import gcd
    return reduce(gcd, numbers)

def lcm(*numbers):
    def lcm(a, b):
        return (a * b) // gcd(a, b)
    return reduce(lcm, numbers, 1)

def GetAdderTree(variables, count):
    if len(variables)>1:
        r1 = GetAdderTree(variables[0:int(len(variables)/2)], count)
        r2 = GetAdderTree(variables[int(len(variables)/2):], r1[1])
        print('        "int32_t tmp_adder_%d = %s+%s;",' % (r2[1], r1[0], r2[0]))
        return ('tmp_adder_%d' % r2[1], r2[1]+1)
    return( variables[0], count)
        

A = (GetA(A, num_iter))
print('"A": '+str([x[0] for x in A.values()]))
#print('"coeff": '+str(['%d/%d' % (x[1].numerator, x[1].denominator) for x in A.values()]))
den_lcm = lcm(*[a[1].denominator for a in A.values()])
final_var = GetAdderTree(['input_%s*%d' % ('_'.join([str(x) for x in a[0]]), a[1].numerator*den_lcm/a[1].denominator) for a in A.values()], 0)
#print('output_type output_point = '+' + '.join(['input_%s*%d/%d' % ('_'.join([str(x) for x in a[0]]), a[1].numerator, a[1].denominator) for a in A.values()])+'0;')
print('        "output_type output_point = %s/%d;",' % (final_var[0], den_lcm))
