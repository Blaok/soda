#!/usr/bin/python3
import functools
from math import ceil, floor
import operator
import sys

# design parameters
# tile size
St = [2000, 0]

# unroll factor
k = 16

# burst length
Sb = 100032

# application-specific parameters
# output image size
S = [2000, 1000]
St[-1] = S[-1]

# input data element width in bit
Wi = [16]

# output data element width in bit
Wo = [8, 8, 8]

# stencil reuse distance
#Sr = st[0]*10+2

# platform-specific parameters
# DRAM read latency in cycle (542ns)
Lr = 542/5

# DRAM write latency in cycle (356ns)
Lw = 356/5

# stencil size
Ss = [23, 19]

# maximum DRAM read bandwidth in bit/cycle (9.5GB/s)
BWr = 9.5*8*5

# maximum DRAM write bandwidth in bit/cycle (8.9GB/s)
BWw = 8.9*8*5

# frequency in MHz
f = 200

# burst width in bit
Wb = 512

# deduced parameters
# total execution latency
def te():
    return (mul(nt())*ceil(mul(St)/Sb)+2)*(max(tc(), tl(), ts()))

# tile number
def nt():
    return [ceil((Si-Ssi+1)/(Sti-Ssi+1)) for (Si, Sti, Ssi) in zip(S, St, Ss)]

# compute latency
def tc():
    return ceil(Sb/k)

# load latency
def tl():
    return ceil(sum(Wi)*Sb/BWr+Lr)*5/3

# store latency
def ts():
    return ceil(sum(Wo)*Sb/BWw+Lw)*5/2

# helper
def mul(l):
    return functools.reduce(operator.mul, l)

default_St0 = St[0]
for St0 in [100, 200, 500, 1000, 1500, 2000]:
    St[0] = St0
    print("St0 = %d,\tk = %d,\tSb = %d, \t%.3f pixel/ns" % (St[0], k, Sb, mul(S)/((te())/f*1000)))
St[0] = default_St0

default_k = k
for k in [1, 2, 4, 8, 16, 32]:
    if k == default_k:
        continue
    print("St0 = %d,\tk = %d,\tSb = %d, \t%.3f pixel/ns" % (St[0], k, Sb, mul(S)/((te())/f*1000)))
k = default_k

default_Sb = Sb
for Sb in [4992, 9984, 20032, 49984, 100032, 200000]:
    if Sb == default_Sb:
        continue
    print("St0 = %d,\tk = %d,\tSb = %d, \t%.3f pixel/ns" % (St[0], k, Sb, mul(S)/((te())/f*1000)))
Sb = default_Sb

default_S0 = S[0]
S[0] = 20000
default_St0 = St[0]
for St0 in [100, 200, 500, 1000, 1500, 2000]:
    St[0] = St0
    print("St0 = %d,\tk = %d,\tSb = %d, \t%.3f pixel/ns" % (St[0], k, Sb, mul(S)/((te())/f*1000)))
St[0] = default_St0
S[0] = default_S0
