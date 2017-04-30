#!/usr/bin/python3
import functools
from math import ceil, floor
import operator
import sys

# design parameters
# tile size
st = [128, 128]

# unroll factor
k = 1

# application-specific parameters
# output image size
S = [2560, 1920]

# input data element width in bit
Wi = [16]

# output data element width in bit
Wo = [8, 8, 8]

# stencil reuse distance
Sr = st[0]*10+2

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
    return (mul(nt())+2)*(max(tl(), ts()))

# tile number
def nt():
    return [ceil(Si/(sti-Ssi+1)) for (Si, sti, Ssi) in zip(S, st, Ss)]

# compute latency
def tc():
    return ceil((mul(st)+Sr-1)*IIc()/k)

# load latency
def tl():
    #    return ceil(mul(st)/floor(Wb/sum(Wi)))*Wb/BWr+Lr
    return ceil(mul(st)/floor(Wb/sum(Wi)))*Wb/BWr*2+Lr
#return max(ceil(mul(st)/floor(Wb/sum(Wi)))*Wb/(BWr*sum(Wi)/(sum(Wi)+sum(Wo)))+Lr, IIl()*ceil(mul(st)*sum(Wi)/Wb))

# store latency
def ts():
    #return ceil(mul(st)/floor(Wb/sum(Wo)))*Wb/BWw+Lw
    return ceil(mul(st)/floor(Wb/sum(Wo)))*Wb/BWw*2+Lw
#return max(ceil(mul(st)/floor(Wb/sum(Wo)))*Wb/(BWw*sum(Wo)/(sum(Wi)+sum(Wo)))+Lw, IIs()*ceil(mul(st)*sum(Wo)/Wb))

# compute II
def IIc():
    return 1
#    return max(ceil(k/pi), ceil(k/po))

# load II
def IIl():
    print("load II: %d" % ceil(floor(Wb/max(Wi))/pi))
    return ceil(floor(Wb/max(Wi))/pi)

# store II
def IIs():
    print("store II: %d" % ceil(floor(Wb/max(Wo))/po))
    return ceil(floor(Wb/max(Wo))/po)

# helper
def mul(l):
    return functools.reduce(operator.mul, l)

# FPGA BRAM size in bit
BRAM = 6000000*8*.06

k = 16
pi = 16
po = 32

#S = [1998, 998]
#Ss = [5, 5]
st = [256, 498]
#Wi = [16]
#Wo = [16]
#Lr = 120
#Lw = 120
#BWr = 12.8*8*5
#BWw = 12.8*8*5

print("%d us" % round((te())/200))

stmin = st
temin = 10000000000000000000000000

pixel_group = int(Wb/max(max(Wi),max(Wo)))

for st0 in range(round(ceil((Ss[0]-1)/(Wb/max(max(Wi),max(Wo))))*Wb/max(max(Wi),max(Wo))/2), round(min(S[0], round(BRAM/2/(sum(Wi)+sum(Wo))/(Ss[1]-1)/10))/2)+1):
    for st1 in range((Ss[1]-1)*10, min(S[1], round(BRAM/2/(sum(Wi)+sum(Wo))/st0/2))+1):
        st = [2*st0, 2*st1]
        newte = te()
        if newte < temin and mul(st) % po == 0:
            temin = newte
            kmin = k
            stmin = st

print("best we can do")
print("min te: "+str(temin/f))
st = stmin
Lr = 120
Lw = 120
BWr = 12.8*8*5
BWw = 12.8*8*5
print("min te: "+str(te()/f))
print("k: "+str(kmin))
print("st: "+str(stmin))
print("pi: "+str(pi))
print("po: "+str(po))

sys.exit(0)
print("let's try to get within 5% of the best")

for k in reversed(range(kmin, kmin+1)):
    print(k)
    for st0 in range((Ss[0]-1)*10, min(S[0], round(BRAM/2/(Wi+Wo)/(Ss[1]-1)/10))+1):
        for st1 in range((Ss[1]-1)*10, min(S[1], round(BRAM/2/(Wi+Wo)/st0))+1):
            st = [st0, st1]
            for pi in reversed(range(pi0(), pimax+1)):
                cont = True
                for po in reversed(range(po0(), pomax+1)):
                    newte = te()
                    if newte < temin:
                        temin = newte
                        kmin = k
                        stmin = st
                        pimin = pi
                        pomin = po
                    else:
                        cont = False
                        break
                if not cont:
                    break

print("min te: "+str(temin))
print(" k: "+str(kmin))
print(" st: "+str(stmin))
print(" pi: "+str(pimin))
print(" po: "+str(pomin))

