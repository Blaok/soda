#!/usr/bin/python3
import functools
import math
import operator

# input
St = [256,256]
A = [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]]
#A = [[0,-1],[-1,0],[0,0],[1,0],[0,1]]
k = 16

# serialize
def Serialize(vec, St):
    # convert vec to scalar coordinates
    result = vec[0]
    for i in range(1, len(St)):
        result += vec[i]*functools.reduce(operator.mul, St[0:i])
    return result

def GetChains(St, A, k):
    A_dag = set()
    for i in range(0, k):
        A_dag = A_dag.union([x+i for x in [Serialize(x, St) for x in A]])
    A_dag = list(A_dag)
    A_dag.sort()
    A_dag = [x for x in A_dag]
    chains = []
    for i in range(0, k):
        chains += [[x for x in A_dag if x%k == i]]
    return chains

def PrintChains(chains):
    k = len(chains)
    length_chain = int(math.log(k-1, 10)+1)
    length_buffer = int(math.log(max([x[-1] for x in chains])-1, 10)+1)
    if(min([x[0] for x in chains]) < 0):
        length_buffer_neg = int(math.log(-min([x[0] for x in chains])-1, 10)+1)
        if length_buffer < length_buffer_neg+1:
            length_buffer = length_buffer_neg+1
    for i in range(0, k):
        for j in range(0,len(chains[i])-1):
            interval_length = chains[i][j+1] - chains[i][j]
            if interval_length == k:
                print(("Chain %%%dd: FF   between %%%dd and %%%dd" % (length_chain, length_buffer, length_buffer)) % (i, chains[i][j], chains[i][j+1]))
            else:
                print(("Chain %%%dd: FIFO between %%%dd and %%%dd" % (length_chain, length_buffer, length_buffer)) % (i, chains[i][j], chains[i][j+1]))

chains = GetChains(St, A, k)
#PrintChains(chains)

def GetPoints(A, k):
    all_points = {} # {offset:{unroll_index:point_index}}
    for i in range(0, k):
        points = [Serialize(x, St) for x in A]
        for idx, j in enumerate(points):
            #            chain_idx = (j+i)%k
#            fifo_idx = chains[chain_idx].index(j+i)
            all_points[j+i] = all_points.get(j+i, {})
            all_points[j+i][i] = idx
#            print("Access %d @ Chain %d, FIFO %d" % (j+i, chain_idx,fifo_idx))
#    print(all_points)
    return all_points


def GetBuffer(chains, A):
    k = len(chains)
    FFs = []    # [outputs]
    FIFOs = {}  # {length:[outputs]}
    inputs = []
    for chain in chains:
        inputs += [chain[-1]]
        for j in range(0, len(chain)-1):
            interval_length = chain[j+1]-chain[j]
            if interval_length == k:
                FFs += [chain[j]]
            else:
                FIFOs[interval_length] = FIFOs.get(interval_length, []) + [chain[j]]
    FFs.sort()
    inputs.sort()
    for FIFO in FIFOs.values():
        FIFO.sort()
    return {'FFs':FFs, 'FIFOs':FIFOs, 'k':k, 'inputs':inputs}

def MapBuffer(buf):
    mapping = {}    # maps an offset to what consumes it
    for idx, item in enumerate(buf['FFs']):
        mapping[item+k] = {'FFs':idx}
    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            mapping[item+fifo_length] = {'FIFOs':[fifo_length, idx]}
    return mapping

def Map2String(mapping):
    if 'FIFOs' in mapping:
        return 'FIFO_%d[%d]' % (mapping['FIFOs'][0], mapping['FIFOs'][1])
    if 'FFs' in mapping:
        return 'FF[%d]' % (mapping['FFs'])

def PrintBuffer(chains, A):
    k = len(chains)
    buf = GetBuffer(chains, A)
    points = GetPoints(A, k)
    mapping = MapBuffer(buf)
#    print(buf)
#    print(points)
#    print(mapping)
    print('input[%d]' % len(buf['inputs']))
    print('FF[%d]' % len(buf['FFs']))
    for fifo_length in buf['FIFOs'].keys():
        print('FIFO_%d[%d]' % (fifo_length,len(buf['FIFOs'][fifo_length])))
    for idx, item in enumerate(buf['inputs']):
        for unroll_index in points[item].keys():
            print("access %4d from input[%d]     when unroll_index = %d for stencil point %s" % (item, idx, unroll_index, str(A[points[item][unroll_index]])))
    for idx, item in enumerate(buf['FFs']):
        for unroll_index in points[item].keys():
            print("access %4d from FF[%d]        when unroll_index = %d for stencil point %s" % (item, idx, unroll_index, str(A[points[item][unroll_index]])))
    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            for unroll_index in points[item].keys():
                print("access %4d from FIFO_%d[%d]  when unroll_index = %d for stencil point %s" % (item, fifo_length, idx, unroll_index, str(A[points[item][unroll_index]])))
    for idx, item in enumerate(buf['inputs']):
        if item in mapping:
            print('feed %s to %s' % ('inputs[%d]'%idx, Map2String(mapping[item])))

    for idx, item in enumerate(buf['FFs']):
        if item in mapping:
            print('feed %s to %s' % ('FF[%d]'%idx, Map2String(mapping[item])))
        else:
            print('pop FF[%d]' % idx)
    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            if item in mapping:
                print('feed %s to %s' % ('FIFO_%d[%d]'%(fifo_length, idx), Map2String(mapping[item])))
            else:
                print('pop FIFO_%d[%d]' % (fifo_length, idx))

PrintBuffer(chains, A)

