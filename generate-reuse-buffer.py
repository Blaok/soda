#!/usr/bin/python3
import functools
import math
import operator

# input
St = [300,304]
A = [[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[0,2],[1,2],[2,2]]
#A = [[0,-1],[-1,0],[0,0],[1,0],[0,1]]
k = 3

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

def GetPoints(A, k):
    all_points = {} # {offset:{unroll_index:point_index}}
    for i in range(0, k):
        points = [Serialize(x, St) for x in A]
        for idx, j in enumerate(points):
            all_points[j+i] = all_points.get(j+i, {})
            all_points[j+i][i] = idx
    return all_points

def GetBuffer(St, A, k):
    FFs = []    # [outputs]
    FIFOs = {}  # {length:[outputs]}
    inputs = []
    for chain in GetChains(St, A, k):
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

def GetConsumeMap(buf):
    mapping = {'k':buf['k']}    # maps an offset to what consumes it
    for idx, item in enumerate(buf['FFs']):
        mapping[item+k] = {'FFs':{'index':idx, 'produce':item}}
    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            mapping[item+fifo_length] = {'FIFOs':{'length':fifo_length, 'index':idx, 'produce':item}}
    return mapping

def GetProduceMap(buf):
    mapping = {'k':buf['k']}    # maps an offset to what produces it
    for idx, item in enumerate(buf['inputs']):
        mapping[item] = {'inputs':{'index':idx, 'produce':item}}
    for idx, item in enumerate(buf['FFs']):
        mapping[item] = {'FFs':{'index':idx, 'produce':item}}
    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            mapping[item] = {'FIFOs':{'length':fifo_length, 'index':idx, 'produce':item}}
    return mapping

def PrintLine(line = ''):
    global indent
    if line:
        print('%s%s' % (' '*indent*4,line))
    else:
        print()

def PrintUpdate(producer_map, consumer_map, src):
    if 'FIFOs' in src:
        dst = consumer_map.get(src['FIFOs']['produce'])
        if not dst:
            return
        fifo_length = src['FIFOs']['length']/producer_map['k']
        fifo_index = src['FIFOs']['index']
        src_str = 'FIFO_%d[%d][FIFO_%d_p[%d]]' % (fifo_length, fifo_index, fifo_length, fifo_index)
    elif 'FFs' in src:
        dst = consumer_map.get(src['FFs']['produce'])
        if not dst:
            return
        src_str = 'FF[%d]' % (src['FFs']['index'])
    else:
        dst = consumer_map.get(src['inputs']['produce'])
        if not dst:
            return
        src_str = 'input_buffer[%d]' % (src['inputs']['index'])

    PrintUpdate(producer_map, consumer_map, dst)

    if 'FIFOs' in dst:
        fifo_length = dst['FIFOs']['length']/consumer_map['k']
        fifo_index = dst['FIFOs']['index']
        dst_str = 'FIFO_%d[%d][FIFO_%d_p[%d]]' % (fifo_length, fifo_index, fifo_length, fifo_index)
    elif 'FFs' in dst:
        dst_str = 'FF[%d]' % (dst['FFs']['index'])
    else:
        return

    PrintLine('%s = %s;' % (dst_str, src_str))

def PrintBuffer(St, A, k):
    global indent
    buf = GetBuffer(St, A, k)
    points = GetPoints(A, k)
    indent_save = indent
    indent = 0
    PrintLine('#if %d!=(UNROLL_FACTOR)' % k)
    PrintLine('#error UNROLL_FACTOR != %d' % k)
    PrintLine('#endif')
    PrintLine()
    for dim, size in enumerate(St):
        PrintLine('#if %d!=(TILE_SIZE_DIM%d)' % (size, dim))
        PrintLine('#error TILE_SIZE_DIM%d != %d' % (dim, size))
        PrintLine('#endif')
        PrintLine()
    PrintLine('#if ((%s*(PIXEL_WIDTH))%%(BURST_WIDTH))!=0' % '*'.join(['(TILE_SIZE_DIM%d)'%x for x in range(0, len(St))]))
    PrintLine('#error TILE_SIZE%BURST_WIDTH != 0')
    PrintLine('#endif')
    PrintLine()
    indent = indent_save

    PrintLine('input_type input_points[(UNROLL_FACTOR)][%d];' % len(A))
    for idx, item in enumerate(A):
        PrintLine('//         input_points[(UNROLL_FACTOR)][%d] <=> (%s)' % (idx, str(item)[1:-1]))
    PrintLine()
    PrintLine('input_type input_buffer[%d];' % len(buf['inputs']))
    PrintLine('input_type FF[%d];' % len(buf['FFs']))
    PrintLine()
    for fifo_length in buf['FIFOs'].keys():
        PrintLine('input_type FIFO_%d[%d][%d];' % (fifo_length/k, len(buf['FIFOs'][fifo_length]), fifo_length/k))
        PrintLine('uint32_t FIFO_%d_p[%d] = {%s};' % (fifo_length/k, len(buf['FIFOs'][fifo_length]), str([0]*len(buf['FIFOs'][fifo_length]))[1:-1]))
        PrintLine()

    PrintLine('for(uint32_t unroll_index = 0; unroll_index<(UNROLL_FACTOR); ++unroll_index)')
    PrintLine('{')
    indent_save = indent
    indent = 0
    PrintLine('#pragma HLS unroll')
    indent = indent_save+1
    PrintLine('if(input_index*(UNROLL_FACTOR)+unroll_index < %s)' % ('*'.join(['(TILE_SIZE_DIM%d)'%x for x in range(0, len(St))])))
    PrintLine('{')
    indent = indent_save+2
    PrintLine('input_buffer[unroll_index] = input[input_index*(UNROLL_FACTOR)+unroll_index];')
    indent = indent_save+1
    PrintLine('}')
    indent = indent_save
    PrintLine('}')
    PrintLine()

    for idx, item in enumerate(buf['inputs']):
        for unroll_index in points[item].keys():
            PrintLine("input_points[%d][%d] = input_buffer[%d]; // (%s)" % (unroll_index, points[item][unroll_index], idx, str(A[points[item][unroll_index]])[1:-1]))
    for idx, item in enumerate(buf['FFs']):
        for unroll_index in points[item].keys():
            PrintLine("input_points[%d][%d] = FF[%d]; // (%s)" % (unroll_index, points[item][unroll_index], idx, str(A[points[item][unroll_index]])[1:-1]))
    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            for unroll_index in points[item].keys():
                PrintLine("input_points[%d][%d] = FIFO_%d[%d][FIFO_%d_p[%d]]; // (%s)" % (unroll_index, points[item][unroll_index], fifo_length/k, idx, fifo_length/k, idx, str(A[points[item][unroll_index]])[1:-1]))
    PrintLine()

    for idx, item in enumerate(buf['inputs']):
        PrintUpdate(GetProduceMap(buf), GetConsumeMap(buf), {'inputs':{'index':idx, 'produce':item}})
        PrintLine()

    for fifo_length in buf['FIFOs'].keys():
        for idx, item in enumerate(buf['FIFOs'][fifo_length]):
            fifo = 'FIFO_%d_p[%d]' % (fifo_length/k, idx)
            PrintLine('%s = %s==%d-1 ? 0 : %s+1;' % (fifo, fifo, fifo_length/k, fifo))

indent = 4
PrintBuffer(St, A, k)

