import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import sys
import time
from pycuda.reduction import ReductionKernel
import pycuda.autoinit


def reduction_cpu(x, n):
	res = 0
	for i in xrange(n):
		res += x[i]
	return res

'''
size = 5
knl = ReductionKernel(dtype_out = np.float32, neutral = "0", reduce_expr = "a+b", map_expr = "x[i]",arguments = "float *x")
a = np.random.randint(5, size = size).astype(np.float32)
a_gpu = gpuarray.to_gpu(a)
result_gpu = knl(a_gpu)

print a
print "\n"
print reduction_cpu(a, size)
print "\n"
print result_gpu.get()
'''

knl = ReductionKernel(dtype_out = np.float32, neutral = "0", reduce_expr = "a+b", map_expr = "x[i]",arguments = "float *x")

time_cpu = []
time_knl = []

N = range(1, 3000)
for i in N:
  size = 32 * i
  a = np.random.randint(5, size = size).astype(np.float32)
  a_gpu = gpuarray.to_gpu(a)

  start = time.time()
  reduction_cpu(a, size)
  time_cpu.append(time.time() - start)

  start = time.time()
  knl(a_gpu)
  time_knl.append(time.time() - start)


MAKE_PLOT = True
if MAKE_PLOT:
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pyplot as plt

  plt.gcf()
  plt.subplot(311)
  plt.plot(N, time_cpu,'r')
  plt.plot(N, time_knl,'b')
  plt.legend(['cpu time', 'reduction kernel'], loc='upper left')
  plt.xlabel('array ratio increase factor')
  plt.ylabel('two coding times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.savefig('lots.png')
