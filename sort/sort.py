#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import sys
import time
 
import pycuda.autoinit

template = """


__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; 
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i ^ j;

  if (ixj > i) 
  {
    if (i & k == 0) 
    {
      if (dev_values[i] > dev_values[ixj]) 
      {
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if (i & k != 0) {
      if (dev_values[i] < dev_values[ixj]) 
      {
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}
"""

mod = compiler.SourceModule(template)
sort = mod.get_function("bitonic_sort_step")

def sort_gpu_arb(x, n):
  tn = 512
  bn = (n - 1) // tn + 1
  numall = bn * tn
  if(n < numall):
    temp = [np.finfo(np.float64).max] * (numall - n)
    array_all =  np.append(x, temp).astype(np.float32)
    a_gpu = gpuarray.to_gpu(array_all)
    k = 2
    while k <= numall:
      j = k >> 1
      while j > 0:
        sort(a_gpu, np.int32(j), np.int32(k), block = (tn, 1, 1), grid = (bn, 1, 1))
        j = j >> 1
      k = k << 1
  else:
    a_gpu = gpuarray.to_gpu(x)
    k = 2
    while k <= numall:
      j = k >> 1
      while j > 0:
        sort(a_gpu, np.int32(j), np.int32(k), block = (tn, 1, 1), grid = (bn, 1, 1))
        j = j >> 1
      k = k << 1
  res = a_gpu.get()
  return res[:n]




THREAD_NUM = 128
#BLOCK_NUM = 1
#NUM_VALS = THREAD_NUM * BLOCK_NUM
'''
a = np.random.randint(NUM_VALS, size = NUM_VALS).astype(np.float32)
a_gpu = gpuarray.to_gpu(a)

k = 2
while k <= NUM_VALS:
  j = k >> 1
  while j > 0:
    sort(a_gpu, np.int32(j), np.int32(k), block = (THREAD_NUM, 1, 1), grid = (BLOCK_NUM, 1, 1))
    j = j >> 1
  k = k << 1

print a_gpu.get()
print "#####"
print np.sort(a)
'''

N = range(1, 500)
time_cpu = []
time_gpu = []

for i in N:
  BLOCK_NUM = i
  NUM_VALS = THREAD_NUM * BLOCK_NUM
  
  a = np.random.randint(3000, size = NUM_VALS).astype(np.float32)
  a_gpu = gpuarray.to_gpu(a)
  

  start = time.time()
  np.sort(a)
  time_cpu.append(time.time() - start)


  start = time.time()
  k = 2
  while k <= NUM_VALS:
    j = k >> 1
    while j > 0:
      sort(a_gpu, np.int32(j), np.int32(k), block = (THREAD_NUM, 1, 1), grid = (BLOCK_NUM, 1, 1))
      j = j >> 1
    k = k << 1
  time_gpu.append(time.time() - start)


'''
a = np.random.randint(3000, size = 512 * 1000).astype(np.float32)
sort_gpu_arb(a, 512 * 1000)
'''


MAKE_PLOT = True
if MAKE_PLOT:
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pyplot as plt

  plt.gcf()
  plt.subplot(311)
  plt.plot(N, time_cpu,'r')
  plt.plot(N, time_gpu,'g')
  plt.legend(['cpu time', 'bitonic time'], loc='upper left')
  plt.xlabel('array ratio increase factor')
  plt.ylabel('two coding times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.savefig('lots.png')




