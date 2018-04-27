#!/usr/bin/env python
 
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import sys
import time
 
import pycuda.autoinit
import string


def filter(x, n):
  dest = []
  for i in xrange(n):
    if(x[i] > 0):
      dest.append(x[i])
  return np.array(dest)





template = """
#define BS 32
#define NPER_THREAD 8
#define WARP_SZ 32


__global__ void filter_naive(int *dst, int *nres, const int *src, int n) 
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n && src[i] > 0)
    dst[atomicAdd(nres, 1)] = src[i];
}

__global__ void filter_shared(int *dst, int *nres, const int* src, int n) 
{
  __shared__ int l_n;
  int i = blockIdx.x * (NPER_THREAD * BS) + threadIdx.x;

  for (int iter = 0; iter < NPER_THREAD; iter++) {

    if (threadIdx.x == 0)
      l_n = 0;
    __syncthreads();

    int d, pos;

    if(i < n) {
      d = src[i];
      if(d > 0)
        pos = atomicAdd(&l_n, 1);
    }
    __syncthreads();

    if(threadIdx.x == 0)
      l_n = atomicAdd(nres, l_n);
    __syncthreads();

    if(i < n && d > 0) {
      pos += l_n; 
      dst[pos] = d;
    }
    __syncthreads();

    i += BS;
  }
}

__device__ inline int lane_id(void)
{
  return threadIdx.x % WARP_SZ;
}

__device__ int warp_bcast(int v, int leader)
{
  return __shfl(v, leader);
}

__device__ int atomicAggInc(int *ctr)
{
  int mask = __ballot(1);
  int leader = __ffs(mask) - 1;
  int res;
  if(lane_id() == leader)
  {
    res = atomicAdd(ctr, __popc(mask));
  }
  res = warp_bcast(res, leader);

  return res + __popc(mask & ((1 << lane_id()) - 1));
}

__global__ void filter_warp(int *dst, const int *src, int *nres, int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i >= n)
  {
    return;
  }
  if(src[i] > 0)
  {
    dst[atomicAggInc(nres)] = src[i];
  }
}

"""




mod = compiler.SourceModule(template)
filter_naive = mod.get_function("filter_naive")
filter_shared = mod.get_function("filter_shared")
filter_warp = mod.get_function("filter_warp")

TEST = False
if TEST:

  size = 32
  array_cpu = (np.random.random(size = size) - 0.5).astype(np.float32)
  array_out_cpu = filter(array_cpu, size)

  array_in = gpuarray.to_gpu(array_cpu)

  nes_in1 = gpuarray.zeros(1, dtype = np.int32)
  nes_in2 = gpuarray.zeros(1, dtype = np.int32)
  nes_in3 = gpuarray.zeros(1, dtype = np.int32)


  array_out = gpuarray.empty(size, np.float32)

  array_out_shared = gpuarray.empty(size, np.float32)

  array_out_warp = gpuarray.empty(size, np.float32)

  
  filter_naive(array_out, nes_in1, array_in,  np.int32(size) , block = (32, 1, 1), grid = ((size - 1) // 32 + 1, 1, 1))

  filter_shared(array_out_shared, nes_in2, array_in,  np.int32(size), block = (32, 1, 1), grid = ((size - 1) // (32 * 8) + 1, 1, 1))

  filter_warp(array_out_warp, array_in, nes_in3,  np.int32(size) , block = (32, 1, 1), grid = ((size - 1) // 32 + 1, 1, 1))

  result_naive = np.empty(int(nes_in1.get()), np.float32)
  result_naive = array_out_shared.get()[: int(nes_in1.get())]

  result_shared = np.empty(int(nes_in2.get()), np.float32)
  result_shared = array_out_shared.get()[: int(nes_in2.get())]

  result_warp = np.empty(int(nes_in2.get()), np.float32)
  result_warp = array_out_warp.get()[: int(nes_in3.get())]


  print array_cpu
  print array_out_cpu
  print result_naive
  print result_shared
  print result_warp



N = range(1, 300)
time_cpu = []
time_naive = []
time_shared = []
time_warp = []

block_size = 32
NPER = 8
for i in N:
  size = i * 32
  array_cpu = (np.random.random(size = size) - 0.5).astype(np.float32)

  start = time.time()
  array_out_cpu = filter(array_cpu, size)
  time_cpu.append(time.time() - start)
  
  array_in = gpuarray.to_gpu(array_cpu)

  nes_in1 = gpuarray.zeros(1, dtype = np.int32)
  nes_in2 = gpuarray.zeros(1, dtype = np.int32)
  nes_in3 = gpuarray.zeros(1, dtype = np.int32)

  array_out = gpuarray.empty(size, np.float32)

  array_out_shared = gpuarray.empty(size, np.float32)

  array_out_warp = gpuarray.empty(size, np.float32)

  start = time.time()
  filter_naive(array_out, nes_in1, array_in,  np.int32(size) , block = (block_size, 1, 1), grid = ((size - 1) // block_size + 1, 1, 1))
  time_naive.append(time.time() - start)

  #print array_out.get()

  start = time.time()
  filter_shared(array_out_shared, nes_in2, array_in,  np.int32(size), block = (block_size, 1, 1), grid = ((size - 1) // (block_size * NPER) + 1, 1, 1))
  time_shared.append(time.time() - start)

  start = time.time()
  filter_warp(array_out_warp, array_in, nes_in3,  np.int32(size) , block = (32, 1, 1), grid = ((size - 1) // 32 + 1, 1, 1))
  time_warp.append(time.time() - start)

MAKE_PLOT = True
if MAKE_PLOT:
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pyplot as plt

  plt.gcf()
  plt.subplot(311)
  plt.plot(N, time_cpu,'r')
  plt.plot(N, time_naive,'g')
  plt.plot(N, time_shared,'b')
  plt.plot(N, time_warp,'k')
  plt.legend(['cpu time', 'gpu naive', 'gpu share', 'gpu warp'], loc='upper left')
  plt.xlabel('array ratio increase factor')
  plt.ylabel('all coding times')
  plt.gca().set_xlim((min(N), max(N)))
  
  plt.subplot(312)
  plt.plot(N, time_naive,'g')
  plt.plot(N, time_shared,'b')
  plt.plot(N, time_warp,'k')
  plt.legend(['gpu naive', 'gpu share', 'gpu warp'], loc='upper left')
  plt.xlabel('array ratio increase factor')
  plt.ylabel('gpu coding times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.subplot(313)
  plt.plot(N, time_shared,'b')
  plt.plot(N, time_warp,'k')
  plt.legend(['gpu share', 'gpu warp'], loc='upper left')
  plt.xlabel('array ratio increase factor')
  plt.ylabel('two coding times')
  plt.gca().set_xlim((min(N), max(N)))


  plt.savefig('lots.png')







