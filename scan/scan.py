#!/usr/bin/env python

 
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import sys
import time
import pycuda.scan as scan
 
import pycuda.autoinit


def scan_cpu(x, n):
  out = np.zeros(n, dtype = np.float32)
  out[0] = x[0]
  for i in xrange(1, n):
    out[i] = out[i - 1] + x[i]
  return out


template = '''
#define BLOCK_SIZE 1024

__global__ void scan_inefficient(float *odata, float *idata, int inputsize )
{
  __shared__ float mem[BLOCK_SIZE];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < inputsize)
  {
    mem[threadIdx.x] = idata[i];
  }

  for(unsigned int stride = 1; stride <= threadIdx.x; stride *= 2)
  {

    __syncthreads();

    float inl = mem[threadIdx.x - stride];

    __syncthreads();

    mem[threadIdx.x] += inl;
  }

  if(i < inputsize)
  {
    odata[i] = mem[threadIdx.x];
  }

}



__global__ void scan_efficient(float *odata, float *idata, int inputsize )
{
  __shared__ float mem[BLOCK_SIZE * 2];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < inputsize)
  {
    mem[threadIdx.x] = idata[i];
  }
  
  for(unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
  {
    
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    
    if(index < 2 * BLOCK_SIZE)
    {
      mem[index] += mem[index - stride];
    }
    
    __syncthreads();
  }

  for(unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
  {
    __syncthreads();

    int index = (threadIdx.x + 1) * stride * 2 - 1;

    if(index + stride < 2 * BLOCK_SIZE)
    {
      mem[index + stride] += mem[index];
    }
  }
  __syncthreads();

  if(i < inputsize)
  {
    odata[i] = mem[threadIdx.x];
  }

}


'''

BLOCK_S = 1024
N = range(1, 300)
time_cpu = []
time_inefficient = []
time_efficient = []
time_scan_knl = []

for i in N:

  size = 32 * i

  mod = compiler.SourceModule(template)
  scan_inefficient = mod.get_function("scan_inefficient")
  scan_efficient = mod.get_function("scan_efficient")


  a = np.random.randint(5, size = size).astype(np.float32)
  a_gpu = gpuarray.to_gpu(a)
  b_gpu = gpuarray.empty_like(a_gpu)

  a_gpu2 = gpuarray.to_gpu(a)
  b_gpu2 = gpuarray.empty_like(a_gpu2)

  start = time.time()
  b_cpu = scan_cpu(a, size)
  time_cpu.append(time.time() - start)

  start = time.time()
  scan_inefficient(b_gpu, a_gpu, np.int32(size),  block = (BLOCK_S, 1, 1), grid = ((size - 1) // BLOCK_S + 1, 1, 1))
  time_inefficient.append(time.time() - start)

  start = time.time()
  scan_efficient(b_gpu2, a_gpu2, np.int32(size),  block = (BLOCK_S, 1, 1), grid = ((size - 1) // BLOCK_S + 1, 1, 1))
  time_efficient.append(time.time() - start)
  
  knl_gpu = gpuarray.to_gpu(a)
  scan_knl = scan.InclusiveScanKernel(np.float32, "a + b", "0")
  start = time.time()
  scan_knl(knl_gpu)
  time_scan_knl.append(time.time() - start)
  
  '''
  print a
  print "\n"
  print b_gpu.get()
  print "\n"
  print scan_cpu(a, size)
  '''


MAKE_PLOT = True
if MAKE_PLOT:
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pyplot as plt

  plt.gcf()
  plt.subplot(311)
  plt.plot(N, time_cpu,'r')
  plt.plot(N, time_inefficient,'g')
  plt.plot(N, time_efficient,'b')
  plt.legend(['cpu time', 'scan inefficient', 'scan efficient'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('three coding times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.subplot(312)
  plt.plot(N, time_inefficient,'g')
  plt.plot(N, time_efficient,'b')
  plt.legend(['scan inefficient', 'scan efficient'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('efficience comparision')
  plt.gca().set_xlim((min(N), max(N)))

  plt.subplot(313)
  plt.plot(N, time_scan_knl,'r')
  plt.plot(N, time_inefficient,'g')
  plt.plot(N, time_efficient,'b')
  plt.legend(['scan_knl time', 'scan inefficient', 'scan efficient'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('different gpu times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.savefig('lots.png')


