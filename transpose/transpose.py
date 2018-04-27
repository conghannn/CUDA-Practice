#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import sys
import time
 
import pycuda.autoinit

template = """
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_naive(float *odata, const float *idata, const unsigned int N) 
{
 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
 
    if(i < N && j < N )
    {
      odata[j + N * i] = idata[i + N * j];
    }
}

__global__ void transpose_naive_multi(float *odata, const float *idata, const unsigned int N)
{
  unsigned int x = blockIdx.x * TILE_DIM + threadIdx.x;
  unsigned int y = blockIdx.y * TILE_DIM + threadIdx.y;
  
  for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
  {
    odata[x * N + (y + j)] = idata[(y + j) * N + x];
  }
}

__global__ void transpose_share(float *odata, const float *idata, const unsigned int N)
{
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  tile[threadIdx.y][threadIdx.x] = idata[y * N + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  odata[y * N + x] = tile[threadIdx.x][threadIdx.y];
}


__global__ void transpose_share_multi(float *odata, const float *idata, const unsigned int N)
{
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
  {
     tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * N + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  
  y = blockIdx.x * TILE_DIM + threadIdx.y;
  
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
  {
     odata[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

__global__ void transpose_share_multi_conflict(float *odata, const float *idata, const unsigned int N)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
  {
     tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * N + x];
  }

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
  {
     odata[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
  }
}

"""


mod = compiler.SourceModule(template)
transpose_naive = mod.get_function("transpose_naive")
transpose_naive_multi = mod.get_function("transpose_naive_multi")
transpose_share = mod.get_function("transpose_share")
transpose_share_multi = mod.get_function("transpose_share_multi")
transpose_share_multi_conflict = mod.get_function("transpose_share_multi_conflict")

TEST = False
if TEST:
	block_size = 32
	i = 2
	size = i * 32
	matrix = np.random.random(size = (size, size)).astype(np.float32)
	matrix_out_cpu = np.empty_like(matrix).astype(np.float32)

	matrix_in = gpuarray.to_gpu(matrix)

	matrix_out_naive = gpuarray.empty((size, size), np.float32)
	matrix_out_naive_multi = gpuarray.empty((size, size), np.float32)
	matrix_out_share = gpuarray.empty((size, size), np.float32)
	matrix_out_share_multi = gpuarray.empty((size, size), np.float32)
	matrix_out_share_multi_conflict = gpuarray.empty((size, size), np.float32)


	matrix_out_cpu = np.transpose(matrix)

	transpose_naive(matrix_out_naive, matrix_in, np.uint32(size), block = (32, 32, 1), grid = (size / block_size, size / block_size, 1 ))
	
	transpose_naive_multi(matrix_out_naive_multi, matrix_in, np.uint32(size), block = (32, 8, 1), grid = (size / block_size, size / block_size, 1 ))

	transpose_share(matrix_out_share, matrix_in, np.uint32(size), block = (32, 32, 1), grid = (size / block_size, size / block_size, 1 ))

	transpose_share_multi(matrix_out_share_multi, matrix_in, np.uint32(size), block = (32, 8, 1), grid = (size / block_size, size / block_size, 1 ))

	transpose_share_multi_conflict(matrix_out_share_multi_conflict, matrix_in, np.uint32(size), block = (32, 8, 1), grid = (size / block_size, size / block_size, 1 ))

	print (np.allclose(matrix_out_cpu, matrix_out_naive.get()) & np.allclose(matrix_out_cpu, matrix_out_naive_multi.get()) & np.allclose(matrix_out_cpu, matrix_out_share.get()) & np.allclose(matrix_out_cpu, matrix_out_share_multi.get()) & np.allclose(matrix_out_cpu, matrix_out_share_multi_conflict.get()))
	
	'''
	print "cpu\n"
	print matrix_out_cpu
	
	print "naive\n"
	print matrix_out_naive.get()
	
	print "naive_multi\n"
	print matrix_out_naive_multi.get()
	
	print "share\n"
	print matrix_out_share.get()
	
	print "share_multi\n"
	print matrix_out_share_multi.get()
	
	print "share_multi_conflict\n"
	print matrix_out_share_multi_conflict.get()
	'''
	




N = range(1, 300)
time_cpu = []
time_naive = []
time_naive_multi = []
time_share = []
time_share_multi = []
time_share_multi_conflict = []

block_size = 32

for i in N:
    size = i * 32
    matrix = np.random.random(size = (size, size)).astype(np.float32)
    matrix_out_cpu = np.empty_like(matrix).astype(np.float32)

    matrix_in = gpuarray.to_gpu(matrix)
    matrix_out = gpuarray.empty((size, size), np.float32)

    start = time.time()
    matrix_out_cpu = np.transpose(matrix)
    time_cpu.append(time.time() - start)

    start = time.time()
    transpose_naive(matrix_out, matrix_in, np.uint32(size), block = (32, 32, 1), grid = (size / block_size, size / block_size, 1 ))
    time_naive.append(time.time() - start)

    start = time.time()
    transpose_naive_multi(matrix_out, matrix_in, np.uint32(size), block = (32, 8, 1), grid = (size / block_size, size / block_size, 1 ))
    time_naive_multi.append(time.time() - start)

    start = time.time()
    transpose_share(matrix_out, matrix_in, np.uint32(size), block = (32, 32, 1), grid = (size / block_size, size / block_size, 1 ))
    time_share.append(time.time() - start)

    start = time.time()
    transpose_share_multi(matrix_out, matrix_in, np.uint32(size), block = (32, 8, 1), grid = (size / block_size, size / block_size, 1 ))
    time_share_multi.append(time.time() - start)

    start = time.time()
    transpose_share_multi_conflict(matrix_out, matrix_in, np.uint32(size), block = (32, 8, 1), grid = (size / block_size, size / block_size, 1 ))
    time_share_multi_conflict.append(time.time() - start)


MAKE_PLOT = True
if MAKE_PLOT:
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pyplot as plt

  plt.gcf()
  plt.subplot(411)
  plt.plot(N, time_cpu,'r')
  plt.plot(N, time_naive_multi,'g')
  plt.plot(N, time_share_multi,'b')
  plt.legend(['cpu time', 'gpu naive', 'gpu share'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('three coding times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.subplot(412)
  plt.plot(N, time_naive,'g')
  plt.plot(N, time_naive_multi,'b')
  plt.legend(['plain', 'multi'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('naive')
  plt.gca().set_xlim((min(N), max(N)))

  plt.subplot(413)
  plt.plot(N, time_share,'g')
  plt.plot(N, time_share_multi,'b')
  plt.legend(['plain', 'multi'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('share')
  plt.gca().set_xlim((min(N), max(N)))

  plt.subplot(414)
  plt.plot(N, time_share,'g')
  plt.plot(N, time_share_multi,'b')
  plt.legend(['conflict', 'no conflict'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('share memory conflict')
  plt.gca().set_xlim((min(N), max(N)))
  

  plt.savefig('lots.png')

