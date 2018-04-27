#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import sys
import time
import pycuda.autoinit
import string
 
kernel_code_template = """

#define TILE_WIDTH $TILE_WIDTH

//advanced matrix multiplication
__global__ void Mul_Tile_Adv(float *A, float *B, float *C, int rA, int common, int lB)
{

  const uint bx = blockIdx.x;
  const uint by = blockIdx.y;
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;

  const uint row = blockDim.y * by + ty;
  const uint col = blockDim.x * bx + tx;

  const uint a_begin = TILE_WIDTH  * by * common;
  const uint b_begin = TILE_WIDTH  * bx;
  //const uint a_end = a_begin + common ;
  const uint a_step = TILE_WIDTH ;
  const uint b_step = TILE_WIDTH * lB;

  float cvalue = 0;

  for(int a = a_begin, b = b_begin, c = 0; c < common; a += a_step, b += b_step, c += TILE_WIDTH)
  {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    if(row < rA && c + tx < common ){
      As[ty][tx] = A[a + common * ty + tx];
    }else{

      As[ty][tx] = 0.0;
    }
    if(c + ty < common && col < lB){
      Bs[ty][tx] = B[b + lB * ty + tx];
    }else{
      Bs[ty][tx] = 0.0;
    }
        

    __syncthreads();

    for(uint k = 0; k < TILE_WIDTH; ++k)
    {
      cvalue += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  const uint c_begin = by * TILE_WIDTH * lB + TILE_WIDTH * bx;
  if(row < rA && col < lB){
    C[c_begin + lB * ty + tx] = cvalue;
  }

}
 
 
//Mul_Tile
__global__ void Mul_Tile(float *A, float *B, float *C, int rA, int common, int lB)
{

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = blockDim.y * by + ty;
  int col = blockDim.x * bx + tx;


  float cvalue = 0;

  for(int t = 0; t < (common - 1)/TILE_WIDTH + 1; ++t)
  {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    if(row < rA && t * TILE_WIDTH + tx < common ){
      As[ty][tx] = A[row * common + t*TILE_WIDTH + tx];
    }else{

      As[ty][tx] = 0.0;
    }
    if(t*TILE_WIDTH + ty < common && col < lB){
      Bs[ty][tx] = B[(t*TILE_WIDTH + ty) * lB + col];
    }else{
      Bs[ty][tx] = 0.0;
    }

    __syncthreads();

    for(int k = 0; k < TILE_WIDTH; ++k)
    {
      cvalue += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  if(row < rA && col < lB){
    C[row * lB + col] = cvalue;
  }

}
"""

BLOCK_SIZE = 16
template = string.Template(kernel_code_template)
code = template.substitute(TILE_WIDTH = BLOCK_SIZE) 
mod = compiler.SourceModule(code)
mat_mul_adv = mod.get_function("Mul_Tile_Adv")
mat_mul_tile = mod.get_function("Mul_Tile")


x = np.arange(0,300,1)
py_times = np.zeros_like(x).astype(np.float64) 
adv_cu_times = np.zeros_like(x).astype(np.float64)
tile_cu_times = np.zeros_like(x).astype(np.float64)
 
for r in xrange(1,300,1):

  MATRIX_A_ROWS = 3 * r
  MATRIX_A_COLS = 4 * r
  MATRIX_B_ROWS = 4 * r
  MATRIX_B_COLS = 5 * r

  
  a_cpu = np.random.rand(MATRIX_A_ROWS, MATRIX_A_COLS).astype(np.float32)
  b_cpu = np.random.rand(MATRIX_B_ROWS, MATRIX_B_COLS).astype(np.float32)

  start = time.time()
  c_cpu = np.dot(a_cpu, b_cpu)
  py_times[r] = time.time() - start

  a_gpu = gpuarray.to_gpu(a_cpu)
  b_gpu = gpuarray.to_gpu(b_cpu)
  c_gpu = gpuarray.empty((MATRIX_A_ROWS, MATRIX_B_COLS), dtype = np.float32)
  c_gpu2 = gpuarray.empty((MATRIX_A_ROWS, MATRIX_B_COLS), dtype = np.float32)

  start = time.time()
  mat_mul_adv(a_gpu, b_gpu, c_gpu, np.int32(MATRIX_A_ROWS) , np.int32(MATRIX_A_COLS) ,np.int32(MATRIX_B_COLS), grid = ((MATRIX_B_COLS - 1) // BLOCK_SIZE  + 1, (MATRIX_A_ROWS - 1) // BLOCK_SIZE  + 1, 1) , block = (BLOCK_SIZE , BLOCK_SIZE , 1))
  adv_cu_times[r] = time.time() - start

  start = time.time()
  mat_mul_tile(a_gpu, b_gpu, c_gpu2, np.int32(MATRIX_A_ROWS) , np.int32(MATRIX_A_COLS) ,np.int32(MATRIX_B_COLS), grid = ((MATRIX_B_COLS - 1) // BLOCK_SIZE  + 1, (MATRIX_A_ROWS - 1) // BLOCK_SIZE  + 1, 1) , block = (BLOCK_SIZE , BLOCK_SIZE , 1))
  tile_cu_times[r] = time.time() - start
  #print 'c_cpu', c_cpu
  #print 'c_gpu1', c_gpu.get()
  #print 'c_gpu2', c_gpu2.get()
  print r, np.allclose(c_cpu, c_gpu.get(), atol = 0.001) and np.allclose(c_cpu, c_gpu2.get(), atol = 0.001)


tile_size_array = [2, 4, 8, 16, 32]
x_block = np.arange(0, 5, 1)
adv_cu_times_blocksize = np.zeros_like(x_block).astype(np.float64)
tile_cu_times_blocksize = np.zeros_like(x_block).astype(np.float64)

r = 300
MATRIX_A_ROWS = 3 * r
MATRIX_A_COLS = 4 * r
MATRIX_B_ROWS = 4 * r
MATRIX_B_COLS = 5 * r

a_cpu_block = np.random.rand(MATRIX_A_ROWS, MATRIX_A_COLS).astype(np.float32)
b_cpu_block = np.random.rand(MATRIX_B_ROWS, MATRIX_B_COLS).astype(np.float32)

a_gpu_block = gpuarray.to_gpu(a_cpu)
b_gpu_block = gpuarray.to_gpu(b_cpu)

c_gpu_block = gpuarray.empty((MATRIX_A_ROWS, MATRIX_B_COLS), dtype = np.float32)
c_gpu2_block = gpuarray.empty((MATRIX_A_ROWS, MATRIX_B_COLS), dtype = np.float32)

for s in xrange(0, len(tile_size_array)):
  BLOCK_SIZE = tile_size_array[s]
  template = string.Template(kernel_code_template)
  code = template.substitute(TILE_WIDTH = BLOCK_SIZE) 
  mod = compiler.SourceModule(code)
  mat_mul_adv = mod.get_function("Mul_Tile_Adv")
  mat_mul_tile = mod.get_function("Mul_Tile")

  iterative = 50
  temp_adv = np.zeros_like(np.arange(0, iterative, 1)).astype(np.float64)
  temp_tile = np.zeros_like(np.arange(0, iterative, 1)).astype(np.float64)

  for j in xrange(0, iterative):
    start = time.time()
    mat_mul_adv(a_gpu_block, b_gpu_block, c_gpu_block, np.int32(MATRIX_A_ROWS) , np.int32(MATRIX_A_COLS) ,np.int32(MATRIX_B_COLS), grid = ((MATRIX_B_COLS - 1) // BLOCK_SIZE  + 1, (MATRIX_A_ROWS - 1) // BLOCK_SIZE  + 1, 1) , block = (BLOCK_SIZE , BLOCK_SIZE , 1))
    temp_adv[j] = time.time() - start

    start = time.time()
    mat_mul_tile(a_gpu_block, b_gpu_block, c_gpu2_block, np.int32(MATRIX_A_ROWS) , np.int32(MATRIX_A_COLS) ,np.int32(MATRIX_B_COLS), grid = ((MATRIX_B_COLS - 1) // BLOCK_SIZE  + 1, (MATRIX_A_ROWS - 1) // BLOCK_SIZE  + 1, 1) , block = (BLOCK_SIZE , BLOCK_SIZE , 1))
    temp_tile[j] = time.time() - start
    print s, j, np.allclose(c_gpu.get(), c_gpu2.get(), atol = 0.001)

  adv_cu_times_blocksize[s] = np.mean(temp_adv)
  tile_cu_times_blocksize[s] = np.mean(temp_tile)
  



 
MAKE_PLOT = True
if MAKE_PLOT:
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pyplot as plt

  plt.gcf()
  plt.subplot(311)
  plt.plot(x, py_times,'r')
  plt.plot(x, adv_cu_times,'g')
  plt.plot(x, tile_cu_times,'b')
  plt.legend(['python mul', 'gpu adv mult', 'gpu tile mult'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('three coding times')
  plt.gca().set_xlim((min(x), max(x)))

  plt.subplot(312)
  plt.plot(x, adv_cu_times,'g')
  plt.plot(x, tile_cu_times,'b')
  plt.legend(['gpu adv mult', 'gpu tile mult'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('only gpu times')
  plt.gca().set_xlim((min(x), max(x)))
  
  plt.subplot(313)
  plt.plot(np.array(tile_size_array).astype(np.int32), adv_cu_times_blocksize,'g')
  plt.plot(np.array(tile_size_array).astype(np.int32), tile_cu_times_blocksize,'b')
  plt.legend(['gpu adv mult', 'gpu tile mult'], loc='upper left')
  plt.xlabel('tile size')
  plt.ylabel('speed at 300 ratio factor')
  plt.gca().set_xlim((min(tile_size_array), max(tile_size_array)))
  
  plt.savefig('lots.png')