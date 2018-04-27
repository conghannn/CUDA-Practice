#!python 
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from numpy import linalg as la
from pycuda import driver, compiler, gpuarray, tools
import sys
#the following module is used to mark the time stamps
import time

import pycuda.autoinit



kernel_code_template = """
#include <stdio.h>
__global__ void MatrixMulTile(float *A, float *B, float *C)
{
    const uint TILE_WIDTH = %(BLOCK_SIZE)s;

	const uint rA = %(MATRIX_A_ROWS)s;
	const uint common = %(MATRIX_A_COLS)s;
	const uint lB = %(MATRIX_B_COLS)s;


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
"""



mod1 = compiler.SourceModule("""

__global__ void Mat(float *A, float *B, float *C, int rA, int common, int lB)
{

    const int TILE_WIDTH = 32;
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
""")


BLOCK_SIZE = 32

mul1 = mod1.get_function("Mat")

x = np.arange(0,100,1)
py_times = np.zeros_like(x).astype(np.float64)
gpu_times = np.zeros_like(x).astype(np.float64)
gpu_improved_times = np.zeros_like(x).astype(np.float64)

#for r in xrange(1,10,1):

r = 3000
MATRIX_A_ROWS = 2 * r
MATRIX_A_COLS = 3 * r
MATRIX_B_ROWS = 3 * r
MATRIX_B_COLS = 4 * r

a_cpu = np.random.rand(MATRIX_A_ROWS, MATRIX_A_COLS).astype(np.float32)
b_cpu = np.random.rand(MATRIX_B_ROWS, MATRIX_B_COLS).astype(np.float32)

#start = time.time()
#c_cpu = np.dot(a_cpu, b_cpu)
#py_times[r] = time.time() - start
#print time.time() - start


a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)
c_gpu = gpuarray.empty((MATRIX_A_ROWS, MATRIX_B_COLS), dtype = np.float32)

c_gpu2 = gpuarray.empty((MATRIX_A_ROWS, MATRIX_B_COLS), dtype = np.float32)


'''
kernel_code = kernel_code_template % {
	'MATRIX_A_ROWS' : MATRIX_A_ROWS,
	'MATRIX_A_COLS' : MATRIX_A_COLS,
	'MATRIX_B_COLS' : MATRIX_B_COLS,
	'BLOCK_SIZE' : BLOCK_SIZE
}

mod = compiler.SourceModule(kernel_code)
mul = mod.get_function("MatrixMulTile")

start = time.time()
mul(a_gpu, b_gpu, c_gpu, grid = ((MATRIX_B_COLS - 1) // BLOCK_SIZE  + 1, (MATRIX_A_ROWS - 1) // BLOCK_SIZE  + 1, 1) , block = (BLOCK_SIZE , BLOCK_SIZE , 1))
#gpu_improved_times[r] = time.time() - start
print time.time() - start
'''
start = time.time()
mul1(a_gpu, b_gpu, c_gpu2, np.int32(MATRIX_A_ROWS) , np.int32(MATRIX_A_COLS) ,np.int32(MATRIX_B_COLS), grid = ((MATRIX_B_COLS - 1) // BLOCK_SIZE  + 1, (MATRIX_A_ROWS - 1) // BLOCK_SIZE  + 1, 1) , block = (BLOCK_SIZE , BLOCK_SIZE , 1))
#gpu_times[r] = time.time() - start
print (time.time() - start, r, BLOCK_SIZE)

#print np.allclose(c_cpu, c_gpu.get()) and np.allclose(c_cpu, c_gpu2.get())

'''
print "-" * 80
print "Matrix A (GPU):"
print a_gpu.get()

print "-" * 80
print "Matrix B (GPU):"
print b_gpu.get()

print "-" * 80
print "Matrix C (GPU):"
print c_gpu2.get()

print "-" * 80
print "Matrix C (CPU):"
print c_cpu

print "-" * 80
print "CPU-GPU difference:"
print c_cpu - c_gpu2.get()
print "L2 norm:", la.norm(c_cpu - c_gpu2.get())
print np.allclose(c_cpu, c_gpu.get()) and np.allclose(c_cpu, c_gpu2.get())

'''

MAKE_PLOT = False
if MAKE_PLOT:
	import matplotlib as mpl
	mpl.use('agg')
	import matplotlib.pyplot as plt

	plt.gcf()
	plt.subplot(311)
	plt.plot(x, py_times,'r')
	plt.plot(x, gpu_times,'g')
	plt.plot(x, gpu_improved_times,'b')
	plt.legend(['python mul', 'gpu tile mult', 'gpu improved mult'], loc='upper left')
	plt.xlabel('matrix ratio increase factor')
	plt.ylabel('output coding times')
	plt.gca().set_xlim((min(x), max(x)))


	plt.savefig('lots.png')



