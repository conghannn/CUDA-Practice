#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import matplotlib as mpl
from PIL import Image
import scipy
from scipy import misc
mpl.use('agg')
import matplotlib.pyplot as plt
from scipy import signal
import string

def create_img(filename, cols, rows):
    size = (cols, rows)
    im = Image.open(filename).convert('L')  # .convert('L') converts the image to grayscale
    im = im.resize(size)
    return np.array(im)

filters = {
    'identity': np.array([[0,0,0], [0,1,0], [0,0,0]]).astype(np.int32),
    'sharpen': np.array([[0,-1,0], [-1,5,-1],[0,-1,0]]).astype(np.int32),
    'blur': np.array([[1,1,1], [1, 1, 1], [1, 1, 1]]).astype(np.int32),
    'edge_det': np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).astype(np.int32),
    'emboss': np.array([[2, 1., 0.], [1., 1, -1.], [0, -1, -2]]).astype(np.int32),
    'sob_x': np.array([[-1, 0., 1.], [-2., 0, 2.], [-1, 0, 1]]).astype(np.int32),
    'sob_y': np.array([[-1, -2., -1.], [0., 0, 0.], [1, 2, 1.]]).astype(np.int32),
    'smooth_5x5': np.array(
        [[0,1,2, 1, 0], [1, 4, 8, 4, 1], [2, 8, 16, 8, 2], [1, 4, 8, 4, 1], [0, 1, 2, 1, 0]]).astype(np.int32)
}

Template = """
 
#define TILE_WIDTH  $TILE_WIDTH

__global__ void convolution(int* in, int* out, const int * __restrict__ M, int height, int width, int mask_width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int ROW_TILE_WIDTH = (TILE_WIDTH - (mask_width - 1));
    
    int row_o = blockIdx.y * ROW_TILE_WIDTH + ty;
    int col_o = blockIdx.x * ROW_TILE_WIDTH + tx;
    int row_i = row_o - (mask_width - 1) / 2;
    int col_i = col_o - (mask_width - 1) / 2;
    
    __shared__ int Ns[TILE_WIDTH][TILE_WIDTH];
    
    if((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
    {
    	Ns[ty][tx] = in[row_i * width + col_i];
    }
    else
    {
    	Ns[ty][tx] = 0;
    }
    
    __syncthreads();
    

    int output = 0;
    if(ty < ROW_TILE_WIDTH  && tx < ROW_TILE_WIDTH )
    {
        for(int i = 0; i < mask_width; i++)
        {
            for(int j = 0; j < mask_width; j++)
            {
                output += M[mask_width * mask_width - 1 - j - i * mask_width] * Ns[ty + i][tx + j];
            }
        }
        if(row_o < height && col_o < width)
            {
            	out[row_o * width + col_o] = output;
            }
    }
}


"""

BLOCK_SIZE = 32
template = string.Template(Template)
code = template.substitute(TILE_WIDTH = BLOCK_SIZE) 
mod = SourceModule(code)

func = mod.get_function("convolution")

def GPU_convolution(matrix, mask):
    (M_height, M_width) = mask.shape
    M_width = np.int32(M_width)
    (M, N) = matrix.shape
    M = np.int32(M)
    N = np.int32(N)
    sourceImage_gpu = cuda.mem_alloc(matrix.nbytes)
    destImage_gpu = cuda.mem_alloc(matrix.nbytes)
    mask_gpu = cuda.mem_alloc(mask.nbytes)
    destImage_cpu = np.empty_like(matrix).astype(np.int32)
    cuda.memcpy_htod(sourceImage_gpu, matrix)
    cuda.memcpy_htod(mask_gpu, mask)
    
    start = time.time()
    func(sourceImage_gpu, destImage_gpu, mask_gpu, M, N, M_width, block=(BLOCK_SIZE, BLOCK_SIZE, 1),
         grid=((N - 1) / (BLOCK_SIZE - (M_width - 1)) + 1, (M - 1) / (BLOCK_SIZE - (M_width - 1)) + 1, 1))
    time1.append(time.time() - start)
    
    start = time.time()
    cpu = signal.convolve2d(matrix, mask, boundary='fill', mode='same')
    time2.append(time.time() - start)
    
    cuda.memcpy_dtoh(destImage_cpu, destImage_gpu)
    print "whether cpu and gpu results are equal:",np.allclose(destImage_cpu, cpu)
    return destImage_cpu

time_gpu = []
time_cpu = []

MASK = np.random.randint(-5, 5, (5,5)).astype(np.int32)
N = range(1, 65)
m = 5
for i in N:
	time1 = []
	time2 = []
	M = np.int32(i * 32)
	N = np.int32(i * 32)
	SourceMatrix= np.random.randint(0,256,(M, N)).astype(np.int32)

	for j in xrange(m):
		matrix_out = GPU_convolution(SourceMatrix, MASK)
	time_gpu.append(np.average(time1))
	time_cpu.append(np.average(time2))

TEST_BLOCK = True
if TEST_BLOCK:
	BLOCK_SIZE = 16
	template = string.Template(Template)
	code = template.substitute(TILE_WIDTH = BLOCK_SIZE) 
	mod = SourceModule(code)

	func = mod.get_function("convolution")


	time_gpu2 = []

	N = range(1, 65)
	m = 5
	for i in N:
		time1 = []
		time2 = []
		M = np.int32(i * 32)
		N = np.int32(i * 32)
		SourceMatrix = np.random.randint(0,256,(M, N)).astype(np.int32)

		for j in xrange(m):
			(M_height, M_width) = MASK.shape
			M_width = np.int32(M_width)
			(M, N) = SourceMatrix.shape
			M = np.int32(M)
			N = np.int32(N)
			sourceImage_gpu2 = cuda.mem_alloc(SourceMatrix.nbytes)
			destImage_gpu2 = cuda.mem_alloc(SourceMatrix.nbytes)
			mask_gpu2 = cuda.mem_alloc(MASK.nbytes)
			destImage_cpu2 = np.empty_like(SourceMatrix).astype(np.int32)
			cuda.memcpy_htod(sourceImage_gpu2, SourceMatrix)
			cuda.memcpy_htod(mask_gpu2, MASK)

			start = time.time()
			func(sourceImage_gpu2, destImage_gpu2, mask_gpu2, M, N, M_width, block=(BLOCK_SIZE, BLOCK_SIZE, 1), grid=((N - 1) / (BLOCK_SIZE - (M_width - 1)) + 1, (M - 1) / (BLOCK_SIZE - (M_width - 1)) + 1, 1))
			time1.append(time.time() - start)

			start = time.time()
			cpu = signal.convolve2d(SourceMatrix, MASK, boundary='fill', mode='same')
			time2.append(time.time() - start)

			cuda.memcpy_dtoh(destImage_cpu2, destImage_gpu2)
			print "whether cpu and gpu results are equal:",np.allclose(destImage_cpu2, cpu)
		time_gpu2.append(np.average(time1))


N = range(1, 65)

MAKE_PLOT = True
if MAKE_PLOT:
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pyplot as plt

  plt.gcf()
  plt.subplot(311)
  plt.plot(N, time_cpu,'g')
  plt.plot(N, time_gpu,'b')
  plt.legend(['cpu time', 'gpu time'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('cpu and gpu time')
  plt.gca().set_xlim((min(N), max(N)))

  plt.subplot(312)
  plt.plot(N, time_gpu,'b')
  plt.legend(['gpu tile mult'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('only gpu times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.subplot(313)
  plt.plot(N, time_gpu2,'r')
  plt.plot(N, time_gpu,'b')
  plt.legend(['size 16', 'size 32'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('kernel size comparision')
  plt.gca().set_xlim((min(N), max(N)))

  plt.savefig('lots.png')


image = create_img("./thrones-002-2.jpg", 640,640).astype(np.int32)
for filter in filters:
    temp_gpu = GPU_convolution(image,filters[filter])
    temp_cpu = signal.convolve2d(image, filters[filter], boundary='fill', mode='same')
    scipy.misc.imsave('cuda_cpu_%r.jpg'%filter, temp_cpu)
    scipy.misc.imsave('cuda_gpu_%r.jpg'%filter, temp_gpu)
