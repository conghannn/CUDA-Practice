#!/usr/bin/env python
 
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import sys
import time
import pycuda.autoinit
import string
import skcuda.linalg as culinalg
import skcuda.misc as cumisc
from skcuda.cublas import *

culinalg.init()

N = range(1, 300)
time_cpu = []
time_linalg = []
time_cula = []
for i in N:
	t = np.float32
	n = i * 4
    
	a = np.asarray(np.random.rand(n,n), t)
	b = np.asarray(np.random.rand(n,n), t)

	start = time.time()
	c = np.dot(a, b)
	time_cpu.append(time.time() - start)

	a_gpu = gpuarray.to_gpu(a)
	b_gpu = gpuarray.to_gpu(b)

	start = time.time()
	c_gpu = culinalg.dot(a_gpu, b_gpu)
	time_linalg.append(time.time() - start)


	a_gpu2 = gpuarray.to_gpu(a)
	b_gpu2 = gpuarray.to_gpu(b)
	c_gpu2 = gpuarray.empty((n, n), np.float32)

	


	h = cublasCreate()

	start = time.time()
	#cublasSgemm(h, 'n', 'n', np.int32(n), np.int32(n), np.int32(n), np.float32(1.0), a_gpu2, np.int32(n), b_gpu2, np.int32(n), np.float32(1.0), c_gpu2, np.int32(n))
	cublasSgemm(h, 'n', 'n', a.shape[0], a.shape[0], a.shape[0], 1.0, a_gpu2.gpudata, a.shape[0], b_gpu2.gpudata, a.shape[0], 1.0, c_gpu2.gpudata, a.shape[0])
	time_cula.append(time.time() - start)
	
	cublasDestroy(h)


	#cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
	

MAKE_PLOT = True
if MAKE_PLOT:
  import matplotlib as mpl
  mpl.use('agg')
  import matplotlib.pyplot as plt

  plt.gcf()
  plt.subplot(311)
  plt.plot(N, time_cpu,'r')
  plt.plot(N, time_linalg,'g')
  plt.plot(N, time_cula,'b')
  plt.legend(['cpu time', 'gpu linalg', 'gpu cula'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('three coding times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.subplot(312)
  plt.plot(N, time_linalg,'g')
  plt.plot(N, time_cula,'b')
  plt.legend(['gpu linalg', 'gpu cula'], loc='upper left')
  plt.xlabel('matrix ratio increase factor')
  plt.ylabel('two coding times')
  plt.gca().set_xlim((min(N), max(N)))

  plt.savefig('matrixmul.png')